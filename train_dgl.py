import uuid
from functools import partial
from itertools import product
import time
import numpy as np
# from torch_geometric.nn import TAGConv, SAGEConv, GraphConv, ChebConv
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import (AGNNConv, APPNPConv, ChebConv, GATConv, GINConv,
                            GraphConv, SAGEConv, SGConv, TAGConv, EdgeConv, GatedGraphConv, GMMConv)
from torch.nn import Linear, ReLU, Sequential

from ag.pyg_utils import generate_pyg_data
from ag.worker_executor import Executor
from data_utils import read_dataset
# from torch_geometric.nn import (ARMAConv, ChebConv, FeaStConv, GATConv,
#                                 GatedGraphConv, GCNConv, GINConv, GraphConv,
#                                 GravNetConv, RGCNConv, SAGEConv, SGConv,
#                                 SplineConv, TAGConv)
from torch_geometric.utils import dropout_adj, to_networkx
from uxils.serialization import dump
from uxils.timer import Timer


class GraphNet(nn.Module):
    def __init__(self, input_size, n_classes, conv_class, in_dropout, out_dropout, n_hidden, n_layers, activation):
        super().__init__()

        self.layers = nn.ModuleList()
        self.in_nn = nn.Linear(input_size, n_hidden)

        for _ in range(n_layers):
            if 'GINConv' in conv_class.func.__name__:
                self.layers.append(GINConv(
                    nn.Sequential(nn.Dropout(0.5), nn.Linear(n_hidden, n_hidden), nn.ReLU()),
                    'mean'))
            else:
                self.layers.append(conv_class(n_hidden, n_hidden))

        self.out_nn = nn.Linear(n_hidden, n_classes)
        self.in_dropout = nn.Dropout(in_dropout)
        self.out_dropout = nn.Dropout(out_dropout)
        self.activation = activation()

    def forward(self, x, g, data):
        x = self.in_nn(x)
        x = self.activation(x)
        x = self.in_dropout(x)

        for layer in self.layers:
            x = layer(g, x)
            # x = layer(x, data.edge_index)
            x = self.activation(x).squeeze()  # GAT num heads (b, H, d)

        x = self.out_dropout(x)
        x = self.out_nn(x)
        return x


while True:

    task = np.random.choice(['a', 'b', 'c', 'd', 'e'])
    dataset, y_test = read_dataset(task)
    n_classes = dataset.get_metadata()['n_class']
    gdata = generate_pyg_data(dataset.get_data())
    input_size = gdata.x.shape[1]
    g = DGLGraph(to_networkx(gdata))

    gpt = 0.3 if task in ['a', 'b', 'e'] else 1

    # degs = g.in_degrees().float()
    # norm = torch.pow(degs, -0.5)
    # norm[torch.isinf(norm)] = 0
    # norm = norm.cuda()
    # g.ndata['norm'] = norm.unsqueeze(1)

    def func(config):
        model = GraphNet(
            input_size=input_size, n_classes=n_classes, conv_class=config['conv_class'],
            in_dropout=config['in_dropout'], out_dropout=config['out_dropout'], n_layers=config['n_layers'],
            n_hidden=config['hidden_size'], activation=config['activation']
            ).cuda()
        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = config['optimizer'](model.parameters(), lr=config['lr'], weight_decay=config['wd'])
        data = gdata.to(torch.device('cuda'))

        start_time = time.time()
        results = []
        for epoch in range(700+1):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, g, data)
            loss = loss_fcn(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch in [100, 300, 500, 700]:
                with torch.no_grad():
                    model.eval()
                    out = model(data.x, g, data)[data.test_mask].max(1)[1].cpu().numpy()
                    test_acc = (y_test.flatten() == out.flatten()).mean()
                    results.append((epoch, test_acc))

        return (results, time.time() - start_time)

    ray.init(
        num_gpus=2, num_cpus=20, memory=2e10, object_store_memory=2e10,
        configure_logging=False, ignore_reinit_error=True,
        log_to_driver=False,
        include_webui=False
    )
    n = 2 if gpt == 1 else 6
    executor = Executor(n, gpu_per_trial=gpt)

    search_space = {
        'conv_class': [
            partial(GraphConv, norm='both'),
            partial(GraphConv, norm='none'),

            partial(TAGConv, k=1),
            partial(TAGConv, k=2),
            partial(TAGConv, k=4),
            partial(TAGConv, k=7),

            partial(GATConv, num_heads=1),

            partial(EdgeConv, batch_norm=True),
            partial(EdgeConv, batch_norm=False),

            partial(SAGEConv, aggregator_type='mean'),
            partial(SAGEConv, aggregator_type='gcn', feat_drop=0.5),
            partial(SAGEConv, aggregator_type='gcn'),

            partial(SGConv, k=1),
            partial(SGConv, k=3),
            partial(SGConv, k=5),

            partial(GINConv, aggregator_type='sum'),
            partial(GINConv, aggregator_type='mean'),
        ],
        'n_layers': [1, 2],
        'hidden_size': [32, 64, 96],
        'in_dropout': [0.5],
        'out_dropout': [0.5],
        'wd': [0, 1e-3],
        'lr': [0.001, 0.01],
        'optimizer': [torch.optim.Adam, torch.optim.SGD],
        'activation': [nn.ReLU, nn.GELU, nn.SELU, nn.Tanh, nn.Tanhshrink]
    }

    SEARCH_SPACE_FLAT = [dict(zip(search_space.keys(), x)) for x in product(*search_space.values())]
    np.random.shuffle(SEARCH_SPACE_FLAT)
    print(len(SEARCH_SPACE_FLAT))
    out_path = f'dgl_task_{task}_{uuid.uuid4()}.pkl'

    idx = 0
    results = []
    st = time.time()
    while len(results) != len(SEARCH_SPACE_FLAT) and ((time.time() - st) < 60*60):
        if idx < len(SEARCH_SPACE_FLAT):
            config = SEARCH_SPACE_FLAT[idx]
            idx += 1
            name = f'{config["conv_class"]} {config}'
            executor.apply(partial(func, config=config), name)

        r = executor.get(timeout=2)
        if r is None:
            continue

        results.append(r)
        print(task, len(results), r)
        dump(results, out_path)

    executor.stop()
    ray.shutdown(True)
    time.sleep(5)
    # TODO APPNPConv Monet

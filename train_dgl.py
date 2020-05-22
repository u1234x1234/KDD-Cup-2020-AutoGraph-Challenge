import time
import uuid
from functools import partial
from itertools import product

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import pytorch as dgl_layers
from torch.nn import Linear, ReLU, Sequential

from ag.graph_net import GraphNet
from ag.pyg_utils import generate_pyg_data
from ag.worker_executor import Executor
from data_utils import read_dataset
from torch_geometric import nn as pyg_layers
from torch_geometric.utils import dropout_adj, to_networkx
from uxils.serialization import dump
from uxils.timer import Timer
from uxils.torch_ext import (available_activations, available_optimizers,
                             init_optimizer)

from uxils.system import suppres_all_output


while True:

    task = np.random.choice(['a', 'b', 'c', 'd', 'e'])
    # task = 'a'
    dataset, y_test = read_dataset(task)
    n_classes = dataset.get_metadata()['n_class']
    gdata = generate_pyg_data(dataset.get_data())
    input_size = gdata.x.shape[1]
    g = DGLGraph(to_networkx(gdata))
    # edges = g.reverse().all_edges()
    # g.add_edges(*edges)

    gpt = 0.25 if task in ['a', 'b', 'e'] else 1

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
        optimizer = init_optimizer(config['optimizer'])(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
        data = gdata.to(torch.device('cuda'))

        start_time = time.time()
        results = []
        for epoch in range(700+1):
            model.train()
            optimizer.zero_grad()
            logits = model(g, data)
            mask = data.train_mask + data.val_mask
            loss = loss_fcn(logits[mask], data.y[mask])
            loss.backward()
            optimizer.step()
            if epoch in [100, 300, 500, 700]:
                with torch.no_grad():
                    model.eval()
                    out = model(g, data)[data.test_mask].max(1)[1].cpu().numpy()
                    test_acc = (y_test.flatten() == out.flatten()).mean()
                    results.append((epoch, test_acc))

        return (results, time.time() - start_time)

    ray.init(
        num_gpus=2, num_cpus=20, memory=2e10, object_store_memory=2e10,
        configure_logging=False, ignore_reinit_error=True,
        log_to_driver=True,
        include_webui=False
    )
    n = 2 if gpt == 1 else 8
    executor = Executor(n, gpu_per_trial=gpt)

    search_space = {
        'conv_class': [
            partial(dgl_layers.GraphConv, norm='both'),
            partial(dgl_layers.GraphConv, norm='none'),

            partial(dgl_layers.TAGConv, k=1),
            partial(dgl_layers.TAGConv, k=3),
            partial(dgl_layers.TAGConv, k=4),
            # partial(pyg_layers.TAGConv, K=4, normalize=False),
            partial(dgl_layers.TAGConv, k=5),

            partial(dgl_layers.GATConv, num_heads=1),

            # partial(dgl_layers.EdgeConv, batch_norm=True),
            # partial(EdgeConv, batch_norm=False),

            partial(dgl_layers.SAGEConv, aggregator_type='mean'),
            partial(dgl_layers.SAGEConv, aggregator_type='gcn', feat_drop=0.5),
            partial(dgl_layers.SAGEConv, aggregator_type='gcn'),

            partial(dgl_layers.SGConv, k=1),
            partial(dgl_layers.SGConv, k=3),
            partial(dgl_layers.SGConv, k=5),

            partial(dgl_layers.GINConv, aggregator_type='sum'),
            partial(dgl_layers.GINConv, aggregator_type='mean'),

            # partial(dgl_layers.GatedGraphConv, n_steps=2, n_etypes=1),
            partial(dgl_layers.ChebConv, k=7),
            partial(dgl_layers.AGNNConv, learn_beta=True),

            # partial(dgl_layers.APPNPConv, k=10, alpha=0.1, edge_drop=0),
            # partial(dgl_layers.APPNPConv, k=10, alpha=0.1, edge_drop=0.5),
            # partial(dgl_layers.APPNPConv, k=10, alpha=0.5, edge_drop=0),
        ],
        'n_layers': [1, 2, 3],
        'hidden_size': [32, 64, 96],
        'in_dropout': [0.5],
        'out_dropout': [0.5],
        'wd': [1e-3, 0],
        'lr': [0.01, 0.001],
        'optimizer': available_optimizers(),
        'activation': available_activations(),
    }

    SEARCH_SPACE_FLAT = [dict(zip(search_space.keys(), x)) for x in product(*search_space.values())]
    np.random.shuffle(SEARCH_SPACE_FLAT)
    print(len(SEARCH_SPACE_FLAT))
    out_path = f'dgl2_task_{task}_{uuid.uuid4()}.pkl'

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
        # print(r[1][0][-1][1], '\n', r)
        print(task, len(results))
        dump(results, out_path)

    with suppres_all_output():
        executor.stop()
        ray.shutdown(True)
        time.sleep(5)

import os
import random
from functools import partial
import copy
import time
import numpy as np
import torch
from dgl import DGLGraph
from dgl.nn import pytorch as dgl_layers
from torch import optim

from torch_geometric import nn as pyg_layers
from torch_geometric.utils import to_networkx

from .graph_net import GraphNet
from .module_utils import init_optimizer


def bc(**kwargs):
    base = {
        'in_dropout': 0.5,
        'out_dropout': 0.5,
        'wd': 0.001,
        'activation': 'relu',
        'optimizer': 'adam',
    }
    base.update(**kwargs)
    return base


# SEARCH_SPACE_FLAT = [
    # bc(conv_class=partial(dgl_layers.TAGConv, k=4), n_layers=1, n_iter=700, lr=0.01, hidden_size=64, wd=0, activation='tanh', optimizer='adam'),
    # bc(conv_class=partial(pyg_layers.TAGConv, K=5), hidden_size=64, n_layers=1, n_iter=70, lr=0.001),
    # bc(conv_class=partial(pyg_layers.SGConv, K=2), hidden_size=64, n_layers=1, n_iter=150, lr=0.001),
    # bc(conv_class=partial(pyg_layers.SAGEConv, normalize=True), hidden_size=128, n_layers=1, n_iter=500, lr=0.001, wd=0),
    # bc(conv_class=partial(pyg_layers.GraphConv, aggr='add'), hidden_size=64, n_layers=1, n_iter=200, lr=0.001),

    # bc(conv_class=partial(pyg_layers.GATConv), hidden_size=64, n_layers=1, n_iter=200, lr=0.001),
    # bc(conv_class=partial(pyg_layers.GraphConv, aggr='add'), hidden_size=64, n_layers=1, n_iter=200, lr=0.001),
    # bc(conv_class=partial(dgl_layers.SAGEConv, aggregator_type='gcn', feat_drop=0.5), hidden_size=96, wd=0, n_layers=2, n_iter=700, lr=0.01, optimizer='adamw', activation='elu'),
    # bc(conv_class=partial(dgl_layers.TAGConv, k=1), n_layers=3, n_iter=500, lr=0.01, hidden_size=64, wd=0, activation='selu', optimizer='adamw'),
    # bc(conv_class=partial(dgl_layers.GINConv, aggregator_type='sum'), hidden_size=64, n_layers=1, n_iter=200, lr=0.001),
    # bc(conv_class=partial(dgl_layers.SGConv, k=5), hidden_size=96, n_layers=1, n_iter=500, lr=0.01, wd=0, optimizer='adam', activation='softsign'),
    # bc(conv_class=partial(dgl_layers.GraphConv, norm='both'), hidden_size=64, n_layers=3, n_iter=200, lr=0.01, optimizer='adamax', activation='leakyrelu'),
    # bc(conv_class=partial(dgl_layers.TAGConv, k=3), hidden_size=256, n_layers=1, n_iter=100, wd=0.01, lr=0.001, optimizer='adamax', activation='prelu'),
    # bc(conv_class=partial(pyg_layers.ChebConv, K=9), hidden_size=64, n_layers=1, n_iter=500, lr=0.001),

    # bc(conv_class=partial(dgl_layers.AGNNConv, learn_beta=True), hidden_size=64, n_layers=2, n_iter=200, lr=0.001),

    # bc(conv_class=partial(dgl_layers.TAGConv, k=4), n_layers=2, n_iter=500, lr=0.01, hidden_size=64, wd=1e-3, activation='selu', optimizer='sgd'),
    # bc(conv_class=partial(dgl_layers.TAGConv, k=4), n_layers=2, n_iter=500, lr=0.01, hidden_size=32, wd=0, activation='selu', optimizer='sgd'),
    # bc(conv_class=partial(dgl_layers.TAGConv, k=1), n_layers=2, n_iter=500, lr=0.01, hidden_size=64, wd=0, activation='selu', optimizer='sgd'),
# ]


SEARCH_SPACE_FLAT = [
    bc(conv_class=partial(pyg_layers.SAGEConv, normalize=True), hidden_size=96, n_layers=2, n_iter=500, lr=0.01, wd=0), # 1
    bc(conv_class=partial(dgl_layers.TAGConv, k=4), n_layers=1, n_iter=700, lr=0.01, hidden_size=48, wd=0, activation='tanh', optimizer='adam'), # 1, 2, 3
    bc(conv_class=partial(pyg_layers.GraphConv, aggr='add'), hidden_size=64, n_layers=1, n_iter=200, lr=0.001), # 2?, 3, 4
    bc(conv_class=partial(pyg_layers.GraphConv, aggr='add'), hidden_size=64, n_layers=2, n_iter=300, lr=0.01, wd=0, optimizer='adamw', activation='elu'), # 5, 1

    bc(conv_class=partial(dgl_layers.SGConv, k=5), hidden_size=96, n_layers=1, n_iter=500, lr=0.01, wd=0, optimizer='adam', activation='softsign'), # 4?

    # bc(conv_class=partial(dgl_layers.TAGConv, k=4), n_layers=2, n_iter=700, lr=0.01, hidden_size=64, wd=1e-3, activation='selu', optimizer='sgd'),

    # bc(conv_class=partial(pyg_layers.TAGConv, K=5), hidden_size=64, n_layers=1, n_iter=100, lr=0.001),
    # bc(conv_class=partial(pyg_layers.SGConv, K=2), hidden_size=32, n_layers=1, n_iter=100, lr=0.001),
    # bc(conv_class=partial(pyg_layers.GraphConv, aggr='mean'), hidden_size=64, n_layers=1, n_iter=100, lr=0.01),
    # bc(conv_class=partial(dgl_layers.TAGConv, k=4), n_layers=2, n_iter=700, lr=0.01, hidden_size=64, wd=1e-3, activation='selu', optimizer='sgd'),
    # bc(conv_class=partial(dgl_layers.TAGConv, k=4), n_layers=2, n_iter=700, lr=0.01, hidden_size=32, wd=0, activation='selu', optimizer='sgd'),
    # bc(conv_class=partial(dgl_layers.TAGConv, k=1), n_layers=2, n_iter=700, lr=0.01, hidden_size=64, wd=0, activation='selu', optimizer='sgd'),
]


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class PYGModel:

    def __init__(self, n_classes, conv_class, hidden_size, n_layers, in_dropout, out_dropout,
                 n_iter, wd, lr, optimizer, activation):
        self.conv_class = conv_class
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = torch.device('cuda')
        self.n_iter = n_iter
        self.n_classes = n_classes
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        self.lr = lr
        self.wd = wd
        self.optimizer_str = optimizer
        self.activation = activation
        self.model = None

    def init_model(self, data):
        input_size = data.x.shape[1]
        self.model = GraphNet(
            input_size=input_size, n_classes=self.n_classes, n_nodes=len(data.x),
            n_layers=self.n_layers, conv_class=self.conv_class, n_hidden=self.hidden_size,
            in_dropout=self.in_dropout, out_dropout=self.out_dropout, activation=self.activation)
        self.model = self.model.to(self.device)
        self.optimizer = init_optimizer(self.optimizer_str)(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, data, g, mask, n_iter):
        self.model.train()
        st = time.time()
        for epoch_idx in range(n_iter):
            self.optimizer.zero_grad()
            out = self.model(g, data)
            loss = self.criterion(out[mask], data.y[mask])
            loss.backward()
            self.optimizer.step()
            if (time.time() - st) > 70:
                break

    def predict(self, data, g, mask=None):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(g, data)
            if mask is not None:
                pred = pred[mask]
        return pred

    def fit_predict(self, data, g):
        data = data.to(self.device)
        g = g.to(self.device)

        # Train + evaluate score on validation
        scores = []
        preds = []
        for train_idx, val_idx in data.cv:
            self.init_model(data)

            train_mask = torch.zeros(len(data.x), dtype=torch.bool)
            train_mask[np.array(data.train_indices)[train_idx]] = 1
            val_mask = torch.zeros(len(data.x), dtype=torch.bool)
            val_mask[np.array(data.train_indices)[val_idx]] = 1
            train_mask = train_mask.to(self.device)
            val_mask = val_mask.to(self.device)

            self.train(data, g, mask=train_mask, n_iter=self.n_iter)
            y_val_pred = self.predict(data, g, mask=val_mask).argmax(1)
            score = (y_val_pred == data.y[val_mask]).sum().cpu().numpy() / len(y_val_pred)
            scores.append(score)

            preds.append(torch.nn.functional.softmax(self.predict(data, g, mask=data.test_mask), dim=1))
            # for _ in range(3):
            #     self.train(data, g, mask=train_mask+val_mask, n_iter=5)
            #     preds.append(torch.nn.functional.softmax(self.predict(data, g, mask=data.test_mask), dim=1))

        score = np.mean(scores)

        # self.train(data, g, mask=train_mask+val_mask)
        # self.train(data, g, mask=data.train_mask + data.val_mask)
        # pred = self.predict(data, g, mask=data.test_mask)
        # pred = sum(preds)
        # self.n_iter = 1
        # for i in range(5):
        #     self.train(data, g, mask=data.train_mask + data.val_mask)
        #     pred += self.predict(data, g, mask=data.test_mask)
        # pred /= 6

        # t_pred = self.predict(data, mask=None)
        # t_pred = F.softmax(t_pred, dim=1).cpu().numpy()
        # test_mask = ~(data.train_mask + data.val_mask).cpu().numpy()
        # perc = np.percentile(t_pred[test_mask].max(axis=1), 15)
        # nmask = ((t_pred * test_mask[:, np.newaxis]).max(axis=1) > perc)
        # nmask = torch.tensor(nmask, dtype=torch.bool).cuda()
        # data.train_mask += nmask
        # data.y[test_mask] = torch.tensor(t_pred[test_mask].argmax(axis=1), dtype=torch.long).cuda()

        # self.n_iter = 5
        # self.train(data, mask=data.train_mask+data.val_mask)
        # pred1 = self.predict(data, mask=data.test_mask).cpu().numpy()
        # pred = (pred + pred1) / 3

        pred = torch.stack(preds).cpu().numpy()
        return pred, score


def create_factory_method(n_classes):

    def create_model(**config):
        return PYGModel(
            n_classes,
            conv_class=config['conv_class'], hidden_size=config['hidden_size'],
            n_layers=config['n_layers'], in_dropout=config['in_dropout'], out_dropout=config['out_dropout'],
            n_iter=config['n_iter'], wd=config['wd'], lr=config['lr'],
            activation=config['activation'], optimizer=config['optimizer'],
            )

    return create_model

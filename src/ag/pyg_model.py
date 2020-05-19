import os
import random
from functools import partialmethod

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import (ARMAConv, ChebConv, FeaStConv, GATConv,
                                GCNConv, GINConv, GraphConv, TAGConv,
                                JumpingKnowledge, RGCNConv, SAGEConv, SGConv,
                                SplineConv)

from .pyg_utils import generate_pyg_data
from itertools import product


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    NewCls.__name__ = f'{cls.__name__}[{kwargs}]'
    return NewCls


SEARCH_SPACE = {
    'conv_class': [
        # GCNConv,
        # partialclass(GCNConv, improved=True),
        # partialclass(GCNConv, normalize=True),

        # partialclass(ChebConv, K=3),
        # partialclass(ChebConv, K=7),

        # partialclass(TAGConv, K=3),
        # partialclass(TAGConv, K=5),

        # partialclass(SGConv, K=2),
        # partialclass(ARMAConv, num_layers=3, num_stacks=2),
        # partialclass(GraphConv, aggr='mean'),

        # partialclass(SAGEConv),
        partialclass(SAGEConv, normalize=True),
    ],
    'hidden_size': [96],
    'num_layers': [2],
    'in_dropout': [0.7],
    'out_dropout': [0.7],
    'n_iter': [500],
    'weight_decay': [0.001],
    'learning_rate': [0.01],
}

# SEARCH_SPACE_FLAT = [dict(zip(SEARCH_SPACE.keys(), x)) for x in product(*SEARCH_SPACE.values())]
# np.random.shuffle(SEARCH_SPACE_FLAT)


def bc(**kwargs):
    base = {
        'in_dropout': 0.5,
        'out_dropout': 0.5,
        'weight_decay': 0.001,
    }
    base.update(**kwargs)
    return base


SEARCH_SPACE_FLAT = [
    bc(conv_class=partialclass(TAGConv, K=5), hidden_size=64, num_layers=1, n_iter=100, learning_rate=0.001),
    bc(conv_class=partialclass(SGConv, K=2), hidden_size=32, num_layers=1, n_iter=100, learning_rate=0.001),
    bc(conv_class=partialclass(SAGEConv, normalize=True), hidden_size=96, num_layers=2, n_iter=500, learning_rate=0.01, weight_decay=0),
    bc(conv_class=partialclass(GraphConv, aggr='mean'), hidden_size=64, num_layers=1, n_iter=100, learning_rate=0.01),
]


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


WITH_EDGE_WEIGHTS = ['ChebConv', 'GCNConv', 'SAGEConv', 'GraphConv', 'TAGConv', 'ARMAConv']


def with_edge_weights(conv_mod):
    for c in WITH_EDGE_WEIGHTS:
        if f'.{c}' in str(type(conv_mod)):
            return True
    return False


class GCN(torch.nn.Module):

    def __init__(self, *, conv_class, num_layers, hidden, features_num=32, num_class=2, in_dropout=0, out_dropout=0):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(conv_class(hidden, hidden))

        self.in_nn = Linear(features_num, hidden)
        self.out_nn = Linear(hidden, num_class)

        self.in_drop = torch.nn.Dropout(in_dropout)
        self.out_drop = torch.nn.Dropout(out_dropout)

    def reset_parameters(self):
        self.in_nn.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = F.relu(self.in_nn(x))
        x = self.in_drop(x)

        for conv in self.convs:
            if with_edge_weights(conv):
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)

            x = F.relu(x)

        x = self.out_drop(x)
        x = self.out_nn(x)

        return x

    def __repr__(self):
        return self.__class__.__name__


class PYGModel:

    def __init__(self, n_classes, conv_class, hidden_size, num_layers, in_dropout, out_dropout,
                 n_iter, weight_decay, learning_rate):
        self.conv_class = conv_class
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device('cuda')
        self.n_iter = n_iter
        self.n_classes = n_classes
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = None

    def train(self, data, mask):
        if self.model is None:
            input_size = data.x.shape[1]
            self.model = GCN(
                features_num=input_size, num_class=self.n_classes,
                num_layers=self.num_layers, conv_class=self.conv_class, hidden=self.hidden_size,
                in_dropout=self.in_dropout, out_dropout=self.out_dropout)
            self.model = self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            # import torchcontrib
            # self.optimizer = torchcontrib.optim.SWA(self.optimizer)

        self.model.train()
        val_accs = []
        for epoch_idx in range(self.n_iter):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = F.cross_entropy(out[mask], data.y[mask])
            loss.backward()
            self.optimizer.step()

            if epoch_idx > 50 and epoch_idx % 10 == 0:
                p = self.predict(data, mask=data.val_mask).max(1)[1]
                acc = (p == data.y[data.val_mask]).sum().cpu().numpy() / len(p)
                val_accs.append(acc)
                # if np.max(val_accs) != np.max(val_accs[-5:]):  # no impr in last steps
                    # break

            # if epoch_idx > 100 and epoch_idx % 10 == 0:
                # self.optimizer.update_swa()
        # self.optimizer.swap_swa_sgd()

        return np.mean(val_accs[-3:])

    def predict(self, data, mask=None):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(data)
            if mask is not None:
                pred = pred[mask]
        return pred

    def fit_predict(self, data):
        data = data.to(self.device)

        score = self.train(data, mask=data.train_mask)
        self.n_iter = 10
        self.train(data, mask=data.train_mask + data.val_mask)
        pred = self.predict(data, mask=data.test_mask)
        self.n_iter = 1
        for i in range(10):
            self.train(data, mask=data.train_mask + data.val_mask)
            pred += self.predict(data, mask=data.test_mask)
        pred /= 11

        pred = pred.cpu().numpy()

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

        return pred, score


def create_factory_method(n_classes):

    def create_model(**config):
        return PYGModel(
            n_classes,
            conv_class=config['conv_class'], hidden_size=config['hidden_size'],
            num_layers=config['num_layers'], in_dropout=config['in_dropout'], out_dropout=config['out_dropout'],
            n_iter=config['n_iter'], weight_decay=config['weight_decay'], learning_rate=config['learning_rate']
            )

    return create_model

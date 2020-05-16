import os
import random
from functools import partialmethod

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (ARMAConv, ChebConv, GCNConv, GINConv, SAGEConv,
                                JumpingKnowledge, SGConv, SplineConv)
from torch_geometric.nn import SAGEConv, SplineConv, GraphConv, GravNetConv, GINConv, ARMAConv, SGConv, RGCNConv, FeaStConv, GATConv

from .pyg_utils import generate_pyg_data


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    NewCls.__name__ = f'{cls.__name__}[{kwargs}]'
    return NewCls


SEARCH_SPACE = {
    'conv_class': [
        GCNConv,
        partialclass(GCNConv, improved=True),
        partialclass(GCNConv, normalize=True),

        partialclass(ChebConv, K=3),
        partialclass(ChebConv, K=7),

        # partialclass(SAGEConv, concat=True),
        partialclass(SAGEConv, normalize=True),
    ],
    'hidden_size': [64, 96],
    'num_layers': [2],
}


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# fix_seed(12345)


WITH_EDGE_WEIGHTS = ['ChebConv', 'GCNConv']



def with_edge_weights(conv_mod):
    for c in WITH_EDGE_WEIGHTS:
        if f'.{c}' in str(type(conv_mod)):
            return True
    return False


class GCN(torch.nn.Module):

    def __init__(self, *, conv_class, num_layers, hidden, features_num=32, num_class=2, in_dropout=0, out_dropout=0):
        super().__init__()
        # self.convs = torch.nn.ModuleList()
        # for i in range(num_layers - 1):
            # self.convs.append(conv_class(32, 64))

        # self.lin2 = Linear(128, 64)
        self.lin3 = Linear(100, num_class)

        # self.lin2 = Linear(hidden, num_class)

        self.first_lin = Linear(features_num, 32)
        self.in_drop = torch.nn.Dropout(in_dropout)
        self.out_drop = torch.nn.Dropout(out_dropout)
        # self.bn = torch.nn.BatchNorm1d(64)
        self.conv1 = ChebConv(32, 50, K=7)
        self.conv2 = ARMAConv(32, 50, dropout=0.5, num_layers=2, num_stacks=2)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = self.in_drop(x)
        # for conv in self.convs:
        #     if with_edge_weights(conv):
        #         x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
        #     else:
        #         x = F.relu(conv(x, edge_index))
        x1 = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x2 = F.relu(self.conv2(x, edge_index))

        x = torch.cat([x1, x2], dim=1)

        # x = F.relu(self.lin2(x))
        x = self.out_drop(x)
        x = self.lin3(x)

        return x

    def __repr__(self):
        return self.__class__.__name__


class Restorable:
    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))

    def restore(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt')))


class PYGModel(Restorable):

    def __init__(self, n_classes, input_size, conv_class, hidden_size, num_layers, in_dropout, out_dropout, n_iter):
        self.conv_class = conv_class
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device('cuda')
        self.n_iter = n_iter

        self.model = GCN(
            features_num=input_size, num_class=n_classes,
            num_layers=self.num_layers, conv_class=self.conv_class, hidden=self.hidden_size,
            in_dropout=in_dropout, out_dropout=out_dropout,
        )
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-3)

    def train(self, data, full=False, m=None):
        data = data.to(self.device)

        self.model.train()
        for epoch in range(1, self.n_iter):
            self.optimizer.zero_grad()
            if full:
                mask = data.train_mask + data.val_mask
            else:
                mask = data.train_mask

            loss = F.cross_entropy(self.model(data)[mask], data.y[mask])
            loss.backward()
            self.optimizer.step()

        print(mask.sum())
        with torch.no_grad():
            self.model.eval()
            y_pred_val = self.model(data).cpu().numpy()
        # y_val_true = data.y[data.val_mask].cpu().numpy()
        # val_acc = (y_pred_val == y_val_true).mean()

        return y_pred_val

    def predict(self, data):
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            pred = self.model(data)[data.test_mask]
        return pred.cpu().numpy()

    def fit_predict(self, data, full=False, m=None):

        score = self.train(data, full=full, m=m)
        pred = self.predict(data)

        return pred, score


def create_factory_method(n_classes, input_size):

    def create_model(**config):
        return PYGModel(
            n_classes, input_size=input_size,
            conv_class=config['conv_class'], hidden_size=config['hidden_size'],
            num_layers=config['num_layers'])

    return create_model

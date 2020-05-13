import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (ARMAConv, ChebConv, GCNConv, GINConv,
                                JumpingKnowledge, SGConv, SplineConv)

# from uxils.functools_ext import partialclass

from .pyg_utils import generate_pyg_data

SEARCH_SPACE = {
    'conv_class': [
        GCNConv,
    ],
    'hidden_size': [32, 64],
    'num_layers': [2],
}


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


fix_seed(12345)


WITH_EDGE_WEIGHTS = ['ChebConv', 'GCNConv']


def with_edge_weights(conv_mod):
    for c in WITH_EDGE_WEIGHTS:
        if c in str(type(conv_mod)):
            return True
    return False


class GCN(torch.nn.Module):

    def __init__(self, *, conv_class, num_layers, hidden, features_num=32, num_class=2):
        super(GCN, self).__init__()
        self.conv1 = conv_class(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(conv_class(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.8, training=self.training)
        for conv in self.convs:
            if with_edge_weights(conv):
                x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
            else:
                x = F.relu(conv(x, edge_index))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

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

    def __init__(self, conv_class, hidden_size, num_layers):
        self.conv_class = conv_class
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device('cuda')

        self.model = GCN(
            features_num=data.x.size()[1], num_class=int(max(data.y)) + 1,
            num_layers=self.num_layers, conv_class=self.conv_class, hidden=self.hidden_size,
        )
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)

    def train(self, data):
        data = data.to(self.device)

        min_loss = float('inf')
        self.model.train()
        for epoch in range(1, 600):
            self.optimizer.zero_grad()
            loss = F.nll_loss(self.model(data)[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()

        y_pred_val = self.model(data)[data.val_mask].max(1)[1].cpu().numpy()
        y_val_true = data.y[data.val_mask].cpu().numpy()

        val_acc = (y_pred_val == y_val_true).mean()

        return val_acc

    def predict(self, data):
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            pred = self.model(data)[data.test_mask].max(1)[1]
        return pred.cpu().numpy().flatten()

    def fit_predict(self, data):

        score = self.train(data)
        pred = self.predict(data)

        return pred, score

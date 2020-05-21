import os
import random
from functools import partialmethod, partial
from itertools import product

import numpy as np
import torch
from torch import optim
from dgl import DGLGraph
from dgl.nn import pytorch as dgl_layers

from torch_geometric import nn as pyg_layers
from torch_geometric.utils import to_networkx

from .graph_net import GraphNet


def is_subclass(obj, classinfo):
    try:
        return issubclass(obj, classinfo)
    except Exception:
        pass
    return False


OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'adamw': optim.AdamW,
}


def init_optimizer(optimizer):
    if isinstance(optimizer, str) and optimizer.lower() in OPTIMIZERS:
        return OPTIMIZERS[optimizer.lower()]
    if is_subclass(optimizer, optim.Optimizer):
        return optimizer
    raise ValueError('No such optimizer: "{}"'.format(optimizer))


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    NewCls.__name__ = f'{cls.__name__}[{kwargs}]'
    return NewCls


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


SEARCH_SPACE_FLAT = [
    bc(conv_class=partial(dgl_layers.TAGConv, k=4), n_layers=1, n_iter=700, lr=0.01, hidden_size=32, wd=0, activation='tanh', optimizer='adam'),
    bc(conv_class=partial(pyg_layers.TAGConv, K=5), hidden_size=64, n_layers=1, n_iter=100, lr=0.001),
    bc(conv_class=partial(pyg_layers.SGConv, K=2), hidden_size=32, n_layers=1, n_iter=100, lr=0.001),
    bc(conv_class=partial(pyg_layers.SAGEConv, normalize=True), hidden_size=96, n_layers=2, n_iter=500, lr=0.01, wd=0),
    bc(conv_class=partial(pyg_layers.GraphConv, aggr='mean'), hidden_size=64, n_layers=1, n_iter=100, lr=0.01),
    bc(conv_class=partial(dgl_layers.TAGConv, k=4), n_layers=2, n_iter=700, lr=0.01, hidden_size=64, wd=1e-3, activation='selu', optimizer='sgd'),
    bc(conv_class=partial(dgl_layers.TAGConv, k=4), n_layers=2, n_iter=700, lr=0.01, hidden_size=32, wd=0, activation='selu', optimizer='sgd'),
    bc(conv_class=partial(dgl_layers.TAGConv, k=1), n_layers=2, n_iter=700, lr=0.01, hidden_size=64, wd=0, activation='selu', optimizer='sgd'),
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
        self.optimizer = optimizer
        self.activation = activation
        self.model = None

    def train(self, data, g, mask):
        if self.model is None:
            input_size = data.x.shape[1]
            self.model = GraphNet(
                input_size=input_size, n_classes=self.n_classes,
                n_layers=self.n_layers, conv_class=self.conv_class, n_hidden=self.hidden_size,
                in_dropout=self.in_dropout, out_dropout=self.out_dropout, activation=self.activation)

            self.model = self.model.to(self.device)
            self.optimizer = init_optimizer(self.optimizer)(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
            self.criterion = torch.nn.CrossEntropyLoss()
            # import torchcontrib
            # self.optimizer = torchcontrib.optim.SWA(self.optimizer)

        self.model.train()
        # val_accs = []
        for epoch_idx in range(self.n_iter):
            self.optimizer.zero_grad()
            out = self.model(g, data)
            loss = self.criterion(out[mask], data.y[mask])
            loss.backward()
            self.optimizer.step()

            # if epoch_idx > 50 and epoch_idx % 10 == 0:
            #     p = self.predict(data, g, mask=data.val_mask).max(1)[1]
            #     acc = (p == data.y[data.val_mask]).sum().cpu().numpy() / len(p)
            #     val_accs.append(acc)
            #     # if np.max(val_accs) != np.max(val_accs[-5:]):  # no impr in last steps
                    # break
            # if epoch_idx > 100 and epoch_idx % 10 == 0:
                # self.optimizer.update_swa()
        # self.optimizer.swap_swa_sgd()

    def predict(self, data, g, mask=None):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(g, data)
            if mask is not None:
                pred = pred[mask]
        return pred

    def fit_predict(self, data):
        data = data.to(self.device)
        g = DGLGraph(to_networkx(data))

        # Train + evaluate score on validation
        self.train(data, g, mask=data.train_mask)
        y_val_pred = self.predict(data, g, mask=data.val_mask).argmax(1)
        score = (y_val_pred == data.y[data.val_mask]).sum().cpu().numpy() / len(y_val_pred)

        # Refit on all data
        self.n_iter = 30
        self.train(data, g, mask=data.train_mask + data.val_mask)
        pred = self.predict(data, g, mask=data.test_mask)
        self.n_iter = 1
        for i in range(5):
            self.train(data, g, mask=data.train_mask + data.val_mask)
            pred += self.predict(data, g, mask=data.test_mask)
        pred /= 6

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

        pred = pred.cpu().numpy()
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

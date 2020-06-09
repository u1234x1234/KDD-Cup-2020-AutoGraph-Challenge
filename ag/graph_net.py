import torch.nn as nn
from dgl import DGLGraph

from torch_geometric.data import Data

from .module_utils import init_activation

WITH_EDGE_WEIGHTS = ['ChebConv', 'GCNConv', 'SAGEConv', 'GraphConv', 'TAGConv', 'ARMAConv']


def _get_name(obj):
    if hasattr(obj, 'func'):
        return str(obj.func)
    else:
        return str(obj)


def _is_dgl(obj):
    # TODO check dgl, pyg or raise
    if hasattr(obj, 'func'):
        obj = obj.func
    module_name = str(obj.__module__)
    return 'dgl' in module_name


def with_edge_weights(conv_mod):
    for c in WITH_EDGE_WEIGHTS:
        if f'.{c}' in str(type(conv_mod)):
            return True
    return False


class GraphNet(nn.Module):
    """
            x
            |
            FC
            |
    Sequence of GraphConv
            |
            FC
            |
            out
    """
    def __init__(self, input_size, n_classes, n_nodes, conv_class, in_dropout, out_dropout, n_hidden, n_layers, activation):
        super().__init__()

        if input_size == 1:
            input_size = 128
            self.emb = nn.Embedding(n_nodes, input_size)
        else:
            self.emb = None

        self.layers = nn.ModuleList()
        self.in_nn = nn.Linear(input_size, n_hidden)

        for _ in range(n_layers):
            if 'GINConv' in _get_name(conv_class):
                self.layers.append(conv_class(
                    apply_func=nn.Linear(n_hidden, n_hidden),
                    aggregator_type='mean'))
            elif 'AGNNConv' in _get_name(conv_class):
                self.layers.append(conv_class())
            elif 'APPNPConv' in _get_name(conv_class):
                self.layers.append(nn.Linear(n_hidden, n_hidden))
            else:
                self.layers.append(conv_class(n_hidden, n_hidden))

        self.out_nn = nn.Linear(n_hidden, n_classes)
        self.in_dropout = nn.Dropout(in_dropout)
        self.out_dropout = nn.Dropout(out_dropout)
        self.activation = init_activation(activation)

    def forward(self, g: DGLGraph, data: Data):
        x = data.x

        if self.emb is not None:
            x = self.emb(x).squeeze()

        x = self.in_nn(x)
        x = self.activation(x)
        x = self.in_dropout(x)

        for layer in self.layers:
            if _is_dgl(layer):
                x = layer(g, x)
            else:
                if with_edge_weights(layer):
                    x = layer(x, edge_index=data.edge_index, edge_weight=data.edge_weight)
                else:
                    x = layer(x, edge_index=data.edge_index)

            x = self.activation(x).squeeze()  # GAT num heads (b, H, d)

        x = self.out_dropout(x)
        x = self.out_nn(x)
        return x

import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import sparse as sp
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.utils import to_scipy_sparse_matrix


class general_GCN_layer(Module):
    def __init__(self):
        super(general_GCN_layer, self).__init__()

    @staticmethod
    def multiplication(A, B):
        if str(A.layout) == 'torch.sparse_coo':
            return torch.spmm(A, B)
        else:
            return torch.mm(A, B)


class snowball_layer(general_GCN_layer):
    def __init__(self, in_features, out_features):
        super(snowball_layer, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight, self.bias = Parameter(torch.FloatTensor(self.in_features, self.out_features).cuda()), Parameter(torch.FloatTensor(self.out_features).cuda())
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv_weight, stdv_bias = 1. / math.sqrt(self.weight.size(1)), 1. / math.sqrt(self.bias.size(0))
        torch.nn.init.uniform_(self.weight, -stdv_weight, stdv_weight)
        torch.nn.init.uniform_(self.bias, -stdv_bias, stdv_bias)
    
    def forward(self, input, adj, eye=False):
        XW = torch.mm(input, self.weight)
        if eye:
            return XW + self.bias
        else:
            return self.multiplication(adj, XW) + self.bias


class truncated_krylov_layer(general_GCN_layer):
    def __init__(self, in_features, n_blocks, out_features, LIST_A_EXP=None, LIST_A_EXP_X_CAT=None):
        super(truncated_krylov_layer, self).__init__()
        self.LIST_A_EXP = LIST_A_EXP
        self.LIST_A_EXP_X_CAT = LIST_A_EXP_X_CAT
        self.in_features, self.out_features, self.n_blocks = in_features, out_features, n_blocks
        self.shared_weight, self.output_bias = Parameter(torch.FloatTensor(self.in_features * self.n_blocks,self.out_features).cuda()), Parameter(torch.FloatTensor(self.out_features).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv_shared_weight, stdv_output_bias = 1. / math.sqrt(self.shared_weight.size(1)), 1. / math.sqrt(self.output_bias.size(0))
        torch.nn.init.uniform_(self.shared_weight, -stdv_shared_weight, stdv_shared_weight)
        torch.nn.init.uniform_(self.output_bias, -stdv_output_bias, stdv_output_bias)

    def forward(self, input, adj, eye=True):
        if self.n_blocks == 1:
            output = torch.mm(input, self.shared_weight)
        elif self.LIST_A_EXP_X_CAT is not None:
            output = torch.mm(self.LIST_A_EXP_X_CAT, self.shared_weight)
        elif self.LIST_A_EXP is not None:
            feature_output = []
            for i in range(self.n_blocks):
                AX = self.multiplication(self.LIST_A_EXP[i], input)
                feature_output.append(AX)
            output = torch.mm(torch.cat(feature_output, 1), self.shared_weight)
        if eye:
            return output + self.output_bias
        else:
            return self.multiplication(adj, output) + self.output_bias


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).float().mean()
    return correct


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def train(model, optimizer, features, adj, labels, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    labels_train, output_train = labels[idx_train], output[idx_train]
    loss_train = F.cross_entropy(output_train, labels_train)
    acc_train = accuracy(output_train, labels_train)
    loss_train.backward()
    optimizer.step()
    return 100 * acc_train.item(), loss_train.item()


class graph_convolutional_network(nn.Module):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout):
        super(graph_convolutional_network, self).__init__()
        self.nfeat, self.nlayers, self.nhid, self.nclass = nfeat, nlayers, nhid, nclass
        self.dropout = dropout
        self.hidden = nn.ModuleList()

    def reset_parameters(self):
        for layer in self.hidden:
            layer.reset_parameters()
        self.out.reset_parameters()


class snowball(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, activation):
        super(snowball, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        self.activation = activation
        for k in range(nlayers):
            self.hidden.append(snowball_layer(k * nhid + nfeat, nhid))
        self.out = snowball_layer(nlayers * nhid + nfeat, nclass)
    
    def forward(self, x, adj):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(F.dropout(self.activation(layer(x, adj)), self.dropout, training=self.training))
            else:
                list_output_blocks.append(F.dropout(self.activation(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj)), self.dropout, training=self.training))
        output = self.out(torch.cat([x] + list_output_blocks, 1), adj, eye=False)
        # return F.log_softmax(output, dim=1)
        return output


class truncated_krylov(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, activation, n_blocks, adj, features):
        super(truncated_krylov, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        self.activation = activation
        LIST_A_EXP, LIST_A_EXP_X, A_EXP = [], [], torch.eye(adj.size()[0], dtype=adj.dtype).cuda()
        if str(adj.layout) == 'torch.sparse_coo':
            dense_adj = adj.to_dense()
        else:
            dense_adj = adj
        for _ in range(n_blocks):
            if nlayers > 1:
                indices = torch.nonzero(A_EXP).t()
                values = A_EXP[indices[0], indices[1]]
                LIST_A_EXP.append(torch.sparse.FloatTensor(indices, values, A_EXP.size()))
            LIST_A_EXP_X.append(torch.mm(A_EXP, features))
            torch.cuda.empty_cache()
            A_EXP = torch.mm(A_EXP, dense_adj)
        self.hidden.append(truncated_krylov_layer(nfeat, n_blocks, nhid, LIST_A_EXP_X_CAT=torch.cat(LIST_A_EXP_X, 1)))
        for _ in range(nlayers - 1):
            self.hidden.append(truncated_krylov_layer(nhid, n_blocks, nhid, LIST_A_EXP=LIST_A_EXP))
        self.out = truncated_krylov_layer(nhid, 1, nclass)
    
    def forward(self, x, adj):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(F.dropout(self.activation(layer(x, adj)), self.dropout, training=self.training))
            else:
                list_output_blocks.append(F.dropout(self.activation(layer(list_output_blocks[layer_num - 1], adj)), self.dropout, training=self.training))
        output = self.out(list_output_blocks[self.nlayers - 1], adj, eye=True)
        return output


def train_krylov(data, activation=nn.Tanh, n_blocks=16, layers=1, hidden=100, dropout=0.3, weight_decay=0.02):

    adj = to_scipy_sparse_matrix(data.edge_index)
    adj = normalize(sp.eye(adj.shape[0]) + adj)
    if data.x.shape[1] == 1:
        features = torch.FloatTensor(np.array(pd.get_dummies(data.x.numpy().flatten())))
    else:
        features = data.x

    labels = torch.LongTensor(data.y.numpy())
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()

    # model = snowball(nfeat=features.shape[1], nlayers=args.layers, nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout, activation=activation)
    model = truncated_krylov(nfeat=features.shape[1], nlayers=layers, nhid=hidden, nclass=labels.max().item() + 1, dropout=dropout, activation=activation(), n_blocks=n_blocks, adj=adj, features=features)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    predictions = []
    n_epochs = 1400
    mask = data.train_mask + data.val_mask
    for epoch in range(n_epochs):
        acc_train, loss_train = train(model, optimizer, features, adj, labels, idx_train=mask)

        if epoch > n_epochs - 10:
            model.eval()
            with torch.no_grad():
                output = F.softmax(model(features, adj)[data.test_mask], dim=1)
                predictions.append(output.cpu().numpy())

    del model
    del features
    del adj
    del labels
    torch.cuda.empty_cache()
    return np.mean(predictions, axis=0)

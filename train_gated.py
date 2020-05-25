# %%
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch import nn

from ag.pyg_utils import generate_pyg_data
from data_utils import read_dataset
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINEConv, GraphUNet, global_add_pool
from torch_geometric.utils import dropout_adj
from dgl.nn.pytorch import GATConv
from dgl import DGLGraph
from torch.nn import init
import dgl.function as fn

sys.path.append('/home/u1234x1234/autograph2020/src')


dataset, y_test = read_dataset(sys.argv[1])
n_classes = dataset.get_metadata()['n_class']
data = generate_pyg_data(dataset.get_data(), 1)
print(data)
g = DGLGraph((data.edge_index[0], data.edge_index[1]))
N = len(data.x)
E = len(data.edge_index[0])
snorm_n = torch.FloatTensor(N).fill_(1./float(N)).sqrt().cuda()
snorm_e = torch.FloatTensor(E).fill_(1./float(E)).sqrt().cuda()





class Aggregator(nn.Module):
    """
    Base Aggregator class. 
    """

    def __init__(self):
        super().__init__()

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}

    def aggre(self, neighbour):
        # N x F
        raise NotImplementedError


class MeanAggregator(Aggregator):
    """
    Mean Aggregator for graphsage
    """

    def __init__(self):
        super().__init__()

    def aggre(self, neighbour):
        mean_neighbour = torch.mean(neighbour, dim=1)
        return mean_neighbour


class MaxPoolAggregator(Aggregator):
    """
    Maxpooling aggregator for graphsage
    """

    def __init__(self, in_feats, out_feats, activation, bias):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation
        # Xavier initialization of weight
#         nn.init.xavier_uniform_(self.linear.weight,
#                                 gain=nn.init.calculate_gain('relu'))

    def aggre(self, neighbour):
        neighbour = self.linear(neighbour)
        if self.activation:
            neighbour = self.activation(neighbour)
        maxpool_neighbour = torch.max(neighbour, dim=1)[0]
        return maxpool_neighbour


class LSTMAggregator(Aggregator):
    """
    LSTM aggregator for graphsage
    """

    def __init__(self, in_feats, hidden_feats):
        super().__init__()
        self.lstm = nn.LSTM(in_feats, hidden_feats, batch_first=True)
        self.hidden_dim = hidden_feats
        self.hidden = self.init_hidden()

        nn.init.xavier_uniform_(self.lstm.weight,
                                gain=nn.init.calculate_gain('relu'))

    def init_hidden(self):
        """
        Defaulted to initialite all zero
        """
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def aggre(self, neighbours):
        """
        aggregation function
        """
        # N X F
        rand_order = torch.randperm(neighbours.size()[1])
        neighbours = neighbours[:, rand_order, :]

        (lstm_out, self.hidden) = self.lstm(neighbours.view(neighbours.size()[0],
                                                            neighbours.size()[
            1],
            -1))
        return lstm_out[:, -1, :]

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}

class NodeApply(nn.Module):
    """
    Works -> the node_apply function in DGL paradigm
    """

    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_feats * 2, out_feats, bias)
        self.activation = activation

#         nn.init.xavier_uniform_(self.linear.weight,
#                                 gain=nn.init.calculate_gain('relu'))

    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        bundle = self.linear(bundle)
        return bundle

    def forward(self, node):
        h = node.data['h']
        c = node.data['c']
        bundle = self.concat(h, c)
        bundle = F.normalize(bundle, p=2, dim=1)
        if self.activation:
            bundle = self.activation(bundle)
        return {"h": bundle}

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class GraphSageLayer(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, graph_norm, batch_norm, residual=False, bias=True,
                 dgl_builtin=False):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.aggregator_type = aggregator_type
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.dgl_builtin = dgl_builtin
        
        if in_feats != out_feats:
            self.residual = False
        
        self.dropout = nn.Dropout(p=dropout)

        if dgl_builtin == False:
            self.nodeapply = NodeApply(in_feats, out_feats, activation, dropout,
                                   bias=bias)
            if aggregator_type == "pool":
                self.aggregator = MaxPoolAggregator(in_feats, in_feats,
                                                    activation, bias)
            elif aggregator_type == "lstm":
                self.aggregator = LSTMAggregator(in_feats, in_feats)
            else:
                self.aggregator = MeanAggregator()
        else:
            self.sageconv = SAGEConv(in_feats, out_feats, aggregator_type,
                    dropout, activation=activation)
        
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)

    def forward(self, g, h, snorm_n=None):
        h_in = h              # for residual connection
        
        if self.dgl_builtin == False:
            h = self.dropout(h)
            g.ndata['h'] = h
            g.update_all(fn.copy_src(src='h', out='m'), self.aggregator,
                         self.nodeapply)
            h = g.ndata['h']
        else:
            h = self.sageconv(g, h)

        if self.graph_norm:
            h = h * snorm_n

        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        if self.residual:
            h = h_in + h       # residual connection
        
        return h
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, aggregator={}, residual={})'.format(self.__class__.__name__,
                                              self.in_channels,
                                              self.out_channels, self.aggregator_type, self.residual)


class Net(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """

    def __init__(self):
        super().__init__()

        in_dim_node = data.x.shape[1]
        hidden_dim = 64
        out_dim = 64
        in_feat_dropout = 0.5
        dropout = 0.5
        aggregator_type = 'pool'
        n_layers = 2
        graph_norm = False
        batch_norm = True
        residual = True
        self.device = torch.device('cuda')

        # self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, graph_norm, batch_norm, residual) for _ in range(n_layers-1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, graph_norm, batch_norm, residual))
        # self.MLP_layer = MLPReadout(out_dim, n_classes)
        self.MLP_layer = nn.Linear(out_dim, n_classes)

    def forward(self):
        h = data.x

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        # graphsage
        for conv in self.layers:
            h = conv(g, h, snorm_n)

        # output
        h_out = self.MLP_layer(h)

        return h_out
    

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)


def train(epoch):
    global data
    model.train()
    loss_all = 0
    data = data.to(device)
    optimizer.zero_grad()
    output = model()
    mask = data.train_mask + data.val_mask
    loss = F.cross_entropy(output[mask], data.y[mask])
    loss.backward()
    loss_all += loss.item()

    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)

    optimizer.step()
    return loss_all


def test():
    model.eval()
    with torch.no_grad():
        output = model()[data.test_mask]

    pred = output.max(dim=1)[1].cpu().numpy()
    return pred


for epoch in range(1, 1500):
    train_loss = train(epoch)
    if epoch % 5 == 0:
        y_pred = test()
        acc = (y_test.flatten() == y_pred.flatten()).mean()
        print(epoch, train_loss, acc)

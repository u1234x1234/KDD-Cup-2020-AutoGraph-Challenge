# %%
import sys

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

sys.path.append('/home/u1234x1234/autograph2020/src')


dataset, y_test = read_dataset(sys.argv[1])
n_class = dataset.get_metadata()['n_class']
data = generate_pyg_data(dataset.get_data(), 1)
print(data)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pool_ratios = [2000 / data.num_nodes, 0.5]
        self.unet = GraphUNet(16, 8, 16,
                              depth=1, pool_ratios=pool_ratios)

        in_size = data.x.shape[1]
        self.in_nn = nn.Linear(in_size, 16)
        self.out_nn = nn.Linear(16, n_class)

    def forward(self):
        x = self.in_nn(data.x)
        edge_index, _ = dropout_adj(data.edge_index, p=0.2,
                                    force_undirected=True,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.unet(x, edge_index)
        # x = F.relu(x)
        x = self.out_nn(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


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

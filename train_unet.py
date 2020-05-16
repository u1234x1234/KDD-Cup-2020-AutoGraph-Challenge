# %%
import sys

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn import Linear, ReLU, Sequential

from ag.pyg_utils import generate_pyg_data
from data_utils import read_dataset
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINEConv, GraphUNet, global_add_pool
from torch_geometric.utils import dropout_adj

sys.path.append('/home/u1234x1234/autograph2020/src')


dataset, y_test = read_dataset('a')
n_class = dataset.get_metadata()['n_class']
schema = dataset.get_metadata()['schema']
time_budget = dataset.get_metadata()['time_budget']
pyg_data = generate_pyg_data(dataset.get_data())
print(pyg_data)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pool_ratios = [2000 / pyg_data.num_nodes, 0.5]
        self.unet = GraphUNet(pyg_data.num_features, 16, n_class,
                              depth=1, pool_ratios=pool_ratios)

    def forward(self):
        # edge_index, _ = dropout_adj(pyg_data.edge_index, p=0.2,
        #                             force_undirected=True,
        #                             num_nodes=pyg_data.num_nodes,
        #                             training=self.training)
        # x = F.dropout(pyg_data.x, p=0.92, training=self.training)

        x = self.unet(pyg_data.x, pyg_data.edge_index)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    global pyg_data
    model.train()

    loss_all = 0

    pyg_data = pyg_data.to(device)
    optimizer.zero_grad()
    output = model()
    mask = pyg_data.train_mask + pyg_data.val_mask
    loss = F.cross_entropy(output[mask], pyg_data.y[mask])
    loss.backward()
    loss_all += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)

    optimizer.step()
    return loss_all


def test():
    model.eval()
    with torch.no_grad():
        output = model()[pyg_data.test_mask]

    pred = output.max(dim=1)[1].cpu().numpy()
    return pred


for epoch in range(1, 1500):
    train_loss = train(epoch)
    if epoch % 5 == 0:
        y_pred = test()
        acc = (y_test.flatten() == y_pred.flatten()).mean()
        print(epoch, train_loss, acc)

# %%
import sys
sys.path.append('/home/u1234x1234/autograph2020/src')
from data_utils import read_dataset

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from ag.pyg_utils import generate_pyg_data
from sklearn.metrics import accuracy_score


dataset, y_test = read_dataset('e')
n_class = dataset.get_metadata()['n_class']
schema = dataset.get_metadata()['schema']
time_budget = dataset.get_metadata()['time_budget']
pyg_data = generate_pyg_data(dataset.get_data())
print(pyg_data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_features = pyg_data.num_features
        dim = 48

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINEConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINEConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        # nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv3 = GINConv(nn3)
        # self.bn3 = torch.nn.BatchNorm1d(dim)

        # nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv4 = GINConv(nn4)
        # self.bn4 = torch.nn.BatchNorm1d(dim)

        # nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv5 = GINConv(nn5)
        # self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, n_class)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        # x = F.relu(self.conv3(x, edge_index))
        # x = self.bn3(x)
        # x = F.relu(self.conv4(x, edge_index))
        # x = self.bn4(x)
        # x = F.relu(self.conv5(x, edge_index))
        # x = self.bn5(x)
        # x = global_add_pool(x)
        # x = x.sum(dim=2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    global pyg_data
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0

    pyg_data = pyg_data.to(device)
    optimizer.zero_grad()
    output = model(pyg_data.x, pyg_data.edge_index)
    loss = F.cross_entropy(output[pyg_data.train_mask], pyg_data.y[pyg_data.train_mask])
    loss.backward()
    loss_all += loss.item()
    optimizer.step()
    return loss_all


def test():
    model.eval()
    with torch.no_grad():
        output = model(pyg_data.x, pyg_data.edge_index)[pyg_data.test_mask]

    pred = output.max(dim=1)[1].cpu().numpy()
    return pred


for epoch in range(1, 1500):
    train_loss = train(epoch)
    if epoch % 50 == 0:
        y_pred = test()
        acc = (y_test.flatten() == y_pred.flatten()).mean()
        print(epoch, train_loss, acc)

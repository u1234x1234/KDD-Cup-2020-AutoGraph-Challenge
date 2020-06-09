# %%
import sys
sys.path.append('/home/u1234x1234/autograph2020/src')
from data_utils import read_dataset

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, GCNConv, ChebConv
from ag.pyg_utils import generate_pyg_data
from sklearn.metrics import accuracy_score

# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets = []
for name in ['a', 'b', 'c', 'e']:
    dataset, y_test = read_dataset(name)
    n_class = dataset.get_metadata()['n_class']
    schema = dataset.get_metadata()['schema']
    time_budget = dataset.get_metadata()['time_budget']
    pyg_data = generate_pyg_data(dataset.get_data())
    pyg_data.to(device)
    datasets.append((y_test, n_class, pyg_data))

# %%


class Net(torch.nn.Module):
    def __init__(self, in_sizes, out_sizes):
        super().__init__()

        emb_size = 48
        hidden_size = 48

        self.in_sizes = in_sizes
        self.in_nn = torch.nn.ModuleList()
        self.out_nn = torch.nn.ModuleList()
        for size in out_sizes:
            self.out_nn.append(Linear(hidden_size, size))
        for size in in_sizes:
            self.in_nn.append(Linear(size, emb_size))

        self.in_drop = torch.nn.Dropout(0.5)
        self.out_drop = torch.nn.Dropout(0.5)

        self.conv1 = ChebConv(emb_size, hidden_size, K=7)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        idx = self.in_sizes.index(x.shape[1])

        x = self.in_nn[idx](x)
        x = F.relu(x)
        xf = self.in_drop(x)

        x = F.relu(self.conv1(xf, edge_index, edge_weight=edge_weight))

        x = self.out_drop(x)
        x = self.out_nn[idx](x)

        return x, xf


def test(model, data):
    model.eval()
    with torch.no_grad():
        output, f = model(data)
        output = output[data.test_mask]

    pred = output.max(dim=1)[1].cpu().numpy()
    return pred, f


def train(model, optimizer, data, n_epoch=1):
    loss_all = 0

    model.train()
    for _ in range(n_epoch):
        optimizer.zero_grad()
        output, _ = model(data)
        mask = data.val_mask + data.train_mask
        loss = F.cross_entropy(output[mask], data.y[mask])
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

    return loss_all


n_classes = [x[1] for x in datasets]
in_sizes = [x[2].num_features for x in datasets]
print(n_classes, in_sizes)


model = Net(in_sizes, n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

y_test1, _, data1 = datasets[0]
y_test2, _, data2 = datasets[1]
y_test3, _, data3 = datasets[2]

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

for g_epoch_idx in range(100):

    # train(model, optimizer, data4, n_epoch=5)
    train(model, optimizer, data3, n_epoch=1)
    train(model, optimizer, data2, n_epoch=1)
    train(model, optimizer, data1, n_epoch=1)

    y_pred1, X = test(model, data1)

    # mask = data1.train_mask + data1.val_mask
    # y_train = data1.y[mask].cpu().numpy()
    # X_train, X_test = X[mask].cpu().numpy(), X[data1.test_mask].cpu().numpy()
    # smodel = ExtraTreesClassifier(n_estimators=200)
    # smodel.fit(X_train, y_train)
    # y_pred = smodel.predict(X_test)
    # acc_s1 = (y_pred.flatten() == y_test1.flatten()).mean()

    y_pred2, _ = test(model, data2)

    acc1 = (y_test1.flatten() == y_pred1.flatten()).mean()
    acc2 = (y_test2.flatten() == y_pred2.flatten()).mean()

    print(g_epoch_idx, acc1, acc2)

import sys
sys.path.append('/home/u1234x1234/autograph2020/starting_kit/ingestion')
from starting_kit.ingestion.dataset import Dataset

import os
import pandas as pd

ROOT_PATH = '/home/u1234x1234/autograph2020/data/public/'


def read_dataset(name):
    dataset = Dataset(f'{ROOT_PATH}/{name}/train.data')
    labels = pd.read_csv(f'{ROOT_PATH}/{name}/test_label.tsv', sep='\t').values[:, 1:]
    return dataset, labels


# def read_dataset(name):

#     X_feat = pd.read_csv(os.path.join(ROOT_PATH, f'{name}/train.data/feature.tsv'), sep='\t')
#     # edge = pd.read_csv(os.path.join(ROOT_PATH, f'{name}/train.data/edge.tsv'), sep='\t')

#     train_labels = pd.read_csv(os.path.join(ROOT_PATH, f'{name}/train.data/train_label.tsv'), sep='\t')
#     test_labels = pd.read_csv(os.path.join(ROOT_PATH, f'{name}/test_label.tsv'), sep='\t')

#     df = pd.merge(train_labels, X_feat, on='node_index')
#     X_train = df.values[:, 2:]
#     y_train = df.label.values
#     assert len(X_train) == len(y_train)

#     df = pd.merge(test_labels, X_feat, on='node_index')
#     X_test = df.values[:, 2:]
#     y_test = df.label.values
#     assert len(X_test) == len(y_test)


#     return X_train, y_train, X_test, y_test


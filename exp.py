# %%
from data_utils import read_dataset
from ag.model import Model
from sklearn.metrics import accuracy_score

dataset, y_test = read_dataset('a')
n_class = dataset.get_metadata()['n_class']
schema = dataset.get_metadata()['schema']
time_budget = dataset.get_metadata()['time_budget']
model = Model()

y_pred = model.train_predict(dataset.get_data(), time_budget=100, n_class=n_class, schema=schema)

score = accuracy_score(y_test, y_pred)
print(score)


# X_train, y_train, X_test, y_test = read_dataset('a')


# %%

from uxils.ml.automl import show_glance

import xgboost as xgb
import lightgbm as lgb

# model = xgb.sklearn.XGBClassifier(n_jobs=20, gpu_id=0, colsample_bytree=0.9, subsample=0.9, n_estimators=50)
# model = lgb.sklearn.LGBMClassifier(n_jobs=20, device='gpu', n_estimators=200)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(accuracy_score(y_test, y_pred))

show_glance(X_train, y_train, X_test, y_test, accuracy_score, 15, method_name='predict')

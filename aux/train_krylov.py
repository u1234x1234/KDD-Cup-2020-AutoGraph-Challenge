from ag.pyg_utils import generate_pyg_data
from ag.krylov import train_krylov
from data_utils import read_dataset



# from torch import nn
# from uxils.ray_ext import optimize_function, show_results
# config = {
#     'activation': [nn.SELU, nn.Tanh, nn.ReLU],
#     'n_blocks': [4, 8, 16, 32],
#     'layers': [1],
#     'hidden': [32, 64, 128, 256],
#     'dropout': [0.3, 0.5, 0.8],
#     'weight_decay': [0.1, 0.01, 0.001],
# }
# r = optimize_function(func, config, time_budget=60*10, num_gpus=2, num_cpus=20, gpu_per_trial=0.5)
# show_results(r)


dataset, y_test = read_dataset('a')
n_class = dataset.get_metadata()['n_class']
data = generate_pyg_data(dataset.get_data(), 1)

y_pred = train_krylov(data)
print(y_pred.shape)
print((y_test.flatten() == y_pred.argmax(axis=1)).mean())

# func(
#     data=data, activation=nn.Tanh, n_blocks=16, layers=1, hidden=16, dropout=0.5, weight_decay=0.02,
# )

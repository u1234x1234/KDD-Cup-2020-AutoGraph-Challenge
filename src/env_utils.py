import os
import sys
from subprocess import check_call


def install_pip_package(name, upgrade: bool = True):
    # options = '--upgrade' if upgrade else ''
    command = f'{sys.executable} -m pip install {name}'
    retcode = check_call([command], shell=True)
    return retcode


def prepare_env():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    dn = os.path.dirname(__file__)

    install_pip_package('torch==1.4.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html')
    install_pip_package(f'torch-scatter==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html')
    install_pip_package(f'torch-sparse==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html')
    install_pip_package(f'torch-cluster==latest+cu100 -f https://pytorch-geometric.com/whl/torch-1.4.0.html')
    install_pip_package(f'-U torch-geometric')
    install_pip_package(f'ray[tune]')
    install_pip_package(f'zstandard')

    install_pip_package(f'{dn}/')
    return dn

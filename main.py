
from Utils import helper

import argparse 
parser = argparse.ArgumentParser(description='Deep Residual Method for American Basket Options')
parser.add_argument('-n', '--n_dim', default=1, type=int, metavar='DIMENSION', help='number of dimension')
args = parser.parse_args()
n_dim = args.n_dim

for seed in range(2, 22, 2):

    print(f'start training ... [ndim: {n_dim}, seed: {seed}]')

    if n_dim == 1:
        process = helper.run_cmd(f'python .\PINN.py --n_dim {n_dim} --seed {seed} --steps_sol 20')
    elif n_dim > 1:
        process = helper.run_cmd(f'python .\PINN.py --n_dim {n_dim} --seed {seed} --steps_sol 10')

    process.wait()
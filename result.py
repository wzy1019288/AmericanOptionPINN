
import os
import torch
import numpy as np
import pandas as pd
from itertools import cycle

from config import OptionConfig
from Utils import plot
from Model.pde import pde_dim1, pde_dim2
from Model.fnn import FNN
from Sampler.dim1 import (
    get_test_dataloaders
)
from Model.train.run_dim1 import (
    get_test_data
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("pytorch version", torch.__version__, "\n")
print ("Training on:", DEVICE)

config = OptionConfig()
n_dim = config.n_dim
r = config.r
T = config.T
K = config.K
sigma = config.sigma
seed = config.seed
image_path = config.image_path
checkpoint_path = config.checkpoint_path

##############################################################################################

# config.n_dim = 2
# config.steps_fb_per_pde = 10
# config.seed = 999


# total_losses = []
# for _seed in range(2, 22, 2):
#     checkpoint_path = config.checkpoint_path
#     df_loss = pd.read_csv(os.path.join(checkpoint_path, 'log_test.txt'), sep='\t').dropna(axis=1)
#     total_losses.append({'seed': _seed, 'total_loss_min': df_loss['Total Loss'].min()})
# pd.DataFrame(total_losses)
# u_losses = df_loss['Unsupervised Loss']
# sol_init_losses = df_loss['sol_init_loss']
# sol_dir_losses = df_loss['sol_dir_loss']
# sol_neu_losses = df_loss['sol_neu_loss']
# fb_losses = df_loss['Free Boundary Loss']
# total_losses = df_loss['Total Loss']

# plot.plot_loss(os.path.join(image_path, 'LOSS.png'), 
#                df_loss['Unsupervised Loss'], 
#                df_loss['sol_init_loss'], 
#                df_loss['sol_dir_loss'], 
#                df_loss['sol_neu_loss'], 
#                df_loss['Free Boundary Loss'], 
#                df_loss['Total Loss'])
##############################################################################################



##############################################################################################
## Testset
sol_conditions_test, fb_conditions_test = get_test_dataloaders(
    n_dim=n_dim, r=r, K=K, T=T,
    lb=config.lb, ub=config.ub, DTYPE=config.DTYPE, 
    N_samples_testset_pde=config.N_samples_testset_pde,
    N_samples_testset_others=config.N_samples_testset_others,
)

all_data = get_test_data(
                sol_conditions_test['Interior'],
                sol_conditions_test['Initial'],
                sol_conditions_test['Dirichlet'],
                fb_conditions_test['Initial'],
                fb_conditions_test['Dirichlet'],
                fb_conditions_test['Neumann'],
                DEVICE)

S_fb_init_pred_records = []
S_fb_dir_pred_records = []
for _seed in range(2, 22, 2):
    config.seed = _seed

    sol_model = FNN(
                dim_in=config.sol_layers[0], 
                width=config.sol_layers[1], 
                dim_out=config.sol_layers[-1], 
                depth=len(config.sol_layers)-2, 
                activation={'in': config.sol_activation,
                            'hid': config.sol_activation,
                            'out': config.sol_output_act }
                )
    fb_model  = FNN(
                dim_in=config.fb_layers[0], 
                width=config.fb_layers[1], 
                dim_out=config.fb_layers[-1], 
                depth=len(config.fb_layers)-2, 
                activation={'in': config.fb_activation,
                            'hid': config.fb_activation,
                            'out': config.fb_output_act }
                )

    best_model = torch.load(os.path.join(config.checkpoint_path, 'model_best.pth.tar'))
    # df_loss = pd.read_csv(os.path.join(config.checkpoint_path, 'log_train.txt'), sep='\t').dropna(axis=1)
    # df_loss[df_loss.Epoch == best_model['epoch']]

    sol_model.load_state_dict(best_model['sol_state_dict'])
    fb_model.load_state_dict(best_model['fb_state_dict'])
    sol_model = sol_model.to(DEVICE)
    fb_model = fb_model.to(DEVICE)

    sol_model.eval()
    fb_model.eval()

    (x_sol_intrr, t_sol_intrr, x_sol_init, y_sol_init, x_sol_dir, y_sol_dir,
                x_fb_init, y_fb_init, x_fb_dir, x_fb_neu, y_fb_neu) = all_data

    # S_sol_intrr = x_sol_intrr.cpu().detach().numpy()
    # t_sol_intrr = t_sol_intrr.cpu().detach().numpy()
    # St_sol_init = x_sol_init.cpu().detach().numpy()
    # St_sol_dir = x_sol_dir.cpu().detach().numpy()

    # S_fb_init_pred = fb_model(x_fb_init)
    S_fb_dir_pred = fb_model(x_fb_dir)
    # S_fb_init_pred = S_fb_init_pred.cpu().detach().numpy()
    # t_fb_init = x_fb_init.cpu().detach().numpy()
    S_fb_dir_pred = S_fb_dir_pred.cpu().detach().numpy()
    t_fb_dir = x_fb_dir.cpu().detach().numpy()


    S_fb_dir_pred_records.append(pd.DataFrame([{f'S_fb_dir_pred_{_seed}': i} for i in S_fb_dir_pred.flatten()]).T)

S_fb_dir_pred_all = pd.concat(S_fb_dir_pred_records)



S_fb_dir_pred_interval = pd.DataFrame({
    'S_fb_dir_pred_mean': S_fb_dir_pred_all.mean(),
    'S_fb_dir_pred_max': 10,
    'S_fb_dir_pred_min': S_fb_dir_pred_all.mean() - S_fb_dir_pred_all.std(),
})
S_fb_dir_pred_interval = S_fb_dir_pred_interval.assign(
                                                    t = t_fb_dir, 
                                                    MAX = config.ub[0],
                                                    MIN = config.lb[0],
                                                ).sort_values('t')

# St_sol_init = x_sol_init.cpu().detach().numpy()
# St_sol_dir = x_sol_dir.cpu().detach().numpy()


from matplotlib import pyplot as plt
plot.plot_free_boundary(
    t = S_fb_dir_pred_interval['t'],
    fb_max = S_fb_dir_pred_interval['S_fb_dir_pred_max'],
    fb_mean = S_fb_dir_pred_interval['S_fb_dir_pred_mean'],
    fb_min = S_fb_dir_pred_interval['S_fb_dir_pred_min'],
    pde_max = S_fb_dir_pred_interval['MAX'],
    path = '/'.join(config.image_path.split('/')[:-1]+['/free_boundary.png'])
)



# Define PDE
# pde = lambda xs, ts, u_val, u_x: _pde(xs, ts, u_val, u_x, r, sigma)



##############################################################################################


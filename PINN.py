
import os
import time
import random
import argparse 
import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from itertools import cycle
from scipy.stats import norm
from functools import lru_cache

import torch

from config import OptionConfig
from Utils import helper, plot

from Model.fnn import FNN
from Model.pde import pde_dim1, pde_dim2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("pytorch version", torch.__version__, "\n")
print ("Training on:", DEVICE)

##############################################################################################
## Settings
## ------------------------- ##
config = OptionConfig()

## parser arguments
# python .\PINN.py --n_dim 1 --seed 10 --steps_sol 20
parser = argparse.ArgumentParser(description='Deep Residual Method for American Basket Options')
parser.add_argument('-n', '--n_dim', default=config.n_dim, type=int, metavar='DIMENSION', help='number of dimension')
parser.add_argument('-sd', '--seed', default=config.seed, type=int, metavar='SEED', help='random seed')
parser.add_argument('-ss', '--steps_sol', default=config.steps_fb_per_pde, type=int, metavar='STEPS OF SOL_MODEL', help='steps_fb_per_pde')
args = parser.parse_args()

# args.n_dim = 4
# args.steps_sol = 10

if args.n_dim == 1:
    from Sampler.dim1 import (
        get_train_dataloaders, get_test_dataloaders
    )
    from Model.train.run_dim1 import (
        train_sol_model,
        train_fb_model,
        train_fb_sol_model,
        test
    )

    n_dim = args.n_dim
    alphas = None

    # Define PDE
    pde = lambda xs, ts, u_val, u_x: pde_dim1(xs, ts, u_val, u_x, config.r, config.sigma)

elif args.n_dim > 1:
    from Sampler.dim2 import (
        get_train_dataloaders, get_test_dataloaders
    )
    from Model.train.run_dim2 import (
        train_sol_model,
        train_fb_model,
        train_fb_sol_model,
        test
    )

    config.N_samples_trainset_pde = 60000
    config.N_samples_trainset_others = 1500
    config.n_dim = args.n_dim
    
    if config.n_dim == 2:
        config.sigma = [[0.05, 0.01], [0.01, 0.06]]
    elif config.n_dim == 3:
        config.sigma = [[0.05, 0.01, 0.1], [0.01, 0.06, -0.03], [0.1, -0.03, 0.4]]
    elif config.n_dim == 4:
        config.sigma = [[0.05, 0.01, 0.1, 0], [0.01, 0.06, -0.03, 0],
                        [0.1, -0.03, 0.4, 0.2], [0, 0, 0.2, 0.3]]
        config.N_samples_testset_pde = 120000
        config.N_samples_testset_others = 1200
    n_dim = config.n_dim
    if isinstance(config.sigma, np.ndarray):
        alphas = np.zeros((n_dim, n_dim))
        for i in range(n_dim):
            alphas[i, i] = np.sum( np.dot(config.sigma[i], config.sigma[i]) )
            for j in range(i):
                alphas[i, j] = alphas[j, i] = np.sum( np.dot(config.sigma[i], config.sigma[j]) )

    # Define PDE
    pde = lambda variables, u_val, derivatives: pde_dim2(variables, u_val, derivatives, 
                                                        config.r, config.d, alphas)

r = config.r
T = config.T
K = config.K
d = config.d
sigma = config.sigma

config.seed = args.seed
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(config.seed)

config.steps_fb_per_pde = args.steps_sol

image_path = config.image_path
checkpoint_path = config.checkpoint_path
##############################################################################################


##############################################################################################
## Point sampling
## ------------------------- ##

## Trainset
sol_conditions, fb_conditions = get_train_dataloaders(
    n_dim=n_dim, r=r, K=K, T=T,
    lb=config.lb, ub=config.ub, DTYPE=config.DTYPE, batch_num=config.batch_num,
    N_samples_trainset_pde=config.N_samples_trainset_pde, 
    N_samples_trainset_others=config.N_samples_trainset_others,
)

## Testset
sol_conditions_test, fb_conditions_test = get_test_dataloaders(
    n_dim=n_dim, r=r, K=K, T=T,
    lb=config.lb, ub=config.ub, DTYPE=config.DTYPE, 
    N_samples_testset_pde=config.N_samples_testset_pde,
    N_samples_testset_others=config.N_samples_testset_others,
)
##############################################################################################


##############################################################################################
## Training Models
# region ------------------- ##
## ------------------------- ##


print('*', '-' * 45, '*')
print('===> neural network training ...')

sol_model = FNN(
            dim_in=config.sol_layers[0], 
            width=config.sol_layers[1], 
            dim_out=config.sol_layers[-1], 
            depth=len(config.sol_layers)-2, 
            activation={'in': config.sol_activation,
                        'hid': config.sol_activation,
                        'out': config.sol_output_act }
            )
sol_model.Xavier_initi()
print('FNN_Sol Architecture:', "\n", sol_model)
print('Total number of trainable parameters = ', sum(p.numel() for p in sol_model.parameters() if p.requires_grad))
fb_model  = FNN(
            dim_in=config.fb_layers[0], 
            width=config.fb_layers[1], 
            dim_out=config.fb_layers[-1], 
            depth=len(config.fb_layers)-2, 
            activation={'in': config.fb_activation,
                        'hid': config.fb_activation,
                        'out': config.fb_output_act }
            )
fb_model.Xavier_initi()
print('FNN_Fb Architecture:', "\n", fb_model)
print('Total number of trainable parameters = ', sum(p.numel() for p in fb_model.parameters() if p.requires_grad))

# create optimizer 
sol_optimizer = torch.optim.RMSprop(sol_model.parameters(), lr=config.sol_lr)
fb_optimizer = torch.optim.RMSprop(fb_model.parameters(), lr=config.fb_lr)
# sol_optimizer = torch.optim.Adam(sol_model.parameters(), lr=config.sol_lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
# fb_optimizer = torch.optim.Adam(fb_model.parameters(), lr=config.fb_lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

# create learning rate schedular
sol_schedular = torch.optim.lr_scheduler.ExponentialLR(sol_optimizer, gamma=0.95)
fb_schedular = torch.optim.lr_scheduler.ExponentialLR(fb_optimizer, gamma=0.95)

# load model to device
sol_model = sol_model.to(DEVICE)
fb_model = fb_model.to(DEVICE)

# create log file
logger_train = helper.Logger(os.path.join(checkpoint_path, 'log_train.txt'), title='American-Option-PINN-1d-Train')
logger_train.set_names(['Epoch', 'Learning Rate', 'Unsupervised Loss', 'Supervised Loss', 'Free Boundary Loss', 'Total Loss', 
                  'sol_init_loss', 'sol_dir_loss', 'sol_neu_loss', 'fb_init_loss', 'fb_dir_loss', 'fb_neu_loss'])
logger_test = helper.Logger(os.path.join(checkpoint_path, 'log_test.txt'), title='American-Option-PINN-1d-Test')
logger_test.set_names(['Epoch', 'Learning Rate', 'Unsupervised Loss', 'Supervised Loss', 'Free Boundary Loss', 'Total Loss', 
                  'sol_init_loss', 'sol_dir_loss', 'sol_neu_loss', 'fb_init_loss', 'fb_dir_loss', 'fb_neu_loss'])
vis_logger = helper.VisdomLogger(env='main', env_path='/'.join([checkpoint_path, 'visdom']))
vis_textlogger = vis_logger.record_texts('LOSSES RECORDS')
## ------------------------- ##
# Unsupervised Loss  = interior_loss
# Supervised Loss    = sol_init_loss + sol_dir_loss + sol_neu_loss
# Free Boundary Loss = fb_init_loss + fb_dir_loss + fb_neu_loss
# Total Loss         = Unsupervised Loss 
#                      + sol_init_loss + sol_dir_loss + sol_neu_loss
#                      + fb_dir_loss + fb_neu_loss
## ------------------------- ##


# Early warning initialization
early_warning = {'Target': 1e10, 'n_steps': 0, 'n_steps_updateLR': 0}

# Training
since = time.time()
for epoch in tqdm(range(config.epochs), desc='PINNs - Training'):
    
    # Case 1: more mdl steps for a single fb step
    # for _ in range(config.steps_fb_per_pde - 1):
    sol_model, fb_model, sol_optimizer, fb_optimizer = train_sol_model(
                                                        sol_model, fb_model,
                                                        sol_optimizer, fb_optimizer,
                                                        pde,
                                                        sol_conditions, fb_conditions,
                                                        config, DEVICE,
                                                        K=K, n_dim=n_dim
                                                    )
    
    # Case 2: more fb steps for a single mdl step
    # for _ in range(0, config.steps_fb_per_pde+1, -1):
    sol_model, fb_model, sol_optimizer, fb_optimizer = train_fb_model(
                                                        sol_model, fb_model,
                                                        sol_optimizer, fb_optimizer,
                                                        sol_conditions, fb_conditions,
                                                        config, DEVICE,
                                                        K=K, n_dim=n_dim
                                                    )

    (sol_model, fb_model, sol_optimizer, fb_optimizer), \
        (u_loss, sol_i_l, sol_d_l, sol_n_l, fb_i_l, fb_d_l, fb_n_l, fb_l, total_l) = train_fb_sol_model(
                                                                                        sol_model, fb_model,
                                                                                        sol_optimizer, fb_optimizer,
                                                                                        pde,
                                                                                        sol_conditions, fb_conditions,
                                                                                        config, DEVICE,
                                                                                        K=K, n_dim=n_dim, epoch=epoch
                                                                                    )

    # save current and best models to checkpoint
    is_best = total_l < early_warning['Target']
    helper.save_checkpoint({'epoch': epoch+1,
                            'sol_state_dict': sol_model.state_dict(),
                            'fb_state_dict': fb_model.state_dict(),
                            'Unsupervised Loss': u_loss,
                            'Supervised Loss': sol_i_l + sol_d_l + sol_n_l,
                            'Free Boundary Loss': fb_l,
                            'Total Loss': total_l,
                            'sol_init_loss': sol_i_l,
                            'sol_dir_loss': sol_d_l,
                            'sol_neu_loss': sol_n_l,
                            'fb_init_loss': fb_i_l,
                            'fb_dir_loss': fb_d_l,
                            'fb_neu_loss': fb_n_l,
                            'sol_optimizer': sol_optimizer.state_dict(),
                            'fb_optimizer': fb_optimizer.state_dict(),
                           }, is_best, checkpoint=checkpoint_path)
    # save training process to log file
    logger_train.append([epoch+1, sol_optimizer.param_groups[0]['lr'], 
                   u_loss, sol_i_l + sol_d_l + sol_n_l, fb_l, total_l, 
                   sol_i_l, sol_d_l, sol_n_l, fb_i_l, fb_d_l, fb_n_l, 
                   ])
    (u_loss_test, sol_i_l_test, sol_d_l_test, sol_n_l_test, fb_i_l_test, 
                        fb_d_l_test, fb_n_l_test, fb_l_test, total_l_test) = test(
                                                                                sol_model, fb_model,
                                                                                sol_conditions_test, fb_conditions_test,
                                                                                pde, 
                                                                                config, DEVICE,
                                                                                K=K, n_dim=n_dim
                                                                            )
    logger_test.append([epoch+1, sol_optimizer.param_groups[0]['lr'], 
                   u_loss_test, sol_i_l_test + sol_d_l_test + sol_n_l_test, fb_l_test, total_l_test, 
                   sol_i_l_test, sol_d_l_test, sol_n_l_test, fb_i_l_test, fb_d_l_test, fb_n_l_test, 
                   ])
    # save visdom
    vis_logger.record_lines(
        Y=[sol_optimizer.param_groups[0]['lr']], X=[epoch+1], 
        legend=['Learning Rate'], 
        panel_name='TRAIN LR', title='learning rate', append=True if epoch>10 else False)
    vis_logger.record_lines(
        Y=[np.log10(u_loss), np.log10(sol_i_l + sol_d_l + sol_n_l), np.log10(fb_l), np.log10(total_l)], 
        X=[epoch+1], 
        legend=['Unsupervised Loss', 'Supervised Loss', 'Free Boundary Loss', 'Total Loss'], 
        panel_name='TRAIN LOSS', title='train loss (log10)', append=True if epoch>10 else False)
    vis_logger.record_lines(
        Y=[np.log10(u_loss_test), np.log10(sol_i_l_test + sol_d_l_test + sol_n_l_test), np.log10(fb_l_test), np.log10(total_l_test)], 
        X=[epoch+1], 
        legend=['Unsupervised Loss', 'Supervised Loss', 'Free Boundary Loss', 'Total Loss'], 
        panel_name='TEST LOSS', title='test loss (log10)', append=True if epoch>10 else False)
    vis_logger.record_lines(Y=[np.log10(u_loss)], X=[epoch+1], legend=['pde_loss'], 
        panel_name='TRAIN pde_loss', title='pde_loss (log10)', append=True if epoch>10 else False)
    vis_logger.record_lines(Y=[np.log10(sol_i_l)], X=[epoch+1], legend=['sol_i_l'], 
        panel_name='TRAIN sol_i_l', title='sol_init_loss (log10)', append=True if epoch>10 else False)
    vis_logger.record_lines(Y=[np.log10(sol_d_l)], X=[epoch+1], legend=['sol_d_l'], 
        panel_name='TRAIN sol_d_l', title='sol_dir_loss (log10)', append=True if epoch>10 else False)
    vis_logger.record_lines(Y=[np.log10(fb_i_l)], X=[epoch+1], legend=['fb_i_l'], 
        panel_name='TRAIN fb_i_l', title='fb_init_loss (log10)', append=True if epoch>10 else False)
    vis_logger.record_lines(Y=[np.log10(fb_d_l)], X=[epoch+1], legend=['fb_d_l'], 
        panel_name='TRAIN fb_d_l', title='fb_dir_loss (log10)', append=True if epoch>10 else False)
    vis_logger.record_lines(Y=[np.log10(fb_n_l)], X=[epoch+1], legend=['fb_n_l'], 
        panel_name='TRAIN fb_n_l', title='fb_neu_loss (log10)', append=True if epoch>10 else False)
    

    # adjust learning rate according to predefined schedule
    if epoch <= 50:
        sol_schedular.step()
        fb_schedular.step()
    else:
        if is_best:
            early_warning['n_steps_updateLR'] = 0
        else:
            early_warning['n_steps_updateLR'] += 1
            
            if n_dim < 4:
                if epoch <= 500:
                    patience_updateLR = 15
                elif 500 < epoch <= 1000:
                    patience_updateLR = 25
                else:
                    patience_updateLR = 100
            else:
                if epoch <= 500:
                    patience_updateLR = 50
                elif 500 < epoch <= 1000:
                    patience_updateLR = 100
                else:
                    patience_updateLR = 150

            if early_warning['n_steps_updateLR'] >= patience_updateLR:
                print(f'==> Updating LR in epoch:{epoch}')
                early_warning['n_steps_updateLR'] = 0
                for _ in range(min(int((patience_updateLR // 5) / 2) + 1, 5)):
                    sol_schedular.step()
                    fb_schedular.step()

    # update early_warning
    early_warning['Target'] = min(total_l, early_warning['Target'])
    if is_best:
        vis_logger.record_texts('==> Saving best model ...', vis_textlogger, True)
        early_warning['n_steps'] = 0
    else:
        early_warning['n_steps'] += 1
        if early_warning['n_steps'] >= config.patience:
            break

    # print results
    print_base = "{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}"
    if epoch == 0:
        TITLE = print_base.format(
                '|Epoch', '|Learning Rate', '|Unsupervised Loss', '|Supervised Loss', '|Free Boundary Loss', '|Total Loss', 
                '|sol_init_loss', '|sol_dir_loss', '|sol_neu_loss', '|fb_init_loss', '|fb_dir_loss', '|fb_neu_loss')
        SPLIT = print_base.format('='*10, '='*20, '='*20, '='*20, '='*20, '='*20, '='*20, '='*20, '='*20, '='*20, '='*20, '='*20)
        vis_logger.record_texts(TITLE, vis_textlogger, True)
        vis_logger.record_texts(SPLIT, vis_textlogger, True)
        print(TITLE)
        print(SPLIT)

    LOSS = print_base.format(epoch+1,
                            format(sol_optimizer.param_groups[0]['lr'], '.20f')[:10],
                            format(u_loss, '.20f')[:10],
                            format(sol_i_l + sol_d_l + sol_n_l, '.20f')[:10],
                            format(fb_l, '.20f')[:10],
                            format(total_l, '.20f')[:10],
                            format(sol_i_l, '.20f')[:10],
                            format(sol_d_l, '.20f')[:10],
                            format(sol_n_l, '.20f')[:10],
                            format(fb_i_l, '.20f')[:10],
                            format(fb_d_l, '.20f')[:10],
                            format(fb_n_l, '.20f')[:10],
                            )
    if epoch % config.verbose == 0 or is_best:
        print(LOSS)
    vis_logger.record_texts(LOSS, vis_textlogger, True)

time_elapsed = time.time() - since
logger_train.close()
logger_test.close()
vis_logger.save()
vis_logger.close()

print('Done in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
print('*', '-' * 45, '*', "\n", "\n")
# endregion
##############################################################################################



##############################################################################################
## Save results
## ------------------------- ##

# config.n_dim = 2
# config.steps_fb_per_pde = 10
# config.seed = 2
# checkpoint_path = config.checkpoint_path
# image_path = config.image_path

df_loss_train = pd.read_csv(os.path.join(checkpoint_path, 'log_train.txt'), sep='\t')
plot.plot_loss(os.path.join(image_path, f'LOSS_TRAIN.png'), 
               df_loss_train['Unsupervised Loss'], 
               df_loss_train['sol_init_loss'], 
               df_loss_train['sol_dir_loss'], 
               df_loss_train['sol_neu_loss'], 
               df_loss_train['Free Boundary Loss'], 
               df_loss_train['Total Loss'])
df_loss_test = pd.read_csv(os.path.join(checkpoint_path, 'log_test.txt'), sep='\t')
plot.plot_loss(os.path.join(image_path, f'LOSS_TEST.png'), 
               df_loss_test['Unsupervised Loss'], 
               df_loss_test['sol_init_loss'], 
               df_loss_test['sol_dir_loss'], 
               df_loss_test['sol_neu_loss'], 
               df_loss_test['Free Boundary Loss'], 
               df_loss_test['Total Loss'])
##############################################################################################



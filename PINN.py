
import os
import time
import random
import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from itertools import cycle
from scipy.stats import norm

import torch

from config import OptionConfig
from Utils import helper, plot
from Sampler.dim1 import (
    get_train_dataloaders, get_test_dataloaders
)
from Model.fnn import FNN
from Model.pde import _pde



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("pytorch version", torch.__version__, "\n")
print ("Training on:", DEVICE)

##############################################################################################
## Settings
## ------------------------- ##
config = OptionConfig()

r = config.r
T = config.T
K = config.K
sigma = config.sigma

seed = config.seed
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(seed)

image_1d_path = config.image_1d_path
checkpoint_path = config.checkpoint_path
##############################################################################################


##############################################################################################
## Point sampling
## ------------------------- ##

## Trainset
sol_conditions, fb_conditions = get_train_dataloaders(
    lb=config.lb, ub=config.ub, K=K, T=T, DTYPE=config.DTYPE, batch_num=config.batch_num,
    N_samples_trainset_pde=config.N_samples_trainset_pde,
    N_samples_trainset_others=config.N_samples_trainset_others,
)

## Testset
sol_conditions_test, fb_conditions_test = get_test_dataloaders(
    lb=config.lb, ub=config.ub, K=K, T=T, DTYPE=config.DTYPE, 
    N_samples_testset_pde=config.N_samples_testset_pde,
    N_samples_testset_others=config.N_samples_testset_others,
)
##############################################################################################


##############################################################################################
## PINN Models
# region ------------------- ##
## ------------------------- ##
# Define PDE
pde = lambda xs, ts, u_val, u_x: _pde(xs, ts, u_val, u_x, r, sigma)

def print_grad():
    for name, param in sol_model.named_parameters():
        if param.requires_grad:
            print('='*40)
            print(f'[sol_model]Accumulation for parameter {name}: {torch.sum(param)}')
            break
    for name, param in fb_model.named_parameters():
        if param.requires_grad:
            print(f'[fb_model ]Accumulation for parameter {name}: {torch.sum(param)}')
            break

def train_sol_model():

    # set model to training mode
    sol_model.train()

    (Unsupervised_loss_batches, 
    sol_init_loss_batches, sol_dir_loss_batches, sol_neu_loss_batches,
    fb_init_loss_batches, fb_dir_loss_batches, fb_neu_loss_batches,
    FB_loss_batches, Total_loss_batches) = [], [], [], [], [], [], [], [], []

    for i, (data_sol_intrr, 
            data_sol_init, data_sol_dir, #data_sol_neu,
            data_fb_init, data_fb_dir, data_fb_neu) in enumerate(zip(sol_conditions['Interior'], 
                                                                    cycle(sol_conditions['Initial']), 
                                                                    cycle(sol_conditions['Dirichlet']),
                                                                    # cycle(sol_conditions['Neumann']),
                                                                    cycle(fb_conditions['Initial']), 
                                                                    cycle(fb_conditions['Dirichlet']),
                                                                    cycle(fb_conditions['Neumann']),
                                                                    )):

        x_sol_intrr, t_sol_intrr = data_sol_intrr
        x_sol_intrr = x_sol_intrr.to(DEVICE)
        t_sol_intrr = t_sol_intrr.to(DEVICE)
        x_sol_intrr.requires_grad = True
        t_sol_intrr.requires_grad = True
        x_sol_init, y_sol_init = data_sol_init
        x_sol_init = x_sol_init.to(DEVICE)
        y_sol_init = y_sol_init.to(DEVICE)
        x_sol_dir, y_sol_dir = data_sol_dir
        x_sol_dir = x_sol_dir.to(DEVICE)
        y_sol_dir = y_sol_dir.to(DEVICE)

        x_fb_init, y_fb_init = data_fb_init
        x_fb_init = x_fb_init.to(DEVICE)
        y_fb_init = y_fb_init.to(DEVICE)
        x_fb_dir = data_fb_dir
        x_fb_dir = x_fb_dir.to(DEVICE)
        x_fb_neu, y_fb_neu = data_fb_neu
        x_fb_neu = x_fb_neu.to(DEVICE)
        y_fb_neu = y_fb_neu.to(DEVICE)
        x_fb_neu.requires_grad = True

        # zero parameter gradients and then compute NN prediction of gradient
        sol_model.zero_grad()
        fb_model.zero_grad()
        
        #--------------- Compute Free Boundary losses
        # Compute Initial Free Boundary Condition
        fb_init_NN = fb_model(x_fb_init)
        fb_init_loss = torch.mean((fb_init_NN - y_fb_init) ** 2)

        # Compute Dirichlet Free Boundary Condition
        s_values = fb_model(x_fb_dir)
        fb_dir_NN = sol_model(torch.cat([s_values, x_fb_dir], dim=1))
        fb_dir_target = torch.relu(torch.ones_like(s_values) * K - s_values)
        fb_dir_loss = torch.mean((fb_dir_NN - fb_dir_target)**2)

        # Compute Neumann Free Boundary Condition
        s_values = fb_model(x_fb_neu)
        fb_neu_NN = sol_model(torch.cat([s_values, x_fb_neu], dim=1))
        fb_neu_NN_gradS = torch.autograd.grad(outputs=fb_neu_NN, inputs=s_values, grad_outputs=torch.ones_like(fb_neu_NN), create_graph=True)[0]
        fb_neu_loss = torch.mean((fb_neu_NN_gradS - y_fb_neu)**2)

        # Compute FB loss
        FB_loss = config.fb_weight[0] * fb_init_loss + \
                  config.fb_weight[1] * fb_dir_loss + \
                  config.fb_weight[2] * fb_neu_loss

        #--------------- Compute PINN losses
        # Compute unsupervised loss
        s_values = fb_model(torch.cat([t_sol_intrr], dim=-1))
        x_sol_intrr_gt_Bt = x_sol_intrr[ x_sol_intrr > s_values ].view(-1,1)    # 大于B(t)的内点，S取值
        t_sol_intrr_gt_Bt = t_sol_intrr[ x_sol_intrr > s_values ].view(-1,1)    # 大于B(t)的内点，t取值
        sol_intrr_NN = sol_model(torch.cat([x_sol_intrr_gt_Bt, t_sol_intrr_gt_Bt], dim=1))
        sol_intrr_NN_gradS = torch.autograd.grad(outputs=sol_intrr_NN, inputs=x_sol_intrr_gt_Bt, grad_outputs=torch.ones_like(sol_intrr_NN), create_graph=True)[0]
        Unsupervised_loss = torch.mean((pde(x_sol_intrr_gt_Bt, t_sol_intrr_gt_Bt, sol_intrr_NN, sol_intrr_NN_gradS))**2)

        # Compute Initial Condition
        sol_init_NN = sol_model(x_sol_init)
        if len(sol_init_NN.shape) > 1:
            sol_init_loss = torch.mean((sol_init_NN - y_sol_init.unsqueeze(-1))**2)
        else:
            sol_init_loss = torch.mean((sol_init_NN - y_sol_init)**2)

        # Compute Dirichlet Boundary Condition
        sol_dir_NN = sol_model(x_sol_dir)
        sol_dir_loss = torch.mean((sol_dir_NN - y_sol_dir)**2)

        # Compute Neumann Boundary Condition
        sol_neu_loss = 0

        #--------------- Compute total loss
        Total_loss = (config.pde_weight * Unsupervised_loss +
                      config.sup_weight[0] * sol_init_loss +
                      config.sup_weight[1] * sol_dir_loss +
                      config.sup_weight[2] * sol_neu_loss +
                      config.fb_weight[1] * fb_dir_loss +
                      config.fb_weight[2] * fb_neu_loss )
        
        # print_grad()

        # zero parameter gradients
        sol_optimizer.zero_grad()
        fb_optimizer.zero_grad()
        # backpropagation
        Total_loss.backward()
        # parameter update    [ fb_model don't need update ]
        sol_optimizer.step()

        # integrate loss over the entire training datset
        Unsupervised_loss_batches.append(Unsupervised_loss.item())
        sol_init_loss_batches.append(sol_init_loss.item())
        sol_dir_loss_batches.append(sol_dir_loss.item())
        sol_neu_loss_batches.append(sol_neu_loss)
        fb_init_loss_batches.append(fb_init_loss.item())
        fb_dir_loss_batches.append(fb_dir_loss.item())
        fb_neu_loss_batches.append(fb_neu_loss.item())
        FB_loss_batches.append(FB_loss.item())
        Total_loss_batches.append(Total_loss.item())

    
    return (np.mean(Unsupervised_loss_batches), 
            np.mean(sol_init_loss_batches), np.mean(sol_dir_loss_batches), np.mean(sol_neu_loss_batches),
            np.mean(fb_init_loss_batches), np.mean(fb_dir_loss_batches), np.mean(fb_neu_loss_batches),
            np.mean(FB_loss_batches), np.mean(Total_loss_batches)
            )

def train_fb_model():

    # set model to training mode
    fb_model.train()

    (fb_init_loss_batches, fb_dir_loss_batches, fb_neu_loss_batches,
    FB_loss_batches) = [], [], [], []

    for i, (data_sol_intrr, 
            data_sol_init, data_sol_dir, #data_sol_neu,
            data_fb_init, data_fb_dir, data_fb_neu) in enumerate(zip(sol_conditions['Interior'], 
                                                                    cycle(sol_conditions['Initial']), 
                                                                    cycle(sol_conditions['Dirichlet']),
                                                                    # cycle(sol_conditions['Neumann']),
                                                                    cycle(fb_conditions['Initial']), 
                                                                    cycle(fb_conditions['Dirichlet']),
                                                                    cycle(fb_conditions['Neumann']),
                                                                    )):
        
        x_sol_intrr, t_sol_intrr = data_sol_intrr
        x_sol_intrr = x_sol_intrr.to(DEVICE)
        t_sol_intrr = t_sol_intrr.to(DEVICE)
        x_sol_intrr.requires_grad = True
        t_sol_intrr.requires_grad = True
        x_sol_init, y_sol_init = data_sol_init
        x_sol_init = x_sol_init.to(DEVICE)
        y_sol_init = y_sol_init.to(DEVICE)
        x_sol_dir, y_sol_dir = data_sol_dir
        x_sol_dir = x_sol_dir.to(DEVICE)
        y_sol_dir = y_sol_dir.to(DEVICE)

        x_fb_init, y_fb_init = data_fb_init
        x_fb_init = x_fb_init.to(DEVICE)
        y_fb_init = y_fb_init.to(DEVICE)
        x_fb_dir = data_fb_dir
        x_fb_dir = x_fb_dir.to(DEVICE)
        x_fb_neu, y_fb_neu = data_fb_neu
        x_fb_neu = x_fb_neu.to(DEVICE)
        y_fb_neu = y_fb_neu.to(DEVICE)
        x_fb_neu.requires_grad = True

        # zero parameter gradients and then compute NN prediction of gradient
        sol_model.zero_grad()
        fb_model.zero_grad()
        
        #--------------- Compute Free Boundary losses
        # Compute Initial Free Boundary Condition
        fb_init_NN = fb_model(x_fb_init)
        fb_init_loss = torch.mean((fb_init_NN - y_fb_init) ** 2)

        # Compute Dirichlet Free Boundary Condition
        s_values = fb_model(x_fb_dir)
        fb_dir_NN = sol_model(torch.cat([s_values, x_fb_dir], dim=1))
        fb_dir_target = torch.relu(torch.ones_like(s_values) * K - s_values)
        fb_dir_loss = torch.mean((fb_dir_NN - fb_dir_target)**2)

        # Compute Neumann Free Boundary Condition
        s_values = fb_model(x_fb_neu)
        fb_neu_NN = sol_model(torch.cat([s_values, x_fb_neu], dim=1))
        fb_neu_NN_gradS = torch.autograd.grad(outputs=fb_neu_NN, inputs=s_values, grad_outputs=torch.ones_like(fb_neu_NN), create_graph=True)[0]
        fb_neu_loss = torch.mean((fb_neu_NN_gradS - y_fb_neu)**2)

        # Compute FB loss
        FB_loss = config.fb_weight[0] * fb_init_loss + \
                  config.fb_weight[1] * fb_dir_loss + \
                  config.fb_weight[2] * fb_neu_loss

        # print_grad()

        # zero parameter gradients
        sol_optimizer.zero_grad()
        fb_optimizer.zero_grad()
        # backpropagation
        FB_loss.backward()
        # parameter update  [ sol_model don't need update ]
        fb_optimizer.step()

        # integrate loss over the entire training datset
        fb_init_loss_batches.append(fb_init_loss.item())
        fb_dir_loss_batches.append(fb_dir_loss.item())
        fb_neu_loss_batches.append(fb_neu_loss.item())
        FB_loss_batches.append(FB_loss.item())

    
    return np.mean(fb_init_loss_batches), np.mean(fb_dir_loss_batches), np.mean(fb_neu_loss_batches), np.mean(FB_loss_batches)

def train_fb_sol_model():

    # set model to training mode
    sol_model.train()
    fb_model.train()

    (Unsupervised_loss_batches, 
    sol_init_loss_batches, sol_dir_loss_batches, sol_neu_loss_batches,
    fb_init_loss_batches, fb_dir_loss_batches, fb_neu_loss_batches,
    FB_loss_batches, Total_loss_batches) = [], [], [], [], [], [], [], [], []

    for i, (data_sol_intrr, 
            data_sol_init, data_sol_dir, #data_sol_neu,
            data_fb_init, data_fb_dir, data_fb_neu) in enumerate(zip(sol_conditions['Interior'], 
                                                                    cycle(sol_conditions['Initial']), 
                                                                    cycle(sol_conditions['Dirichlet']),
                                                                    # cycle(sol_conditions['Neumann']),
                                                                    cycle(fb_conditions['Initial']), 
                                                                    cycle(fb_conditions['Dirichlet']),
                                                                    cycle(fb_conditions['Neumann']),
                                                                    )):

        x_sol_intrr, t_sol_intrr = data_sol_intrr
        x_sol_intrr = x_sol_intrr.to(DEVICE)
        t_sol_intrr = t_sol_intrr.to(DEVICE)
        x_sol_intrr.requires_grad = True
        t_sol_intrr.requires_grad = True
        x_sol_init, y_sol_init = data_sol_init
        x_sol_init = x_sol_init.to(DEVICE)
        y_sol_init = y_sol_init.to(DEVICE)
        x_sol_dir, y_sol_dir = data_sol_dir
        x_sol_dir = x_sol_dir.to(DEVICE)
        y_sol_dir = y_sol_dir.to(DEVICE)

        x_fb_init, y_fb_init = data_fb_init
        x_fb_init = x_fb_init.to(DEVICE)
        y_fb_init = y_fb_init.to(DEVICE)
        x_fb_dir = data_fb_dir
        x_fb_dir = x_fb_dir.to(DEVICE)
        x_fb_neu, y_fb_neu = data_fb_neu
        x_fb_neu = x_fb_neu.to(DEVICE)
        y_fb_neu = y_fb_neu.to(DEVICE)
        x_fb_neu.requires_grad = True

        # zero parameter gradients and then compute NN prediction of gradient
        sol_model.zero_grad()
        fb_model.zero_grad()
        
        #--------------- Compute Free Boundary losses
        # Compute Initial Free Boundary Condition
        fb_init_NN = fb_model(x_fb_init)
        fb_init_loss = torch.mean((fb_init_NN - y_fb_init) ** 2)

        # Compute Dirichlet Free Boundary Condition
        s_values = fb_model(x_fb_dir)
        fb_dir_NN = sol_model(torch.cat([s_values, x_fb_dir], dim=1))
        fb_dir_target = torch.relu(torch.ones_like(s_values) * K - s_values)
        fb_dir_loss = torch.mean((fb_dir_NN - fb_dir_target)**2)

        # Compute Neumann Free Boundary Condition
        s_values = fb_model(x_fb_neu)
        fb_neu_NN = sol_model(torch.cat([s_values, x_fb_neu], dim=1))
        fb_neu_NN_gradS = torch.autograd.grad(outputs=fb_neu_NN, inputs=s_values, grad_outputs=torch.ones_like(fb_neu_NN), create_graph=True)[0]
        fb_neu_loss = torch.mean((fb_neu_NN_gradS - y_fb_neu)**2)

        # Compute FB loss
        FB_loss = config.fb_weight[0] * fb_init_loss + \
                  config.fb_weight[1] * fb_dir_loss + \
                  config.fb_weight[2] * fb_neu_loss

        # print('@'*50)
        # print_grad()

        # zero parameter gradients
        sol_optimizer.zero_grad()
        fb_optimizer.zero_grad()
        # backpropagation
        FB_loss.backward(retain_graph=True)
        # parameter update

        #--------------- Compute PINN losses
        # Compute unsupervised loss
        s_values = fb_model(torch.cat([t_sol_intrr], dim=-1))
        x_sol_intrr_gt_Bt = x_sol_intrr[ x_sol_intrr > s_values ].view(-1,1)    # 大于B(t)的内点，S取值
        t_sol_intrr_gt_Bt = t_sol_intrr[ x_sol_intrr > s_values ].view(-1,1)    # 大于B(t)的内点，t取值
        sol_intrr_NN = sol_model(torch.cat([x_sol_intrr_gt_Bt, t_sol_intrr_gt_Bt], dim=1))
        sol_intrr_NN_gradS = torch.autograd.grad(outputs=sol_intrr_NN, inputs=x_sol_intrr_gt_Bt, grad_outputs=torch.ones_like(sol_intrr_NN), create_graph=True)[0]
        Unsupervised_loss = torch.mean((pde(x_sol_intrr_gt_Bt, t_sol_intrr_gt_Bt, sol_intrr_NN, sol_intrr_NN_gradS))**2)

        # Compute Initial Condition
        sol_init_NN = sol_model(x_sol_init)
        if len(sol_init_NN.shape) > 1:
            sol_init_loss = torch.mean((sol_init_NN - y_sol_init.unsqueeze(-1))**2)
        else:
            sol_init_loss = torch.mean((sol_init_NN - y_sol_init)**2)

        # Compute Dirichlet Boundary Condition
        sol_dir_NN = sol_model(x_sol_dir)
        sol_dir_loss = torch.mean((sol_dir_NN - y_sol_dir)**2)

        # Compute Neumann Boundary Condition
        sol_neu_loss = 0

        #--------------- Compute total loss
        Total_loss = (config.pde_weight * Unsupervised_loss +
                      config.sup_weight[0] * sol_init_loss +
                      config.sup_weight[1] * sol_dir_loss +
                      config.sup_weight[2] * sol_neu_loss +
                      config.fb_weight[1] * fb_dir_loss +
                      config.fb_weight[2] * fb_neu_loss )
        
        # print_grad()

        # backpropagation
        Total_loss.backward()
        # parameter update
        fb_optimizer.step()
        sol_optimizer.step()

        # integrate loss over the entire training datset
        Unsupervised_loss_batches.append(Unsupervised_loss.item())
        sol_init_loss_batches.append(sol_init_loss.item())
        sol_dir_loss_batches.append(sol_dir_loss.item())
        sol_neu_loss_batches.append(sol_neu_loss)
        fb_init_loss_batches.append(fb_init_loss.item())
        fb_dir_loss_batches.append(fb_dir_loss.item())
        fb_neu_loss_batches.append(fb_neu_loss.item())
        FB_loss_batches.append(FB_loss.item())
        Total_loss_batches.append(Total_loss.item())

    
    return (np.mean(Unsupervised_loss_batches), 
            np.mean(sol_init_loss_batches), np.mean(sol_dir_loss_batches), np.mean(sol_neu_loss_batches),
            np.mean(fb_init_loss_batches), np.mean(fb_dir_loss_batches), np.mean(fb_neu_loss_batches),
            np.mean(FB_loss_batches), np.mean(Total_loss_batches)
            )
# endregion
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

# create optimizer and learning rate schedular
sol_optimizer = torch.optim.RMSprop(sol_model.parameters(), lr=config.sol_lr)
sol_schedular = torch.optim.lr_scheduler.ExponentialLR(sol_optimizer, gamma=0.9)
fb_optimizer = torch.optim.RMSprop(fb_model.parameters(), lr=config.fb_lr)
fb_schedular = torch.optim.lr_scheduler.ExponentialLR(fb_optimizer, gamma=0.9)

# load model to device
sol_model = sol_model.to(DEVICE)
fb_model = fb_model.to(DEVICE)

# create log file
logger = helper.Logger(os.path.join(checkpoint_path, 'log.txt'), title='American-Option-PINN-1d')
logger.set_names(['Epoch', 'Learning Rate', 'Unsupervised Loss', 'Supervised Loss', 'Free Boundary Loss', 'Total Loss', 
                  'sol_init_loss', 'sol_dir_loss', 'sol_neu_loss', 'fb_init_loss', 'fb_dir_loss', 'fb_neu_loss'])
## ------------------------- ##
# Unsupervised Loss  = interior_loss
# Supervised Loss    = sol_init_loss + sol_dir_loss + sol_neu_loss
# Free Boundary Loss = fb_init_loss + fb_dir_loss + fb_neu_loss
# Total Loss         = Unsupervised Loss 
#                      + sol_init_loss + sol_dir_loss + sol_neu_loss
#                      + fb_dir_loss + fb_neu_loss
## ------------------------- ##


#Early warning initialization
early_warning = {'Target': 1e10, 'n_steps': 0}

# Training
(u_losses,
sol_init_losses, sol_dir_losses, sol_neu_losses,
fb_init_losses, fb_dir_losses, fb_neu_losses,
fb_losses, total_losses) = [], [], [], [], [], [], [], [], []

since = time.time()
for epoch in tqdm(range(config.epochs), desc='PINNs - Training'):
    
    # print('Epoch {}/{}'.format(epoch, params['epochs']-1), 'with LR = {:.1e}'.format(sol_optimizer.param_groups[0]['lr']))  

    # Case 1: more mdl steps for a single fb step
    for _ in range(config.steps_fb_per_pde - 1):
        train_sol_model()
    
    # Case 2: more fb steps for a single mdl step
    for _ in range(0, config.steps_fb_per_pde+1, -1):
        train_fb_model()

    u_loss, sol_i_l, sol_d_l, sol_n_l, fb_i_l, fb_d_l, fb_n_l, fb_l, total_l = train_fb_sol_model()

    # save current and best models to checkpoint
    is_best = total_l < early_warning['Target']
    helper.save_checkpoint({'epoch': epoch+1,
                            'state_dict': sol_model.state_dict(),
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
                            'optimizer': sol_optimizer.state_dict(),
                           }, is_best, checkpoint=checkpoint_path)
    # save training process to log file
    logger.append([epoch+1, sol_optimizer.param_groups[0]['lr'], 
                   u_loss, sol_i_l + sol_d_l + sol_n_l, fb_l, total_l, 
                   sol_i_l, sol_d_l, sol_n_l, fb_i_l, fb_d_l, fb_n_l, 
                   ])
    
    # adjust learning rate according to predefined schedule
    sol_schedular.step()
    fb_schedular.step()

    # record losses
    u_losses.append(u_loss)
    sol_init_losses.append(sol_i_l)
    sol_dir_losses.append(sol_d_l)
    sol_neu_losses.append(sol_n_l)
    fb_init_losses.append(fb_i_l)
    fb_dir_losses.append(fb_d_l)
    fb_neu_losses.append(fb_n_l)
    fb_losses.append(fb_l)
    total_losses.append(total_l)

    # print results
    print_base = "{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}"
    if epoch == 0:
        print(print_base.format(
                '|Epoch', '|Learning Rate', '|Unsupervised Loss', '|Supervised Loss', '|Free Boundary Loss', '|Total Loss', 
                '|sol_init_loss', '|sol_dir_loss', '|sol_neu_loss', '|fb_init_loss', '|fb_dir_loss', '|fb_neu_loss'))
        print(print_base.format('='*10, '='*20, '='*20, '='*20, '='*20, '='*20, '='*20, '='*20, '='*20, '='*20, '='*20, '='*20))
        print('\n')
    if epoch % config.verbose == 0:
        print(print_base.format(epoch+1,
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
                                ))
    
    # update early_warning
    early_warning['Target'] = min(total_l, early_warning['Target'])
    if is_best:
        print('==> Saving best model ...')
        early_warning['n_steps'] = 0
    else:
        early_warning['n_steps'] += 1
        if early_warning['n_steps'] >= config.patience:
            break

logger.close()
time_elapsed = time.time() - since

print('Done in {}'.format(str(datetime.timedelta(seconds=time_elapsed))), '!')
print('*', '-' * 45, '*', "\n", "\n")
# endregion
##############################################################################################


##############################################################################################
## Save results
## ------------------------- ##
plot.plot_loss(os.path.join(image_1d_path, f'LOSS_seed_{seed}.png'), 
               u_losses, 
               sol_init_losses, 
               sol_dir_losses, 
               sol_neu_losses, 
               fb_losses, 
               total_losses)
##############################################################################################






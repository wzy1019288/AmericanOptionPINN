
import os
import torch
import numpy as np
import pandas as pd
from itertools import cycle
from functools import lru_cache

from Utils.helper import (
    record_grad,
    all_model_zero_grad,
    all_optimizer_zero_grad,
)


##############################################################################################
## Train
## ------------------------- ##
def train_sol_model(
        sol_model, fb_model,
        sol_optimizer, fb_optimizer,
        pde,
        sol_conditions, fb_conditions,
        config, DEVICE,
        **kwargs
):

    n_dim = kwargs['n_dim']
    K = kwargs['K']

    # set model to training mode
    sol_model.train()

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

        for _ in range(config.steps_fb_per_pde - 1):
            
            x_sol_intrr, t_sol_intrr = data_sol_intrr
            x_sol_intrr = x_sol_intrr.to(DEVICE)
            t_sol_intrr = t_sol_intrr.to(DEVICE)
            x_sol_intrr.requires_grad = True
            t_sol_intrr.requires_grad = True
            x_sol_init, y_sol_init = data_sol_init
            x_sol_init = x_sol_init.to(DEVICE)
            y_sol_init = y_sol_init.to(DEVICE)
            x_sol_init.requires_grad = True
            y_sol_init.requires_grad = True
            x_sol_dir, y_sol_dir = data_sol_dir
            x_sol_dir = x_sol_dir.to(DEVICE)
            y_sol_dir = y_sol_dir.to(DEVICE)
            x_sol_dir.requires_grad = True

            x_fb_init, y_fb_init = data_fb_init
            x_fb_init = x_fb_init.to(DEVICE)
            y_fb_init = y_fb_init.to(DEVICE)
            x_fb_init.requires_grad = True
            x_fb_dir = data_fb_dir
            x_fb_dir = x_fb_dir.to(DEVICE)
            x_fb_dir.requires_grad = True
            x_fb_neu = data_fb_neu
            x_fb_neu = x_fb_neu.to(DEVICE)
            x_fb_neu.requires_grad = True

            # Compute Initial Free Boundary Condition
            fb_init_NN = fb_model(x_fb_init)

            # Compute Dirichlet Free Boundary Condition
            s_values1 = fb_model(x_fb_dir)
            fb_dir_NN = sol_model(torch.cat([s_values1, x_fb_dir], dim=1))

            # Compute Neumann Free Boundary Condition
            s_values2 = fb_model(x_fb_neu)
            fb_neu_NN = sol_model(torch.cat([s_values2, x_fb_neu], dim=1))

            # Compute Interior Condition
            s_values3 = fb_model(torch.cat([t_sol_intrr], dim=-1))
            temp = torch.sum((x_sol_intrr < s_values3).int(),
                            dim=-1) < torch.ones(t_sol_intrr.shape[0]).to(DEVICE)   # 判断是否所有资产S都大于B(t)
            x_f = x_sol_intrr[ temp ]   # 大于B(t)的内点，S取值
            t_f = torch.reshape(t_sol_intrr[ temp ], (-1,1) )   # 大于B(t)的内点，t取值
            variables = list()
            derivatives = list()
            for i in range(x_f.shape[1]):
                variables.append( x_f[:, i:i+1] )
            variables.append( t_f )
            u_val = sol_model(torch.cat(variables, dim=1)).requires_grad_()

            # Compute Initial Condition
            sol_init_NN = sol_model(x_sol_init)

            # Compute Dirichlet Boundary Condition
            sol_dir_NN = sol_model(x_sol_dir)

            # Compute Neumann Boundary Condition
            sol_neu_loss = 0

            # zero parameter gradients and then compute NN prediction of gradient
            sol_model, fb_model = all_model_zero_grad(sol_model, fb_model)

            #--------------- Compute Free Boundary losses
            fb_init_loss = torch.mean((fb_init_NN - y_fb_init) ** 2)

            fb_dir_target = torch.relu(torch.ones_like(s_values1[:,-1]) * K - torch.min(s_values1, dim=1).values)
            fb_dir_loss = torch.mean((fb_dir_NN - fb_dir_target)**2)

            fb_neu_NN_gradS = torch.autograd.grad(outputs=fb_neu_NN, inputs=s_values2, grad_outputs=torch.ones_like(fb_neu_NN), retain_graph=True, create_graph=True)[0]
            fb_neu_target = -torch.nn.functional.one_hot(
                                torch.argmin( s_values2, dim=1),
                                num_classes=n_dim
                                )
            fb_neu_loss = torch.mean((fb_neu_NN_gradS - fb_neu_target)**2)

            #--------------- Compute PINN losses
            if len(sol_init_NN.shape) > 1:
                sol_init_loss = torch.mean((sol_init_NN - y_sol_init.unsqueeze(-1))**2)
            else:
                sol_init_loss = torch.mean((sol_init_NN - y_sol_init)**2)
            
            if len(sol_dir_NN.shape) > 1:
                sol_dir_loss = torch.mean((sol_dir_NN - y_sol_dir.unsqueeze(-1))**2)
            else:
                sol_dir_loss = torch.mean((sol_dir_NN - y_sol_dir)**2)

            #--------------- Compute unsupervised loss
            for i, var in enumerate(variables[:-1]):
                # derivatives = [gradu_x1, gradu_x2]
                derivatives.append( torch.autograd.grad(outputs=u_val,
                                                        inputs=var,
                                                        grad_outputs=torch.ones_like(u_val),
                                                        retain_graph=True, create_graph=True)[0] )
            Unsupervised_loss = torch.mean((pde(variables, u_val, derivatives))**2)

            #--------------- Compute FB loss
            FB_loss = config.fb_weight[0] * fb_init_loss + \
                      config.fb_weight[1] * fb_dir_loss + \
                      config.fb_weight[2] * fb_neu_loss

            #--------------- Compute total loss
            Total_loss = (config.pde_weight * Unsupervised_loss +
                          config.sup_weight[0] * sol_init_loss +
                          config.sup_weight[1] * sol_dir_loss +
                          config.sup_weight[2] * sol_neu_loss +
                          config.fb_weight[1] * fb_dir_loss +
                          config.fb_weight[2] * fb_neu_loss )
            
            # zero parameter gradients
            sol_optimizer, fb_optimizer = all_optimizer_zero_grad(sol_optimizer, fb_optimizer)
            # backpropagation
            Total_loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=sol_model.parameters(), max_norm=1, norm_type=float('inf'))
            # parameter update    [ fb_model don't need update ]
            sol_optimizer.step()

    return sol_model, fb_model, sol_optimizer, fb_optimizer


def train_fb_model(
        sol_model, fb_model,
        sol_optimizer, fb_optimizer,
        sol_conditions, fb_conditions,
        config, DEVICE,
        **kwargs
):

    n_dim = kwargs['n_dim']
    K = kwargs['K']

    # set model to training mode
    fb_model.train()

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
        
        for _ in range(0, config.steps_fb_per_pde+1, -1):
        
            x_sol_intrr, t_sol_intrr = data_sol_intrr
            x_sol_intrr = x_sol_intrr.to(DEVICE)
            t_sol_intrr = t_sol_intrr.to(DEVICE)
            x_sol_intrr.requires_grad = True
            t_sol_intrr.requires_grad = True
            x_sol_init, y_sol_init = data_sol_init
            x_sol_init = x_sol_init.to(DEVICE)
            y_sol_init = y_sol_init.to(DEVICE)
            x_sol_init.requires_grad = True
            y_sol_init.requires_grad = True
            x_sol_dir, y_sol_dir = data_sol_dir
            x_sol_dir = x_sol_dir.to(DEVICE)
            y_sol_dir = y_sol_dir.to(DEVICE)
            x_sol_dir.requires_grad = True

            x_fb_init, y_fb_init = data_fb_init
            x_fb_init = x_fb_init.to(DEVICE)
            y_fb_init = y_fb_init.to(DEVICE)
            x_fb_init.requires_grad = True
            x_fb_dir = data_fb_dir
            x_fb_dir = x_fb_dir.to(DEVICE)
            x_fb_dir.requires_grad = True
            x_fb_neu = data_fb_neu
            x_fb_neu = x_fb_neu.to(DEVICE)
            x_fb_neu.requires_grad = True

            # Compute Initial Free Boundary Condition
            fb_init_NN = fb_model(x_fb_init)

            # Compute Dirichlet Free Boundary Condition
            s_values1 = fb_model(x_fb_dir)
            fb_dir_NN = sol_model(torch.cat([s_values1, x_fb_dir], dim=1))

            # Compute Neumann Free Boundary Condition
            s_values2 = fb_model(x_fb_neu)
            fb_neu_NN = sol_model(torch.cat([s_values2, x_fb_neu], dim=1))
            
            sol_model, fb_model = all_model_zero_grad(sol_model, fb_model)

            #--------------- Compute Free Boundary losses
            fb_init_loss = torch.mean((fb_init_NN - y_fb_init) ** 2)

            fb_dir_target = torch.relu(torch.ones_like(s_values1[:,-1]) * K - torch.min(s_values1, dim=1).values)
            fb_dir_loss = torch.mean((fb_dir_NN - fb_dir_target)**2)

            fb_neu_NN_gradS = torch.autograd.grad(outputs=fb_neu_NN, inputs=s_values2, grad_outputs=torch.ones_like(fb_neu_NN), retain_graph=True, create_graph=True)[0]
            fb_neu_target = -torch.nn.functional.one_hot(
                                torch.argmin( s_values2, dim=1),
                                num_classes=n_dim
                                )
            fb_neu_loss = torch.mean((fb_neu_NN_gradS - fb_neu_target)**2)

            #--------------- Compute FB loss
            FB_loss = config.fb_weight[0] * fb_init_loss + \
                    config.fb_weight[1] * fb_dir_loss + \
                    config.fb_weight[2] * fb_neu_loss

            # zero parameter gradients
            sol_optimizer, fb_optimizer = all_optimizer_zero_grad(sol_optimizer, fb_optimizer)
            # backpropagation
            FB_loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=fb_model.parameters(), max_norm=1, norm_type=float('inf'))
            # parameter update  [ sol_model don't need update ]
            fb_optimizer.step()

    return sol_model, fb_model, sol_optimizer, fb_optimizer


def train_fb_sol_model(
        sol_model, fb_model,
        sol_optimizer, fb_optimizer,
        pde,
        sol_conditions, fb_conditions,
        config, DEVICE,
        **kwargs
):

    n_dim = kwargs['n_dim']
    K = kwargs['K']
    epoch = kwargs['epoch']

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
        x_sol_init.requires_grad = True
        y_sol_init.requires_grad = True
        x_sol_dir, y_sol_dir = data_sol_dir
        x_sol_dir = x_sol_dir.to(DEVICE)
        y_sol_dir = y_sol_dir.to(DEVICE)
        x_sol_dir.requires_grad = True

        x_fb_init, y_fb_init = data_fb_init
        x_fb_init = x_fb_init.to(DEVICE)
        y_fb_init = y_fb_init.to(DEVICE)
        x_fb_init.requires_grad = True
        x_fb_dir = data_fb_dir
        x_fb_dir = x_fb_dir.to(DEVICE)
        x_fb_dir.requires_grad = True
        x_fb_neu = data_fb_neu
        x_fb_neu = x_fb_neu.to(DEVICE)
        x_fb_neu.requires_grad = True

        # Compute Initial Free Boundary Condition
        fb_init_NN = fb_model(x_fb_init)

        # Compute Dirichlet Free Boundary Condition
        s_values1 = fb_model(x_fb_dir)
        fb_dir_NN = sol_model(torch.cat([s_values1, x_fb_dir], dim=1))

        # Compute Neumann Free Boundary Condition
        s_values2 = fb_model(x_fb_neu)
        fb_neu_NN = sol_model(torch.cat([s_values2, x_fb_neu], dim=1))

        # Compute Interior Condition
        s_values3 = fb_model(torch.cat([t_sol_intrr], dim=-1))
        temp = torch.sum((x_sol_intrr < s_values3).int(),
                        dim=-1) < torch.ones(t_sol_intrr.shape[0]).to(DEVICE)   # 判断是否所有资产S都大于B(t)
        x_f = x_sol_intrr[ temp ]   # 大于B(t)的内点，S取值
        t_f = torch.reshape(t_sol_intrr[ temp ], (-1,1) )   # 大于B(t)的内点，t取值
        variables = list()
        derivatives = list()
        for i in range(x_f.shape[1]):
            variables.append( x_f[:, i:i+1] )
        variables.append( t_f )
        u_val = sol_model(torch.cat(variables, dim=1)).requires_grad_()

        # Compute Initial Condition
        sol_init_NN = sol_model(x_sol_init)

        # Compute Dirichlet Boundary Condition
        sol_dir_NN = sol_model(x_sol_dir)

        # Compute Neumann Boundary Condition
        sol_neu_loss = 0

        # print and save grad (check gradient vanishing/explosion )
        record_grad(sol_model, fb_model, config, epoch)

        # zero parameter gradients and then compute NN prediction of gradient
        sol_model, fb_model = all_model_zero_grad(sol_model, fb_model)

        #--------------- Compute Free Boundary losses
        fb_init_loss = torch.mean((fb_init_NN - y_fb_init) ** 2)

        fb_dir_target = torch.relu(torch.ones_like(s_values1[:,-1]) * K - torch.min(s_values1, dim=1).values)
        fb_dir_loss = torch.mean((fb_dir_NN - fb_dir_target)**2)

        fb_neu_NN_gradS = torch.autograd.grad(outputs=fb_neu_NN, inputs=s_values2, grad_outputs=torch.ones_like(fb_neu_NN), retain_graph=True, create_graph=True)[0]
        fb_neu_target = -torch.nn.functional.one_hot(
                            torch.argmin( s_values2, dim=1),
                            num_classes=n_dim
                            )
        fb_neu_loss = torch.mean((fb_neu_NN_gradS - fb_neu_target)**2)

        #--------------- Compute PINN losses
        if len(sol_init_NN.shape) > 1:
            sol_init_loss = torch.mean((sol_init_NN - y_sol_init.unsqueeze(-1))**2)
        else:
            sol_init_loss = torch.mean((sol_init_NN - y_sol_init)**2)
        
        if len(sol_dir_NN.shape) > 1:
            sol_dir_loss = torch.mean((sol_dir_NN - y_sol_dir.unsqueeze(-1))**2)
        else:
            sol_dir_loss = torch.mean((sol_dir_NN - y_sol_dir)**2)

        #--------------- Compute unsupervised loss
        for i, var in enumerate(variables[:-1]):
            # derivatives = [gradu_x1, gradu_x2]
            derivatives.append( torch.autograd.grad(outputs=u_val,
                                                    inputs=var,
                                                    grad_outputs=torch.ones_like(u_val),
                                                    retain_graph=True, create_graph=True)[0] )
        Unsupervised_loss = torch.mean((pde(variables, u_val, derivatives))**2)
          
        #--------------- Compute FB loss
        FB_loss = config.fb_weight[0] * fb_init_loss + \
                  config.fb_weight[1] * fb_dir_loss + \
                  config.fb_weight[2] * fb_neu_loss

        #--------------- Compute total loss
        Total_loss = (config.pde_weight * Unsupervised_loss +
                      config.sup_weight[0] * sol_init_loss +
                      config.sup_weight[1] * sol_dir_loss +
                      config.sup_weight[2] * sol_neu_loss +
                      config.fb_weight[1] * fb_dir_loss +
                      config.fb_weight[2] * fb_neu_loss )
            
        # zero parameter gradients
        sol_optimizer, fb_optimizer = all_optimizer_zero_grad(sol_optimizer, fb_optimizer)
        # backpropagation / parameter update
        FB_loss.backward(retain_graph=True)
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=fb_model.parameters(), max_norm=1, norm_type=float('inf'))
        fb_optimizer.step()
        sol_init_loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=sol_model.parameters(), max_norm=1, norm_type=float('inf'))
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
    
    return (sol_model, fb_model, sol_optimizer, fb_optimizer), (np.mean(Unsupervised_loss_batches), 
            np.mean(sol_init_loss_batches), np.mean(sol_dir_loss_batches), np.mean(sol_neu_loss_batches),
            np.mean(fb_init_loss_batches), np.mean(fb_dir_loss_batches), np.mean(fb_neu_loss_batches),
            np.mean(FB_loss_batches), np.mean(Total_loss_batches)
            )
##############################################################################################


##############################################################################################
## Test
## ------------------------- ##
@ lru_cache(maxsize=None)
def get_test_data(
    sol_conditions_test_intrr, 
    sol_conditions_test_init, 
    sol_conditions_test_dir, 
    fb_conditions_test_init,
    fb_conditions_test_dir,
    fb_conditions_test_neu,
    DEVICE):

    for i, (data_sol_intrr, 
            data_sol_init, data_sol_dir, #data_sol_neu,
            data_fb_init, data_fb_dir, data_fb_neu) in enumerate(zip(sol_conditions_test_intrr, 
                                                                    cycle(sol_conditions_test_init), 
                                                                    cycle(sol_conditions_test_dir),
                                                                    # cycle(sol_conditions['Neumann']),
                                                                    cycle(fb_conditions_test_init), 
                                                                    cycle(fb_conditions_test_dir),
                                                                    cycle(fb_conditions_test_neu),
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
        x_fb_neu = data_fb_neu
        x_fb_neu = x_fb_neu.to(DEVICE)
        x_fb_neu.requires_grad = True

        return (x_sol_intrr, t_sol_intrr, x_sol_init, y_sol_init, x_sol_dir, y_sol_dir,
                x_fb_init, y_fb_init, x_fb_dir, x_fb_neu)


def test(
        sol_model, fb_model,
        sol_conditions_test, fb_conditions_test,
        pde, 
        config, DEVICE,
        **kwargs
):

    n_dim = kwargs['n_dim']
    K = kwargs['K']

    # set model to testing mode
    sol_model.eval()
    fb_model.eval()

    (Unsupervised_loss_batches, 
    sol_init_loss_batches, sol_dir_loss_batches, sol_neu_loss_batches,
    fb_init_loss_batches, fb_dir_loss_batches, fb_neu_loss_batches,
    FB_loss_batches, Total_loss_batches) = [], [], [], [], [], [], [], [], []

    (x_sol_intrr, t_sol_intrr, x_sol_init, y_sol_init, x_sol_dir, y_sol_dir,
        x_fb_init, y_fb_init, x_fb_dir, x_fb_neu) = get_test_data(
                                                        sol_conditions_test['Interior'],
                                                        sol_conditions_test['Initial'],
                                                        sol_conditions_test['Dirichlet'],
                                                        fb_conditions_test['Initial'],
                                                        fb_conditions_test['Dirichlet'],
                                                        fb_conditions_test['Neumann'],
                                                        DEVICE)

    # Compute Initial Free Boundary Condition
    fb_init_NN = fb_model(x_fb_init)

    # Compute Dirichlet Free Boundary Condition
    s_values1 = fb_model(x_fb_dir)
    fb_dir_NN = sol_model(torch.cat([s_values1, x_fb_dir], dim=1))

    # Compute Neumann Free Boundary Condition
    s_values2 = fb_model(x_fb_neu)
    fb_neu_NN = sol_model(torch.cat([s_values2, x_fb_neu], dim=1))

    # Compute Interior Condition
    s_values3 = fb_model(torch.cat([t_sol_intrr], dim=-1))
    temp = torch.sum((x_sol_intrr < s_values3).int(),
                    dim=-1) < torch.ones(t_sol_intrr.shape[0]).to(DEVICE)   # 判断是否所有资产S都大于B(t)
    x_f = x_sol_intrr[ temp ]   # 大于B(t)的内点，S取值
    t_f = torch.reshape(t_sol_intrr[ temp ], (-1,1) )   # 大于B(t)的内点，t取值
    variables = list()
    derivatives = list()
    for i in range(x_f.shape[1]):
        variables.append( x_f[:, i:i+1] )
    variables.append( t_f )
    u_val = sol_model(torch.cat(variables, dim=1)).requires_grad_()

    # Compute Initial Condition
    sol_init_NN = sol_model(x_sol_init)

    # Compute Dirichlet Boundary Condition
    sol_dir_NN = sol_model(x_sol_dir)

    # Compute Neumann Boundary Condition
    sol_neu_loss = 0

    # zero parameter gradients and then compute NN prediction of gradient
    sol_model, fb_model = all_model_zero_grad(sol_model, fb_model)
    
    #--------------- Compute Free Boundary losses
    fb_init_loss = torch.mean((fb_init_NN - y_fb_init) ** 2)

    fb_dir_target = torch.relu(torch.ones_like(s_values1[:,-1]) * K - torch.min(s_values1, dim=1).values)
    fb_dir_loss = torch.mean((fb_dir_NN - fb_dir_target)**2)

    fb_neu_NN_gradS = torch.autograd.grad(outputs=fb_neu_NN, inputs=s_values2, grad_outputs=torch.ones_like(fb_neu_NN), retain_graph=True, create_graph=True)[0]
    fb_neu_target = -torch.nn.functional.one_hot(
                        torch.argmin( s_values2, dim=1),
                        num_classes=n_dim
                        )
    fb_neu_loss = torch.mean((fb_neu_NN_gradS - fb_neu_target)**2)

    #--------------- Compute PINN losses
    if len(sol_init_NN.shape) > 1:
        sol_init_loss = torch.mean((sol_init_NN - y_sol_init.unsqueeze(-1))**2)
    else:
        sol_init_loss = torch.mean((sol_init_NN - y_sol_init)**2)
    
    if len(sol_dir_NN.shape) > 1:
        sol_dir_loss = torch.mean((sol_dir_NN - y_sol_dir.unsqueeze(-1))**2)
    else:
        sol_dir_loss = torch.mean((sol_dir_NN - y_sol_dir)**2)

    #--------------- Compute unsupervised loss
    for i, var in enumerate(variables[:-1]):
        # derivatives = [gradu_x1, gradu_x2]
        derivatives.append( torch.autograd.grad(outputs=u_val,
                                                inputs=var,
                                                grad_outputs=torch.ones_like(u_val),
                                                retain_graph=True, create_graph=True)[0] )
    Unsupervised_loss = torch.mean((pde(variables, u_val, derivatives))**2)

    #--------------- Compute FB loss
    FB_loss = config.fb_weight[0] * fb_init_loss + \
              config.fb_weight[1] * fb_dir_loss + \
              config.fb_weight[2] * fb_neu_loss
    
    #--------------- Compute total loss
    Total_loss = (Unsupervised_loss +
                  sol_init_loss +
                  sol_dir_loss +
                  sol_neu_loss +
                  fb_dir_loss +
                  fb_neu_loss )

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
##############################################################################################

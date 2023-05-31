
import os
import torch
import pandas as pd

from config import OptionConfig
from Utils import plot
from Model.fnn import FNN


config = OptionConfig()
seed = config.seed
image_1d_path = config.image_1d_path
checkpoint_path = config.checkpoint_path

df_loss = pd.read_csv(os.path.join(checkpoint_path, 'log.txt'), sep='\t').dropna(axis=1)
u_losses = df_loss['Unsupervised Loss']
sol_init_losses = df_loss['sol_init_loss']
sol_dir_losses = df_loss['sol_dir_loss']
sol_neu_losses = df_loss['sol_neu_loss']
fb_losses = df_loss['Free Boundary Loss']
total_losses = df_loss['Total Loss']

plot.plot_loss(os.path.join(image_1d_path, f'LOSS_seed_{seed}.png'), 
               u_losses, 
               sol_init_losses, 
               sol_dir_losses, 
               sol_neu_losses, 
               fb_losses, 
               total_losses)


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

checkpoint = torch.load(os.path.join(checkpoint_path, 'model_best.pth.tar'))
sol_model.load_state_dict(checkpoint['sol_state_dict'])
fb_model.load_state_dict(checkpoint['fb_state_dict'])


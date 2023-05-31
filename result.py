
import os
import pandas as pd

from config import OptionConfig
from Utils import plot


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
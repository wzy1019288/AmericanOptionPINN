
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(path, 
              u_losses, 
              sol_init_losses, 
              sol_dir_losses, 
              sol_neu_losses, 
              fb_losses, 
              total_losses
            ):
    plt.figure( figsize=(12,8) )
    plt.semilogy(u_losses, label='Unsupervised')
    plt.semilogy(np.array(sol_init_losses) +\
                    np.array(sol_dir_losses) +\
                    np.array(sol_neu_losses),
                    label='Supervised')
    plt.semilogy(fb_losses, label='Free Boundary')
    plt.semilogy(total_losses, label='PINN')
    plt.legend()
    plt.title('Losses')
    plt.savefig(path)
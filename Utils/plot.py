
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

def plot_free_boundary(
        t, fb_max, fb_mean, fb_min, pde_max,
        path,
    ):
    fig = plt.figure()
    plt.plot(t, fb_max, c='orange', linestyle='--')
    plt.plot(t, fb_mean, c=(0, 0, 0.5), alpha=0.5, label='Mean Value' )
    plt.plot(t, fb_min, c='orange', linestyle='--', label='Bollinger Bands' )

    plt.fill_between(t, fb_max, pde_max, where=(pde_max >= fb_max), interpolate=True, 
                    color=(0., 0., 1.0), alpha=0.8, label='Continuation Region')
    plt.fill_between(t, fb_min, fb_max, where=(fb_max >= fb_min), interpolate=True, 
                    color=(0.5, 0.5, 1.0), alpha=0.3, label='Continuation Region\nConfidence Interval')

    plt.xlabel('Time(t)')
    plt.ylabel('Underlying Value(S)')
    plt.ylim((6., 14.))
    plt.legend(loc = 'lower right')
    plt.title('Free Boundary')
    plt.savefig(path)
    plt.close(fig)

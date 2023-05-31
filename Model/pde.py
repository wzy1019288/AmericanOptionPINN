
import torch


def _pde(xs, ts, u_val, u_x, r, sigma):

    u_xx = torch.autograd.grad(u_x, xs, create_graph=True, grad_outputs=torch.ones_like(u_x))[0]
    u_t = torch.autograd.grad(u_val, ts, create_graph=True, grad_outputs=torch.ones_like(u_val))[0]
    f = (r * xs * u_x) + u_t + (sigma**2 * xs**2 * u_xx)/2 - (r * u_val)
    return f

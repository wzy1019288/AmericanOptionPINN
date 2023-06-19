
import torch


def pde_dim1(xs, ts, u_val, u_x, r, sigma):
    u_xx = torch.autograd.grad(u_x, xs, create_graph=True, grad_outputs=torch.ones_like(u_x))[0]
    u_t = torch.autograd.grad(u_val, ts, create_graph=True, grad_outputs=torch.ones_like(u_val))[0]
    f = (r * xs * u_x) + u_t + (sigma**2 * xs**2 * u_xx)/2 - (r * u_val)
    return f

def pde_dim2(variables, u_val, derivatives, r, d, alphas):
    second_order = 0
    first_order = 0
    zero_order = 0
    u_t = torch.autograd.grad(outputs=u_val,
                                inputs=variables[-1],
                                grad_outputs=torch.ones_like(u_val),
                                retain_graph=True, create_graph=True)[0]
    zero_order += u_t - r * u_val
    for i, var_i in enumerate(variables[:-1]):
        # break
        gradu_xi = derivatives[i]
        first_order += d[i] * var_i * gradu_xi
        gradu2_xixi = torch.autograd.grad(outputs=gradu_xi,
                                            inputs=var_i,
                                            grad_outputs=torch.ones_like(gradu_xi),
                                            retain_graph=True, create_graph=True)[0]
        # [0, 0]
        # [1, 1], then [1, 0] (*2) (, [0, 1])
        # [2, 2], then [2, 0], [2, 1] (*2) (, [0, 2], [1, 2])
        second_order += gradu2_xixi * (var_i**2) * alphas[i, i]
        for j, var_j in enumerate(variables[:i]):
            gradu2_xixj = torch.autograd.grad(outputs=gradu_xi,
                                                inputs=var_j,
                                                grad_outputs=torch.ones_like(gradu_xi),
                                                retain_graph=True, create_graph=True)[0]
            second_order += 2 * gradu2_xixj * var_i * var_j * alphas[i, j]
    f = second_order/2 + first_order + zero_order
    return f

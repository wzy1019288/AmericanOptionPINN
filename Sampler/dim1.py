
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


# Define sampler
class Sampler_IBC():
    def __init__(self, lb, ub, cond=None, N_points=100,
                 method='uniform', grid=None, split=False, DTYPE=torch.float64):
        self.lb = lb
        self.ub = ub
        self.cond = cond
        self.DTYPE = DTYPE
        self.sample(N_points, method, grid, split)
    
    def sample(self, N_points, method, grid, split):
        if method == 'uniform':
            x_ibc = np.random.uniform(0, 1, size=(N_points, self.ub.shape[0]))
            x_ibc = self.lb + (self.ub - self.lb)*x_ibc
        elif method == 'latin':
            from pyDOE import lhs
            x_ibc = self.lb + (self.ub - self.lb)*lhs(self.ub.shape[0],N_points)
        elif method == 'sobol':
            import sobol
            x_ibc = sobol.sample(dimension=self.ub.shape[0], n_points=N_points)
            x_ibc = self.lb + (self.ub - self.lb)*x_ibc
        elif method == 'equi':
            x_ibc = np.linspace(self.lb, self.ub, N_points)
        elif method == 'grid':
            x_ibc = np.linspace(self.lb, self.ub, N_points).T
            temp_final = list()
            for val in x_ibc[0]:
                temp_final.append( [val] )
            dim = 1
            while dim < x_ibc.shape[0]:
                temp = list()
                for t1 in range(x_ibc.shape[1]):
                    for t2 in range(len(temp_final)):
                        temp_val = temp_final[t2].copy()
                        temp_val.append( x_ibc[dim, t1] )
                        temp.append( temp_val )
                temp_final = temp
                dim += 1
            x_ibc = np.array(temp_final)
        elif method == 'grid_old':
            idx = np.random.choice(range(grid.shape[0]),N_points,replace=False)
            x_ibc = grid[idx]
        if self.cond != None:
            y_ibc = self.cond(x_ibc)
            self.y = torch.tensor(y_ibc, dtype=self.DTYPE, requires_grad=False)
        if split:
            x_ibc, t_ibc = x_ibc[:, :-1], x_ibc[:, -1:]
            self.t = torch.tensor(t_ibc, dtype=self.DTYPE, requires_grad=False)
        self.x = torch.tensor(x_ibc, dtype=self.DTYPE, requires_grad=False)

# Define sampler dataset
class SamplerDataset(Dataset):
    def __init__(self, sampler, return_var='x,y') -> None:
        self.return_var = return_var.split(',')
        self.len_vars = len(return_var)
        if self.return_var == ['x', 'y']:
            self.x = sampler.x
            self.y = sampler.y
        elif self.return_var == ['x', 't']:
            self.x = sampler.x
            self.t = sampler.t
        elif self.return_var == ['x']:
            self.x = sampler.x
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.return_var == ['x', 'y']:
            return [self.x[idx], self.y[idx]]
        elif self.return_var == ['x', 't']:
            return [self.x[idx], self.t[idx]]
        elif self.return_var == ['x']:
            return self.x[idx]

# Define condition function
def h_1(inp, K):
    res = list()
    for inp_val in inp:
        res.append( K - inp_val[0] )
    return np.array(res)

def h_2(inp):
    res = - np.ones( inp.shape[0] )
    return res

def g(inp):
    res = np.zeros( inp.shape[0] )
    return np.array(res)

def u_0(inp, K):
    res = list()
    for inp_val in inp:
        x, t = inp_val
        res.append( np.max([0, K-x]) )
    return np.array(res)

def s_0(inp, K):
    res = np.ones( inp.shape[0] ) * K
    return np.array(res)


def get_train_dataloaders(lb, ub, K, T, DTYPE, N_samples_trainset_pde, N_samples_trainset_others, batch_num):

    # PDE Conditions
    print('PDE Conditions\n')
    # Interior x: 30000*1 t: 30000*1
    interior_sampler = Sampler_IBC(lb, ub, cond=None, DTYPE=DTYPE,
                        N_points=N_samples_trainset_pde, method='sobol', split=True)
    print('[interior_sampler]')
    print( f'x: {interior_sampler.x.shape}     t: {interior_sampler.t.shape}' )
    dataset_intrr = SamplerDataset(interior_sampler, 'x,t')
    dataloader_intrr = DataLoader(dataset_intrr, batch_size=len(dataset_intrr)//batch_num, shuffle=True)

    # Initial  x: 300*2 y: 300
    init_sampler = Sampler_IBC(np.array([0., T]), np.array([3.*K, T]),
                            lambda inp: u_0(inp, K), N_samples_trainset_others, DTYPE=DTYPE, method='sobol' )
    print('[init_sampler]')
    print( f'x: {init_sampler.x.shape}     y: {init_sampler.y.shape}' )
    dataset_init = SamplerDataset(init_sampler, 'x,y')
    dataloader_init = DataLoader(dataset_init, batch_size=len(dataset_init)//batch_num, shuffle=True)

    # Dirichlet  x: 300*2 y: 300
    dir_sampler = Sampler_IBC(np.array([3.*K, 0.]), np.array([3.*K, T]),
                            g, N_samples_trainset_others, DTYPE=DTYPE, method='sobol' )
    print('[dir_sampler]')
    print( f'x: {dir_sampler.x.shape}     y: {dir_sampler.y.shape}' )
    dataset_dir = SamplerDataset(dir_sampler, 'x,y')
    dataloader_dir = DataLoader(dataset_dir, batch_size=len(dataset_dir)//batch_num, shuffle=True)

    #Neumann
    print('[neu_sampler]')
    print('No')

    # Free Boundary Conditions
    print('\nFree Boundary Conditions\n')
    # Initial  x: 1*1 y: 1
    fb_init_sampler = Sampler_IBC(np.array([T]), np.array([T]),
                                lambda inp: s_0(inp, K), 1, DTYPE=DTYPE )
    print('[fb_init_sampler]')
    print( f'x: {fb_init_sampler.x.shape}     y: {fb_init_sampler.y.shape}' )
    dataset_fb_init = SamplerDataset(fb_init_sampler, 'x,y')
    dataloader_fb_init = DataLoader(dataset_fb_init, batch_size=len(dataset_fb_init), shuffle=False)

    # Dirichlet  x: 300*1
    fb_dir_sampler = Sampler_IBC(np.array([0.]), np.array([T]),
                                None, N_samples_trainset_others, DTYPE=DTYPE, method='sobol' )
    print('[fb_dir_sampler]')
    print( f'x: {fb_dir_sampler.x.shape}' )
    dataset_fb_dir = SamplerDataset(fb_dir_sampler, 'x')
    dataloader_fb_dir = DataLoader(dataset_fb_dir, batch_size=len(dataset_fb_dir)//batch_num, shuffle=False)

    # Neumann  x: 300*1 y: 300
    fb_neu_sampler = Sampler_IBC(np.array([0.]), np.array([T]),
                                h_2, N_samples_trainset_others, DTYPE=DTYPE, method='sobol' )
    print('[fb_neu_sampler]')
    print( f'x: {fb_neu_sampler.x.shape}     y: {fb_neu_sampler.y.shape}' )
    dataset_fb_neu = SamplerDataset(fb_neu_sampler, 'x,y')
    dataloader_fb_neu = DataLoader(dataset_fb_neu, batch_size=len(dataset_fb_neu)//batch_num, shuffle=False)

    # Conditions passed to PINN
    sol_conditions = {'Interior': dataloader_intrr,
                    'Initial':dataloader_init,
                    'Dirichlet':dataloader_dir,
                    'Neumann':None}
    fb_conditions = {'Initial':dataloader_fb_init,
                    'Dirichlet':dataloader_fb_dir,
                    'Neumann':dataloader_fb_neu}
    
    return sol_conditions, fb_conditions


def get_test_dataloaders(lb, ub, K, T, DTYPE, N_samples_testset_pde, N_samples_testset_others):
    # PDE Conditions
    print('PDE Conditions\n')
    # Interior x: 1000000*1 t: 1000000*1
    interior_sampler_test = Sampler_IBC(lb, ub, cond=None, DTYPE=DTYPE,
                                N_points=N_samples_testset_pde, method='uniform', split=True)
    print('[interior_sampler_test]')
    print( f'x: {interior_sampler_test.x.shape}     t: {interior_sampler_test.t.shape}' )
    dataset_intrr_test = SamplerDataset(interior_sampler_test, 'x,t')
    dataloader_intrr_test = DataLoader(dataset_intrr_test, batch_size=len(dataset_intrr_test), shuffle=False)

    # Initial  x: 1000*2 y: 1000
    init_sampler_test = Sampler_IBC(np.array([0., T]), np.array([3.*K, T]),
                            lambda inp: u_0(inp, K), N_samples_testset_others, DTYPE=DTYPE, method='uniform' )
    print('[init_sampler_test]')
    print( f'x: {init_sampler_test.x.shape}     y: {init_sampler_test.y.shape}' )
    dataset_init_test = SamplerDataset(init_sampler_test, 'x,y')
    dataloader_init_test = DataLoader(dataset_init_test, batch_size=len(dataset_init_test), shuffle=False)

    # Dirichlet  x: 1000*2 y: 1000
    dir_sampler_test = Sampler_IBC(np.array([3.*K, 0.]), np.array([3.*K, T]),
                            g, N_samples_testset_others, DTYPE=DTYPE, method='uniform' )
    print('[dir_sampler_test]')
    print( f'x: {dir_sampler_test.x.shape}     y: {dir_sampler_test.y.shape}' )
    dataset_dir_test = SamplerDataset(dir_sampler_test, 'x,y')
    dataloader_dir_test = DataLoader(dataset_dir_test, batch_size=len(dataset_dir_test), shuffle=False)

    #Neumann
    print('[neu_sampler_test]')
    print('No')

    # Free Boundary Conditions
    print('\nFree Boundary Conditions\n')
    # Initial  t: 1*1 y: 1
    fb_init_sampler_test = Sampler_IBC(np.array([T]), np.array([T]),
                                lambda inp: s_0(inp, K), 1, DTYPE=DTYPE )
    print('[fb_init_sampler_test]')
    print( f't: {fb_init_sampler_test.x.shape}     y: {fb_init_sampler_test.y.shape}' )
    dataset_fb_init_test = SamplerDataset(fb_init_sampler_test, 'x,y')
    dataloader_fb_init_test = DataLoader(dataset_fb_init_test, batch_size=len(dataset_fb_init_test), shuffle=False)

    # Dirichlet  t: 1000*1
    fb_dir_sampler_test = Sampler_IBC(np.array([0.]), np.array([T]),
                                None, N_samples_testset_others, DTYPE=DTYPE, method='uniform' )
    print('[fb_dir_sampler_test]')
    print( f't: {fb_dir_sampler_test.x.shape}' )
    dataset_fb_dir_test = SamplerDataset(fb_dir_sampler_test, 'x')
    dataloader_fb_dir_test = DataLoader(dataset_fb_dir_test, batch_size=len(dataset_fb_dir_test), shuffle=False)

    # Neumann  t: 1000*1 y: 1000
    fb_neu_sampler_test = Sampler_IBC(np.array([0.]), np.array([T]),
                                h_2, N_samples_testset_others, DTYPE=DTYPE, method='uniform' )
    print('[fb_neu_sampler_test]')
    print( f't: {fb_neu_sampler_test.x.shape}     y: {fb_neu_sampler_test.y.shape}' )
    dataset_fb_neu_test = SamplerDataset(fb_neu_sampler_test, 'x,y')
    dataloader_fb_neu_test = DataLoader(dataset_fb_neu_test, batch_size=len(dataset_fb_neu_test), shuffle=False)

    # Conditions passed to PINN
    sol_conditions_test = {'Interior': dataloader_intrr_test,
                        'Initial':dataloader_init_test,
                        'Dirichlet':dataloader_dir_test,
                        'Neumann':None}
    fb_conditions_test = {'Initial':dataloader_fb_init_test,
                        'Dirichlet':dataloader_fb_dir_test,
                        'Neumann':dataloader_fb_neu_test}

    return sol_conditions_test, fb_conditions_test




import os
import numpy as np
from dataclasses import dataclass, field

import torch


@dataclass
class OptionConfig:
    
    # Define dimmension
    _n_dim = 1
    
    # Define option params
    r = 0.01
    _T = 3
    _K = 10
    _sigma = 0.05
    d: np.ndarray = field(init=False)   # r-q

    # Define sampler params
    lb: np.ndarray = field(init=False)
    ub: np.ndarray = field(init=False)
    N_samples_trainset_pde = 30000
    N_samples_trainset_others = 300
    N_samples_testset_pde = 200000
    N_samples_testset_others = 3000

    # Define global params
    DTYPE = torch.float32
    _seed = 88888

    # Define path
    image_path: str = field(init=False)
    checkpoint_path: str = field(init=False)

    # Define PINN params
    batch_num = 1
    verbose = 100

    sample_method = 'sobol'
    sol_layers: list = field(init=False)
    sol_activation = 'hard_swish'
    sol_output_act = None
    sol_initializer = 'glorot_normal'
    sol_lr = 1e-2
    sol_optimizer = 'rmsprop'

    fb_layers: list = field(init=False)
    fb_activation = 'hard_swish'
    fb_output_act = None
    fb_initializer = 'glorot_normal'
    fb_lr = 1e-2
    fb_optimizer = 'rmsprop'

    pde_weight = 1
    sup_weight = [1., 1., 1.]
    fb_weight = [1., 1., 1.]
    _steps_fb_per_pde = 20
    epochs = 50000
    patience = 1000

    def __post_init__(self):
        self.lb = np.array([0.]*self._n_dim + [0.])
        self.ub = np.array([3.*self._K]*self._n_dim + [self._T])
        self.sol_layers = [self._n_dim+1, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        self.fb_layers = [1, 100, 100, 100, self._n_dim]
        self.d = np.array([self.r]*self._n_dim) - np.array([0]*self._n_dim)

        self.image_path = f'Images/{self._n_dim}d/steps_sol_{self._steps_fb_per_pde}/seed_{self._seed}'
        self.checkpoint_path = f'Checkpoints/{self._n_dim}d/steps_sol_{self._steps_fb_per_pde}/seed_{self._seed}'
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

    @property
    def n_dim(self):
        return self._n_dim

    @n_dim.setter
    def n_dim(self, value):
        self._n_dim = value

        self.lb = np.array([0.]*self._n_dim + [0.])
        self.ub = np.array([3.*self._K]*self._n_dim + [self._T])
        self.sol_layers = [self._n_dim+1, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        self.fb_layers = [1, 100, 100, 100, self._n_dim]
        self.d = np.array([self.r]*self._n_dim) - np.array([0]*self._n_dim)
        
        self.image_path = f'Images/{self._n_dim}d/steps_sol_{self._steps_fb_per_pde}/seed_{self._seed}'
        self.checkpoint_path = f'Checkpoints/{self._n_dim}d/steps_sol_{self._steps_fb_per_pde}/seed_{self._seed}'
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value

        self.ub = np.array([3.*self._K]*self._n_dim + [self._T])

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value

        self.ub = np.array([3.*self._K]*self._n_dim + [self._T])

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if isinstance(value, float):
            self._sigma = value
        if isinstance(value, list):
            value = np.array(value)
        if isinstance(value, np.ndarray):
            if np.linalg.det(value) != 0:
                if np.allclose(value, value.T):
                    if np.all(np.linalg.eigvals(value) > 0):
                        self._sigma = value
                    else:
                        print('[Set Wrong!]Matrix is not positive semidefinite')
                else:
                    print('[Set Wrong!]Matrix is not symmetric')
            else:
                print('[Set Wrong!]Matrix is singular')

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

        self.image_path = f'Images/{self._n_dim}d/steps_sol_{self._steps_fb_per_pde}/seed_{self._seed}'
        self.checkpoint_path = f'Checkpoints/{self._n_dim}d/steps_sol_{self._steps_fb_per_pde}/seed_{self._seed}'
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
    
    @property
    def steps_fb_per_pde(self):
        return self._steps_fb_per_pde
    
    @steps_fb_per_pde.setter
    def steps_fb_per_pde(self, value):
        self._steps_fb_per_pde = value

        self.image_path = f'Images/{self._n_dim}d/steps_sol_{self._steps_fb_per_pde}/seed_{self._seed}'
        self.checkpoint_path = f'Checkpoints/{self._n_dim}d/steps_sol_{self._steps_fb_per_pde}/seed_{self._seed}'
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

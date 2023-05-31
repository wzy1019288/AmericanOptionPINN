
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
    sigma = 0.05

    # Define sampler params
    lb = np.array([0., 0.])
    ub: np.array = field(init=False)
    N_samples_trainset_pde = 30000
    N_samples_trainset_others = 300
    N_samples_testset_pde = 1000000
    N_samples_testset_others = 1000

    # Define global params
    DTYPE = torch.float32
    _seed = 10

    # Define path
    image_1d_path = 'Images/1d'
    checkpoint_path: str = field(init=False)

    # Define PINN params
    batch_num = 10
    verbose = 50

    sample_method = 'sobol'
    sol_layers: list = field(init=False)
    sol_activation = 'tanh'
    sol_output_act = None
    sol_initializer = 'glorot_normal'
    sol_lr = 1e-2
    sol_optimizer = 'rmsprop'

    fb_layers: list = field(init=False)
    fb_activation = 'tanh'
    fb_output_act = None
    fb_initializer = 'glorot_normal'
    fb_lr = 1e-2
    fb_optimizer = 'rmsprop'

    pde_weight = 1
    sup_weight = [1., 1., 1.]
    fb_weight = [1., 1., 1.]
    steps_fb_per_pde = 20
    epochs = 1000
    patience = 200

    def __post_init__(self):
        self.ub = np.array([3.*self._K, self._T])
        self.checkpoint_path = f'Checkpoints/1d/seed_{self._seed}'
        self.sol_layers = [self._n_dim+1, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        self.fb_layers = [1, 100, 100, 100, self._n_dim]
        
        os.makedirs(self.image_1d_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

    @property
    def n_dim(self):
        return self._n_dim

    @n_dim.setter
    def n_dim(self, value):
        self._n_dim = value
        self.sol_layers = [self._n_dim+1, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        self.fb_layers = [1, 100, 100, 100, self._n_dim]
    
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, value):
        self._T = value
        self.ub = np.array([3.*self._K, self._T])
    
    @property
    def K(self):
        return self._K
    
    @K.setter
    def K(self, value):
        self._K = value
        self.ub = np.array([3.*self._K, self._T])

    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, value):
        self._seed = value
        self.checkpoint_path = f'Checkpoints/1d/seed_{self._seed}'
        os.makedirs(self.checkpoint_path, exist_ok=True)

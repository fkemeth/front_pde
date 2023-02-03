"""Config file with hyperparameters.

See README.md in the main directory for details.
"""

import torch
import numpy as np

config = {}
config["dt"] = 5e-2
config["T"] = 500
config["N"] = 400
config["L"] = 90

config["a"] = -0.1
config["D"] = 0.1

# config["boundary_conditions"] = 'no-flux'
config["boundary_conditions"] = 'periodic'
config["ic"] = 'mixed_random'
config["use_fd_dt"] = True
config["fd_dt_acc"] = 4
config["rescale_dx"] = 1.0
config["load_data"] = True
config["noise_augment"] = False

config["n_train"] = 100
config["n_test"] = 10

config["MODEL"] = {}
config["MODEL"]["kernel_size"] = 5
config["MODEL"]['device'] = 'cuda'
config["MODEL"]['use_param'] = False
config["MODEL"]['num_params'] = 2
config["MODEL"]['n_filters'] = 96
config["MODEL"]['n_layers'] = 4
config["MODEL"]["n_derivs"] = 2

config["TRAINING"] = {}
config["TRAINING"]['batch_size'] = 128
config["TRAINING"]['num_workers'] = 8
config["TRAINING"]["reduce_factor"] = .5
config["TRAINING"]["patience"] = 15
config["TRAINING"]["lr"] = 2e-3
config["TRAINING"]['epochs'] = 100
config["TRAINING"]['proceed_training'] = False
config["TRAINING"]['dtype'] = torch.float32

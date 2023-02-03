# Learning PDEs for Interfaces

In this repository, there are scripts to learn effective PDEs that describe the evolution of a one-dimensional front obtained from
two-dimensional phase field models.
The main scripts are

- `run_black_box_model.py`. This trains and evaluates a black box neural network PDE for the front dynamics.
- `run_additive_model.py`. This trains and evaluates a gray box additive correction to the KPZ equation.
- `run_function_model.py`. This trains and evaluates a gray box functional correction to the KPZ equation.

The scripts import a config file, `config.py`, where you have to specify the parameters for learning the PDE.
Furthermore, there is a `utils.py` file that contains the definitions of the neural network architectures, and scripts for training the PDE.
For training the PDE, PyTorch is used.
In addition there is a `tests.py` file that contains functions used for plotting/testing the code, which are called in the `run` files.

In the KPZ folder, there is a script called `dataset.py` and `config.py`.
In the `config.py` files, all the hyperparameters are defined as a dictionary.
In the `dataset.py` files, a `torch.utils.data.Dataset` class called `Dataset` must be located.

In the specific `config.py` files of the examples, a few hyper parameters are specified, of which some are explained below

- `dt`: The time step between snapshots of the simulation data.
- `T`: The number of snapshots per simulation.
- `N`: The number of spatial grid points.
- `L`: The length of the spatial domain.
- Furthermore, parameters specific to the problem can be specified.

- `load_data`: If False, this boolean parameter indicates that the data for learning the PDE has to be simulated. If it is already simulated, one can set it to True and load the data from disk.
- `use_fd_dt`: If True, then finite differences in time are used to calculate time derivatives. If False, a function returning the time derivative has to be provided.
- `fd_dt_acc`: If `use_fd_dt` is True, it specifies the accuracy order of the finite differences in time.
- `rescale_dx`: Default is 1.0. Can be tuned if the spatial derivatives are on different orders of magnitude due to very small/very large spatial domain.
- `ic`: Specifies the type of initial conditions used for simulation.
- `noise_augment`: Augment the training data by adding noise. Default should be False.
- `n_train`: The number of traning trajectories.
- `n_test`: The number of test trajectories.

Furthermore, there are parameters concerning only the model architecture. These include:

- `kernel_size`: The length of the finite difference stencil used for obtaining the space derivatives.
- `device`: If the model shall be trained on the gpu or cpu.
- `use_param`: If a parameter changes from simulation to simulation, this must be set to True.
- `num_params`: If `use_param` is True, then here the number of parameters that change have to be specified.
- `n_filters`: The number of neurons in each layer of the PDE model.
- `n_layers`: The number of layers of the PDE model.
- `n_derivs`: The number of derivatives used as input to the PDE model.

In addition, there are parameters concerning only the model architecture. These include:

- `batch_size`: The batch size used for training.
- `num_workers`: Number of workers used for loading data. Default should be 8.
- `reduce_factor`: Reduce the learning rate by this factor when loss does not decrease.
- `patience`: Reduce the learning rate if loss does not decrease for this amount of epochs.
- `lr`: Initial learning rate.
- `epochs`: Total number of epochs to train.
- `proceed_training`: If True, loads a stored model and continues training it.
- `dtype`: Data type used. Default is `torch.float32`. If the finite difference space derivatives are inaccurate, use `torch.float64`. But this can be very slow.

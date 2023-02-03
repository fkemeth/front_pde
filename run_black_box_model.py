import os
import shutil

import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

import tests

from config import config
from utils import Network, Model, progress

# The KPZ model with small D
if config["GENERAL"]["example"] == 'kpz':
    from kpz.dataset import Dataset
    import kpz.config as ex_cfg
    import kpz.tests as ex_tests

# The KPZ model with large D
if config["GENERAL"]["example"] == 'kpz_largeD':
    from kpz_largeD.dataset import Dataset
    import kpz_largeD.config as ex_cfg
    import kpz_largeD.tests as ex_tests

# The biologically motivated cell model
if config["GENERAL"]["example"] == 'cell_model':
    from cell_model.dataset import Dataset
    import cell_model.config as ex_cfg

# Create directories if they do not exist already
if not os.path.exists(config["GENERAL"]["save_dir"]):
    os.makedirs(config["GENERAL"]["save_dir"])

if not os.path.exists(config["GENERAL"]["fig_path"]):
    os.makedirs(config["GENERAL"]["fig_path"])

if not os.path.exists(config["GENERAL"]["save_dir"]+'/log'):
    os.makedirs(config["GENERAL"]["save_dir"]+'/log')
else:
    shutil.rmtree(config["GENERAL"]["save_dir"]+'/log')
    os.makedirs(config["GENERAL"]["save_dir"]+'/log')

# Set the data type as specified in the config file
torch.set_default_dtype(ex_cfg.config["TRAINING"]['dtype'])


def main(config):
    """Integrate system and train model."""

    verbose = config["GENERAL"]["verbose"]

    # Create Dataset
    dataset_train = Dataset(ex_cfg.config, ex_cfg.config["n_train"])
    dataset_test = Dataset(ex_cfg.config, ex_cfg.config["n_test"],
                           start_idx=ex_cfg.config["n_train"])

    if verbose:
        tests.visualize_dynamics(dataset_train, path=config["GENERAL"]["fig_path"])

    # Create Dataloader
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=int(ex_cfg.config["TRAINING"]['batch_size']), shuffle=True,
        num_workers=int(ex_cfg.config["TRAINING"]['num_workers']), pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=int(ex_cfg.config["TRAINING"]['batch_size']), shuffle=False,
        num_workers=int(ex_cfg.config["TRAINING"]['num_workers']), pin_memory=True)

    # Create the network architecture
    network = Network(ex_cfg.config["MODEL"], n_vars=dataset_train.x_data.shape[1])

    if verbose:
        tests.visualize_derivatives(network, dataset_train, path=config["GENERAL"]["fig_path"])

    # Create a model wrapper around the network architecture
    # Contains functions for training
    model = Model(dataloader_train, dataloader_test, network, ex_cfg.config["TRAINING"],
                  path=config["GENERAL"]["save_dir"])

    logger = SummaryWriter(config["GENERAL"]["save_dir"]+'/log/')

    progress_bar = tqdm.tqdm(range(0, int(ex_cfg.config["TRAINING"]['epochs'])),
                             total=int(ex_cfg.config["TRAINING"]['epochs']),
                             leave=True, desc=progress(0, 0))

    # Load an already trained model if desired
    if ex_cfg.config["TRAINING"]['proceed_training']:
        model.load_network(ex_cfg.config["boundary_conditions"]+'test.model')

    # Train the model
    train_loss_list = []
    val_loss_list = []
    for epoch in progress_bar:
        train_loss = model.train()
        val_loss = model.validate()
        progress_bar.set_description(progress(train_loss, val_loss))

        logger.add_scalar('Loss/train', train_loss, epoch)
        logger.add_scalar('Loss/val', val_loss, epoch)
        logger.add_scalar('learning rate', model.optimizer.param_groups[-1]["lr"], epoch)

        model.save_network(ex_cfg.config["boundary_conditions"]+'test.model')

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    # Plot the loss curves
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_loss_list, label='training loss')
    ax.plot(val_loss_list, label='validation loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('')
    ax.set_yscale('log')
    plt.savefig(config["GENERAL"]["fig_path"]+'loss_curves.pdf')
    plt.show()

    # Visualize the predictions of the model
    tests.visualize_predictions(dataset_test, model, path=config["GENERAL"]["fig_path"])

    # Visualize different models
    if config["GENERAL"]["example"] == 'kpz_largeD' or config["GENERAL"]["example"] == 'kpz':
        # ex_tests.test_analytic_models()
        ex_tests.test_integrate_wo_kpz()


if __name__ == "__main__":
    main(config)

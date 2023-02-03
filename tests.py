import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import findiff


def visualize_derivatives(network, dataset, path):
    """Plot the calculated derivatives."""
    xx = dataset.config["L"]*np.linspace(0, 1, num=dataset.config["N"], endpoint=False)

    input_x, dx, _, _ = dataset[0]

    if dataset.boundary_conditions == 'periodic':
        input_x = F.pad(input_x.unsqueeze(0), (network.off_set,
                                               network.off_set), mode='circular')[0]
    elif dataset.boundary_conditions == 'no-flux':
        input_x = F.pad(input_x.unsqueeze(0), (network.off_set, network.off_set), mode='reflect')[0]

    derivs = network.calc_derivs(input_x.unsqueeze(0)[:, :1].to(network.device),
                                 dx.unsqueeze(0).to(network.device)).cpu().numpy()[0].T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(1, network.n_derivs+1):
        ax.plot(xx, derivs[:, i], label='Network derivative '+str(i))
        d_dx = findiff.FinDiff(0, float(dx.numpy()), i, acc=4)
        derivs_fd = d_dx(input_x.numpy()[0])[network.off_set:-network.off_set]
        ax.plot(xx, derivs_fd, '--', label='Numeric derivative '+str(i))
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'd/dx')
    plt.legend()
    plt.savefig(path + 'derivatives.pdf')
    plt.close()


def visualize_dynamics(dataset, path):
    xx = dataset.config["L"]*np.linspace(0, 1, num=dataset.config["N"], endpoint=False)
    tt = np.linspace(0, dataset.config["T"]*dataset.delta_t, dataset.config["T"]+1)
    if dataset.use_fd_dt:
        tt = tt[:-dataset.config["fd_dt_acc"]]
        trajectory = dataset.x_data[:dataset.config["T"]-dataset.config["fd_dt_acc"]+1, 0]
        dudt_trajectory = dataset.y_data[:dataset.config["T"]-dataset.config["fd_dt_acc"]+1, 0]
    else:
        trajectory = dataset.x_data[:dataset.config["T"]+1, 0]
        dudt_trajectory = dataset.y_data[:dataset.config["T"]+1, 0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pl1 = ax.pcolor(xx, tt, trajectory, rasterized=True)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r't')
    plt.colorbar(pl1, label='u')
    plt.savefig(path+'space_time_dynamics.pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pl1 = ax.pcolor(xx, tt, dudt_trajectory, rasterized=True)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r't')
    plt.colorbar(pl1, label='du/dt')
    plt.savefig(path+'space_time_dudt.pdf')
    plt.close()


def visualize_predictions(dataset, model, path):
    xx = dataset.config["L"]*np.linspace(0, 1, num=dataset.config["N"], endpoint=False)
    tt = np.linspace(0, dataset.config["T"] *
                     dataset.delta_t, dataset.config["T"]+1)

    if dataset.use_fd_dt:
        tt = tt[:-dataset.config["fd_dt_acc"]]
        trajectory = dataset.x_data[:dataset.config["T"]-dataset.config["fd_dt_acc"]+1, 0]
        dudt_trajectory = dataset.y_data[:dataset.config["T"]-dataset.config["fd_dt_acc"]+1, 0]
    else:
        trajectory = dataset.x_data[:dataset.config["T"]+1, 0]
        dudt_trajectory = dataset.y_data[:dataset.config["T"]+1, 0]

    initial_condition, delta_x, _, param = dataset[0]
    # initial_condition = initial_condition[:, ::8]
    _, prediction = model.integrate(initial_condition.detach().numpy().flatten(),
                                    [delta_x.detach().numpy(), param.detach().numpy()],
                                    tt)

    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121)
    pl1 = ax1.pcolor(xx, tt, trajectory, rasterized=True)
    ax1.set_xlabel(r'x')
    ax1.set_ylabel(r't')
    plt.title('True')
    plt.colorbar(pl1, label='u')
    ax2 = fig.add_subplot(122)
    pl2 = ax2.pcolor(xx, tt, prediction[:, 0], rasterized=True)
    ax2.set_xlabel(r'x')
    ax2.set_ylabel(r't')
    plt.title('Predictions')
    plt.colorbar(pl2, label='u')
    plt.subplots_adjust(wspace=0.35)
    plt.savefig(path+'space_time_predictions.pdf')
    plt.show()

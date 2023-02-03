import numpy as np
import matplotlib.pyplot as plt

import torch

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


from kpz.config import config
from kpz.utils import create_initial_conditions, integrate, get_front, filter_front, \
    integrate_front, integrate_kpz, f_front, f_kpz

from scipy import ndimage
from scipy.optimize import curve_fit


def test_initial_condition():
    initial_condition = create_initial_conditions(config, ic='sergio')
    xx = np.linspace(0, config["L"], config["N"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pl1 = ax.pcolor(xx, xx, initial_condition, vmin=-1, vmax=1, rasterized=True, cmap='plasma')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(pl1, label='\phi')
    plt.savefig('kpz/fig/initial_condition.pdf')
    plt.show()


def test_integration():
    dx = config["L"]/config["N"]
    xx = np.linspace(0, config["L"], config["N"])
    act_config = config.copy()
    act_config["dt"] = 2*act_config["dt"]

    data, _, _ = integrate(act_config)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pl1 = ax.pcolor(xx[::2], xx[::2], data[-1, ::2, ::2], vmin=-1, vmax=1, rasterized=True,
                    cmap='plasma')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(pl1, label=r'\phi')
    plt.savefig('kpz/fig/last_snapshot.pdf')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    print("Creating images. This may take a few seconds.")
    pl1 = ax.pcolor(xx[:], xx[50:150], data[0, 50:150, :], vmin=-
                    1, vmax=1, rasterized=True, cmap='plasma')
    plt.colorbar(pl1, label=r'$\phi$')
    scas = []
    for i in range(0, data.shape[0], 2):
        sca = ax.pcolor(xx[:], xx[50:150], data[i, 50:150, :], vmin=-1, vmax=1, rasterized=True,
                        cmap='plasma')
        scas.append([sca])

    ani = animation.ArtistAnimation(
        fig, scas, interval=25, blit=True, repeat_delay=0, repeat=True)
    fps = 50
    bitrate = 2000
    FFMpegWriter = animation.writers['ffmpeg']
    FFwriter = FFMpegWriter(fps=fps, bitrate=bitrate)
    ani.save('kpz/fig/simulation_phase_field.avi', writer=FFwriter, dpi=200)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    print("Creating images. This may take a few seconds.")
    pl1 = plt.imshow(data[0, 50:150, :], vmin=-1, vmax=1, cmap='plasma',
                     extent=[xx[0], xx[-1], xx[0], xx[-1]], origin='lower')
    plt.colorbar(pl1, label=r'$\phi$')
    scas = []
    for i in range(0, data.shape[0], 2):
        sca = ax.imshow(data[i, 50:150, :], vmin=-1, vmax=1, cmap='plasma',
                        extent=[xx[0], xx[-1], xx[0], xx[-1]], origin='lower')
        scas.append([sca])

    ani = animation.ArtistAnimation(
        fig, scas, interval=25, blit=True, repeat_delay=0, repeat=True)
    fps = 50
    bitrate = 2000
    FFMpegWriter = animation.writers['ffmpeg']
    FFwriter = FFMpegWriter(fps=fps, bitrate=bitrate)
    ani.save('kpz/fig/simulation_phase_field_imshow.mp4', writer=FFwriter, dpi=200)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx, get_front(data[:1], dx)[0], color='k')
    ax.set_xlabel('x')
    ax.set_ylabel('h')
    ax.set_xlim((0, xx[-1]))
    ax.set_ylim((15, 23))
    plt.savefig('kpz/fig/front_initial_condition.pdf')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx, get_front(data[-1:], dx)[0], color='k')
    ax.set_xlabel('x')
    ax.set_ylabel('h')
    ax.set_xlim((0, xx[-1]))
    ax.set_ylim((15, 23))
    plt.savefig('kpz/fig/front_last_snapshot.pdf')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'h')
    ax.set_ylim((10, 30))
    ax.set_xlim((xx[0], xx[-1]))
    print("Creating images. This may take a few seconds.")
    scas = []
    for i in range(0, data.shape[0], 2):
        sca, = ax.plot(xx, get_front(data[i:i+1], dx)[0], color='k')
        scas.append([sca])

    ani = animation.ArtistAnimation(
        fig, scas, interval=25, blit=True, repeat_delay=0, repeat=True)
    fps = 50
    bitrate = 2000
    FFMpegWriter = animation.writers['ffmpeg']
    FFwriter = FFMpegWriter(fps=fps, bitrate=bitrate)
    ani.save('kpz/fig/simulation_front_position.mp4', writer=FFwriter, dpi=200)
    plt.show()


def test_analytic_models():
    dx = config["L"]/config["N"]
    xx = np.linspace(0, config["L"], config["N"])
    act_config = config.copy()
    act_config["dt"] = act_config["dt"]

    tt = np.linspace(0, act_config["dt"]*act_config["T"], act_config["T"]+1)

    data, _, _ = integrate(act_config)
    initial_front = get_front(data[:1], dx)[0]
    front_pf = get_front(data[-1:], dx)[0]

    front_prediction, _, _ = integrate_front(act_config, tt, initial_front)

    kpz_prediction, _, _ = integrate_kpz(act_config, tt, initial_front)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx, front_pf, label='Phase field front')
    ax.plot(xx, front_prediction[-1], label='1-d front model')
    ax.plot(xx, kpz_prediction[-1], label='KPZ model')
    ax.set_xlabel('x')
    ax.set_ylabel('')
    plt.legend()
    plt.savefig('kpz/fig/analytic_models_predictions_last_snapshot.pdf')
    plt.show()


def test_front_interpolation():
    """Test the effectiveness of finding the front."""
    dx = config["L"]/config["N"]
    xx = np.linspace(0, config["L"], config["N"])

    data, _, _ = integrate(config)

    def func(x, a, b):
        return np.tanh(a*x-b)

    fronts = []
    for snapshot in data:
        idxs = []
        for strip in snapshot.T:
            params, _ = curve_fit(func, xx, strip)
            idxs.append(params[1]/params[0])
        fronts.append(idxs)
    fronts = np.array(fronts)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx, strip)
    ax.plot(xx, func(xx, *params))
    ax.set_xlabel('')
    ax.set_ylabel('')
    # plt.savefig('')
    plt.show()

    y_data = (fronts[1:] - fronts[:-1])/config["dt"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pl1 = ax.pcolor(y_data)
    cbar = plt.colorbar(pl1)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # plt.savefig('')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'h')
    ax.set_ylim((10, 30))
    print("Creating images. This may take a few seconds.")
    scas = []
    for i in range(0, data.shape[0]):
        sca1, = ax.plot(xx, get_front(data[i], dx), color='k')
        sca2, = ax.plot(xx, fronts[i], color='blue')
        scas.append([sca1, sca2])

    ani = animation.ArtistAnimation(
        fig, scas, interval=25, blit=True, repeat_delay=0, repeat=True)
    fps = 25
    bitrate = 2000
    FFMpegWriter = animation.writers['ffmpeg']
    FFwriter = FFMpegWriter(fps=fps, bitrate=bitrate)
    ani.save('kpz/fig/simulation_front_position_w_fit.avi', writer=FFwriter, dpi=200)
    plt.show()


def test_integrate():
    from kpz.dataset import Dataset
    import kpz.config as ex_cfg
    from utils import Network, Model, progress

    ex_cfg.config["load_data"] = True
    ex_cfg.config["n_train"] = 2

    dataset_train = Dataset(ex_cfg.config, ex_cfg.config["n_train"])
    dataset_test = Dataset(ex_cfg.config, ex_cfg.config["n_test"],
                           start_idx=ex_cfg.config["n_train"])

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=int(ex_cfg.config["TRAINING"]['batch_size']), shuffle=True,
        num_workers=int(ex_cfg.config["TRAINING"]['num_workers']), pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=int(ex_cfg.config["TRAINING"]['batch_size']), shuffle=False,
        num_workers=int(ex_cfg.config["TRAINING"]['num_workers']), pin_memory=True)

    network = Network(ex_cfg.config["MODEL"], n_vars=dataset_train.x_data.shape[1])

    model = Model(dataloader_train, dataloader_test, network, ex_cfg.config["TRAINING"],
                  path='kpz/')

    model.load_network(ex_cfg.config["boundary_conditions"]+'test.model')

    xx = ex_cfg.config["L"] * np.linspace(0, 1, num=int(ex_cfg.config["N"]), endpoint=False)
    tt = np.linspace(0, ex_cfg.config["T"] * ex_cfg.config["dt"], ex_cfg.config["T"]+1)

    _, _, _, param = dataset_test[0]

    np.random.seed(540)
    initial_condition = create_initial_conditions(ex_cfg.config, ic='sergio')

    data, dx, _ = integrate(ex_cfg.config, initial_condition)

    fig = plt.figure(figsize=(6, 2.5))
    ax1 = fig.add_subplot(121)
    pl1 = ax1.pcolor(xx, xx, initial_condition, vmin=-1, vmax=1, rasterized=True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xticks((0, 45, 90))
    ax1.set_yticks((0, 45, 90))
    plt.colorbar(pl1, label=r'\phi')
    ax2 = fig.add_subplot(122)
    pl2 = ax2.pcolor(xx, xx, data[-1], vmin=-1, vmax=1, rasterized=True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xticks((0, 45, 90))
    ax2.set_yticks((0, 45, 90))
    plt.colorbar(pl2, label=r'\phi')
    plt.tight_layout()
    plt.savefig('kpz/fig/initial_condition_and_snapshot.pdf')
    plt.show()

    fronts = get_front(data, dx)

    fig = plt.figure(figsize=(6, 2.5))
    ax = fig.add_subplot(111)
    for ic in dataset_train.x_data[:int(2*(ex_cfg.config["T"])):int(ex_cfg.config["T"]-3)]:
        ax.plot(xx, ic[0], color='k')
    ax.plot(xx, fronts[0], color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('h')
    plt.tight_layout()
    plt.savefig('kpz/fig/initial_conditions_train_test.pdf')
    plt.show()

    initial_front = torch.tensor(
        fronts[0], dtype=torch.get_default_dtype()).unsqueeze(0)

    dx = torch.tensor(ex_cfg.config["L"]/ex_cfg.config["N"], dtype=torch.get_default_dtype())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx, model.dfdt(0, fronts[100], dx, param), label='learned')
    ax.plot(xx, f_front(0, fronts[100], ex_cfg.config), label='1-d front model')
    ax.plot(xx, (fronts[101]-fronts[100])/ex_cfg.config["dt"], label='finite differences')
    ax.set_xlabel('')
    ax.set_ylabel('x')
    ax.set_xticks((0, 45, 90))
    plt.savefig('kpz/fig/dudt_learned_true.pdf')
    plt.show()

    vmin, vmax = np.min(fronts), np.max(fronts)

    plt.set_cmap('gnuplot2')
    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_subplot(111)
    pl1 = ax.pcolor(xx, tt, fronts, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_xticks((0, 45, 90))
    plt.colorbar(pl1, label=r'h')
    plt.tight_layout()
    plt.savefig('kpz/fig/fronts_true.pdf')
    plt.show()

    _, prediction = model.integrate(initial_front.detach().numpy().flatten(),
                                    [dx.detach().numpy(), param.detach().numpy()],
                                    tt)

    front_prediction, _, _ = integrate_front(ex_cfg.config, tt, initial_front.numpy()[0])

    kpz_prediction, _, _ = integrate_kpz(ex_cfg.config, tt, initial_front.numpy()[0])

    fig = plt.figure(figsize=(7, 2.5))
    ax1 = fig.add_subplot(131)
    pl1 = ax1.pcolor(xx, tt, fronts, rasterized=True, vmin=vmin, vmax=vmax)
    ax1.set_ylabel(r't')
    ax1.set_xticks((0, 45, 90))
    plt.title('Phase field front')
    plt.colorbar(pl1, ax=ax1, label='h', orientation='horizontal')
    ax2 = fig.add_subplot(132)
    pl2 = ax2.pcolor(xx, tt, front_prediction, rasterized=True, vmin=vmin, vmax=vmax)
    ax2.set_ylabel(r't')
    ax2.set_xticks((0, 45, 90))
    plt.title('1-d front model')
    plt.colorbar(pl2, label='h', orientation='horizontal')
    ax3 = fig.add_subplot(133)
    pl3 = ax3.pcolor(xx, tt, prediction[:, 0], rasterized=True, vmin=vmin, vmax=vmax)
    ax3.set_ylabel(r't')
    ax3.set_xticks((0, 45, 90))
    plt.title('NN predictions')
    plt.colorbar(pl3, label='h', orientation='horizontal')
    ax1.set_ylim((0, 15))
    ax2.set_ylim((0, 15))
    ax3.set_ylim((0, 15))
    plt.subplots_adjust(wspace=0.35)
    plt.savefig('kpz/fig/space_time_front_predictions.pdf')
    plt.show()

    emax = 0.05
    fig = plt.figure(figsize=(7, 2.5))
    ax1 = fig.add_subplot(131)
    pl1 = ax1.pcolor(xx, tt, np.abs(fronts-front_prediction), rasterized=True, vmin=0, vmax=emax,
                     cmap='plasma')
    # ax1.set_xlabel(r'x')
    ax1.set_ylabel(r't')
    ax1.set_xticks((0, 45, 90))
    plt.title('1-d front model')
    plt.colorbar(pl1, ax=ax1, label='error', orientation='horizontal')
    ax2 = fig.add_subplot(132)
    pl2 = ax2.pcolor(xx, tt, np.abs(fronts-kpz_prediction), rasterized=True, vmin=0, vmax=emax,
                     cmap='plasma')
    # ax2.set_xlabel(r'x')
    ax2.set_ylabel(r't')
    ax2.set_xticks((0, 45, 90))
    plt.title('KPZ model')
    plt.colorbar(pl2, label='error', orientation='horizontal')
    ax3 = fig.add_subplot(133)
    pl3 = ax3.pcolor(xx, tt, np.abs(fronts-prediction[:, 0]), rasterized=True, vmin=0, vmax=emax,
                     cmap='plasma')
    # ax3.set_xlabel(r'x')
    ax3.set_ylabel(r't')
    ax3.set_xticks((0, 45, 90))
    plt.title('NN predictions')
    plt.colorbar(pl3, label='error', orientation='horizontal')
    ax1.set_ylim((0, 25))
    ax2.set_ylim((0, 25))
    ax3.set_ylim((0, 25))
    plt.subplots_adjust(wspace=0.35)
    plt.savefig('kpz/fig/space_time_error.pdf')
    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx, fronts[-1], label='Phase field front')
    ax.plot(xx, front_prediction[-1], label='1-d front model')
    ax.plot(xx, kpz_prediction[-1], label='KPZ model')
    ax.plot(xx, prediction[-1, 0], label='NN predictions')
    ax.set_xlabel('x')
    ax.set_ylabel('')
    plt.legend()
    plt.savefig('kpz/fig/predictions_all_last_snapshot.pdf')
    plt.show()


def test_integrate_wo_kpz():
    from kpz.dataset import Dataset
    import kpz.config as ex_cfg
    from utils import Network, Model, progress

    ex_cfg.config["load_data"] = True
    ex_cfg.config["n_train"] = 2

    dataset_train = Dataset(ex_cfg.config, ex_cfg.config["n_train"])
    dataset_test = Dataset(ex_cfg.config, ex_cfg.config["n_test"],
                           start_idx=ex_cfg.config["n_train"])

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=int(ex_cfg.config["TRAINING"]['batch_size']), shuffle=True,
        num_workers=int(ex_cfg.config["TRAINING"]['num_workers']), pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=int(ex_cfg.config["TRAINING"]['batch_size']), shuffle=False,
        num_workers=int(ex_cfg.config["TRAINING"]['num_workers']), pin_memory=True)

    network = Network(ex_cfg.config["MODEL"], n_vars=dataset_train.x_data.shape[1])

    model = Model(dataloader_train, dataloader_test, network, ex_cfg.config["TRAINING"],
                  path='kpz/')

    model.load_network(ex_cfg.config["boundary_conditions"]+'test.model')

    xx = ex_cfg.config["L"] * np.linspace(0, 1, num=int(ex_cfg.config["N"]), endpoint=False)
    tt = np.linspace(0, ex_cfg.config["T"] * ex_cfg.config["dt"], ex_cfg.config["T"]+1)

    _, _, _, param = dataset_test[0]

    np.random.seed(540)
    initial_condition = create_initial_conditions(ex_cfg.config, ic='sergio_orig')

    data, dx, _ = integrate(ex_cfg.config, initial_condition)

    fig = plt.figure(figsize=(6, 2.5))
    ax1 = fig.add_subplot(121)
    pl1 = ax1.pcolor(xx, xx, initial_condition, vmin=-1, vmax=1, rasterized=True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xticks((0, 45, 90))
    ax1.set_yticks((0, 45, 90))
    plt.colorbar(pl1, label=r'\phi')
    ax2 = fig.add_subplot(122)
    pl2 = ax2.pcolor(xx, xx, data[-1], vmin=-1, vmax=1, rasterized=True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xticks((0, 45, 90))
    ax2.set_yticks((0, 45, 90))
    plt.colorbar(pl2, label=r'\phi')
    plt.tight_layout()
    plt.savefig('kpz/fig/initial_condition_and_snapshot.pdf')
    plt.show()

    fronts = get_front(data, dx)

    fig = plt.figure(figsize=(6, 2.5))
    ax = fig.add_subplot(111)
    for ic in dataset_train.x_data[:int(2*(ex_cfg.config["T"])):int(ex_cfg.config["T"]-3)]:
        ax.plot(xx, ic[0], color='k')
    ax.plot(xx, fronts[0], color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('h')
    plt.tight_layout()
    plt.savefig('kpz/fig/initial_conditions_train_test.pdf')
    plt.show()

    initial_front = torch.tensor(
        fronts[0], dtype=torch.get_default_dtype()).unsqueeze(0)

    dx = torch.tensor(ex_cfg.config["L"]/ex_cfg.config["N"], dtype=torch.get_default_dtype())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx, model.dfdt(0, fronts[100], dx, param), label='learned')
    ax.plot(xx, f_front(0, fronts[100], ex_cfg.config), label='1-d front model')
    ax.plot(xx, (fronts[101]-fronts[100])/ex_cfg.config["dt"], label='finite differences')
    ax.set_xlabel('')
    ax.set_ylabel('x')
    ax.set_xticks((0, 45, 90))
    plt.savefig('kpz/fig/dudt_learned_true.pdf')
    plt.show()

    vmin, vmax = np.min(fronts), np.max(fronts)

    plt.set_cmap('gnuplot2')
    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_subplot(111)
    pl1 = ax.pcolor(xx, tt, fronts, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_xticks((0, 45, 90))
    plt.colorbar(pl1, label=r'h')
    plt.tight_layout()
    plt.savefig('kpz/fig/fronts_true.pdf')
    plt.show()

    _, prediction = model.integrate(initial_front.detach().numpy().flatten(),
                                    [dx.detach().numpy(), param.detach().numpy()],
                                    tt)

    front_prediction, _, _ = integrate_front(ex_cfg.config, tt, initial_front.numpy()[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx, fronts[-1], label='Phase field front')
    ax.plot(xx, front_prediction[-1], label='1-d front model')
    ax.set_xlabel('x')
    ax.set_ylabel('')
    plt.legend()
    plt.savefig('kpz/fig/analytic_models_wo_kpz_predictions_last_snapshot.pdf')
    plt.show()

    fig = plt.figure(figsize=(7, 2.5))
    ax1 = fig.add_subplot(131)
    pl1 = ax1.pcolor(xx, tt, fronts, rasterized=True, vmin=vmin, vmax=vmax)
    ax1.set_ylabel(r't')
    ax1.set_xticks((0, 45, 90))
    plt.title('Phase field front')
    plt.colorbar(pl1, ax=ax1, label='h', orientation='horizontal')
    ax2 = fig.add_subplot(132)
    pl2 = ax2.pcolor(xx, tt, front_prediction, rasterized=True, vmin=vmin, vmax=vmax)
    ax2.set_ylabel(r't')
    ax2.set_xticks((0, 45, 90))
    plt.title('1-d front model')
    plt.colorbar(pl2, label='h', orientation='horizontal')
    ax3 = fig.add_subplot(133)
    pl3 = ax3.pcolor(xx, tt, prediction[:, 0], rasterized=True, vmin=vmin, vmax=vmax)
    ax3.set_ylabel(r't')
    ax3.set_xticks((0, 45, 90))
    plt.title('NN predictions')
    plt.colorbar(pl3, label='h', orientation='horizontal')
    ax1.set_ylim((0, 15))
    ax2.set_ylim((0, 15))
    ax3.set_ylim((0, 15))
    plt.subplots_adjust(wspace=0.35)
    plt.savefig('kpz/fig/space_time_front_predictions_wo_kpz.pdf')
    plt.show()

    emax = 0.05
    fig = plt.figure(figsize=(7, 2.5))
    ax1 = fig.add_subplot(121)
    pl1 = ax1.pcolor(xx, tt, np.abs(fronts-front_prediction), rasterized=True, vmin=0, vmax=emax,
                     cmap='plasma')
    ax1.set_ylabel(r't')
    ax1.set_xticks((0, 45, 90))
    plt.title('1-d front model')
    plt.colorbar(pl1, ax=ax1, label='error', orientation='horizontal')
    ax3 = fig.add_subplot(122)
    pl3 = ax3.pcolor(xx, tt, np.abs(fronts-prediction[:, 0]), rasterized=True, vmin=0, vmax=emax,
                     cmap='plasma')
    ax3.set_ylabel(r't')
    ax3.set_xticks((0, 45, 90))
    plt.title('NN predictions')
    plt.colorbar(pl3, label='error', orientation='horizontal')
    ax1.set_ylim((0, 25))
    ax2.set_ylim((0, 25))
    ax3.set_ylim((0, 25))
    plt.subplots_adjust(wspace=0.35)
    plt.savefig('kpz/fig/space_time_error_wo_kpz.pdf')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx, fronts[-1], label='Phase field front')
    ax.plot(xx, front_prediction[-1], label='1-d front model')
    ax.plot(xx, prediction[-1, 0], label='NN predictions')
    ax.set_xlabel('x')
    ax.set_ylabel('')
    plt.legend()
    plt.savefig('kpz/fig/predictions_all_last_snapshot_wo_kpz.pdf')
    plt.show()

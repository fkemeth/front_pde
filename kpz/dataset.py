import torch

import numpy as np

from scipy import ndimage

from kpz.utils import integrate, get_front, filter_front


class Dataset(torch.utils.data.Dataset):
    """Phase field front dataset."""

    def __init__(self, config, n_runs, start_idx=0):
        self.boundary_conditions = config["boundary_conditions"]
        self.initial_conditions = config["ic"]
        self.use_fd_dt = config["use_fd_dt"]
        if not self.use_fd_dt:
            self.dudt = config["dudt"]
        self.n_runs = n_runs
        self.delta_t = config["dt"]
        self.rescale_dx = float(config["rescale_dx"])
        self.config = config
        self.start_idx = start_idx
        self.x_data, self.delta_x, self.y_data, self.param = self.create_data()
        self.n_samples = self.x_data.shape[0]

    def create_data(self):
        x_data = []
        delta_x = []
        param = []
        y_data = []
        for i in range(self.start_idx, self.start_idx+self.n_runs):
            if self.config["load_data"]:
                data = np.load('kpz/data/fronts_'+str(i)+'.npy')
                dx = np.load('kpz/data/dx_'+str(i)+'.npy')
                pars = np.load('kpz/data/pars_'+str(i)+'.npy')
            else:
                data, dx, pars = integrate(self.config, ic=self.initial_conditions)
                fronts = get_front(data, dx)
                data = fronts
                np.save('kpz/data/fronts_'+str(i)+'.npy', data)
                np.save('kpz/data/dx_'+str(i)+'.npy', dx)
                np.save('kpz/data/pars_'+str(i)+'.npy', pars)

            if self.use_fd_dt:
                if self.config["noise_augment"]:
                    if self.config["fd_dt_acc"] == 2:
                        randn = 1e-5*np.random.randn(*data.shape)
                        y_data.append((data[1:]-(data[:-1]+randn[:-1]))/self.delta_t)
                        x_data.append(data[:-1]+randn[:-1])
                        delta_x.append(np.repeat(dx, len(data)-1))
                        param.append(np.repeat(pars[np.newaxis], len(data)-1, axis=0))
                    else:
                        raise ValueError(
                            "Finite difference in time with noise only implemented for accuracy 2.")
                else:
                    # y_data.append((data[1:]-data[:-1])/self.delta_t)
                    if self.config["fd_dt_acc"] == 2:
                        # accuracy 2
                        y_data.append((data[2:]-data[:-2])/(2*self.delta_t))
                        x_data.append(data[1:-1])
                        delta_x.append(np.repeat(dx, len(data)-2))
                        param.append(np.repeat(pars[np.newaxis], len(data)-2, axis=0))
                    elif self.config["fd_dt_acc"] == 4:
                        # accuracy 4
                        y_data.append((data[:-4]-8*data[1:-3]+8 *
                                       data[3:-1]-data[4:])/(12*self.delta_t))
                        x_data.append(data[2:-2])
                        delta_x.append(np.repeat(dx, len(data)-4))
                        param.append(np.repeat(pars[np.newaxis], len(data)-4, axis=0))
                    else:
                        raise ValueError("Finite difference in time accuracy must be 2 or 4.")
            else:
                y_data.append([self.config["dudt"](0, u, pars) for u in data])
                x_data.append(data)
                delta_x.append(np.repeat(dx, len(data)))
                param.append(np.repeat(pars[np.newaxis], len(data), axis=0))

        x_data = np.concatenate(x_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)
        delta_x = np.concatenate(delta_x, axis=0)*self.rescale_dx
        param = np.concatenate(param, axis=0)
        if len(x_data.shape) == 2:
            return x_data[:, np.newaxis], delta_x, y_data[:, np.newaxis], param
        else:
            return np.transpose(x_data, (0, 2, 1)), delta_x, np.transpose(y_data, (0, 2, 1)), param

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        _x = torch.tensor(self.x_data[index], dtype=torch.get_default_dtype())
        _dx = torch.tensor(self.delta_x[index], dtype=torch.get_default_dtype())
        _y = torch.tensor(self.y_data[index], dtype=torch.get_default_dtype())
        _p = torch.tensor(self.param[index], dtype=torch.get_default_dtype())
        return _x, _dx, _y, _p

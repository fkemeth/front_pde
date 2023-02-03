import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pickle
import findiff

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from scipy import interpolate

from scipy.integrate import solve_ivp


class Swish(nn.Module):
    """Nonlinear activation function."""

    def forward(self, input_tensor):
        """Forward pass through activation function."""
        return input_tensor * torch.sigmoid(input_tensor)


def kpz(D, a, x):
    """Kardar-Parisi-Zhang (KPZ) equation."""
    return D*x[:, 2] - a*torch.sqrt(2*D)*(1.0+(x[:, 1]**2)/2.0)


def diffusion(D, x):
    """Edwards-Wilkinson equation."""
    return D*x[:, 2]



def progress(train_loss, val_loss):
    """TQDM progress bar description."""
    return "Train/Loss: {:.8f} " \
           "Val/Loss: {:.8f}" \
           .format(train_loss, val_loss)


class Network(nn.Module):
    """Pytorch network architecture."""

    def __init__(self, config, n_vars):
        super(Network, self).__init__()
        self.kernel_size = int(config["kernel_size"])
        self.n_derivs = int(config["n_derivs"])
        self.device = config["device"]
        self.use_param = config["use_param"]
        self.num_params = config["num_params"]
        self.off_set = int((config["kernel_size"]-1)/2)
        self.n_vars = n_vars
        self.register_buffer('coeffs', self.get_coeffs(max_deriv=self.n_derivs))

        layers = []
        if self.use_param:
            num_features = int(self.n_vars*(self.n_derivs+1)+self.num_params)
        else:
            num_features = int(self.n_vars*(self.n_derivs+1))
        n_channels = int(config["n_filters"])
        for _ in range(int(config["n_layers"])):
            layers.append(nn.Conv1d(num_features, n_channels, [1], stride=1, padding=0, bias=True))
            # layers.append(nn.Tanh())
            layers.append(Swish())
            num_features = n_channels

        layers.append(nn.Conv1d(num_features, self.n_vars, [1], stride=1, padding=0, bias=True))

        self.network = nn.Sequential(*layers)
        self.trainable_parameters = sum(p.numel()
                                        for p in self.network.parameters() if p.requires_grad)

    def get_coeffs(self, min_deriv=0, max_deriv=5):
        """Get finite difference coefficients."""
        assert max_deriv > min_deriv, "Derivative range not specified"
        assert min_deriv >= 0, "Derivatives should be larger zero"

        coeffs = np.zeros((max_deriv-min_deriv+1, 1, self.kernel_size))
        # Finite difference coefficients
        for i in range(min_deriv, max_deriv+1):
            # Get coefficient for certain derivative with maximal acc order for given kernel_size
            fd_coeff = np.array([1.])
            if i > 0:
                acc_order = 0
                while len(fd_coeff) < self.kernel_size:
                    acc_order += 2
                    fd_coeff = findiff.coefficients(i, acc_order)["center"]["coefficients"]
                assert len(fd_coeff) == self.kernel_size, \
                    "Finite difference coefficients do not match kernel"
                coeffs[i, 0, :] = fd_coeff
            else:
                coeffs[i, 0, int((self.kernel_size-1)/2)] = 1.0
        return torch.tensor(coeffs, requires_grad=False,
                            dtype=torch.get_default_dtype()).to(self.device)

    def calc_derivs(self, x, dx):
        """Calculate derivativers of input x."""
        finite_diffs = F.conv1d(x, self.coeffs, groups=x.shape[1])
        scales = torch.cat([torch.pow(dx.unsqueeze(1), i)
                            for i in range(self.coeffs.shape[0])], axis=-1)
        scales = scales.repeat(1, self.n_vars)
        scales = scales.unsqueeze(2).repeat(1, 1, finite_diffs.shape[-1]).to(self.device)
        return finite_diffs/scales

    def forward(self, x, dx, param=None):
        """Predict temporal derivative of input x."""
        # Calculate derivatives
        x = self.calc_derivs(x, dx)
        if self.use_param:
            x = torch.cat([x, param.unsqueeze(-1).repeat(1, 1, x.shape[-1])], axis=1)
        # Forward through distributed parameter stack
        x = self.network(x)
        return x


class Additive_Network(nn.Module):
    """Gray-box additive network architecture."""

    def __init__(self, config, n_vars, a, D):
        super(Additive_Network, self).__init__()
        self.kernel_size = int(config["kernel_size"])
        self.n_derivs = int(config["n_derivs"])
        self.device = config["device"]
        self.use_param = config["use_param"]
        self.num_params = config["num_params"]
        self.off_set = int((config["kernel_size"]-1)/2)
        self.n_vars = n_vars
        self.a = torch.tensor(a, requires_grad=False,
                              dtype=torch.get_default_dtype()).to(self.device)
        self.D = torch.tensor(D, requires_grad=False,
                              dtype=torch.get_default_dtype()).to(self.device)
        self.register_buffer('coeffs', self.get_coeffs(max_deriv=self.n_derivs))

        layers = []
        if self.use_param:
            num_features = int(self.n_vars*(self.n_derivs+1)+self.num_params)
        else:
            num_features = int(self.n_vars*(self.n_derivs+1))
        n_channels = int(config["n_filters"])
        for _ in range(int(config["n_layers"])):
            layers.append(nn.Conv1d(num_features, n_channels, [1], stride=1, padding=0, bias=True))
            # layers.append(nn.Tanh())
            layers.append(Swish())
            num_features = n_channels

        layers.append(nn.Conv1d(num_features, self.n_vars, [1], stride=1, padding=0, bias=True))

        self.network = nn.Sequential(*layers)
        self.trainable_parameters = sum(p.numel()
                                        for p in self.network.parameters() if p.requires_grad)

    def get_coeffs(self, min_deriv=0, max_deriv=5):
        """Get finite difference coefficients."""
        assert max_deriv > min_deriv, "Derivative range not specified"
        assert min_deriv >= 0, "Derivatives should be larger zero"

        coeffs = np.zeros((max_deriv-min_deriv+1, 1, self.kernel_size))
        # Finite difference coefficients
        for i in range(min_deriv, max_deriv+1):
            # Get coefficient for certain derivative with maximal acc order for given kernel_size
            fd_coeff = np.array([1.])
            if i > 0:
                acc_order = 0
                while len(fd_coeff) < self.kernel_size:
                    acc_order += 2
                    fd_coeff = findiff.coefficients(i, acc_order)["center"]["coefficients"]
                assert len(fd_coeff) == self.kernel_size, \
                    "Finite difference coefficients do not match kernel"
                coeffs[i, 0, :] = fd_coeff
            else:
                coeffs[i, 0, int((self.kernel_size-1)/2)] = 1.0
        return torch.tensor(coeffs, requires_grad=False,
                            dtype=torch.get_default_dtype()).to(self.device)

    def calc_derivs(self, x, dx):
        """Calculate derivativers of input x."""
        finite_diffs = F.conv1d(x, self.coeffs, groups=x.shape[1])
        scales = torch.cat([torch.pow(dx.unsqueeze(1), i)
                            for i in range(self.coeffs.shape[0])], axis=-1)
        scales = scales.repeat(1, self.n_vars)
        scales = scales.unsqueeze(2).repeat(1, 1, finite_diffs.shape[-1]).to(self.device)
        return finite_diffs/scales

    def white_box(self, x):
        return kpz(self.D, self.a, x)

    def forward(self, x, dx, param=None):
        """Predict temporal derivative of input x."""
        # Calculate derivatives
        x = self.calc_derivs(x, dx)
        # Use KPZ
        y = self.white_box(x).unsqueeze(1)
        if self.use_param:
            x = torch.cat([x, param.unsqueeze(-1).repeat(1, 1, x.shape[-1])], axis=1)
        # Forward through distributed parameter stack
        x = y + self.network(x)
        return x


class Function_Network(nn.Module):
    """Gray-box functional network architecture."""

    def __init__(self, config, n_vars, a, D):
        super(Function_Network, self).__init__()
        self.kernel_size = int(config["kernel_size"])
        self.n_derivs = int(config["n_derivs"])
        self.device = config["device"]
        self.use_param = config["use_param"]
        self.num_params = config["num_params"]
        self.off_set = int((config["kernel_size"]-1)/2)
        self.n_vars = n_vars
        self.a = torch.tensor(a, requires_grad=False,
                              dtype=torch.get_default_dtype()).to(self.device)
        self.D = torch.tensor(D, requires_grad=False,
                              dtype=torch.get_default_dtype()).to(self.device)
        self.register_buffer('coeffs', self.get_coeffs(max_deriv=self.n_derivs))

        layers = []
        if self.use_param:
            num_features = int(self.n_vars*(self.n_derivs+1)+self.num_params)
        else:
            num_features = int(self.n_vars*(self.n_derivs+1))
        n_channels = int(config["n_filters"])
        # Use a combination of nearby input values
        # layers.append(nn.Conv1d(num_features, n_channels, [3], stride=1, padding=0, bias=True))
        # layers.append(Swish())
        # num_features = n_channels
        for _ in range(int(config["n_layers"])):
            # layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.Conv1d(num_features, n_channels, [1], stride=1, padding=0, bias=True))
            # layers.append(nn.Tanh())
            layers.append(Swish())
            num_features = n_channels

        layers.append(nn.Conv1d(num_features, self.n_vars, [1], stride=1, padding=0, bias=True))

        self.network = nn.Sequential(*layers)
        self.trainable_parameters = sum(p.numel()
                                        for p in self.network.parameters() if p.requires_grad)

    def get_coeffs(self, min_deriv=0, max_deriv=5):
        """Get finite difference coefficients."""
        assert max_deriv > min_deriv, "Derivative range not specified"
        assert min_deriv >= 0, "Derivatives should be larger zero"

        kernel_size = int(self.kernel_size/2)
        coeffs = np.zeros((max_deriv-min_deriv+1, 1, kernel_size))
        # Finite difference coefficients
        for i in range(min_deriv, max_deriv+1):
            # Get coefficient for certain derivative with maximal acc order for given kernel_size
            fd_coeff = np.array([1.])
            if i > 0:
                acc_order = 0
                while len(fd_coeff) < kernel_size:
                    acc_order += 2
                    fd_coeff = findiff.coefficients(i, acc_order)["center"]["coefficients"]
                assert len(fd_coeff) == kernel_size, \
                    "Finite difference coefficients do not match kernel"
                coeffs[i, 0, :] = fd_coeff
            else:
                coeffs[i, 0, int((kernel_size-1)/2)] = 1.0
        return torch.tensor(coeffs, requires_grad=False,
                            dtype=torch.get_default_dtype()).to(self.device)

    def calc_derivs(self, x, dx):
        """Calculate derivativers of input x."""
        finite_diffs = F.conv1d(x, self.coeffs, groups=x.shape[1])
        scales = torch.cat([torch.pow(dx.unsqueeze(1), i)
                            for i in range(self.coeffs.shape[0])], axis=-1)
        scales = scales.repeat(1, self.n_vars)
        scales = scales.unsqueeze(2).repeat(1, 1, finite_diffs.shape[-1]).to(self.device)
        return finite_diffs/scales

    def white_box(self, x):
        return kpz(self.D, self.a, x)

    def forward(self, x, dx, param=None):
        """Predict temporal derivative of input x."""
        # Calculate derivatives
        x = self.calc_derivs(x, dx)
        # Use KPZ
        x = self.white_box(x).unsqueeze(1)
        # Functional correction to the KPZ
        x = x[:, :, int(self.off_set/2):-int(self.off_set/2)] + \
            self.network(self.calc_derivs(x, dx))
        return x


class Model:
    def __init__(self, dataloader_train, dataloader_val, network, config, path):
        super().__init__()
        self.base_path = path

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        self.boundary_conditions = dataloader_train.dataset.boundary_conditions

        self.net = network
        self.device = self.net.device
        print('Using:', self.device)
        self.net = self.net.to(self.device)

        self.learning_rate = float(config["lr"])

        self.criterion = nn.MSELoss(reduction='sum').to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=int(config["patience"]),
            factor=float(config["reduce_factor"]), min_lr=1e-7)

    def pad(self, data, target):
        """Pad input/target depending on boundary conditions and kernel size."""
        if self.boundary_conditions == 'periodic':
            data = F.pad(data, (self.net.off_set, self.net.off_set), mode='circular')
            return data, target
        elif self.boundary_conditions == 'no-flux':
            # F.pad does not yet support symmetric relection
            # https://github.com/pytorch/pytorch/issues/46240
            data = F.pad(data, (self.net.off_set, self.net.off_set), mode='reflect')
            # left_bound = data[:, :, :self.net.off_set]
            # left_bound = torch.flip(left_bound, dims=[2])
            # right_bound = data[:, :, -self.net.off_set:]
            # right_bound = torch.flip(right_bound, dims=[2])
            # # reflect at outmost border to ensure zero-flux
            # data = torch.cat((left_bound, data, right_bound), axis=2)
            return data, target
        else:
            return data, target[:, :, self.net.off_set:-self.net.off_set]

    def train(self):
        """
        Train model over one epoch.

        Returns
        -------
        avg_loss: float
            Loss averaged over the training data
        """
        self.net = self.net.train()

        sum_loss, cnt = 0, 0
        for (data, delta_x, target, param) in self.dataloader_train:
            data, target = self.pad(data, target)
            data = data.to(self.device)
            delta_x = delta_x.to(self.device)
            target = target.to(self.device)
            if self.net.use_param:
                param = param.to(self.device)

            # backward
            self.optimizer.zero_grad()

            # forward
            if self.net.use_param:
                output = self.net(data, delta_x, param)
            else:
                output = self.net(data, delta_x)

            # compute loss
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            # measure accuracy on batch
            sum_loss += loss
            cnt += 1

        # Learning Rate reduction
        self.scheduler.step(sum_loss / cnt)

        return sum_loss / cnt

    def validate(self):
        """
        Validate model on validation set.

        Updates learning rate using scheduler.

        Updates best accuracy.

        Returns
        -------
        avg_loss: float
            Loss averaged over the validation data
        """
        self.net = self.net.eval()

        sum_loss, cnt = 0, 0
        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(self.dataloader_val):
            for (data, delta_x, target, param) in self.dataloader_val:
                data, target = self.pad(data, target)
                data = data.to(self.device)
                delta_x = delta_x.to(self.device)
                target = target.to(self.device)
                if self.net.use_param:
                    param = param.to(self.device)

                # forward
                if self.net.use_param:
                    output = self.net(data, delta_x, param)
                else:
                    output = self.net(data, delta_x)

                # loss / accuracy
                sum_loss += self.criterion(output, target)
                cnt += 1

        return sum_loss / cnt

    def save_network(self, name):
        """
        Save model to disk.

        Arguments
        -------
        name: str
            Model filename.

        Returns
        -------
        name: str
            Model filename.
        """
        model_file_name = self.base_path+name
        torch.save(self.net.state_dict(), model_file_name)
        return name

    def load_network(self, name):
        """
        Load model from disk.

        Arguments
        -------
        name: str
            Model filename.
        """
        model_file_name = self.base_path+name
        self.net.load_state_dict(torch.load(model_file_name))

    def dfdt(self, t, y, delta_x, param):
        """Return numpy output of model."""
        y = y.reshape(self.net.n_vars, -1)
        y = torch.tensor(y, dtype=torch.get_default_dtype()).unsqueeze(0).to(self.net.device)
        delta_x = torch.tensor(delta_x, dtype=torch.get_default_dtype()
                               ).unsqueeze(0).to(self.net.device)
        if self.net.use_param:
            param = torch.tensor(param, dtype=torch.get_default_dtype()
                                 ).unsqueeze(0).to(self.net.device)
        if self.boundary_conditions == 'periodic' or self.boundary_conditions == 'no-flux':
            y, _ = self.pad(y, None)
        else:
            pass
        if self.net.use_param:
            return self.net.forward(y, delta_x, param)[0].cpu().detach().numpy().flatten()
        else:
            return self.net.forward(y, delta_x)[0].cpu().detach().numpy().flatten()

    def integrate(self, initial_condition, pars, t_eval):
        """Integrate initial condition using the learned model."""
        sol = solve_ivp(self.dfdt, [0, t_eval[-1]], initial_condition,
                        t_eval=t_eval, args=pars, rtol=1e-8, atol=1e-12)
        if sol.status == -1:
            raise ValueError('Integration failed.')
        sol.y = sol.y.T
        sol.y = np.reshape(sol.y, (len(t_eval), self.net.n_vars, -1))
        return sol.t, sol.y

    # def integrate(self, dataset, svd, idx, horizon, use_svd=True):
    #     """Integrate idx'th snapshot of dataset for horizon time steps."""
    #     left_bounds, _, right_bounds, _, _, param = dataset.get_data(True)
    #     data = []
    #     if use_svd:
    #         data0 = svd.inverse_transform(
    #             svd.transform(dataset.x_data[idx].reshape(1, -1)))
    #         data.append(data0.reshape(2, -1))
    #     else:
    #         data.append(dataset.x_data[idx])

    #     for i in range(idx, horizon+idx):
    #         pred_f = self.net.forward(torch.tensor(data[-1], dtype=torch.get_default_dtype()
    #                                                ).unsqueeze(0).to(self.net.device),
    #                                   dataset.__getitem__(i)[1].unsqueeze(0).to(
    #             self.net.device),
    #             torch.tensor(param[idx],
    #                          dtype=torch.get_default_dtype()).unsqueeze(0).to(self.net.device))[0].cpu().detach().numpy()
    #         prediction = data[-1][:, dataset.off_set:-dataset.off_set] + dataset.delta_t*pred_f

    #         prediction = np.concatenate((left_bounds[i+1], prediction, right_bounds[i+1]), axis=1)
    #         if use_svd:
    #             prediction = svd.inverse_transform(
    #                 svd.transform(prediction.reshape(1, -1)))
    #             data.append(prediction.reshape(2, -1))
    #         else:
    #             data.append(prediction)
    #     return np.array(data)


class Additive_Network_Diffusion(nn.Module):
    """Gray-box additive network architecture."""

    def __init__(self, config, n_vars, a, D):
        super(Additive_Network_Diffusion, self).__init__()
        self.kernel_size = int(config["kernel_size"])
        self.n_derivs = int(config["n_derivs"])
        self.device = config["device"]
        self.use_param = config["use_param"]
        self.num_params = config["num_params"]
        self.off_set = int((config["kernel_size"]-1)/2)
        self.n_vars = n_vars
        self.a = torch.tensor(a, requires_grad=False,
                              dtype=torch.get_default_dtype()).to(self.device)
        self.D = torch.tensor(D, requires_grad=False,
                              dtype=torch.get_default_dtype()).to(self.device)
        self.register_buffer('coeffs', self.get_coeffs(max_deriv=self.n_derivs))

        layers = []
        if self.use_param:
            num_features = int(self.n_vars*(self.n_derivs+1)+self.num_params)
        else:
            num_features = int(self.n_vars*(self.n_derivs+1))
        n_channels = int(config["n_filters"])
        for _ in range(int(config["n_layers"])):
            layers.append(nn.Conv1d(num_features, n_channels, [1], stride=1, padding=0, bias=True))
            # layers.append(nn.Tanh())
            layers.append(Swish())
            num_features = n_channels

        layers.append(nn.Conv1d(num_features, self.n_vars, [1], stride=1, padding=0, bias=True))

        self.network = nn.Sequential(*layers)
        self.trainable_parameters = sum(p.numel()
                                        for p in self.network.parameters() if p.requires_grad)

    def get_coeffs(self, min_deriv=0, max_deriv=5):
        """Get finite difference coefficients."""
        assert max_deriv > min_deriv, "Derivative range not specified"
        assert min_deriv >= 0, "Derivatives should be larger zero"

        coeffs = np.zeros((max_deriv-min_deriv+1, 1, self.kernel_size))
        # Finite difference coefficients
        for i in range(min_deriv, max_deriv+1):
            # Get coefficient for certain derivative with maximal acc order for given kernel_size
            fd_coeff = np.array([1.])
            if i > 0:
                acc_order = 0
                while len(fd_coeff) < self.kernel_size:
                    acc_order += 2
                    fd_coeff = findiff.coefficients(i, acc_order)["center"]["coefficients"]
                assert len(fd_coeff) == self.kernel_size, \
                    "Finite difference coefficients do not match kernel"
                coeffs[i, 0, :] = fd_coeff
            else:
                coeffs[i, 0, int((self.kernel_size-1)/2)] = 1.0
        return torch.tensor(coeffs, requires_grad=False,
                            dtype=torch.get_default_dtype()).to(self.device)

    def calc_derivs(self, x, dx):
        """Calculate derivativers of input x."""
        finite_diffs = F.conv1d(x, self.coeffs, groups=x.shape[1])
        scales = torch.cat([torch.pow(dx.unsqueeze(1), i)
                            for i in range(self.coeffs.shape[0])], axis=-1)
        scales = scales.repeat(1, self.n_vars)
        scales = scales.unsqueeze(2).repeat(1, 1, finite_diffs.shape[-1]).to(self.device)
        return finite_diffs/scales

    def white_box(self, x):
        return diffusion(self.D, x)

    def forward(self, x, dx, param=None):
        """Predict temporal derivative of input x."""
        # Calculate derivatives
        x = self.calc_derivs(x, dx)
        # Use KPZ
        y = self.white_box(x).unsqueeze(1)
        if self.use_param:
            x = torch.cat([x, param.unsqueeze(-1).repeat(1, 1, x.shape[-1])], axis=1)
        # Forward through distributed parameter stack
        x = y + self.network(x)
        return x

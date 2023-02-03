import numpy as np

import findiff

import kpz.fun.stencil_2d as stl

from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit


def no_flux(y, N):
    """Enforce no-flux boundaries."""
    y[0:N] = y[N:2 * N]
    y[-N:] = y[-2 * N:-N]
    y[::N] = y[1::N]
    y[N - 1::N] = y[N - 2::N]
    return y


def periodic(y, N):
    y[0:N] = y[N:2 * N]
    y[-N:] = y[-2 * N:-N]
    return y


def f_phase_field(t, y, config, stencil):
    """Phase field equation."""
    # No flux boundaries
    N = config["N"]
    dx = config["L"]/float(N)
    if config["boundary_conditions"] == 'no-flux':
        y = no_flux(y, N)
    elif config["boundary_conditions"] == 'periodic':
        y = periodic(y, N)
    else:
        raise ValueError(
            'Boundary conditions must either be no-flux or periodic, but are ' +
            str(config["boundary_conditions"]))

    return config["D"]*stencil.dot(y) / (dx**2) - (y-config["a"]) * (y**2-1)


def create_initial_conditions(config, ic):
    """Create initial conditions."""
    N = config["N"]
    dx = config["L"]/float(N)
    phi = np.zeros((N, N))
    if config["boundary_conditions"] == 'no-flux':
        if ic == 'sergio':
            for ix in range(N):
                for iy in range(N):
                    roo = 15.0+0.66*np.cos(dx*iy*np.pi/(2*4)) - 0.66 * \
                        np.cos(dx*iy*np.pi/(3*4))+0.33*np.cos(dx*iy*np.pi/4)
                    r = dx*np.sqrt(float((ix-roo)*(ix-roo)))
                    phi[ix][iy] = np.tanh((roo-r)/(np.sqrt(2.0*config["D"])))
        elif ic == 'sergio_orig':
            for ix in range(N):
                for iy in range(N):
                    roo = 15.0+1.0*np.cos(dx*iy*np.pi/4) + 1.0*np.cos(dx*iy*np.pi /
                                                                      (2*4))+1.0*np.cos(dx*iy*np.pi/(3*4))
                    r = dx*np.sqrt(float((ix-roo)*(ix-roo)))
                    phi[ix][iy] = np.tanh((roo-r)/(np.sqrt(2.0*config["D"])))
        else:
            c1 = 10+10*np.random.random()
            a2, a3, a4, a5, a6 = [2*np.random.random(), 2*np.random.random(),
                                  2*np.random.random(), 2*np.random.random(),
                                  np.random.random()]
            c2, c3, c4, c5, c6 = [np.random.randint(0, 2), np.random.randint(0, 5),
                                  np.random.randint(0, 17), np.random.randint(0, 33),
                                  np.random.randint(32, 64)]
            for ix in range(N):
                for iy in range(N):
                    roo = c1 + \
                        a2*np.cos(c2*dx*iy*np.pi/config["L"]) + \
                        a3*np.cos(c3*dx*iy*np.pi/config["L"]) + \
                        a4*np.cos(c4*dx*iy*np.pi/config["L"]) + \
                        a5*np.cos(c5*dx*iy*np.pi/config["L"]) + \
                        a6*np.cos(c6*dx*iy*np.pi/config["L"])
                    r = dx*np.sqrt(float((ix-roo)*(ix-roo)))
                    phi[ix][iy] = np.tanh((roo-r)/(np.sqrt(2.0*config["D"])))
    elif config["boundary_conditions"] == 'periodic':
        if ic == 'sergio':
            for ix in range(N):
                for iy in range(N):
                    roo = 15.0+0.66*np.sin(dx*iy*np.pi/(2*4)) - 0.66 * \
                        np.sin(dx*iy*np.pi/(3*4))+0.33*np.sin(dx*iy*np.pi/4)
                    r = dx*np.sqrt(float((ix-roo)*(ix-roo)))
                    phi[ix][iy] = np.tanh((roo-r)/(np.sqrt(2.0*config["D"])))
        elif ic == 'sergio_orig':
            for ix in range(N):
                for iy in range(N):
                    roo = 15.0+1.0*np.sin(dx*iy*22*np.pi/config["L"]) + \
                        1.0*np.sin(dx*iy*12*np.pi/config["L"]) + \
                        1.0*np.sin(dx*iy*8*np.pi/config["L"])
                    r = dx*np.sqrt(float((ix-roo)*(ix-roo)))
                    phi[ix][iy] = np.tanh((roo-r)/(np.sqrt(2.0*config["D"])))
        else:
            c1 = 10+10*np.random.random()
            a2, a3, a4, a5, a6, a7 = [2*np.random.random(), 2*np.random.random(),
                                      2*np.random.random(), 2*np.random.random(),
                                      np.random.random(),
                                      0.2*np.random.random()]
            c2, c3, c4, c5, c6, c7 = [int(2*np.random.randint(1, 2)),
                                      int(2*np.random.randint(2, 4)),
                                      int(2*np.random.randint(4, 8)),
                                      int(2*np.random.randint(8, 16)),
                                      int(2*np.random.randint(16, 32)),
                                      int(2*np.random.randint(32, 64))]
            for ix in range(N):
                for iy in range(N):
                    roo = c1 + \
                        a2*np.sin(c2*dx*iy*np.pi/config["L"]) + \
                        a3*np.sin(c3*dx*iy*np.pi/config["L"]) + \
                        a4*np.sin(c4*dx*iy*np.pi/config["L"]) + \
                        a5*np.sin(c5*dx*iy*np.pi/config["L"]) + \
                        a6*np.sin(c6*dx*iy*np.pi/config["L"]) + \
                        a7*np.sin(c7*dx*iy*np.pi/config["L"])
                    r = dx*np.sqrt(float((ix-roo)*(ix-roo)))
                    phi[ix][iy] = np.tanh((roo-r)/(np.sqrt(2.0*config["D"])))
    else:
        raise ValueError(
            'Boundary conditions must either be no-flux or periodic, but are ' +
            str(config["boundary_conditions"]))
    return phi


def get_front_linear(data, dx):
    """Get the position of the front using linear fit."""
    idxsh = []
    for y in range(data.shape[1]):
        idx = np.argmax(data[:, y] <= 0.0, axis=0)
        idxh = idx - data[idx, y]/(data[idx+1, y]-data[idx, y])
        idxsh.append(idxh)
    return dx*np.array(idxsh)


def tanh_func(x, a, b):
    return np.tanh(a*x-b)


def jac(x, a, b):
    return np.array([(1-np.tanh(a*x-b)**2)*x, -(1-np.tanh(a*x-b)**2)]).T


def get_front(data, dx):
    """Get the position of the front using tanh fit."""
    xx = np.linspace(0, dx*data.shape[1], data.shape[1], endpoint=False)
    fronts = []
    for snapshot in data:
        idxs = []
        for strip in snapshot.T:
            params, _ = curve_fit(tanh_func, xx, strip, p0=[1, 15], jac=jac)
            idxs.append(params[1]/params[0])
        fronts.append(idxs)
    fronts = np.array(fronts)
    return fronts


def moving_average(x, w):
    """Running average."""
    return np.convolve(x, np.ones(w), 'same') / w


def filter_front(fronts, width=3, iterations=3):
    """Filter the front profiles."""
    filtered_front = np.zeros_like(fronts)
    for _ in range(iterations):
        for i in range(fronts.shape[0]):
            filtered_front[i] = moving_average(fronts[i], width)
            filtered_front[i, 0] = filtered_front[i, 1]
            filtered_front[i, -1] = filtered_front[i, -2]
    return filtered_front


def integrate(config, ic='sergio'):
    """Integrate phase field model."""
    print("Integrating phase field model.")
    if ic == 'sergio' or ic == 'mixed_random' or ic == 'sergio_orig' or ic is None:
        initial_condition = create_initial_conditions(config, ic)
    else:
        initial_condition = ic
    stencil = stl.create_stencil(config["N"], config["N"], 1)

    tt = np.linspace(0, config["T"] * config["dt"], config["T"]+1)

    sol = solve_ivp(f_phase_field, [0, tt[-1]], np.reshape(initial_condition, config["N"]**2),
                    t_eval=tt, args=[config, stencil])
    sol.y = sol.y.T

    # no flux boundaries
    if config["boundary_conditions"] == 'no-flux':
        for i in range(len(tt)):
            sol.y[i] = no_flux(sol.y[i], config["N"])

    sol.y = np.reshape(sol.y, (len(tt), config["N"], config["N"]))
    return sol.y, config["L"]/config["N"], np.array([config["a"], config["D"]])


def f_front(t, y, config):
    """Front dynamics."""
    # No flux boundaries
    dx = config["L"]/config["N"]
    d_dx = findiff.FinDiff(0, dx, 1)
    dd_dxx = findiff.FinDiff(0, dx, 2)

    if config["boundary_conditions"] == 'no-flux':
        y = np.pad(y, (10, 10), 'reflect')
        dudt = config["D"]/(1+d_dx(y)**2)*dd_dxx(y) -\
            np.sqrt(2*config["D"])*config["a"]*np.sqrt(1+d_dx(y)**2)
        return dudt[10:-10]
    else:
        y = np.pad(y, (10, 10), 'wrap')
        dudt = config["D"]/(1+d_dx(y)**2)*dd_dxx(y) -\
            np.sqrt(2*config["D"])*config["a"]*np.sqrt(1+d_dx(y)**2)
        return dudt[10:-10]


def integrate_front(config, tt, ic):
    """Integrate phase field model."""
    print("Integrating front equation.")

    tt = np.linspace(0, config["T"] * config["dt"], config["T"]+1)

    sol = solve_ivp(f_front, [0, tt[-1]], ic, t_eval=tt, args=[config], rtol=1e-6, atol=1e-9)
    sol.y = sol.y.T

    if config["boundary_conditions"] == 'no-flux':
        sol.y[:, :1] = sol.y[:, 1:2]
        sol.y[:, -1:] = sol.y[:, -2:-1]
    return sol.y, config["L"]/config["N"], np.array([config["a"], config["D"]])


def f_kpz(t, y, config):
    """Front dynamics."""
    # No flux boundaries
    dx = config["L"]/config["N"]
    d_dx = findiff.FinDiff(0, dx, 1)
    dd_dxx = findiff.FinDiff(0, dx, 2)

    if config["boundary_conditions"] == 'no-flux':
        y = np.pad(y, (20, 20), 'reflect')
    else:
        y = np.pad(y, (20, 20), 'wrap')
    dudt = config["D"]*dd_dxx(y) -\
        np.sqrt(2*config["D"])*config["a"]*(1+(1.0/2.0)*d_dx(y)**2)

    return dudt[20:-20]


def f_diffusion(t, y, config):
    """Front dynamics diffusion."""
    # No flux boundaries
    dx = config["L"]/config["N"]
    dd_dxx = findiff.FinDiff(0, dx, 2)

    if config["boundary_conditions"] == 'no-flux':
        y = np.pad(y, (20, 20), 'reflect')
    else:
        y = np.pad(y, (20, 20), 'wrap')
    dudt = config["D"]*dd_dxx(y)

    return dudt[20:-20]


def integrate_kpz(config, tt, ic):
    """Integrate KPZ model."""
    print("Integrating KPZ model.")

    tt = np.linspace(0, config["T"] * config["dt"], config["T"]+1)

    sol = solve_ivp(f_kpz, [0, tt[-1]], ic, t_eval=tt, args=[config], rtol=1e-6, atol=1e-9)
    sol.y = sol.y.T

    if config["boundary_conditions"] == 'no-flux':
        sol.y[:, :1] = sol.y[:, 1:2]
        sol.y[:, -1:] = sol.y[:, -2:-1]
    return sol.y, config["L"]/config["N"], np.array([config["a"], config["D"]])


def integrate_diffusion(config, tt, ic):
    """Integrate diffusion model."""
    print("Integrating Edwards-Wilkinson model.")

    tt = np.linspace(0, config["T"] * config["dt"], config["T"]+1)

    sol = solve_ivp(f_diffusion, [0, tt[-1]], ic, t_eval=tt, args=[config], rtol=1e-6, atol=1e-9)
    sol.y = sol.y.T

    if config["boundary_conditions"] == 'no-flux':
        sol.y[:, :1] = sol.y[:, 1:2]
        sol.y[:, -1:] = sol.y[:, -2:-1]
    return sol.y, config["L"]/config["N"], np.array([config["a"], config["D"]])

import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

show_plots = False


def do_method(A, U, t, Nx, Nt, plot_timestep, method='forward_euler',
              submethod='trapezoidal'):
    if method == 'forward_euler':
        for k in range(Nt - 1):
            U[:-1, (k + 1):(k + 2)] = \
                U[:-1, k:(k + 1)] + dt * A @ U[:-1, k:(k + 1)]
    elif method == 'backward_euler':
        for k in range(Nt - 1):
            U[:-1, (k + 1):(k + 2)] = \
                np.linalg.solve(np.eye(Nx - 1) - dt * A, U[:-1, k:(k + 1)])
    elif method == 'trapezoidal':
        for k in range(Nt - 1):
            U[:-1, (k + 1):(k + 2)] = \
                np.linalg.solve(np.eye(Nx - 1) - (dt / 2) * A,
                                (np.eye(Nx - 1) + (dt / 2) * A) @ U[:-1, k:(k + 1)])
    elif method == 'midpoint':
        # Use Trapezoidal as the first step
        if submethod == 'trapezoidal':
            U[:-1, 1:2] = \
                np.linalg.solve(np.eye(Nx - 1) - (dt / 2) * A,
                                (np.eye(Nx - 1) + (dt / 2) * A) @ U[:-1, 0:1])
        else:   # Forward Euler
            U[:-1, 1:2] = U[:-1, 0:1] + dt * A @ U[:-1, 0:1]
        for k in range(Nt - 2):
            U[:-1, (k + 2):(k + 3)] = \
                U[:-1, k:(k + 1)] + 2 * dt * A @ U[:-1, (k + 1):(k + 2)]
    elif method == 'lax_friedrichs':
        B = np.diag(np.ones(Nx - 2), 1) + np.diag(np.ones(Nx - 2), -1)
        B[0, -1] = 1
        B[-1, 0] = 1
        B = 0.5 * B
        for k in range(Nt - 1):
            U[:-1, (k + 1):(k + 2)] = \
                B @ U[:-1, k:(k + 1)] + dt * A @ U[:-1, k:(k + 1)]

    U[-1, :] = U[0, :]

    nrows = 2
    ncols = 4
    fig2, ax2 = plt.subplots(nrows, ncols)
    fig2.suptitle(f'{method}, dt = {dt}')
    time = 0
    for p in range(nrows):
        for q in range(ncols):
            if time <= tf:
                try:
                    ax2[p][q].plot(x, U[:, np.where(t == time)[0][0]])
                    ax2[p][q].set_title(f't = {time}')
                except IndexError:
                    pass
                time += plot_timestep

    if show_plots:
        fig2.show()

    return U


'''Problem 1
u_t = -3 * u_x

BCs:
u(t, -10) = u(t, 10)

IC:
                 { x + 1, -1 < x < 0
u(0, x) = f(x) = { 1 - x, 0 < x < 1
                 { 0    , otherwise

Solve using the method of lines, dx = 0.25, from t = 0 to t = 3.

a) Forward Euler, dt = 0.1.
b) Forward Euler, dt = 0.01.
c) Trapezoidal, dt = 0.1.
d) Trapezoidal, dt = 0.01.
e) Midpoint, dt = 0.1.
f) Midpoint, dt = 0.01.
g) Lax-Friedrichs, dt = 0.1.
h) Lax-Friedrichs, dt = 0.01.
'''
c = 3
#dx = 2
dx = 0.25
xi = -10
xf = 10
Nx = floor((xf - xi) / dx) + 1
x = np.linspace(xi, xf, Nx)
ti = 0
tf = 3


def f1(x):
    if -1 <= x < 0:
        return x + 1
    elif 0 <= x <= 1:
        return 1 - x
    else:
        return 0


def true_solution1(t, x):
    return f1(x - c * t)


def solve_advection_eqn_1(dt, method='forward_euler'):
    Nt = floor((tf - ti) / dt) + 1
    t = np.linspace(ti, tf, Nt)

    A = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
    A[0, -1] = -1
    A[-1, 0] = 1
    A = (- c / (2 * dx)) * A

    U = np.zeros((Nx, Nt))
    U[:-1, 0] = [f1(k) for k in x[:-1]]

    U = do_method(A, U, t, Nx, Nt, 0.5, method=method)

    return U, t


dt = 0.1
U, t = solve_advection_eqn_1(dt, method='forward_euler')
A1 = U[np.where(x == 9)[0][0], np.where(t == 3)[0][0]]
print(f'A1: {A1}')

dt = 0.01
U, t = solve_advection_eqn_1(dt, method='forward_euler')
A2 = U[np.where(x == 9)[0][0], np.where(t == 3)[0][0]]
print(f'A2: {A2}')

dt = 0.1
U, t = solve_advection_eqn_1(dt, method='trapezoidal')
A3 = U[np.where(x == 9)[0][0], np.where(t == 3)[0][0]]
print(f'A3: {A3}')

dt = 0.01
U, t = solve_advection_eqn_1(dt, method='trapezoidal')
A4 = U[np.where(x == 9)[0][0], np.where(t == 3)[0][0]]
print(f'A4: {A4}')

dt = 0.1
U, t = solve_advection_eqn_1(dt, method='midpoint')
A5 = U[np.where(x == 9)[0][0], np.where(t == 3)[0][0]]
print(f'A5: {A5}')

dt = 0.01
U, t = solve_advection_eqn_1(dt, method='midpoint')
A6 = U[np.where(x == 9)[0][0], np.where(t == 3)[0][0]]
print(f'A6: {A6}')

dt = 0.1
U, t = solve_advection_eqn_1(dt, method='lax_friedrichs')
A7 = U[np.where(x == 9)[0][0], np.where(t == 3)[0][0]]
print(f'A7: {A7}')

dt = 0.01
U, t = solve_advection_eqn_1(dt, method='lax_friedrichs')
A8 = U[np.where(x == 9)[0][0], np.where(t == 3)[0][0]]
print(f'A8: {A8}')


'''Problem 2
u_t = -(0.2 + sin(x - 1)^2) * u_x
0 < x < 2*pi

BCs:
u(t, 0) = u(t, 2*pi)

IC:
u(0, x) = cos(x)

Solve using the method of lines, dx = 2*pi / 100.  Solve from t = 0 to
t = 8.

a) Backward Euler, dt = 1.
b) Backward Euler, dt = 0.01.
c) Trapezoidal method, dt = 1.
d) Trapezoidal method, dt = 0.01.
e) Midpoint method, dt = 0.05.
f) Midpoint method, dt = 0.01.
g) Lax-Friedrichs, dt = 0.05.
h) Lax-Friedrichs, dt = 0.01.
'''
dx = 2 * np.pi / 100
xi = 0
xf = 2 * np.pi
Nx = ceil((xf - xi) / dx) + 1
x = np.linspace(xi, xf, Nx)
ti = 0
tf = 8


def f2(x):
    return np.cos(x)


def c(x):
    return 0.2 + np.sin(x - 1) ** 2


def solve_advection_eqn_2(dt, method='backward_euler'):
    Nt = floor((tf - ti) / dt) + 1
    t = np.linspace(ti, tf, Nt)

    A = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
    A[0, -1] = -1
    A[-1, 0] = 1
    A = (-1 / (2 * dx)) * A
    for ix, it in np.ndindex(A.shape):
        A[ix, it] = c(x[ix]) * A[ix, it]

    U = np.zeros((Nx, Nt))
    U[:-1, 0] = [f2(k) for k in x[:-1]]

    U = do_method(A, U, t, Nx, Nt, 1, method=method)

    return U, t


dt = 1
U, t = solve_advection_eqn_2(dt, method='backward_euler')
A9 = U[np.where(np.isclose(x, np.pi))[0][0], np.where(t == 4)[0][0]]
print(f'A9: {A9}')

dt = 0.01
U, t = solve_advection_eqn_2(dt, method='backward_euler')
A10 = U[np.where(np.isclose(x, np.pi))[0][0], np.where(t == 4)[0][0]]
print(f'A10: {A10}')

dt = 1
U, t = solve_advection_eqn_2(dt, method='trapezoidal')
A11 = U[np.where(np.isclose(x, np.pi))[0][0], np.where(t == 4)[0][0]]
print(f'A11: {A11}')

dt = 0.01
U, t = solve_advection_eqn_2(dt, method='trapezoidal')
A12 = U[np.where(np.isclose(x, np.pi))[0][0], np.where(t == 4)[0][0]]
print(f'A12: {A12}')

dt = 0.05
U, t = solve_advection_eqn_2(dt, method='midpoint')
A13 = U[np.where(np.isclose(x, np.pi))[0][0], np.where(t == 4)[0][0]]
print(f'A13: {A13}')

dt = 0.01
U, t = solve_advection_eqn_2(dt, method='midpoint')
A14 = U[np.where(np.isclose(x, np.pi))[0][0], np.where(t == 4)[0][0]]
print(f'A14: {A14}')

dt = 0.05
U, t = solve_advection_eqn_2(dt, method='lax_friedrichs')
A15 = U[np.where(np.isclose(x, np.pi))[0][0], np.where(t == 4)[0][0]]
print(f'A15: {A15}')

dt = 0.01
U, t = solve_advection_eqn_2(dt, method='lax_friedrichs')
A16 = U[np.where(np.isclose(x, np.pi))[0][0], np.where(t == 4)[0][0]]
print(f'A16: {A16}')


'''Problem 3
u_t = -u_x
0 < x < 25

BCs:
u(t, 0) = u(t, 25)

IC:
u(0, x) = f(x) = e^(-20 * (x - 2)^2) + e^(-(x - 5)^2)

Solve using the method of lines.  dx = 0.05, dt = 1/22.  Solve for u(17, 19).

a) Lax-Friedrichs
b) Midpoint method
'''
c = 1
dx = 0.05
xi = 0
xf = 25
Nx = floor((xf - xi) / dx) + 1
x = np.linspace(xi, xf, Nx)
dt = 1 / 22
ti = 0
tf = 17
Nt = floor((tf - ti)/ dt) + 1
t = np.linspace(ti, tf, Nt)


def f3(x):
    return np.exp(-20 * (x - 2) ** 2) + np.exp(-1 * (x - 5) ** 2)


def solve_advection_eqn_3(method='midpoint', submethod='trapezoidal'):
    A = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
    A[0, -1] = -1
    A[-1, 0] = 1
    A = (- c / (2 * dx)) * A

    U = np.zeros((Nx, Nt))
    U[:-1, 0] = [f3(k) for k in x[:-1]]

    U = do_method(A, U, t, Nx, Nt, 2, method=method,
                  submethod=submethod)

    return U


U = solve_advection_eqn_3(method='lax_friedrichs')
A17 = U[np.where(x == 19)[0][0], np.where(t == 17)[0][0]]
print(f'A17: {A17}')

U = solve_advection_eqn_3(method='midpoint')
A18 = U[np.where(x == 19)[0][0], np.where(t == 17)[0][0]]
print(f'A18: {A18}')

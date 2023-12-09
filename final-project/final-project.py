import matplotlib.pyplot as plt
import numpy as np
from math import floor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

show_plots = True


def discretize_time(dt):
    Nt = floor((tN - t0) / dt) + 1
    t = np.linspace(t0, tN, Nt)
    return dt, Nt, t


def plot_solution(t, x, U):
    T, X = np.meshgrid(t, x)
    ax = plt.axes(projection='3d')
    ax.plot_surface(T, X, U)

    # Point of interest for problem 1
    # x = np.where(x == 0)[0][0]
    # t = np.where(t == 0.25)[0][0]
    # ax.plot3D(0.25, 0, U[x, t], 'rp', markersize=5)

    # Point of interest for problem 2
    # x = np.where(x == 0.5)[0][0]
    # t = np.where(t == 0.1)[0][0]
    # ax.plot3D(0.1, 0.5, U[x, t], 'rp', markersize=5)

    if show_plots:
        plt.show()


'''HW4, Problem 1
Heat equation
u_t = 5 * u_xx
-1 < x < 2, t > 0

BCs:
u(t, -1) = 0
u(t, 2) = 0

IC:
u(0, x) = sin((4*pi/3)(x + 1))

Solve using the method of lines from t=0 to t=0.25, using dx = 0.125.
'''
t0 = 0
tN = 0.05
xi = -1
xf = 2
c = 5

dx = 0.125
Nx = floor((xf - xi) / dx) + 1
x = np.linspace(xi, xf, Nx)


def true_solution(t, x):
    return (np.exp(-5 * 16 * np.pi ** 2 * t / 9)
            * np.sin((4 * np.pi / 3) * (x + 1)))


def u0(x):
    return np.sin((4 * np.pi / 3) * (x + 1))


def method_of_lines(t, x, ic, bc_i, bc_f, c, method='forward_euler'):
    dt = t[1] - t[0]

    def f(t, u):
        return A @ u

    def f1(t, u):
        return f(t, u)

    def f2(t, u):
        return f(t + dt / 2, u + (dt / 2) * f1(t, u))

    def f3(t, u):
        return f(t + dt / 2, u + (dt / 2) * f2(t, u))

    def f4(t, u):
        return f(t + dt, u + dt * f3(t, u))

    def k1(t, u):
        return f(t, u)
    def k2(t, u):
        pass
    def k3(t, u):
        pass
    def k4(t, u):
        pass

    U = np.zeros((len(x), len(t)))
    U[:, 0] = ic(x)
    U[0, :] = bc_i
    U[-1, :] = bc_f

    A = (np.diag(-2 * np.ones(Nx - 2)) +
         np.diag(np.ones(Nx - 3), 1) +
         np.diag(np.ones(Nx - 3), -1)) * c / dx ** 2

    if method == 'forward_euler':
        for k in range(len(t) - 1):
            U[1:-1, (k + 1):(k + 2)] = \
                (U[1:-1, k:(k + 1)] + dt * f(t[k], U[1:-1, k:(k + 1)]))
    elif method == 'backward_euler':
        for k in range(len(t) - 1):
            U[1:-1, (k + 1):(k + 2)] = \
                np.linalg.solve(np.eye(len(x) - 2) - dt * A,
                                U[1:-1, k:(k + 1)])
    elif method == 'rk2':
        for k in range(len(t) - 1):
            U[1:-1, (k + 1):(k + 2)] = \
                U[1:-1, k:(k + 1)] + \
                dt * f(t[k] + (dt / 2),
                       U[1:-1, k:(k + 1)] + (dt / 2) * f(t[k],
                                                         U[1:-1, k:(k + 1)]))
    elif method == 'rk4':
        for k in range(len(t) - 1):
            U[1:-1, (k + 1):(k + 2)] = \
                U[1:-1, k:(k + 1)] + (dt / 6) * \
                (f1(t[k], U[1:-1, k:(k + 1)]) +
                 2 * f2(t[k], U[1:-1, k:(k + 1)]) +
                 2 * f3(t[k], U[1:-1, k:(k + 1)]) +
                 f4(t[k], U[1:-1, k:(k + 1)]))

    return U


# dt, Nt, t = discretize_time(dt=0.005)
# T, X = np.meshgrid(t, x)
# plot_solution(t, x, true_solution(T, X))
#
# dt, Nt, t = discretize_time(dt=0.005)
# U = method_of_lines(t, x, u0, 0, 0, 5, method='rk2')
# plot_solution(t, x, U)

dt, Nt, t = discretize_time(dt=0.0005)
U = method_of_lines(t, x, u0, 0, 0, 5, method='rk4')
plot_solution(t, x, U)

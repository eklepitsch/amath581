import matplotlib.pyplot as plt
import numpy as np
from math import floor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

'''Problem 1
Heat equation
u_t = 5 * u_xx
-1 < x < 2, t > 0

BCs:
u(t, -1) = 0
u(t, 2) = 0

IC:
u(0, x) = sin((4*pi/3)(x + 1))

Solve using the method of lines from t=0 to t=0.25, using dx = 0.125.

a) Use Forward Euler, dt = 2^-10.
b) Use Forward Euler, dt = 1/552
c) Use Backward Euler, dt = 0.05
d) Use Backward Euler, dt = 0.005
'''
t0 = 0
tN = 0.25
xi = -1
xf = 2

dx = 0.125
Nx = floor((xf - xi) / dx) + 1
x = np.linspace(xi, xf, Nx)


def u0_1(x):
    return np.sin((4 * np.pi / 3) * (x + 1))


def true_solution_1(t, x):
    return (np.exp(-5 * 16 * np.pi ** 2 * t / 9)
            * np.sin((4 * np.pi / 3) * (x + 1)))


def discretize_time(dt):
    Nt = floor((tN - t0) / dt) + 1
    t = np.linspace(t0, tN, Nt)
    return dt, Nt, t


def method_of_lines_1(t, x, ic, bc_i, bc_f, c, method='forward_euler'):
    def f(t, u):
        return A @ u

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

    return U


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

    plt.show()


def get_error(t, x, U, true_soln):
    T, X = np.meshgrid(t, x)
    return U - true_soln(T, X)


# Part (a)
dt, Nt, t = discretize_time(dt=2**-10)
U = method_of_lines_1(t, x, u0_1, 0, 0, 5, method='forward_euler')
plot_solution(t, x, U)
A1 = U[np.where(x == 0)[0][0], np.where(t == 0.25)[0][0]]
A2 = np.max(np.abs(get_error(t, x, U, true_solution_1)))
print(f'A1: {A1}')
print(f'A2: {A2}')


# Part (b)
dt, Nt, t = discretize_time(dt=1/552)
T, X = np.meshgrid(t, x)
U = method_of_lines_1(t, x, u0_1, 0, 0, 5, method='forward_euler')
plot_solution(t, x, U)
error = U - true_solution_1(T, X)
A3 = U[np.where(x == 0)[0][0], np.where(t == 0.25)[0][0]]
A4 = np.max(np.abs(get_error(t, x, U, true_solution_1)))
print(f'A3: {A3}')
print(f'A4: {A4}')

# part (c)
dt, Nt, t = discretize_time(dt=0.05)
U = method_of_lines_1(t, x, u0_1, 0, 0, 5, method='backward_euler')
plot_solution(t, x, U)
A5 = U[np.where(x == 0)[0][0], np.where(t == 0.25)[0][0]]
A6 = np.max(np.abs(get_error(t, x, U, true_solution_1)))
print(f'A5: {A5}')
print(f'A6: {A6}')


# part (d)
dt, Nt, t = discretize_time(dt=0.005)
U = method_of_lines_1(t, x, u0_1, 0, 0, 5, method='backward_euler')
plot_solution(t, x, U)
A7 = U[np.where(x == 0)[0][0], np.where(t == 0.25)[0][0]]
A8 = np.max(np.abs(get_error(t, x, U, true_solution_1)))
print(f'A7: {A7}')
print(f'A8: {A8}')


'''Problem 2
u_t = u_xx + 10x

BCs:
u(t, 0) = 0
u(t, 1) = 10t

IC:
u(0, x) = sin(pi * x) - 0.8 * sin(3 * pi * x)

True solution is:
u(t, x) = 10tx + e^(-pi^2 * t) * sin(pi * x) - 0.8e^(-9pi^2 * t) * sin(3pi * x)

dx = 0.05, solve from t = 0 to t = 0.1.

a) Solve with Forward Euler, dt = 1/550.
b) Solve with Forward Euler, dt = 0.0005.
c) Solve with Trapezoidal method, dt = 0.01.
d) Solve with Trapezoidal method, dt = 0.001.
'''
t0 = 0
tN = 0.1
xi = 0
xf = 1

dx = 0.05
Nx = floor((xf - xi) / dx) + 1
x = np.linspace(xi, xf, Nx)


def u0_2(x):
    return np.sin(np.pi * x) - 0.8 * np.sin(3 * np.pi * x)


def true_solution_2(t, x):
    return 10 * t * x + np.exp(-1 * np.pi ** 2 * t) * np.sin(np.pi * x) \
        - 0.8 * np.exp(-9 * np.pi ** 2 * t) * np.sin(3 * np.pi * x)


def method_of_lines_2(t, x, ic, bc_i, bc_f, method='forward_euler'):
    U = np.zeros((len(x), len(t)))
    U[:, 0] = ic(x)
    U[0, :] = bc_i
    U[-1, :] = bc_f

    A = (np.diag(-2 * np.ones(Nx - 2)) +
         np.diag(np.ones(Nx - 3), 1) +
         np.diag(np.ones(Nx - 3), -1)) * 1 / dx ** 2

    c = np.zeros((1, len(x) - 2))
    for j in range(1, len(x) - 1):
        c[0, j - 1] = 10 * x[j]
    cf = c[0, -1]

    if method == 'forward_euler':
        for k in range(len(t) - 1):
            c[0, -1] = cf + 10 * t[k] / dx ** 2
            U[1:-1, (k + 1):(k + 2)] = \
                U[1:-1, k:(k + 1)] + dt * (A @ U[1:-1, k:(k + 1)] + c.T)
    elif method == 'trapezoidal':
        c_this = c.copy()
        c_next = c.copy()
        for k in range(len(t) - 1):
            c_this[0, -1] = cf + 10 * t[k] / dx ** 2
            c_next[0, -1] = cf + 10 * t[k + 1] / dx ** 2
            M = dt / 2
            U[1:-1, (k + 1):(k + 2)] = \
                np.linalg.solve(np.eye(len(x) - 2) - M * A,
                                U[1:-1, k:(k + 1)] + M * A @ U[1:-1, k:(k + 1)]
                                + (dt / 2) * (c_this + c_next).T)

    return U


# Part (a)
dt, Nt, t = discretize_time(dt=1/550)
U = method_of_lines_2(t, x, u0_2, 0, 10 * t, method='forward_euler')
plot_solution(t, x, U)
A9 = U[np.where(x == 0.5)[0][0], np.where(t == 0.1)[0][0]]
A10 = np.max(np.abs(get_error(t, x, U, true_solution_2)))
print(f'A9: {A9}')
print(f'A10: {A10}')

# Part (b)
dt, Nt, t = discretize_time(dt=0.0005)
U = method_of_lines_2(t, x, u0_2, 0, 10 * t, method='forward_euler')
plot_solution(t, x, U)
A11 = U[np.where(x == 0.5)[0][0], np.where(t == 0.1)[0][0]]
A12 = np.max(np.abs(get_error(t, x, U, true_solution_2)))
print(f'A11: {A11}')
print(f'A12: {A12}')

# Part (c)
dt, Nt, t = discretize_time(dt=0.01)
U = method_of_lines_2(t, x, u0_2, 0, 10 * t, method='trapezoidal')
plot_solution(t, x, U)
A13 = U[np.where(x == 0.5)[0][0], np.where(t == 0.1)[0][0]]
A14 = np.max(np.abs(get_error(t, x, U, true_solution_2)))
print(f'A13: {A13}')
print(f'A14: {A14}')

# Part (d)
dt, Nt, t = discretize_time(dt=0.001)
U = method_of_lines_2(t, x, u0_2, 0, 10 * t, method='trapezoidal')
plot_solution(t, x, U)
A15 = U[np.where(x == 0.5)[0][0], np.where(t == 0.1)[0][0]]
A16 = np.max(np.abs(get_error(t, x, U, true_solution_2)))
print(f'A15: {A15}')
print(f'A16: {A16}')

# Plot true solution for problem 2
# T, X = np.meshgrid(t, x)
# ax = plt.axes(projection='3d')
# ax.plot_surface(T, X, true_solution_2(T, X))
# plt.show()

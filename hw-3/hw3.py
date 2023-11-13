import matplotlib.pyplot as plt
import numpy as np
import scipy
from math import floor
from mpl_toolkits.mplot3d import Axes3D

show_plots = False

'''Problem 1
y''(t) + 0.1 y'(t) + sin(y(t)) = 0
y(0) = y(6) = 0.5
dt = 0.06

Use a second order central difference scheme to discretize y'(t)
and y''(t).  Since the ODE is nonlinear, solve using Newton's method.

For Newton's method, use a tolerance of 1e-8.

a) Start with an initial guess of all 0.5's.
b) Start with an initial guess of
    y = 0.005t^4 - 0.07t^3 + 0.66t^2 - 2.56t + 0.55
'''
fig0, ax0 = plt.subplots(1, 2)

t0 = 0
tN = 6
y0 = 0.5
yN = 0.5
dt = 0.06

N = floor((tN - t0) / dt) + 1
t = np.linspace(t0, tN, N)


def F(y):
    z = np.zeros_like(y)
    z[0] = y[0] - y0
    z[-1] = y[-1] - yN
    for k in range(1, N - 1):
        z[k] = ((20 - dt) * y[k - 1] - 40 * y[k] + 20 * dt ** 2 * np.sin(y[k])
                + (20 + dt) * y[k + 1])
    return z


def jacobian(y):
    J = np.zeros((N, N))
    J[0, 0] = 1
    J[-1, -1] = 1
    for k in range(1, N - 1):
        J[k, k - 1] = 20 - dt
        J[k, k] = -40 + 20 * dt ** 2 * np.cos(y[k, 0])
        J[k, k + 1] = 20 + dt
    return J


# Part (a) - initial guess is all 0.5's
y = 0.5 * np.ones((N, 1))
k = 0
max_steps = 500
while np.max(np.abs(F(y))) >= 1e-8 and k < max_steps:
    dy = np.linalg.solve(jacobian(y), F(y))
    y = y - dy
    k = k + 1

y = y.reshape(N)
ax0[0].plot(t, y, 'k', t0, y0, 'ro', tN, yN, 'ro')

A1 = y[np.where(t == 3)[0][0]]
A2 = np.max(y)
A3 = np.min(y)
print(f'A1: {A1}')
print(f'A2: {A2}')
print(f'A3: {A3}')

# Part (b) - initial guess is given polynomial
y = 0.005 * t ** 4 - 0.07 * t ** 3 + 0.66 * t ** 2 - 2.56 * t + 0.55
y = np.array(y).reshape(-1, 1)
k = 0
while np.max(np.abs(F(y))) >= 1e-8 and k < max_steps:
    dy = np.linalg.solve(jacobian(y), F(y))
    y = y - dy
    k = k + 1

y = y.reshape(N)
A4 = y[np.where(t == 3)[0][0]]
A5 = np.max(y)
A6 = np.min(y)
print(f'A4: {A4}')
print(f'A5: {A5}')
print(f'A6: {A6}')

ax0[1].plot(t, y, 'k', t0, y0, 'ro', tN, yN, 'ro')

if show_plots:
    fig0.show()


'''Problem 2
u_xx + u_yy = 0, 0 < x < 3, 0 < y < 3

BCs:
u(x, 0) = x^2 - 3x
u(x, 3) = sin(2*pi*x/3)
u(0, y) = sin(pi*y/3)
u(3, y) = 3y - y^2

Solve using the direct method and a 5-point Laplacian.  May need to use sparse
matrices to avoid running out of memory.

a) Use dx = dy = 0.05
b) Use dx = dy = 0.015
'''

x0 = 0
xN = 3
y0 = 0
yN = 3


# Boundary condition u(x, 0)
def a(x):
    return x ** 2 - 3 * x


# Boundary condition u(x, 3)
def b(x):
    return np.sin(2 * np.pi * x / 3)


# Boundary condition u(0, y)
def c(y):
    return np.sin(np.pi * y / 3)


# Boundary condition u(3, y)
def d(y):
    return 3 * y - y ** 2


def solve_laplace_eqn(dx, sparse=True):
    dy = dx
    N = floor((xN - x0) / dx) + 1
    N_total = (N - 2) * (N - 2)  # Only interior points matter
    x = np.linspace(x0, xN, N)
    y = np.linspace(y0, yN, N)

    if sparse:
        A = scipy.sparse.dok_array((N_total, N_total))
    else:
        A = np.zeros((N_total, N_total))

    B = np.zeros((N_total, 1))

    def get_index(m, n):
        return (n - 1) * (N - 2) + m - 1

    for n in range(1, N - 1):
        for m in range (1, N - 1):
            k = get_index(m, n)
            A[k, k] = -4 / dx ** 2
            if m > 1:
                A[k, k - 1] = 1 / dx ** 2
            if n < N - 2:
                A[k, k + N - 2] = 1 / dx ** 2
            if m < N - 2:
                A[k, k + 1] = 1 / dx ** 2
            if n > 1:
                A[k, k - (N - 2)] = 1 / dx ** 2
            if n == 1:
                B[k] = B[k] - a(x[m]) / dx ** 2
            if n == N - 2:
                B[k] = B[k] - b(x[m]) / dx ** 2
            if m == 1:
                B[k] = B[k] - c(y[n]) / dx ** 2
            if m == N - 2:
                B[k] = B[k] - d(y[n]) / dx ** 2

    if sparse:
        A = A.tocsc()
        u_interior = scipy.sparse.linalg.spsolve(A, B).reshape((N - 2, N - 2))
    else:
        u_interior = np.linalg.solve(A, B).reshape((N - 2, N - 2))

    U = np.zeros((N, N))
    U[1:(N - 1), 1:(N - 1)] = u_interior
    U[0, :] = a(x)
    U[N - 1, :] = b(x)
    U[:, 0] = c(y)
    U[:, N - 1] = d(y)

    return U, x, y


# Part (a)
U, x, y = solve_laplace_eqn(dx=0.05, sparse=True)
# In U, the y-index is the row and the x-index is the column.
A7 = U[np.where(y == 1)[0][0], np.where(x == 1)[0][0]]
A8 = U[np.where(y == 2)[0][0], np.where(x == 2)[0][0]]
print(f'A7: {A7}')
print(f'A8: {A8}')

# Part (b)
U, x, y = solve_laplace_eqn(dx=0.015, sparse=True)
A9 = U[np.where(np.isclose(y, 1.005))[0][0],
       np.where(np.isclose(x, 1.005))[0][0]]
A10 = U[np.where(np.isclose(y, 1.995))[0][0],
        np.where(np.isclose(x, 1.995))[0][0]]
print(f'A9: {A9}')
print(f'A10: {A10}')

if show_plots:
    fig2, ax2 = plt.subplots(1, 1)
    X, Y = np.meshgrid(x, y)
    ax2 = plt.axes(projection='3d')
    ax2.plot_surface(X, Y, U)

    zero_vector = np.zeros_like(x)
    n_vector = xN * np.ones_like(x)
    # u(x, 0)
    ax2.plot3D(x, zero_vector, a(x), 'r')
    # u(x, 3)
    ax2.plot3D(x, n_vector, b(x), 'r')
    # u(0, y)
    ax2.plot3D(zero_vector, y, c(y), 'r')
    # u(3, y)
    ax2.plot3D(n_vector, y, d(y), 'r')

    fig2.show()


'''Problem 3
u_xx + u_yy = -e^(-2 * (x^2 + y^2)), -1 < x < 1, -1 < y < 1

BCs:
u(x, 1) = (x^3 - x)/3
u(x, -1) = 0
u(-1, y) = 0
u(1, y) = 0

Solve using the direct method and a 5-point Laplacian.

a) Use dx = 0.1 and dy = 0.05
b) Use dx = 0.01 and dy = 0.025
'''

xi = -1
xf = 1
yi = -1
yf = 1


# Boundary condition u(x, 1)
def a(x):
    return (x ** 3 - x) / 3


# Nonhomogeneous term
def f(x, y):
    return -1 * np.exp(-2 * (x ** 2 + y ** 2))


def solve_laplace_eqn(dx, dy, sparse=True):
    Nx = floor((xf - xi) / dx) + 1
    Ny = floor((yf - yi) / dy) + 1
    N_total = (Nx - 2) * (Ny - 2)  # One eqn for each interior point
    x = np.linspace(xi, xf, Nx)
    y = np.linspace(yi, yf, Ny)
    U = np.zeros((Ny, Nx))

    if sparse:
        A = scipy.sparse.dok_array((N_total, N_total))
    else:
        A = np.zeros((N_total, N_total))

    B = np.zeros((N_total, 1))

    def get_index(m, n):
        return (n - 1) * (Nx - 2) + m - 1

    for n in range(1, Ny - 1):
        for m in range(1, Nx - 1):
            k = get_index(m, n)
            A[k, k] = -2 * ((1 / dx ** 2) + (1 / dy ** 2))
            if m > 1:
                A[k, k - 1] = 1 / dx ** 2
            if n < Ny - 2:
                A[k, k + Nx - 2] = 1 / dy ** 2
            if m < Nx - 2:
                A[k, k + 1] = 1 / dx ** 2
            if n > 1:
                A[k, k - (Nx - 2)] = 1 / dy ** 2
            if n == Ny - 2:
                B[k] = f(x[m], y[n]) - a(x[m]) / dy ** 2
            else:
                B[k] = f(x[m], y[n])

    if sparse:
        A = A.tocsc()
        u_interior = scipy.sparse.linalg.spsolve(A, B).reshape((Ny - 2, Nx - 2))
    else:
        u_interior = np.linalg.solve(A, B).reshape((Ny - 2, Nx - 2))

    U[1:(Ny - 1), 1:(Nx - 1)] = u_interior
    U[-1, :] = a(x)

    return U, x, y


U, x, y = solve_laplace_eqn(dx=0.1, dy=0.05, sparse=True)
A11 = U[np.where(y == 0)[0][0], np.where(x == 0)[0][0]]
A12 = U[np.where(np.isclose(y, 0.5))[0][0],
np.where(np.isclose(x, -0.5))[0][0]]
print(f'A11: {A11}')
print(f'A12: {A12}')

U, x, y = solve_laplace_eqn(dx=0.01, dy=0.025, sparse=True)
A13 = U[np.where(y == 0)[0][0], np.where(x == 0)[0][0]]
A14 = U[np.where(np.isclose(y, 0.5))[0][0],
np.where(np.isclose(x, -0.5))[0][0]]
print(f'A13: {A13}')
print(f'A14: {A14}')

if show_plots:
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax3.plot_surface(X, Y, U)

    nones_vector_x = -1 * np.ones_like(x)
    ones_vector_x = np.ones_like(x)
    nones_vector_y = -1 * np.ones_like(y)
    ones_vector_y = np.ones_like(y)
    # u(x, -1)
    ax3.plot3D(x, -1 * np.ones_like(x), 0, 'r')
    # u(x, 1)
    ax3.plot3D(x, np.ones_like(x), a(x), 'r')
    # u(-1, y)
    ax3.plot3D(-1 * np.ones_like(y), y, 0, 'r')
    # u(1, y)
    ax3.plot3D(np.ones_like(y), y, 0, 'r')
    ax3.view_init(-135, 30)

    fig3.show()

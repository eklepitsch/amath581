import matplotlib.pyplot as plt
import numpy as np
from math import floor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

show_plots = False

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


dt = 0.1
Nt = floor((tf - ti) / dt) + 1
t = np.linspace(ti, tf, Nt)

A = np.diag(np.ones(Nx - 2), 1) - np.diag(np.ones(Nx - 2), -1)
A[0, -1] = -1
A[-1, 0] = 1
A = (- c / (2 * dx)) * A

U = np.zeros((Nx, Nt))
U[:-1, 0] = [f1(k) for k in x[:-1]]
# print(x)
# print([f1(k) for k in x])
# print(U)

# Forward Euler
for k in range(Nt - 1):
    U[:-1, (k + 1):(k + 2)] = U[:-1, k:(k + 1)] + dt * A @ U[:-1, k:(k + 1)]
U[-1, :] = U[0, :]

fig1, ax1 = plt.subplots(2, 4)
fig2, ax2 = plt.subplots(2, 4)
time = 0
for p in range(2):
    for q in range(4):
        if time <= 3:
            ax1[p][q].plot(x, U[:, np.where(t == time)[0][0]])
            ax1[p][q].set_title(f't = {time}')
            ax2[p][q].plot(x, )
            time += 0.5
fig1.show()
fig2.show()

A1 = U[np.where(x == 9)[0][0], np.where(t == 3)[0][0]]
print(f'A1: {A1}')

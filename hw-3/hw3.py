import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil

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
fig1, ax1 = plt.subplots(1, 2)

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
print(f'Steps needed for 1a: {k}')
ax1[0].plot(t, y, 'k', t0, y0, 'ro', tN, yN, 'ro')

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

print(f'Steps needed for 1b: {k}')

y = y.reshape(N)
A4 = y[np.where(t == 3)[0][0]]
A5 = np.max(y)
A6 = np.min(y)
print(f'A4: {A4}')
print(f'A5: {A5}')
print(f'A6: {A6}')

ax1[1].plot(t, y, 'k', t0, y0, 'ro', tN, yN, 'ro')

plt.show()
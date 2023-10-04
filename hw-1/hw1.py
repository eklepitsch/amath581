import numpy as np
import matplotlib.pyplot as plt

''' Problem 1
x'(t) = -4 * x * sin(t)
x(0) = 1
t0 = 0, tN = 8, dt = ?

(a) Forward Euler, dt = 2e-5, dt = 2e-6
(b) Huen, dt = 2e-5, dt = 2e-6
(c) RK2, dt = 2e-5, dt = 2e-6
'''
x0 = 1
t0 = 0
tN = 8


def f(x, t):
    return -4 * x * np.sin(t)


def true_solution(t):
    return np.exp(4 * (np.cos(t) - 1))


# Part (a)
def forward_euler(t0, tN, dt):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = x[k] + dt * f(x[k], t[k])

    return t, x


# tplot = np.linspace(t0, tN, 10000)
# xplot = true_solution(tplot)
# plt.plot(tplot, xplot, 'k')

dt = 2e-5
t, approx_solution = forward_euler(t0, tN, dt)
local_error = abs(true_solution(t[1]) - approx_solution[1])
global_error = abs(true_solution(t[-1]) - approx_solution[-1])
A1 = local_error
A2 = global_error

dt = 2e-6
t, approx_solution = forward_euler(t0, tN, dt)
local_error = abs(true_solution(t[1]) - approx_solution[1])
global_error = abs(true_solution(t[-1]) - approx_solution[-1])
A3 = local_error
A4 = global_error

print(f'A1 = {A1}')
print(f'A2 = {A2}')
print(f'A3 = {A3}')
print(f'A4 = {A4}')


#plt.show()

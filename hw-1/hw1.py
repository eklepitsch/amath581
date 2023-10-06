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
def forward_euler(t0, tN, x0, dt):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = x[k] + dt * f(x[k], t[k])

    return t, x


def heun(t0, tN, x0, dt):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] =\
            x[k] + (dt / 2) * (f(x[k], t[k]) +
                               f(x[k] + dt * f(x[k], t[k]), t[k] + dt))

    return t, x


def rk2(t0, tN, x0, dt):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = (x[k] + dt *
                    f(x[k] + (dt / 2) * f(x[k], t[k]), t[k] + (dt / 2)))

    return t, x


def do_numerical_method(method, t0, tN, x0, dt):
    t, approx_solution = method(t0, tN, x0, dt)
    local_error = abs(true_solution(t[1]) - approx_solution[1])
    global_error = abs(true_solution(t[-1]) - approx_solution[-1])
    return local_error, global_error


dt = 2e-5
A1, A2 = do_numerical_method(forward_euler, t0, tN, x0, dt)

dt = 2e-6
A3, A4 = do_numerical_method(forward_euler, t0, tN, x0, dt)

print('Forward Euler:')
print(f'A1 = {A1}')
print(f'A2 = {A2}')
print(f'A3 = {A3}')
print(f'A4 = {A4}')

dt = 2e-5
A5, A6 = do_numerical_method(heun, t0, tN, x0, dt)

dt = 2e-6
A7, A8 = do_numerical_method(heun, t0, tN, x0, dt)

print('Heun:')
print(f'A5 = {A5}')
print(f'A6 = {A6}')
print(f'A7 = {A7}')
print(f'A8 = {A8}')

dt = 2e-5
A9, A10 = do_numerical_method(rk2, t0, tN, x0, dt)

dt = 2e-6
A11, A12 = do_numerical_method(rk2, t0, tN, x0, dt)

print('rk2:')
print(f'A9 = {A9}')
print(f'A10 = {A10}')
print(f'A11 = {A11}')
print(f'A12 = {A12}')
#plt.show()

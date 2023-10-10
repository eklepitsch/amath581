import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


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


def f1(x, t):
    return -4 * x * np.sin(t)


def true_solution1(t):
    return np.exp(4 * (np.cos(t) - 1))


# Part (a)
def forward_euler(t0, tN, x0, dt, f):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = x[k] + dt * f(x[k], t[k])

    return t, x


def heun(t0, tN, x0, dt, f):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] =\
            x[k] + (dt / 2) * (f(x[k], t[k]) +
                               f(x[k] + dt * f(x[k], t[k]), t[k] + dt))

    return t, x


def rk2(t0, tN, x0, dt, f):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = (x[k] + dt *
                    f(x[k] + (dt / 2) * f(x[k], t[k]), t[k] + (dt / 2)))

    return t, x


def do_numerical_method(method, t0, tN, x0, dt, f, true_solution):
    t, approx_solution = method(t0, tN, x0, dt, f)
    local_error = abs(true_solution(t[1]) - approx_solution[1])
    global_error = abs(true_solution(t[-1]) - approx_solution[-1])
    final_approximation = approx_solution[-1]
    return local_error, global_error, final_approximation


dt = 2e-5
A1, A2, _ = do_numerical_method(forward_euler, t0, tN, x0, dt, f1, true_solution1)

dt = 2e-6
A3, A4, _ = do_numerical_method(forward_euler, t0, tN, x0, dt, f1, true_solution1)

print('Forward Euler:')
print(f'A1 = {A1}')
print(f'A2 = {A2}')
print(f'A3 = {A3}')
print(f'A4 = {A4}')

dt = 2e-5
A5, A6, _ = do_numerical_method(heun, t0, tN, x0, dt, f1, true_solution1)

dt = 2e-6
A7, A8, _ = do_numerical_method(heun, t0, tN, x0, dt, f1, true_solution1)

print('Heun:')
print(f'A5 = {A5}')
print(f'A6 = {A6}')
print(f'A7 = {A7}')
print(f'A8 = {A8}')

dt = 2e-5
A9, A10, _ = do_numerical_method(rk2, t0, tN, x0, dt, f1, true_solution1)

dt = 2e-6
A11, A12, _ = do_numerical_method(rk2, t0, tN, x0, dt, f1, true_solution1)

print('rk2:')
print(f'A9 = {A9}')
print(f'A10 = {A10}')
print(f'A11 = {A11}')
print(f'A12 = {A12}')


''' Problem 2
x'(t) = 8 * sin(x)
x(0) = pi / 4
t0 = 0, tN = 2, dt = ?

(a) predictor-corrector method, dt = 0.1
(b) predictor-corrector method, dt = 0.01
'''
x0 = np.pi / 4
t0 = 0
tN = 2


def f2(x, t):
    return 8 * np.sin(x)


def true_solution2(t):
    return 2 * np.arctan(np.exp(8 * t) / (1 + np.sqrt(2)))


def predictor_corrector(t0, tN, x0, dt, f):
    # Do step 1 using rk2
    t, approx_solution = rk2(t0, dt, x0, dt, f2)

    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0
    x[1] = approx_solution[1]

    # Do remaining steps using predictor-corrector
    for k in range(1, len(t) - 1):
        xp = x[k] + (dt / 2) * (3 * f(x[k], t[k]) - f(x[k-1], t[k-1]))
        x[k + 1] = x[k] + (dt / 2) * (3 * f(xp, t[k+1]) + f(x[k], t[k]))

    return t, x


dt = 0.1
_, ge, final_approximation = do_numerical_method(
    predictor_corrector, t0, tN, x0, dt, f2, true_solution2)
A13 = final_approximation
A14 = ge

dt = 0.01
_, ge, final_approximation = do_numerical_method(
    predictor_corrector, t0, tN, x0, dt, f2, true_solution2)
A15 = final_approximation
A16 = ge

print('Predictor-corrector:')
print(f'A13 = {A13}')
print(f'A14 = {A14}')
print(f'A15 = {A15}')
print(f'A16 = {A16}')


''' Problem 3
FitzHugh-Nagumo model
v = voltage, w = membrane channel activity
v'(t) = v - (1/3)v^3 - w + I(t)  
w'(t) = (a + v - bw) / T
I(t) = (1/10) * (5 + sin(pi * t / 10))

ICs:
v(0) = 0.1, w(0) = 1
t0 = 0, tN = 100
'''
a = 0.7
b = 1
T = 12
t0 = 0
tN = 100
v0 = 0.1
w0 = 1


def fitzhugh_nagumo(t, z, a, b, T):
    v, w = z
    return [
        v - (1 / 3) * pow(v, 3) - w + (1 / 10) * (5 + np.sin(np.pi * t / 10)),
        (a + v - b * w) / T
    ]


def solve_fitzhugh_nagumo(tol):
    sol = solve_ivp(fitzhugh_nagumo, [t0, tN], [v0, w0],
                    args=(a, b, T), atol=tol, rtol=tol, method='RK45')
    t = sol.t
    v = sol.y[0]
    w = sol.y[1]
    dt = [b - a for a, b in zip(t, t[1:])]
    average_time_step = sum(dt) / len(dt)
    final_voltage = v[-1]

    return final_voltage, average_time_step


A17, A18 = solve_fitzhugh_nagumo(tol=1e-4)
A19, A20 = solve_fitzhugh_nagumo(tol=1e-9)

print('fitzhugh_nagumo:')
print(f'A17 = {A17}')
print(f'A18 = {A18}')
print(f'A19 = {A19}')
print(f'A20 = {A20}')

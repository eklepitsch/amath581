import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

show_plots = True


def bisection(f, a, b, tol):
    x = (a + b) / 2
    while np.abs(b - a) >= tol:
        if np.sign(f(x)) == np.sign(f(a)):
            a = x
        else:
            b = x
        x = (a + b) / 2
    return x


'''Problem 1
Eigenvalue problem.  The system is:

p(x), v(x), lambda

where p(x) is position, v(x) = p'(x) = velocity

p' = v
v' = -1 * beta * p
beta = lambda - 100 * (sin(2*x) + 1)

p(-1) = 0
p(1) = 0
v(-1) = ?
v(1) = ?

Given: first eigenvalue is between 23 and 24
'''
print('Problem 1\n------------')


def beta(x, l):
    """
    x: independent variable
    l: lambda (eigenvalue)
    """
    return l - 100 * (np.sin(2*x) + 1)


def f(x, y, beta):
    """
    vector valued function
    x: independent variable
    y: vector = [p, v]
    beta: eigenvalue
    """
    p = y[0]
    v = y[1]
    return np.array([v, -1 * beta * p])


def shoot(l):
    """l: lambda (eigenvalue)"""
    tspan = np.array([xi, xf])
    ic = np.array([pi, vi])
    sol = solve_ivp(lambda x, y: f(x, y, beta(x, l)), tspan, ic)
    # sol.y[0, :] = position
    # sol.y[1, :] = velocity
    return sol.y[0, -1]  # final position


L = 1
xi = -L    # initial value of x
xf = L     # final value of x
pi = 0     # initial position
pf = 0     # final position
A = 1      # Initial velocity guess
vi = A     # initial velocity
vf = None  # final velocity (unknown)

# The first eigenvalue is between 23 and 24,
# so start guessing at 0.1.
l = 0.1   # lambda
dl = 0.1

i = 0
num_eigenvals = 5
eigenvalues = []

# Solve the final position assuming the eigenvalue is l
position_final = shoot(l)
sign = np.sign(position_final)

while i < num_eigenvals:
    # Solve the final position assuming the eigenvalue is l + dl
    position_final_next = shoot(l + dl)
    sign_next = np.sign(position_final_next)
    if sign != sign_next:
        eigenvalues.append(bisection(shoot, l, l + dl, 1e-8))
        i = i + 1
    sign = sign_next
    l = l + dl

for i, e in enumerate(eigenvalues):
    print(f'Eigenvalue {i+1}: {eigenvalues[i]}')


def calculate_p0(eigenvalue):
    tspan = np.array([xi, xf])
    ic = np.array([pi, vi])
    sol = solve_ivp(lambda x, y: f(x, y, beta(x, eigenvalue)),
                    tspan, ic, t_eval=np.array([-1, 0, 1]))
    p0 = sol.y[0, 1]  # position at x = 0
    v0 = sol.y[1, 1]  # velocity at x = 0
    return p0


# Part (a) - Calculate the first eigenvalue
A1 = eigenvalues[0]
print(f'A1: {A1}')

# Part (b) - Calculate the eigenfunction p(x) corresponding
# to the first eigenvalue.  Find p(0).
A2 = calculate_p0(eigenvalues[0])
print(f'A2: {A2}')

# Part (c) - Calculate the second eigenvalue
A3 = eigenvalues[1]
print(f'A3: {A3}')

# Part (d) - Calculate the eigenfunction p(x) corresponding
# to the second eigenvalue.  Find p(0).
A4 = calculate_p0(eigenvalues[1])
print(f'A4: {A4}')

# Part (e) - Calculate the third eigenvalue
A5 = eigenvalues[2]
print(f'A5: {A5}')

# Part (f) - Calculate the eigenfunction p(x) corresponding
# to the third eigenvalue.  Find p(0).
A6 = calculate_p0(eigenvalues[2])
print(f'A6: {A6}')

if show_plots:
    fig, ax = plt.subplots(1,2)
    fig.suptitle('Problem 1')
    ax[0].set_title('Position')
    ax[1].set_title('Velocity')
    for i, e in enumerate(eigenvalues):
        tspan = np.array([xi, xf])
        ic = np.array([pi, vi])
        sol = solve_ivp(lambda x, y: f(x, y, beta(x, e)),
                        tspan, ic, t_eval=np.linspace(xi, xf, 10000))
        ax[0].plot(sol.t, sol.y[0, :], label=f'Mode {i}')
        ax[1].plot(sol.t, sol.y[1, :], label=f'Mode {i}')
        ax[0].legend()
        ax[1].legend()

''' Problem 2, parts (a-b)
IVP:

x'' - x = 0
x(0) = 1
x'(0) = 0
t0 = 0
tN = 1

This 2nd-order IVP can be written as a system of 1st-order ODEs.
Let x = (y z)
y' = z
z' = y
y(0) = 1
z(0) = 0

Given: The true solution is x(t) = 1/2 * (e^2 + e^-t).
'''
print('Problem 2\n------------')


def true_solution(t):
    return (1 / 2) * (np.exp(t) + np.exp(-1 * t))


def ge(t, true_solution, approx_solution):
    # Global error
    return abs(true_solution(t[-1]) - approx_solution[-1])


def trapezoidal_method_for_2_a(t0, tN, x0, dt):
    """x0: vector [y z]"""
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros([2, len(t)])  # vector valued [y z]
    x[:, 0] = x0
    y = x[0, :]
    z = x[1, :]

    # Constants
    A = (2 / dt) + (dt / 2)
    B = (2 / dt) - (dt / 2)

    for k in range(len(t) - 1):
        z[k + 1] = (2 * y[k] + A * z[k]) / B
        y[k + 1] = y[k] + (dt / 2) * (z[k] + z[k + 1])

    return t, x


fig, ax = plt.subplots(1, 2)
fig.suptitle('Problem 2 - parts (a-b)')

t0 = 0
tN = 1
x0 = np.array([[1, 0]])
dt = 0.1
t, x = trapezoidal_method_for_2_a(t0, tN, x0, dt)

A7 = x[0, -1]
A8 = ge(t, true_solution, x[0, :])
print(f'A7: {A7}')
print(f'A8: {A8}')

if show_plots:
    ax[0].plot(t, x[0, :], 'ro', label='Approx')
    ax[0].plot(t, true_solution(t), label='True soln')
    ax[0].set_title(rf'$\Delta t$ = {dt}')
    ax[0].legend()

dt = 0.01
t, x = trapezoidal_method_for_2_a(t0, tN, x0, dt)
A9 = x[0, -1]
A10 = ge(t, true_solution, x[0, :])
print(f'A9: {A9}')
print(f'A10: {A10}')

if show_plots:
    ax[1].plot(t, x[0, :], 'ro', label='Approx')
    ax[1].plot(t, true_solution(t), label='True soln')
    ax[1].set_title(rf'$\Delta t$ = {dt}')
    ax[1].legend()


''' Problem 2, parts (c-d)
IVP:

x'' + x = 0
x(0) = 1
x'(0) = 0
t0 = 0
tN = 1

This 2nd-order IVP can be written as a system of 1st-order ODEs.
Let x = (y z)
y' = z
z' = -y
y(0) = 1
z(0) = 0

Given: The true solution is x(t) = cos(t).
'''


def midpoint_method_for_2_c(t0, tN, x0, dt):
    """x0: vector [y z]"""
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros([2, len(t)])
    x[:, 0] = x0

    y = x[0, :]
    z = x[1, :]

    # Use Forward Euler to calculate x[1]
    y[1] = y[0] + dt * z[0]
    z[1] = z[0] - dt * y[0]

    # Use Midpoint Method to calculate the rest
    for k in range(1, len(t) - 1):
        y[k + 1] = y[k - 1] + 2 * dt * z[k]
        z[k + 1] = z[k - 1] - 2 * dt * y[k]

    return t, x


fig, ax = plt.subplots(1, 2)
fig.suptitle('Problem 2, parts (c-d)')

x0 = np.array([1, 0])
t0 = 0
tN = 1

dt = 0.1
t, x = midpoint_method_for_2_c(t0, tN, x0, dt)
A11 = x[0, -1]
A12 = ge(t, np.cos, x[0, :])
print(f'A11 = {A11}')
print(f'A12 = {A12}')

if show_plots:
    ax[0].plot(t, x[0, :], 'ro', label='Approx')
    ax[0].plot(t, np.cos(t), label='True soln')
    ax[0].set_title(rf'$\Delta t$ = {dt}')
    ax[0].legend()

dt = 0.01
t, x = midpoint_method_for_2_c(t0, tN, x0, dt)
A13 = x[0, -1]
A14 = ge(t, np.cos, x[0, :])
print(f'A13 = {A13}')
print(f'A14 = {A14}')

if show_plots:
    ax[1].plot(t, x[0, :], 'ro', label='Approx')
    ax[1].plot(t, np.cos(t), label='True soln')
    ax[1].set_title(rf'$\Delta t$ = {dt}')
    ax[1].legend()

if show_plots:
    plt.show()

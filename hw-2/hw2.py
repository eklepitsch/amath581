import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

show_plots = False


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


''' Problem 3
Chebyshev equation:

(1 - x^2) * y'' - x * y' + a^2 * y = 0

Solve using 2nd-order central difference scheme (ie. the "Direct Method").

y'' - p(x) * y' + q(x) * y = r(x)

a) a = 1, y(-0.5) = -0.5, y(0.5) = 0.5, dx = 0.1; true solution: y = x.
b) a = 2, y(-0.5) = 0.5, y(0.5) = 0.5, dx = 0.1; true solution: y = 1 - 2x^2
c) a = 3, y(-.5) = -1/3, y(0.5) = 1/3, dx = 0.1; true solution: y = x - (4/3)x^3
'''
print('Problem 3\n------------')


def p(x):
    return -1 * x / (1 - x**2)


def q(x, a):
    return a**2 / (1 - x**2)


def r(x):
    return 0


def direct_method(xi, xf, yi, yf, a, N):
    """2nd-order central difference scheme."""
    x = np.linspace(xi, xf, N)
    dx = x[1] - x[0]

    # We will solve the matrix eqn Ay = B,
    # where A is the discretization matrix for the LHS of the ODE,
    # and B is the discretization matrix for the RHS of the ODE.
    A = np.zeros((N, N))
    B = np.zeros((N, 1))

    # Populate the first and last values which are easy.
    A[0, 0] = 1
    A[-1, -1] = 1
    B[0] = yi
    B[-1] = yf

    # Populate the rest of the matrices
    for k in range(1, N - 1):
        A[k, k - 1] = 1 - (dx / 2) * p(x[k])
        A[k, k] = -2 + (dx ** 2) * q(x[k], a)
        A[k, k + 1] = 1 + (dx / 2) * p(x[k])
        B[k] = (dx ** 2) * r(k)

    # Solve the system
    return x, np.linalg.solve(A, B).reshape(N)


def max_error(approx_soln, true_soln):
    if len(approx_soln) != len(true_soln):
        return -1

    return max([abs(i[1] - i[0]) for i in list(zip(approx_soln, true_soln))])


def true_solution_a(x):
    return x


def true_solution_b(x):
    return 1 - 2 * x**2


def true_solution_c(x):
    return x - (4 / 3) * x**3


fig, ax = plt.subplots(1, 3)
fig.suptitle('Problem 3')

# Part (a)
x, y = direct_method(xi=-0.5, xf=0.5, yi=-0.5, yf=0.5, a=1, N=11)

if show_plots:
    ax[0].plot(x, true_solution_a(x), label='true soln')
    ax[0].plot(x, y, 'ro', label='approx')
    ax[0].set_title('Part (a)')
    ax[0].legend()

A15 = y[5]  # x = 0 is index 5
A16 = max_error(y, true_solution_a(x))
print(f'A15: {A15}')
print(f'A16: {A16}')

# Part (b)
x, y = direct_method(xi=-0.5, xf=0.5, yi=0.5, yf=0.5, a=2, N=11)

if show_plots:
    ax[1].plot(x, true_solution_b(x), label='true soln')
    ax[1].plot(x, y, 'ro', label='approx')
    ax[1].set_title('Part (b)')
    ax[1].legend()

A17 = y[5]  # x = 0 is index 5
A18 = max_error(y, true_solution_b(x))
print(f'A17: {A17}')
print(f'A18: {A18}')

# Part (c)
x, y = direct_method(xi=-0.5, xf=0.5, yi=-1/3, yf=1/3, a=3, N=11)

if show_plots:
    ax[2].plot(x, true_solution_c(x), label='true soln')
    ax[2].plot(x, y, 'ro', label='approx')
    ax[2].set_title('Part (c)')
    ax[2].legend()

A19 = y[5]  # x = 0 is index 5
A20 = max_error(y, true_solution_c(x))
print(f'A19: {A19}')
print(f'A20: {A20}')

if show_plots:
    plt.show()

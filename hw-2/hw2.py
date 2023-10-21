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

if show_plots:
    plt.show()

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import namedtuple

# Set to True to save the figures to .png files
save_figures = False
if save_figures:
    # Bump up the resolution (adds processing time)
    mpl.rcParams['figure.dpi'] = 900


def float_formatter(x):
    """
    Specify the precision and notation (positional vs. scientific)
    for floating point values.
    """
    p = 6
    if abs(x) < 1e-4 or abs(x) > 1e4:
        return np.format_float_scientific(x, precision=p)
    else:
        return np.format_float_positional(x, precision=p)


def ge(t, true_solution, approx_solution):
    """Global error"""
    return abs(true_solution(t[-1]) - approx_solution[-1])


def true_solution_a(t):
    return (1 / 2) * (np.exp(t) + np.exp(-1 * t))


def true_solution_c(t):
    return np.cos(t)


def trapezoidal_method(t0, tN, x0, dt):
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


def midpoint_method(t0, tN, x0, dt):
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

# Trapezoidal method for dt = 0.1
fig1, ax1 = plt.subplots(1, 2)
fig1.suptitle('Trapezoidal method')

t0 = 0
tN = 1
x0 = np.array([[1, 0]])
dt = 0.1
t, x = trapezoidal_method(t0, tN, x0, dt)

ax1[0].plot(t, x[0, :], 'ro', label='Approx')
ax1[0].plot(t, true_solution_a(t), label='True soln')
ax1[0].set_title(rf'$\Delta t$ = {dt}')
ax1[0].set_xlabel('Time')
ax1[0].set_ylabel(rf'x(t)')
ax1[0].legend()

# Trapezoidal method for dt = 0.01
dt = 0.01
t, x = trapezoidal_method(t0, tN, x0, dt)

ax1[1].plot(t, x[0, :], 'ro', label='Approx')
ax1[1].plot(t, true_solution_a(t), label='True soln')
ax1[1].set_title(rf'$\Delta t$ = {dt}')
ax1[1].set_xlabel('Time')
ax1[1].legend()

if save_figures:
    fig1.savefig('images/Trapezoidal-method-true-vs-approx')

# Midpoint method for dt = 0.1
fig2, ax2 = plt.subplots(1, 2)
fig2.suptitle('Midpoint method')

x0 = np.array([1, 0])
t0 = 0
tN = 1

dt = 0.1
t, x = midpoint_method(t0, tN, x0, dt)

ax2[0].plot(t, x[0, :], 'ro', label='Approx')
ax2[0].plot(t, np.cos(t), label='True soln')
ax2[0].set_title(rf'$\Delta t$ = {dt}')
ax2[0].set_xlabel('Time')
ax2[0].set_ylabel(rf'x(t)')
ax2[0].legend()

# Midpoint method for dt = 0.01
dt = 0.01
t, x = midpoint_method(t0, tN, x0, dt)

ax2[1].plot(t, x[0, :], 'ro', label='Approx')
ax2[1].plot(t, np.cos(t), label='True soln')
ax2[1].set_title(rf'$\Delta t$ = {dt}')
ax2[1].set_xlabel('Time')
ax2[1].legend()

if save_figures:
    fig2.savefig('images/Midpoint-method-true-vs-approx')

# Do the trapezoidal and midpoint method using several other values of dt.
# Then plot the log-log graph of dt vs. global error to determine the order
# of each method.
DeltaT = namedtuple('DeltaT', 'decimal str')
delta_t = [DeltaT(pow(2, -5), r'$2^{-5}$'),
           DeltaT(pow(2, -6), r'$2^{-6}$'),
           DeltaT(pow(2, -7), r'$2^{-7}$'),
           DeltaT(pow(2, -8), r'$2^{-8}$'),
           DeltaT(pow(2, -9), r'$2^{-9}$'),
           DeltaT(pow(2, -10), r'$2^{-10}$'),
           DeltaT(pow(2, -11), r'$2^{-11}$'),
           DeltaT(pow(2, -12), r'$2^{-12}$')]

Method = namedtuple('Method', 'name fn soln')

# Define the methods to use
methods = [Method('Trapezoidal', trapezoidal_method, true_solution_a),
           Method('Midpoint', midpoint_method, true_solution_c)]

t0 = 0
tN = 1
x0 = np.array([[1, 0]])

for method in methods:
    # Fig3 is the plot of the true solution vs. the approximation
    fig3, ax3 = plt.subplots(gridspec_kw={'left': 0.15})
    ax3.set_title(f'Approximate solutions for {method.name} method')
    ax3.set_xlabel('time')
    ax3.set_ylabel('x(t)')

    # Fig4 is the plot of the global error vs. delta t
    fig4, ax4 = plt.subplots(nrows=1, ncols=2, width_ratios=[5, 3],
                             gridspec_kw={
                                 'left': 0.08,     # Left padding
                                 'right': 0.96,    # Right padding
                                 'wspace': 0.05})  # Space between axes
    fig4.suptitle(f'Global error for {method.name} method')
    fig4.set_figwidth(fig4.get_figwidth() * 1.5)    # Increase the width

    # Column 0 = delta t, Column 1 = global error
    global_err = np.zeros((len(delta_t), 2))

    # For each delta t, approximate the solution using the given method
    for i, dt in enumerate(delta_t):
        t, approx = method.fn(t0, tN, x0, dt.decimal)

        # Record delta t and global error (for plotting later)
        global_err[i][0] = dt.decimal
        global_err[i][1] = ge(t, method.soln, approx[0])

        ax3.plot(t, approx[0], label=fr'$\Delta$t = {dt.str}')

        # Plot true solution on last iteration
        if i == len(delta_t) - 1:
            ax3.plot(t, method.soln(t),
                     label='true solution', linestyle=':', linewidth=4)

        ax3.ticklabel_format(useOffset=False)

    # Take the natural logarithm of delta t and the global error
    dt = global_err[:, 0]
    en = global_err[:, 1]
    log_dt = np.log(dt)
    log_en = np.log(en)

    # Find the best fit line through ln(En)
    [m, b] = np.polyfit(log_dt, log_en, 1)
    best_fit_fn = np.poly1d([m, b])
    best_fit_data = best_fit_fn(log_dt)

    # Plot the data and best fit line
    ax4[0].set_xlabel(r'ln($\Delta$t)')
    ax4[0].set_ylabel(r'ln($E_N$)')
    ax4[0].plot(log_dt, log_en, label='Actual error',
                marker='o', markersize=6, linestyle='')
    ax4[0].plot(log_dt, best_fit_data, label='Best fit')
    ax4[0].text(np.mean(log_dt),
                np.mean(best_fit_data) - np.ptp(best_fit_data) / 3,
                f'Slope of best fit = {round(m, 5)}')

    # Plot a table containing the En and ln(En) values
    table_values = []
    for dt, err, log_err in zip(delta_t, global_err[:, 1], log_en):
        table_values.append([dt.str,
                             float_formatter(err),
                             float_formatter(log_err)])
    table = plt.table(cellText=table_values,
                      colLabels=[r'$\Delta$t', r'$E_N$', r'ln($E_N$)'],
                      bbox=[0, 0 ,1 , 1])
    ax4[1].add_table(table)
    ax4[1].axis('off')

    # Add legends to the plots
    ax3.legend()
    ax4[0].legend()

    if save_figures:
        fig3.savefig(f'images/{method.name.replace(" ", "-")}-true-vs-approx')
        fig4.savefig(f'images/{method.name.replace(" ", "-")}-global-error')

# Plot the stability region of the trapezoidal method in the complex plane
fig5, ax5 = plt.subplots()
fig5.suptitle('Stability region for trapezoidal method')
ax5.axhline(y=0, color='k')
ax5.axvline(x=0, color='k')
ax5.set_xlim([-10, 10])
ax5.set_ylim([-10, 10])
ax5.set_xticks([-10, 0, 10])
ax5.set_yticks([-10, 0, 10])
ax5.set_xticklabels([rf'$-\infty$', 0, rf'$\infty$'])
ax5.set_yticklabels([rf'$-\infty$', 0, rf'$\infty$'])
ax5.set_xlabel(rf'Re($\lambda$)')
ax5.set_ylabel(rf'Im($\lambda$)')
ax5.fill_between(range(-10, 1), -10, 10,
                 color='g', alpha=0.5, label='stable')
ax5.fill_between(range(0, 11), -10, 10,
                 color='r', alpha=0.5, label='unstable')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
ax5.spines['left'].set_visible(False)
ax5.legend()
if save_figures:
    fig5.savefig(f'images/Trapezoidal-stability-region')

if not save_figures:
    plt.show()

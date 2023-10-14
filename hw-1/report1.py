import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import namedtuple
from math import floor, ceil

# Set to True to save the figures to .png files
save_figures = False
if save_figures:
    # Bump up the resolution (adds processing time)
    mpl.rcParams['figure.dpi'] = 900


def plot(ax, data1, data2, param_dict):
    """
    A helper function to make a graph.
    """
    out = ax.plot(data1, data2, **param_dict)
    return out


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


np.set_printoptions(formatter={'float': float_formatter})

''' Problem 1
ODE: x'(t) = -4 * x * sin(t)

ICs:
x(0) = 1
t0 = 0, tN = 8, dt = ?

Approximate the solution with the following methods
(a) Forward Euler, dt = [2e-5, 2e-12]
(b) Huen, dt = [2e-5, 2e-12]
(c) RK2, dt = [2e-5, 2e-12]
'''
x0 = 1
t0 = 0
tN = 8


def f(x, t):
    """The RHS of the ODE"""
    return -4 * x * np.sin(t)


def true_solution(t):
    """The true solution to the ODE"""
    return np.exp(4 * (np.cos(t) - 1))


# Part (a)
def forward_euler(t0, tN, x0, dt, f):
    """Forward Euler method"""
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = x[k] + dt * f(x[k], t[k])

    return t, x


def heun(t0, tN, x0, dt, f):
    """Heun's method"""
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] =\
            x[k] + (dt / 2) * (f(x[k], t[k]) +
                               f(x[k] + dt * f(x[k], t[k]), t[k] + dt))

    return t, x


def rk2(t0, tN, x0, dt, f):
    """RK2 method"""
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = (x[k] + dt *
                    f(x[k] + (dt / 2) * f(x[k], t[k]), t[k] + (dt / 2)))

    return t, x


def lte(t, true_solution, approx_solution):
    """Local truncation error"""
    return abs(true_solution(t[1]) - approx_solution[1])


def ge(t, true_solution, approx_solution):
    """Global error"""
    return abs(true_solution(t[-1]) - approx_solution[-1])


# Define some namedtuples for convenience
DeltaT = namedtuple('DeltaT', 'decimal str')
delta_t = [DeltaT(pow(2, -5), r'$2^{-5}$'),
           DeltaT(pow(2, -6), r'$2^{-6}$'),
           DeltaT(pow(2, -7), r'$2^{-7}$'),
           DeltaT(pow(2, -8), r'$2^{-8}$'),
           DeltaT(pow(2, -9), r'$2^{-9}$'),
           DeltaT(pow(2, -10), r'$2^{-10}$'),
           DeltaT(pow(2, -11), r'$2^{-11}$'),
           DeltaT(pow(2, -12), r'$2^{-12}$')]

Method = namedtuple('Method', 'name fn')

# Define the methods to use
methods = [Method('Forward Euler', forward_euler),
           Method('Heun', heun),
           Method('RK2', rk2)]

for method in methods:
    # Adjust left margin for Heun and RK2 plots since it looks better
    gridspec = None
    if method.name == 'Heun' or method.name == 'RK2':
        gridspec ={'left': 0.15}

    # Fig1 is the plot of the true solution vs. the approximation
    fig1, ax1 = plt.subplots(gridspec_kw={'left': 0.15})
    ax1.set_title(f'Approximate solutions for {method.name} method')
    ax1.set_xlabel('time')
    ax1.set_ylabel('x(t)')

    # Fig2 is the plot of the global error vs. delta t
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, width_ratios=[5, 3],
                             gridspec_kw={
                                'left': 0.08,     # Left padding
                                'right': 0.96,    # Right padding
                                'wspace': 0.05})  # Space between axes
    fig2.suptitle(f'Global error for {method.name} method')
    fig2.set_figwidth(fig2.get_figwidth() * 1.5)    # Increase the width

    # Column 0 = delta t, Column 1 = global error
    global_err = np.zeros((len(delta_t), 2))

    # For each delta t, approximate the solution using the given method
    for i, dt in enumerate(delta_t):
        t, approx = method.fn(t0, tN, x0, dt.decimal, f)

        # Record delta t and global error (for plotting later)
        global_err[i][0] = dt.decimal
        global_err[i][1] = ge(t, true_solution, approx)

        # For Huen and RK2, show a zoomed in plot because the true solution
        # and the approximate solutions are too close together in the full plot
        # and cannot be distinguished.
        max_point = len(approx)
        if method.name == 'Heun' or method.name == 'RK2':
            max_time = 0.005
            max_point = ceil(max_time / dt.decimal)

            # Plot the approximate solution for select delta t
            if dt.decimal in [pow(2, -8), pow(2, -9), pow(2, -10), pow(2, -12)]:
                plot(ax1, t[:max_point], approx[:max_point],
                     {'label': fr'$\Delta$t = {dt.str}'})
        else:
            plot(ax1, t[:max_point], approx[:max_point],
                 {'label': fr'$\Delta$t = {dt.str}'})

        # Plot true solution on last iteration
        if i == len(delta_t) - 1:
            ax1.plot(t[:max_point], true_solution(t[:max_point]),
                     label='true solution', linestyle=':', linewidth=4)

        ax1.ticklabel_format(useOffset=False)

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
    ax2[0].set_xlabel(r'ln($\Delta$t)')
    ax2[0].set_ylabel(r'ln($E_N$)')
    plot(ax2[0], log_dt, log_en,
         {'label': 'Actual error', 'marker': 'o',
          'markersize': 6, 'linestyle': ''})
    plot(ax2[0], log_dt, best_fit_data, {'label': 'Best fit'})
    ax2[0].text(np.mean(log_dt),
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
    ax2[1].add_table(table)
    ax2[1].axis('off')

    # Add legends to the plots
    ax1.legend()
    ax2[0].legend()

    if save_figures:
        fig1.savefig(f'{method.name.replace(' ', '-')}-true-vs-approx')
        fig2.savefig(f'{method.name.replace(' ', '-')}-global-error')

# Show the plots if we're not saving them to files
if not save_figures:
    plt.show()

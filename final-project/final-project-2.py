import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.integrate
from collections import namedtuple
from math import floor, ceil

# Set to True to save the figures to .png files
save_figures = False
if save_figures:
    # Bump up the resolution (adds processing time)
    mpl.rcParams['figure.dpi'] = 900

# Where to write the generated image files
image_path = r'C:\amath581-images'


'''Methods

The following section of code contains functions for the various numerical
methods examined in this report: rk2, rk4, and rk5.  There are two methods
for rk5.  The first takes f_y as a parameter and can be used when f_y is
known explicitly.  The second takes f' as a parameter and can be used
if it is cheaper to evaluate f' than it is to evaluate f_y.
'''
def rk2(t0, tN, x0, dt, f):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = (x[k] + dt *
                    f(x[k] + (dt / 2) * f(x[k], t[k]), t[k] + (dt / 2)))

    return t, x


def rk4(t0, tN, x0, dt, f):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    def f1(x, t):
        return f(x, t)

    def f2(x, t):
        return f(x + (dt / 2) * f1(x, t), t + dt / 2)

    def f3(x, t):
        return f(x + (dt / 2) * f2(x, t), t + dt / 2)

    def f4(x, t):
        return f(x + dt * f3(x, t), t + dt)

    for k in range(len(t) - 1):
        x[k + 1] = x[k] + (dt / 6) * \
                   (f1(x[k], t[k]) +
                    2 * f2(x[k], t[k]) +
                    2 * f3(x[k], t[k]) +
                    f4(x[k], t[k]))

    return t, x


def rk5_method1(t0, tN, x0, dt, f, yn, first_step_fe=False):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    def k1(x, t):
        return dt * f(x, t)

    def k2(x, t):
        return dt * f(x + (1/3) * k1(x, t)
                      + (1/18) * dt * yn(x, t) * k1(x, t), t)

    def k3(x, t):
        return dt * f(x - (152/125) * k1(x, t) + (252/125) * k2(x, t)
                      - (44/125) * dt * yn(x, t) * k1(x, t), t)

    def k4(x, t):
        return dt * f(x + (19/2) * k1(x, t) - (72/7) * k2(x, t)
                      + (25/14) * k3(x, t)
                      + (5/2) * dt * yn(x, t) * k1(x, t), t)

    start = 0
    if first_step_fe:
        # Use forward Euler for first time step
        x[1] = x[0] + dt * f(x[0], t[0])
        start = 1

    # Use RK5 for remaining time steps
    for k in range(start, len(t) - 1):
        x[k + 1] = (x[k] + (5/48) * k1(x[k], t[k])
                    + (27/56) * k2(x[k], t[k])
                    + (125/336) * k3(x[k], t[k])
                    + (1/24) * k4(x[k], t[k]))

    return t, x


def rk5_method2(t0, tN, x0, dt, f, f_prime, first_step_fe=False):
    t = np.arange(t0, tN + dt / 2, dt)
    x = np.zeros_like(t)
    x[0] = x0

    def k1(x, t):
        return dt * f(x, t)

    def k2(x, t):
        return dt * f(x + (1/3) * k1(x, t)
                      + (1/18) * dt ** 2 * f_prime(x, t) * k1(x, t), t)

    def k3(x, t):
        return dt * f(x - (152/125) * k1(x, t) + (252/125) * k2(x, t)
                      - (44/125) * dt ** 2 * f_prime(x, t) * k1(x, t), t)

    def k4(x, t):
        return dt * f(x + (19/2) * k1(x, t) - (72/7) * k2(x, t)
                      + (25/14) * k3(x, t)
                      + (5/2) * dt ** 2 * f_prime(x, t) * k1(x, t), t)

    start = 0
    if first_step_fe:
        # Use forward Euler for first time step
        x[1] = x[0] + dt * f(x[0], t[0])
        start = 1

    # Use RK5 for remaining time steps
    for k in range(start, len(t) - 1):
        x[k + 1] = (x[k] + (5/48) * k1(x[k], t[k])
                    + (27/56) * k2(x[k], t[k])
                    + (125/336) * k3(x[k], t[k])
                    + (1/24) * k4(x[k], t[k]))

    return t, x


def scipy_solve_ivp(t0, tN, x0, dt, f):
    def f_flip_args(t, y):
        return f(y, t)

    t = np.arange(t0, tN + dt / 2, dt)

    # Handle edge case when the last t value is outside of the t_span
    if t[-1] > tN:
        t = t[:-1]

    sol = scipy.integrate.solve_ivp(f_flip_args, [t0, tN], [x0], method='RK45',
                                    t_eval=t, max_step=dt)

    return t, sol.y.flatten()


'''Helper functions

The following section of code contains functions that I use frequently for
doing simple tasks like plotting, etc.
'''
def ge(t, true_solution, approx_solution):
    """Global error"""
    return abs(true_solution(t[-1]) - approx_solution[-1])


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


'''IVP 1

y' = -y
y(0) = 1

True solution: y(t) = e^-t
'''
def f1(y, t):
    return -1 * y


def f1_prime(y, t):
    return -1


def yn1(y, t):
    return 1 / y


def true_solution1(t):
    return np.exp(-1 * t)


'''IVP 2

y' = (1/4)y - (1/80)y^2
y(0) = 1

True solution: y(t) = 20 / (1 + 19e^(-t/4))
'''
def f2(y, t):
    return (1 / 4) * y - (1 / 80) * y ** 2


def f2_prime(y, t):
    return (1 / 4) - (1 / 40) * y


def yn2(y, t):
    return f2_prime(y, t) / f2(y, t)


def true_solution2(t):
    return 20 / (1 + 19 * np.exp((-1/4) * t))


'''IVP 3 (adapted from HW1, Problem 1)

x'(t) = -4 * x * sin(t)
x(0) = 1
t0 = 0, tN = pi - dt
'''
def f3(x, t):
    return -4 * x * np.sin(t)


def f3_prime(x, t):
    return -4 * x * np.cos(t)


def xn3(x, t):
    return 1/np.tan(t)


def true_solution3(t):
    return np.exp(4 * (np.cos(t) - 1))


'''Analysis

The following section of code uses the various numerical methods and does
some analysis of the results.
'''
# Define some namedtuples for convenience
DeltaT = namedtuple('DeltaT', 'decimal str')

# Comment this in for large delta t values
# delta_t = [DeltaT(0.032, r'0.032'),
#            DeltaT(0.064, r'0.064'),
#            DeltaT(0.128, r'0.128'),
#            DeltaT(0.256, r'0.256'),
#            DeltaT(0.512, r'0.512'),
#            DeltaT(1.024, r'1.024')]

# Comment this in for small delta t values
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
methods = [Method('RK2', rk2),
           Method('RK4', rk4),
           Method('RK5', rk5_method1),
           Method('scipy.solve_ivp', scipy_solve_ivp)]
           #Method('RK5 method 2', rk5_method2)]  # Not used in the final report

Problem = namedtuple('Problem',
                     'name f f_prime yn true_soln t0 tN y0 first_step_fe')

# Define the problems to solve
problems = [Problem('test problem 1', f1, f1_prime, yn1, true_solution1, 0, 20, 1, False),
            Problem('test problem 2', f2, f2_prime, yn2, true_solution2, 0, 20, 1, False),
            Problem('Test problem 3', f3, f3_prime, xn3, true_solution3, 0, np.pi, 1, True)]

for problem in problems:
    # Fig3 is the log-log plot of the errors for each method overlayed in
    # a single graph.
    fig3, ax3 = plt.subplots(nrows=1, ncols=1)
    fig3.suptitle(f'Global error for {problem.name} - comparison of methods'
                  f'\n$\Delta t$ = [2^-5, 2^-6, ..., 2^-12]')
    ax3.set_xlabel(r'$ln(\Delta t)$')
    ax3.set_ylabel(r'$ln(E_N)$')

    for method in methods:
        # Fig1 is the plot of the true solution vs. the approximation
        fig1, ax1 = plt.subplots(gridspec_kw={'left': 0.15})
        ax1.set_title(f'Approximate solutions for {problem.name}'
                      f' using {method.name}')
        ax1.set_xlabel('time')
        ax1.set_ylabel('x(t)')

        # Fig2 is the plot of the global error vs. delta t
        fig2, ax2 = plt.subplots(nrows=1, ncols=2, width_ratios=[5, 3],
                                 gridspec_kw={
                                     'left': 0.08,  # Left padding
                                     'right': 0.96,  # Right padding
                                     'wspace': 0.05})  # Space between axes
        fig2.suptitle(f'Global error for {problem.name} using {method.name}')
        fig2.set_figwidth(fig2.get_figwidth() * 1.5)  # Increase the width

        # Column 0 = delta t, Column 1 = global error
        global_err = np.zeros((len(delta_t), 2))

        # For each delta t, approximate the solution using the given method
        t_cache = None
        s_cache = None
        for i, dt in enumerate(delta_t):
            if (method.name == 'RK2' or method.name == 'RK4'
                    or method.name == 'scipy.solve_ivp'):
                t, approx = method.fn(problem.t0, problem.tN, problem.y0,
                                      dt.decimal, problem.f)
            elif method.name == 'RK5':
                t, approx = method.fn(problem.t0, problem.tN, problem.y0,
                                      dt.decimal, problem.f, problem.yn,
                                      problem.first_step_fe)
            elif method.name == 'RK5 method 2':
                t, approx = method.fn(problem.t0, problem.tN, problem.y0,
                                      dt.decimal, problem.f, problem.f_prime,
                                      problem.first_step_fe)

            # Record delta t and global error (for plotting later)
            global_err[i][0] = dt.decimal
            global_err[i][1] = ge(t, problem.true_soln, approx)

            max_time = 1
            max_point = ceil(max_time / dt.decimal)
            ax1.plot(t[:max_point], approx[:max_point],
                     label=fr'$\Delta$t = {dt.str}')
            if i == 0:
                t_cache = t[:max_point]
                s_cache = problem.true_soln(t[:max_point])

            # Plot true solution on last iteration
            if i == len(delta_t) - 1:
                ax1.plot(t_cache, s_cache,
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

        # Plot global error
        ax2[0].set_xlabel(r'$ln(\Delta t)$')
        ax2[0].set_ylabel(r'$ln(E_N)$')

        # Use this code for generating plots with the same x and y axis
        # ticks as the journal article.
        #ax2[0].set_xlabel(r'Step Size')
        #ax2[0].set_ylabel(r'Relative Error at t=20')
        #ax2[0].set_xscale('log')
        #ax2[0].set_xticks(dt)
        #ax2[0].set_yscale('log')
        #ax2[0].set_yticks([1e-10, 1e-08, 1e-06, 1e-04, 1e-02, 1e00])
        #ax2[0].get_xaxis().set_major_formatter(
        #    mpl.ticker.FuncFormatter(lambda x, p: format(float(x), ','))
        #)

        ax2[0].plot(log_dt, log_en, label='Actual error',
                    marker='o', markersize=6, linestyle='')
        ax2[0].plot(log_dt, best_fit_data, label='Best fit')
        ax3.plot(log_dt, best_fit_data,
                 label=f'{method.name}, slope = {round(m, 5)}')
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
                          bbox=[0, 0, 1, 1])
        ax2[1].add_table(table)
        ax2[1].axis('off')

        # Add legends to the plots
        ax1.legend()
        ax2[0].legend()
        ax3.legend()

        if save_figures:
            img_path = os.path.join(image_path, f'Approximate-solutions-'
                                                f'for-{problem.name}-using'
                                                f'-{method.name}')
            fig1.savefig(img_path)
            img_path = os.path.join(image_path, f'Global-error-for-'
                                                f'{problem.name}-using-'
                                                f'{method.name}')
            fig2.savefig(img_path)
            img_path = os.path.join(image_path, f'Comparison-of-methods-'
                                                f'{problem.name}-using-'
                                                f'small-dt')
            fig3.savefig(img_path)

plt.show()

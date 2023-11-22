clear all; close all; clc;

% u_t = -c * u_x
% u(t, 0) = u(t, 1)
% u(t, x) = u0(x)

c = 1;

u0 = @(x)(exp(-50 * (x - 0.5).^2));
true_sol = @(t, x)(u0(mod(x - c * t, 1)));

x0 = 0;
xf = 1;
Nx = 101;
x = linspace(x0, xf, Nx);
dx = x(2) - x(1);

t0 = 0;
tf = 1;
Nt = 201;
t = linspace(t0, tf, Nt);
dt = t(2) - t(1);

plot_rows = 3;
plot_cols = 3;
num_plots = plot_rows * plot_cols;
t_plot = linspace(t0, tf, num_plots);

figure()
sgtitle("True Solution")
for j = 1:length(t_plot)
    tval = t_plot(j);
    subplot(plot_rows, plot_cols, j);
    plot(x, true_sol(tval, x))
    title(sprintf("t = %0.2f", tval))
    ylim([-1, 1])
end

A = diag(ones(Nx - 2, 1), 1) - diag(ones(Nx - 2, 1), -1);
A(1, end) = -1;
A(end, 1) = 1;
A = (-c / (2 * dx)) * A;

U = zeros(Nx, Nt);
U(1:(end - 1), 1) = u0(x(1:(end-1)));

% Forward Euler
for k = 1:Nt - 1
    U(1:(end - 1), k + 1) = U(1:(end - 1), k) + dt * A * U(1:(end - 1), k);
end
U(end, :) = U(1, :);

figure()
sgtitle("Forward Euler")
for j = 1:length(t_plot)
    tval = t_plot(j);
    k = round(tval / dt) + 1;
    subplot(plot_rows, plot_cols, j);
    plot(x, U(:, k))
    title(sprintf("t = %0.2f", tval))
end

% Backward Euler
A_backward = eye(Nx - 1) - dt * A;
for k = 1:Nt - 1
    U(1:(end - 1), k + 1) = A_backward \ U(1:(end - 1), k);
end
U(end, :) = U(1, :);

figure()
sgtitle("Backward Euler")
for j = 1:length(t_plot)
    tval = t_plot(j);
    k = round(tval / dt) + 1;
    subplot(plot_rows, plot_cols, j);
    plot(x, U(:, k))
    title(sprintf("t = %0.2f", tval))
    ylim([-1, 1])
end

% Midpoint
U(1:(end - 1), 2) = U(1:(end - 1), 1) + dt * A * U(1:(end - 1), 1);
for k = 1:Nt - 2
    U(1:(end - 1), k + 2) = U(1:(end - 1), k) + 2 * dt * A * U(1:(end - 1), k + 1);
end
U(end, :) = U(1, :);

ratio = abs(c * dt / dx)

figure()
sgtitle("Midpoint")
for j = 1:length(t_plot)
    tval = t_plot(j);
    k = round(tval / dt) + 1;
    subplot(plot_rows, plot_cols, j);
    plot(x, U(:, k))
    title(sprintf("t = %0.2f", tval))
    ylim([-1, 1])
end

vals = eig(A);
sort(dt * vals);
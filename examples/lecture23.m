clear all; close all; clc;

% u_t = -c * u_x
% u(t, 0) = u(t, 1)
% u(0, x) = u0(x)
% u(t, x) = true_sol

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
Nt = 101;
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
    ylim([0, 1])
end

% Lax-Friedrichs
A = diag(ones(Nx - 2, 1), 1) - diag(ones(Nx - 2, 1), -1);
A(1, end) = -1;
A(end, 1) = 1;
A = (-c / (2 * dx)) * A;

B = diag(ones(Nx - 2, 1), 1) + diag(ones(Nx - 2, 1), -1);
B(1, end) = 1;
B(end, 1) = 1;
B = 0.5 * B;

U = zeros(Nx, Nt);
U(1:(end - 1), 1) = u0(x(1:(end-1)));

for k = 1:Nt - 1
    U(1:(end - 1), k + 1) = B * U(1:(end - 1), k) + dt * A * U(1:(end - 1), k);
end
U(end, :) = U(1, :);

figure()
sgtitle("Lax-Friedrichs")
for j = 1:length(t_plot)
    tval = t_plot(j);
    k = round(tval / dt) + 1;
    subplot(plot_rows, plot_cols, j);
    plot(x, U(:, k))
    ylim([0, 1])
    title(sprintf("t = %0.2f", tval))
end
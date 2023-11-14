clear all; close all; clc;

% u_t = u_xx
% u(t, 0) = 0
% u(t, 1) = 0
% u(0, x) = sin(pi * x)

Nx = 11
x0 = 0;
xf = 1;
x = linspace(x0, xf, Nx);
dx = x(2) - x(1)

t0 = 0;
tf = 0.5;
% Choose Nt so that dt / dx^2 = 1/4
Nt = 2 * (Nx - 1)^2 + 1
t = linspace(t0, tf, Nt);
dt = t(2) - t(1)

U = zeros(Nx, Nt);
U(:, 1) = sin(pi * x);
U(1, :) = 0;
U(end, :) = 0;

A = (diag(-2 * ones(Nx - 2, 1)) + diag(ones(Nx - 3, 1), 1) + diag(ones(Nx - 3, 1), -1)) / dx^2;

% Check the eigenvalues of A
[~, vals] = eig(A);
k = 1:(Nx - 2);
exact_vals = (2 / dx^2) * (cos(k * pi * dx) - 1);
sort(diag(vals))
sort(exact_vals')
sort(dt * diag(vals))

f = @(t, u)(A * u);

% Solve with forward Euler
for k = 1:Nt - 1
    U(2:end-1, k + 1) = U(2:end-1, k) + dt * f(t(k), U(2:end-1, k));
end

[T, X] = meshgrid(t, x);

figure()
surf(T, X, U)
shading interp

true_solution = @(t, x)(exp(-pi^2 * t) .* sin(pi * x));

err = U - true_solution(T, X);
figure()
surf(T, X, err)
shading interp

global_err = max(abs(err))
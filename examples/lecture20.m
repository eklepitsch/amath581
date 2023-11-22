clear all; close all; clc;

% u_t = u_xx
% u(t, 0) = 0
% u(t, 1) = 0
% u(0, x) = sin(pi * x)

Nx = 101
x0 = 0;
xf = 1;
x = linspace(x0, xf, Nx);
dx = x(2) - x(1)

t0 = 0;
tf = 0.5;
Nt = 101
t = linspace(t0, tf, Nt);
dt = t(2) - t(1)

U = zeros(Nx, Nt);
U(:, 1) = sin(pi * x);
U(1, :) = 0;
U(end, :) = 0;

% Crank-Nicolson Method
% Solve with Trapezoidal Method

A = (diag(-2 * ones(Nx - 2, 1)) + diag(ones(Nx - 3, 1), 1) + diag(ones(Nx - 3, 1), -1)) / dx^2;
% (I - dt / 2 * A) * u_(k+1) = (I + dt / 2 * A) * u_k
I = eye(Nx - 2);
A_lhs = (I - (dt / 2) * A);
A_rhs = (I + (dt / 2) * A);

for k = 1:Nt - 1
    U(2:end-1, k + 1) = A_lhs \ (A_rhs * U(2:end-1, k));
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

global_err = max(abs(err(:)))
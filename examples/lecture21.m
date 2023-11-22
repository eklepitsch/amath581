clear all; close all; clc;

% u_t = -2 * u_x
% u(t, 0) = u(t, 1)
% u(0, x) = sin(pi * x)
% u(t, x) = sin(pi * (x - 2 * t)) <- That's not right!

% Just to make this code operational, I switched the initial
% condition to 
% u(0, x) = sin(2 * pi * x)
% and so the true solution is 
% u(t, x) = sin(2 * pi * (x - 2 * t))

c = 2;

Nx = 101;
x0 = 0;
xN = 1;
x = linspace(x0, xN, Nx);
dx = x(2) - x(1);

t0 = 0;
tf = 4;
Nt = 10001;
t = linspace(t0, tf, Nt);
dt = t(2) - t(1);

A = diag(ones(Nx - 2, 1), 1) - diag(ones(Nx - 2, 1), -1);
A(1, end) = -1;
A(end, 1) = 1;
A = (-c / (2 * dx)) * A;

U = zeros(Nx, Nt);
U(1:(end - 1), 1) = sin(2 * pi * x(1:(end - 1)));

% Forward Euler
% for k = 1:Nt - 1
%     U(1:(end - 1), k + 1) = U(1:(end - 1), k) + dt * A * U(1:(end - 1), k);
% end
% U(end, :) = U(1, :);

% Backward Euler
A_backward = eye(Nx - 1) - dt * A;
for k = 1:Nt - 1
    U(1:(end - 1), k + 1) = A_backward \ U(1:(end - 1), k);
end
U(end, :) = U(1, :);

[T, X] = meshgrid(t, x);
surf(T, X, U)
shading interp
title("Solution")

true_sol = @(t, x)(sin(2 * pi * (x - c * t))) % This was wrong in class
err = U - true_sol(T, X);
figure()
surf(T, X, err)
shading interp
title("Error")

global_err = max(abs(err(:)))

vals = eig(A);
sort(dt * vals)
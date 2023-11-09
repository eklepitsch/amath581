clear all; close all; clc;

% u_xx + u_yy = 0
% u(x, 0) = x^2 - x
% All other boundaries are zero
% u(x, 1) = u(0, y) = u(1, y) = 0

x0 = 0;
xN = 1;
y0 = 0;
yN = 1;

N = 201
x = linspace(x0, xN, N);
y = linspace(x0, xN, N);
dx = x(2) - x(1)

% Boundary condition
a = @(x)(x.^2 - x);

% Jacobi Iteration
U = zeros(N);
U(1, :) = a(x);
Unew = U;

max_steps = 6000;
for k = 1:max_steps
    for m = 2:N-1
        for n = 2:N-1
            Unew(m, n) = (U(m, n - 1) + U(m - 1, n) + U(m + 1, n) + U(m, n + 1)) / 4;
        end
    end
    U = Unew;
end

% Gauss-Seidel Iteration
U = zeros(N);
U(1, :) = a(x);

max_steps = 3000;
for k = 1:max_steps
    for m = 2:N-1
        for n = 2:N-1
            U(m, n) = (U(m, n - 1) + U(m - 1, n) + U(m + 1, n) + U(m, n + 1)) / 4;
        end
    end
end

% These vectors are just to plot the boundary conditions
zero_vector = zeros(size(x));
one_vector = ones(size(x));

[X, Y] = meshgrid(x, y);

% Plot the solution
surf(X, Y, U)
hold on
% u(0, y) = 0
plot3(zero_vector, y, zero_vector, 'r')
% u(x, 1) = 0
plot3(x, one_vector, zero_vector, 'r')
% u(1, y) = 0
plot3(one_vector, y, zero_vector, 'r')
% u(x, 0) = a(x)
plot3(x, zero_vector, a(x), 'r')
hold off
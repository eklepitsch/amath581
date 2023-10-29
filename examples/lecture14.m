clear all; close all; clc;

% u_xx + u_yy = 0
% u(x, 0) = x^2 - x
% All other boundaries are zero
% u(x, 1) = u(0, y) = u(1, y) = 0

x0 = 0;
xN = 1;
y0 = 0;
yN = 1;

N = 11
x = linspace(x0, xN, N);
y = linspace(x0, xN, N);
dx = x(2) - x(1)

% Boundary condition
a = @(x)(x.^2 - x);

% Solve Au = b
N_total = N * N
entries_in_matrix = N_total^2

A = zeros(N_total);
b = zeros(N_total, 1);

point2ind = @(m, n)((n - 1) * N + m);

for n = 1:N
    for m = 1:N
        k = point2ind(m, n);
        if n == 1
            A(k, k) = 1;
            b(k) = a(x(m));
        elseif m == 1 || m == N || n == N
            A(k, k) = 1;
            b(k) = 0;
        else
            A(k, k) = -4 / dx^2;
            A(k, k + 1) = 1 / dx^2;
            A(k, k - 1) = 1 / dx^2;
            A(k, k + N) = 1 / dx^2;
            A(k, k - N) = 1 / dx^2;
            b(k) = 0; % f(x(m), y(n))
        end
    end
end

u = A\b;
size(u)
U = reshape(u, [N, N]);
% MATLAB stores data internally column-by-column instead of 
% row-by-row (the way python and C do), but we chose to 
% order our grid points row-by-row.  The following line 
% switches rows and columns of U.
U = permute(U, [2, 1]);

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
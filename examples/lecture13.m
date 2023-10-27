clear all; close all; clc;

% u_xx + u_yy = 0
% u(x, 0) = x^2 - x
% All other boundaries are zero
% u(x, 1) = u(0, y) = u(1, y) = 0

x0 = 0;
xN = 1;
y0 = 0;
yN = 1;

N = 11;
x = linspace(x0, xN, N);
y = linspace(x0, xN, N);

% Boundary condition
a = @(x)(x.^2 - x);

% THese vectors are just to plot the boundary conditions
zero_vector = zeros(size(x));
one_vector = ones(size(x));

% u(0, y) = 0
plot3(zero_vector, y, zero_vector, 'r')
% u(x, 1) = 0
hold on
plot3(x, one_vector, zero_vector, 'r')
% u(1, y) = 0
plot3(one_vector, y, zero_vector, 'r')
% u(x, 0) = a(x)
plot3(x, zero_vector, a(x), 'r')
hold off
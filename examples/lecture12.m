clear all; close all; clc;

% y'' + y^3 = 0
% y(0) = 0
% y(1) = 2

x0 = 0;
y0 = 0;
xN = 1;
yN = 2;

N = 11;
x = linspace(x0, xN, N)'; % Transpose to make a row vector
% row/column doesn't matter, but it has to match y
dx = x(2) - x(1)

y = ones(size(x));
max_steps = 500;
k = 0;
while max(abs(F(y, y0, yN, dx))) >= 1e-8 && k < max_steps
    change_in_y = jacobian(y, dx)\F(y, y0, yN, dx);
    y = y - change_in_y;
    k = k + 1;
end
k
plot(x, y, 'k', x0, y0, 'ro', xN, yN, 'ro')


function z = F(y, y0, yN, dx)

z = zeros(size(y));
z(1) = y(1) - y0;
z(end) = y(end) - yN;
for k = 2:length(y) - 1
    z(k) = y(k - 1) - 2 * y(k) + y(k + 1) + dx^2 * y(k)^3;
end

end

function J = jacobian(y, dx)

J = zeros(length(y));
J(1, 1) = 1;
J(end, end) = 1;
for k = 2:length(y) - 1
    J(k, k - 1) = 1;
    J(k, k) = -2 + 3 * dx^2 * y(k)^2;
    J(k, k + 1) = 1;
end

end
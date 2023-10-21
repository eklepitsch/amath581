clear all; close all; clc;

% y'' + p(x) * y' + q(x) * y = r(x)
% y'' - x * y' + 5 * y = 0
% y(0) = 0 and y(3) = 18

p = @(x)(-x);
q = 5;
r = 0;

x0 = 0;
y0 = 0;
xN = 3;
yN = 18;

true_sol = @(x)(x.^5 - 10 * x.^3 + 15 * x);
N = 31;
x = linspace(x0, xN, N);
dx = x(2) - x(1)

A = zeros(N, N);
b = zeros(N, 1);

% First boundary
A(1, 1) = 1;
b(1) = y0;

% Second boundary
A(N, N) = 1;
b(N) = yN;

for k = 2:N-1
    A(k, k - 1) = (1 - dx * p(x(k)) / 2);
    A(k, k) = (-2 + dx^2 * q);
    A(k, k + 1) = (1 + dx * p(x(k)) / 2);
end

y = A\b;
y = y'; % x is a row vector and these should be the same shape

plot(x, true_sol(x), 'k')
hold on
plot(x, y, 'b', x0, y0, 'ro', xN, yN, 'ro')
hold off

err = max(abs(y - true_sol(x)))

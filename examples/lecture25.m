clear all; close all; clc;

% u''(x) = f(x)
% u(x0) = 0 = u(xf)

f = @(x)(1 ./ (1 + 16 * x.^2));
true_sol = @(x)((1 / 32) * (-log(16 * x.^2 + 1) + log(17) + 8 * x .* atan(4 * x) - 8 * atan(4)));

N = 5;
x0 = -1;
xf = 1;
x = linspace(x0, xf, N + 1);

A = zeros(N + 1);
b = zeros(N + 1, 1);
b(1) = 0;
b(N + 1) = 0;
for i = 2:N
    b(i) = f(x(i));
end

% Fill in first row: u(x0) = 0
for j = 1:(N + 1)
    A(1, j) = x(1)^(N - j + 1);
end

% Fill in last row: u(xf) = 0
for j = 1:(N + 1)
    A(N + 1, j) = x(N + 1)^(N - j + 1);
end

for i = 2:N
    for j = 1:(N - 1)
        A(i, j) = (N - j + 1) * (N - j) * x(i)^(N - j - 1);
    end
end

c = A\b;

u = @(x)(polyval(c, x));

xplot = linspace(x0, xf, 1000);
plot(xplot, true_sol(xplot), 'k')
hold on
plot(xplot, u(xplot), 'b')
hold off

err = max(abs(true_sol(x) - u(x)))
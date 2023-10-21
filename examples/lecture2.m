clear all; close all; clc;

% x'(t) = 0.5 * x
% x0 = 0.1
% t0 = 0, tN = 2, dt = 0.5

x0 = 0.1;
t0 = 0;
tN = 2;
dt = 0.5;

f = @(x, t)(0.5 * x);
true_solution = @(t)(x0 * exp(0.5 * t));

tplot = linspace(t0, tN, 1000);
xplot = true_solution(tplot);
plot(tplot, xplot, 'k')

hold on
% plot(t0, x0, 'ro')
% 
% x1 = x0 + dt * f(x0, t0);
% t1 = t0 + dt;
% plot(t1, x1, 'ro')
% 
% x2 = x1 + dt * f(x1, t1);
% t2 = t1 + dt;
% plot(t2, x2, 'ro')

t = t0:dt:tN;
x = zeros(size(t));
x(1) = x0;
for k = 1:(length(t) - 1)
    x(k + 1) = x(k) + dt * f(x(k), t(k));
end
plot(t, x, 'ro')
hold off
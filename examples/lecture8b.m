clear all; close all; clc;

L = 4;
x0 = -L;
y0 = 0;
xf = L;
yf = 0;

f = @(t, x, beta)([x(2); -beta * x(1)]);

tspan = [x0, xf];
beta = bisection(@(beta)(shoot(beta, f, tspan, y0) - yf), 0.1, 0.2, 1e-8);

init_condition = [y0; 1];
tspan = linspace(x0, xf, 1000);
[X, Y] = ode45(@(x, y)f(x, y, beta), tspan, init_condition);
plot(X, Y(:, 1), 'k', xf, yf, 'ro')

n = 1;
beta_predicted = ((n * pi) / (2 * L))^2
beta

function y_final = shoot(beta, f, tspan, y0)

v0 = 1;
init_condition = [y0; v0];
[X, Y] = ode45(@(x, y)f(x, y, beta), tspan, init_condition);
y_final = Y(end, 1);

end

function x = bisection(f, a, b, tol)

x = (a + b) / 2;
while abs(f(x)) >= tol
    if sign(f(x)) == sign(f(a))
        a = x;
    else
        b = x;
    end
    x = (a + b) / 2;
end

end
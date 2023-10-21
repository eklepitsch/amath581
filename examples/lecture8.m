clear all; close all; clc;

t0 = 0;
x0 = 4;
tf = 1;
xf = 1;

f = @(t, x)([x(2); 1.5 * x(1)^2]);

tspan = [t0, tf];
v0 = bisection(@(v0)(shoot(v0, f, tspan, x0) - xf), -30, -40, 1e-8);

init_condition = [x0; v0];
tspan = linspace(t0, tf, 1000);
[T, X] = ode45(f, tspan, init_condition);
plot(T, X(:, 1), 'k', tf, xf, 'ro')

function x_final = shoot(v0, f, tspan, x0)

init_condition = [x0; v0];
[T, X] = ode45(f, tspan, init_condition);
x_final = X(end, 1);

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
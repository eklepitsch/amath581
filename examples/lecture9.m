clear all; close all; clc;

L = 4;
x0 = -L;
y0 = 0;
xf = L;
yf = 0;

f = @(t, x, beta)([x(2); -beta * x(1)]);

beta = 0;
A = 1;
tspan = [x0, xf];
init_condition = [y0; A];
sign_current = sign(shoot(beta, f, tspan, y0));
dbeta = 0.1;

num_modes = 5;
eigenvals = zeros(1, num_modes);
k = 1;

while k <= num_modes
    beta_next = beta + dbeta;
    sign_next = sign(shoot(beta_next, f, tspan, y0));
    if sign_current ~= sign_next
        eigenvals(k) = bisection(@(beta)(shoot(beta, f, tspan, y0) - yf), beta, beta_next, 1e-8);
        k = k + 1;
    end
    beta = beta_next;
    sign_current = sign_next;
end

figure()
hold on
for k = 1:num_modes
    eigenval = eigenvals(k)
    predicted_val = (k * pi / (2 * L))^2
    tspan = linspace(x0, xf, 1000);
    init_condition = [y0; A];
    [X, Y] = ode45(@(x, y)f(x, y, eigenval), tspan, init_condition);
    plot(X, Y(:, 1))
end
hold off

function y_final = shoot(beta, f, tspan, y0)

v0 = 1;
init_condition = [y0; v0];
[X, Y] = ode45(@(x, y)f(x, y, beta), tspan, init_condition);
y_final = Y(end, 1);

end

function x = bisection(f, a, b, tol)

x = (a + b) / 2;
while abs(b - a) >= tol
    if sign(f(x)) == sign(f(a))
        a = x;
    else
        b = x;
    end
    x = (a + b) / 2;
end

end

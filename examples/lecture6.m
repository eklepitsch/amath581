clear all; close all; clc;

% Solve the ODE x' = 1x with x(0) = 1
% from t=0 to t=10
f = @(t, x)(1 * x);

t0 = 0;
tf = 10;
tspan = [t0, tf];
x0 = 1;

[T, X] = ode45(f, tspan, x0);
plot(T, X, 'k')

% Can produce more points.  This makes the plot
% smoother, but does NOT increase the accruacy
tspan = linspace(t0, tf, 1000);
[Tplot, Xplot] = ode45(f, tspan, x0);
plot(T, X, 'k')
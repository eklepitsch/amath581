clear all; close all; clc;

% u_xx + u_yy = 0
% u(x, 0) = x^2 - x
% All other boundaries are zero
% u(x, 1) = u(0, y) = u(1, y) = 0

x0 = 0;
xN = 1;
y0 = 0;
yN = 1;

N = 501
x = linspace(x0, xN, N);
y = linspace(x0, xN, N);
dx = x(2) - x(1)

% Boundary condition
a = @(x)(x.^2 - x);

tic
% Set up system Au = b
N_total = (N - 2) * (N - 2)
entries_in_matrix = N_total^2

% A = zeros(N_total);
% A = sparse(N_total, N_total);
row_vec = zeros(5 * N_total, 1);
col_vec = zeros(5 * N_total, 1);
data_vec = zeros(5 * N_total, 1);
b = zeros(N_total, 1);

point2ind = @(m, n)((n - 2) * (N - 2) + m - 1);

ind = 1;
for n = 2:N-1
    for m = 2:N-1
        k = point2ind(m, n);
        % A(k, k) = -4 / dx^2;
        row_vec(ind) = k;
        col_vec(ind) = k;
        data_vec(ind) = -4 / dx^2;
        ind = ind + 1;
        if m > 2
            % A(k, k - 1) = 1 / dx^2;
            row_vec(ind) = k;
            col_vec(ind) = k - 1;
            data_vec(ind) = 1 / dx^2;
            ind = ind + 1;
        end
        if n < N - 1
            % A(k, k + N - 2) = 1 / dx^2;
            row_vec(ind) = k;
            col_vec(ind) = k + N - 2;
            data_vec(ind) = 1 / dx^2;
            ind = ind + 1;
        end
        if m < N - 1
            % A(k, k + 1) = 1 / dx^2;
            row_vec(ind) = k;
            col_vec(ind) = k + 1;
            data_vec(ind) = 1 / dx^2;
            ind = ind + 1;
        end
        if n > 2
            % A(k, k - (N - 2)) = 1 / dx^2;
            row_vec(ind) = k;
            col_vec(ind) = k - (N - 2);
            data_vec(ind) = 1 / dx^2;
            ind = ind + 1;
        else
            b(k) = b(k) - a(x(m)) / dx^2;
        end
    end
end

A = sparse(row_vec(1:ind-1), col_vec(1:ind-1), data_vec(1:ind-1));

t1 = toc;
fprintf("Time to set up matrix = %ds\n", t1)

tic
% u = A\b;

% The default maximum number of iterations for all of 
% the iterative methods is too small, so we have to 
% crank it up to 2000.  The other arguments are just 
% default values.  

% pcg requires a *positive* definite matrix, but our A
% is *negative* definite, so we have to flip the sign.  

% u = -pcg(-A, b, 1e-6, 2000);

% u = bicgstab(A, b, 1e-6, 2000);

% u = minres(A, b, 1e-6, 2000);

u = gmres(A, b, [], 1e-6, 2000);

t2 = toc;
fprintf("Time to solve = %ds\n", t2)
fprintf("Total time = %ds\n", t1 + t2)

U_int = reshape(u, [N-2, N-2]);
% MATLAB stores data internally column-by-column instead of 
% row-by-row (the way python and C do), but we chose to 
% order our grid points row-by-row.  The following line 
% switches rows and columns of U.
U_int = permute(U_int, [2, 1]);
U = zeros(N, N);
U(2:N-1, 2:N-1) = U_int;
U(1, :) = a(x);

% These vectors are just to plot the boundary conditions
zero_vector = zeros(size(x));
one_vector = ones(size(x));

[X, Y] = meshgrid(x, y);

% Plot the solution
surf(X, Y, U)
shading interp
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
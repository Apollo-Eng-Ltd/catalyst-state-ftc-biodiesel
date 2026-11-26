function [v, fval, flag, out] = fmincon(J, v0, A, b, Aeq, beq, lb, ub, nl, opts)
% Octave test shim (not shipped): sqp with linear inequalities and bounds
g = @(v) b - A*v;
if isempty(A), g = []; end
[v, fval, info, iter] = sqp(v0, J, [], g, lb, ub, 100, 1e-8);
flag = info; out.iterations = iter;
end

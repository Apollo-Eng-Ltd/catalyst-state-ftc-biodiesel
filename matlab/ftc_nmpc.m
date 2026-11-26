function [u0, info, Uw] = ftc_nmpc(x, q, TG0, nb0, nA0, u_prev, nact, P, Uw)
%FTC_NMPC  Economic NMPC for the batch, single shooting with fmincon (SQP).
%   Same problem as control/nmpc.py (CasADi/IPOPT, multiple shooting): horizon H = 6 intervals of 5 min,
%   RK4 with 40 substeps per interval, decisions per interval [T_cmd (K), u_cat (mol/min), F_A (mol/min)],
%   bounds [298.15 353.15], [0 1], [0 20], |dT_cmd| <= 15 K per interval.
%   Objective: -100 X_E(end) + 5 sum(u_cat dt)/n_base0 + sum(F_A dt)/n_A0 + 0.05 sum((dT_cmd/10)^2)
%              + 1e3 sum(max(T - T_hi, 0)^2)  (soft limit T_hi = 63.5 C; the penalty equals the optimal slack).
%   Uw: warm start (H x 3), returned shifted by one interval.
H = 6; T_hi = 273.15 + 63.5; dTmax = 15;
lb = repmat([298.15 0 0], H, 1); ub = repmat([353.15 1 20], H, 1);
sc = repmat([10 1 20], H, 1);
if isempty(Uw), Uw = repmat(u_prev(:)', H, 1); end
% rate limit on T_cmd as linear inequalities on scaled decisions v = U ./ sc (column-major)
A = zeros(2*nact, 3*H); b = zeros(2*nact, 1);
for k = 1:nact
    ik = k;                                   % index of T_cmd of interval k in U(:)
    A(2*k-1, ik) = sc(k,1);  A(2*k, ik) = -sc(k,1);
    if k == 1
        b(2*k-1) = dTmax + u_prev(1);  b(2*k) = dTmax - u_prev(1);
    else
        A(2*k-1, ik-1) = -sc(k-1,1);  A(2*k, ik-1) = sc(k-1,1);
        b(2*k-1) = dTmax;  b(2*k) = dTmax;
    end
end
PP = pack(P);
J = @(v) cost(reshape(v, H, 3) .* sc, x, q, TG0, nb0, nA0, u_prev, nact, PP, T_hi);
opts = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off', 'MaxIterations', 100, ...
    'OptimalityTolerance', 1e-7, 'StepTolerance', 1e-9, 'ConstraintTolerance', 1e-8);
tic;
[v, ~, flag, out] = fmincon(J, Uw(:) ./ sc(:), A, b, [], [], lb(:) ./ sc(:), ub(:) ./ sc(:), [], opts);
tw = toc;
U = reshape(v, H, 3) .* sc;
u0 = U(1, :)';
xe = x;
for k = 1:nact, xe = rk4(xe, U(k,:)', q, PP); end
info = struct('t_wall', tw, 'XE_pred_end', xe(4) / (3*TG0), 'flag', flag, 'iters', out.iterations);
Uw = [U(2:end, :); U(end, :)];
end

function j = cost(U, x, q, TG0, nb0, nA0, u_prev, nact, PP, T_hi)
j = 0; up = u_prev(:)';
for k = 1:nact
    x = rk4(x, U(k,:)', q, PP);
    sl = max(x(13) - T_hi, 0);
    j = j + 5*U(k,2)*5/nb0 + U(k,3)*5/nA0 + 0.05*((U(k,1) - up(1))/10)^2 + 1e3*sl^2;
    up = U(k,:);
end
j = j - 100 * x(4) / (3*TG0);
end

function x = rk4(x, u, q, PP)
h = 5 / 40;
for i = 1:40
    k1 = f(x, u, q, PP); k2 = f(x + h/2*k1, u, q, PP); k3 = f(x + h/2*k2, u, q, PP); k4 = f(x + h*k3, u, q, PP);
    x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end
end

function PP = pack(P)
PP.k50 = P.k50(:); PP.EaR = P.Ea_cal(:) / P.R_cal; PP.iTr = 1 / P.T_ref;
PP.v = [P.a P.c_th P.k_H P.k_SE P.Ea_SE P.GL_sat P.w_th P.k_N P.k_S P.dH P.C_REF P.rho_cp P.UA P.V_J P.rho_cp_J P.F_J];
PP.NUt = [ -1  1  0  1 -1  0  0  0  0  0  0;
            0 -1  1  1 -1  0  0  0  0  0  0;
            0  0 -1  1 -1  1  0  0  0  0  0;
            0  0  0  0  1  0 -1 -1  0  1  0;
            0  0  0  0  1  0 -1  0 -1  0  1;
            0  0  0  0  0  0 -1  0  0  0  1;
            0  0  0 -1  1  0  0  0  0  1 -1]';
end

function dx = f(x, u, q, PP)
% identical to ftc_rhs.m, with the parameters packed for speed
v = PP.v;
n = max(x(1:11), 0); V = x(12); T = x(13); TJ = x(14);
c = n / V;
k = PP.k50 .* exp(-PP.EaR * (1/T - PP.iTr));
kSE = v(4) * exp(-v(5) / 8.314462 * (1/T - 1/333.15));
gl = v(6) * (1 - exp(-c(6) / v(6)));
z = (c(7) - v(2)) / v(7);
if z > 30, sp = z; else, sp = log1p(exp(z)); end
g = max(v(7) * sp, 0);
fa = v(1) * q(1) * q(2) * g / v(11);
r = [fa*(k(1)*c(1)*c(5) - k(2)*c(2)*c(4)); fa*(k(3)*c(2)*c(5) - k(4)*c(3)*c(4)); ...
     fa*(k(5)*c(3)*c(5) - k(6)*gl*c(4)); v(8)*c(8)*c(7); v(9)*c(9)*c(7); v(3)*c(7); kSE*c(4)*c(11)];
dn = (PP.NUt * r) * V;
lossA = q(4) * n(5);
dn(9) = dn(9) + q(3);
dn(5) = dn(5) - lossA + u(3);
dn(7) = dn(7) + u(2);
dV = (u(3) - lossA) * 32.042/1000/0.791 + q(3) * 18.015/1000;
Tin = TJ + q(5) * (u(1) - TJ);
dT = (-v(10) * (r(1) + r(2) + r(3)) * V + v(13) * (TJ - T)) / (v(12) * V);
dTJ = (v(16) * v(15) * (Tin - TJ) - v(13) * (TJ - T)) / (v(15) * v(14));
dx = [dn; dV; dT; dTJ];
end

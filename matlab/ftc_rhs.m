function dx = ftc_rhs(x, u, q, P)
%FTC_RHS  Pilot reactor model (mirrors model/plant.py and control/casadi_plant.py).
%   x (14): moles TG DG MG E A GL Cat FFA W Soap OH; V (L); T, T_J (K)
%   u (3):  T_cmd (K, TCU setpoint), u_cat (mol base/min), F_A (mol MeOH/min)
%   q (5):  a_rel, eta, F_W (mol water/min), k_loss (1/min methanol loss), phi (TCU capacity)
%#codegen
n  = max(x(1:11), 0);  V = x(12);  T = x(13);  TJ = x(14);
c  = n / V;
k  = P.k50 .* exp(-P.Ea_cal / P.R_cal * (1/T - 1/P.T_ref));
kSE = P.k_SE * exp(-P.Ea_SE / 8.314462 * (1/T - 1/333.15));
gl_s = P.GL_sat * (1 - exp(-c(6) / P.GL_sat));
z = (c(7) - P.c_th) / P.w_th;
if z > 30, sp = z; else, sp = log1p(exp(z)); end
g = max(P.w_th * sp, 0);
fa = P.a * q(1) * q(2) * g / P.C_REF;
TG = c(1); DG = c(2); MG = c(3); E = c(4); A = c(5);
r1 = fa * (k(1)*TG*A - k(2)*DG*E);
r2 = fa * (k(3)*DG*A - k(4)*MG*E);
r3 = fa * (k(5)*MG*A - k(6)*gl_s*E);
rN = P.k_N * c(8) * c(7);
rW = P.k_S * c(9) * c(7);
rH = P.k_H * c(7);
rSE = kSE * E * c(11);
%       TG  DG  MG  E   A   GL  Cat FFA W   Soap OH
NU = [ -1   1   0   1  -1   0   0   0   0   0    0;    % r1
        0  -1   1   1  -1   0   0   0   0   0    0;    % r2
        0   0  -1   1  -1   1   0   0   0   0    0;    % r3
        0   0   0   0   1   0  -1  -1   0   1    0;    % rN
        0   0   0   0   1   0  -1   0  -1   0    1;    % rW
        0   0   0   0   0   0  -1   0   0   0    1;    % rH
        0   0   0  -1   1   0   0   0   0   1   -1];   % rSE
dn = (NU' * [r1; r2; r3; rN; rW; rH; rSE]) * V;
lossA = q(4) * n(5);
dn(9) = dn(9) + q(3);
dn(5) = dn(5) - lossA + u(3);
dn(7) = dn(7) + u(2);
dV  = (u(3) - lossA) * 32.042/1000/0.791 + q(3) * 18.015/1000;
Tin = TJ + q(5) * (u(1) - TJ);
qrx = -P.dH * (r1 + r2 + r3) * V;
dT  = (qrx + P.UA * (TJ - T)) / (P.rho_cp * V);
dTJ = (P.F_J * P.rho_cp_J * (Tin - TJ) - P.UA * (TJ - T)) / (P.rho_cp_J * P.V_J);
dx = [dn; dV; dT; dTJ];
end

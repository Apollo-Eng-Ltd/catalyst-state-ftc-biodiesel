function L = ftc_octave_loop(cname, mode)
% Emulates the Simulink loop (0.25-min sampling, ZOH inputs) for testing ftc_supervisor in Octave.
addpath('../matlab'); addpath('shims');
P = ftc_params(); P = rmfield(P, 'names');
P.P0_W = 5 * 880 * 2^3 * 0.33^5;  P.m0 = 600 + P.x0(5)*32.042/1000 + P.x0(7)*138.205/1000;
D = jsondecode(fileread('ftc_sup_data.json')); m = D.mag.(cname);
q_f = [1 1 0 0 1]; t_f = 0;
switch cname
  case 'degraded_oil', q_f(1) = m;
  case 'water_ingress', q_f(3) = m/100*600*1000/18.015/10; t_f = 10;
  case 'methanol_loss', q_f(4) = m; t_f = 10;
end
qf = @(t) (t >= t_f) * q_f(:) + (t < t_f) * [1;1;0;0;1];
ftc_supervisor('init', struct('mode', mode, 'P', P, 'D', D, 'st', [4;6;1.088;0.0002;60;53], 't_pid', 35));
x = P.x0; mm = 0; eI = 0; ud = [P.x0(13); 0; 0]; Tsp = 333.15;
opts = odeset('RelTol', 1e-7, 'AbsTol', 1e-9, 'MaxStep', 0.25);
for k = 0:480
  t = 0.25*k; q = qf(t);
  Tc = min(max(ud(1), 298.15), 353.15); Tin = x(14) + q(5)*(Tc - x(14));
  Vbp = (x(5)*32.042/0.791 + x(4)*292/0.875)/1000; CB = (1/138.205)/(6*100/870*32.042/0.791/1000);
  pH = 12 + log10(max(x(7)+x(11),1e-12)/Vbp/CB);
  y = [x(13); x(14); Tin; Tc; q(2); pH; P.m0 + mm; x(4)/(3*P.x0(1)); x(13)];
  [u, stop] = ftc_supervisor(t, y);
  if stop, break; end
  f = @(tt, z) rhsz(tt, z, u, mode, qf, P, Tsp);
  [~, Z] = ode45(f, [t t+0.25], [x; mm; eI], opts);
  z = Z(end,:)'; x = z(1:14); mm = z(15); eI = z(16);
  if mode == 0
    raw = Tsp + P.Kc*((Tsp - x(13)) + eI/P.tauI); ud = [min(max(raw,298.15),353.15); 0; 0];
  else
    ud = u;
  end
end
L = ftc_supervisor('log');
end

function dz = rhsz(t, z, u, mode, qf, P, Tsp)
x = z(1:14); eI = z(16); q = qf(t);
if mode == 0
  raw = Tsp + P.Kc*((Tsp - x(13)) + eI/P.tauI); Tc = min(max(raw,298.15),353.15); u = [Tc; 0; 0];
  deI = (Tsp - x(13)) + (Tc - raw)/P.Kc;
else
  deI = 0;
end
dx = ftc_rhs(x, u, q, P);
dm = (u(3) - q(4)*max(x(5),0))*32.042/1000 + q(3)*18.015/1000 + u(2)*138.205/1000;
dz = [dx; dm; deI];
end

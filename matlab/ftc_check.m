%FTC_CHECK  Reproduce two Python reference batches (PI on the TCU, 60 C setpoint, 60 min) in MATLAB.
%   Python reference: control/plant_sim.py (LSODA), truth trajectories without measurement noise.
%   Pass criterion: |X_E(MATLAB) - X_E(Python)| < 0.1 percentage point at every reference time.
P = ftc_params();
ref.t = [0 5 10 20 30 40 50 60];
ref.normal       = [0 0.7772 0.8886 0.9480 0.9654 0.9701 0.9700 0.9681];
ref.degraded_oil = [0 0.5905 0.7675 0.8684 0.9077 0.9267 0.9363 0.9411];
cases = {'normal', 1.0; 'degraded_oil', 1 - 0.85*0.6};      % F3 at severity 0.6: a_rel = 0.49
Tsp = 333.15;
opts = odeset('RelTol',1e-8,'AbsTol',1e-9,'MaxStep',1);
fprintf('%-13s %6s %9s %9s %8s\n','case','t','X_E MATLAB','X_E Python','diff pp');
worst = 0; tic;
for c = 1:size(cases,1)
    q = [cases{c,2} 1 0 0 1];
    f = @(t,z) pi_loop(z, q, Tsp, P);
    [t, Z] = ode15s(f, ref.t, [P.x0; 0], opts);
    XE = Z(:,4) / (3*P.x0(1));
    r = ref.(cases{c,1});
    for i = 1:numel(t)
        d = 100*(XE(i) - r(i)); worst = max(worst, abs(d));
        fprintf('%-13s %6.1f %9.4f %9.4f %8.4f\n', cases{c,1}, t(i), XE(i), r(i), d);
    end
end
fprintf('worst difference %.4f pp, wall time %.2f s\n', worst, toc);
assert(worst < 0.1, 'MATLAB model does not reproduce the Python reference');
disp('MATLAB plant reproduces the Python pipeline');

function dz = pi_loop(z, q, Tsp, P)
x = z(1:14); eI = z(15);
raw = Tsp + P.Kc * ((Tsp - x(13)) + eI / P.tauI);
Tc = min(max(raw, P.TJin_lim(1)), P.TJin_lim(2));
dx = ftc_rhs(x, [Tc 0 0], q, P);
dz = [dx; (Tsp - x(13)) + (Tc - raw) / P.Kc];
end

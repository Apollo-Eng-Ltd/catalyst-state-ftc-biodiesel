function varargout = ftc_supervisor(t, y)
%FTC_SUPERVISOR  Measurement system, causal estimator and NMPC layer of the Simulink model.
%   ftc_supervisor('init', cfg)    reset; cfg: mode (0 PI, 1 NMPC-nom, 2 FT-NMPC rule v1, 3 FT-NMPC rule v2),
%                                  P, D (ftc_sup_data),
%                                  st (static recipe inputs, 6x1), t_pid (PI stop time, min)
%   [u, stop, est] = ftc_supervisor(t, y)   called every 0.25 min by the Simulink block.
%       y = [T+bias, T_J, T_Jin, T_cmd (K, previous interval), eta, pH, m_ld (kg), X_E, T_true (K)]
%       u = [T_cmd (K), u_cat (mol/min), F_A (mol/min)], held until the next 5-min decision.
%       est = [X_E estimate, a_rel, eta, phi, F_W, k_loss] (NaN when not estimated)
%   L = ftc_supervisor('log')      the run log (measurements, decisions, solve times).
%   Mirrors control/plant_sim.py (measurement model and noise sequence), control/controllers.py (rule v1)
%   and control/controllers2.py (rule v2: calibrated margin, model-free alarms, 5-min confirmation).
%   Rule v2: C1  X_E,hat - delta >= 0.965 and predicted gain < 0.1 point
%            C2  predicted gain < 0.05 point and X_E,hat did not rise over the last interval
%            C3  X_E,hat fell at two consecutive decisions
%            stop if C1 or C2 at two consecutive decisions, or C3, or t = 120 min
%            delta = 0.67 point; 1.03 point once a model-free alarm has fired (latched): load cell
%            |m_ld - (m0 + dosed)| > 0.32 kg, TCU |T_cmd - T_Jin| > 0.25 K, stirrer |P_st/P0 - 1| > 0.03,
%            each on 2.5-min moving means (control/stoprule_calibration.json).
persistent S
if ischar(t)
    switch t
        case 'init'
            S = y;  S.k = 0;  S.h = {zeros(96,1), zeros(96,1)};  S.yhat = [];
            S.u = [NaN; 0; 0];  S.u_prev = [333.15; 0; 0];  S.Uw = [];
            S.stop = 0;  S.est = nan(6,1);
            S.xn = S.P.x0;  S.TG0 = S.P.x0(1);  S.nb0 = S.P.x0(7);  S.nA0 = S.P.x0(5);
            S.log = struct('t', [], 'meas', [], 'XE', [], 'T_true', [], 'dec', [], 'solve', []);
            S.cat_dosed = 0;  S.meoh_dosed = 0;  S.log.cat_kg = 0;  S.log.meoh_kg = 0;
            S.alarm = false;  S.cond_prev = false;  S.XE_prev = NaN;  S.XE_prev2 = NaN;
            S.delta0 = 0.006731091572517833;  S.delta1 = 0.010281060552296162;  S.log.alarm = false;
            varargout = {};
        case 'log'
            varargout = {S.log};
    end
    return
end
if S.stop
    varargout = {S.u, 1, S.est};  return
end
P = S.P;  D = S.D;
k = round(t / 0.25);  e = D.noise(k + 1, :);
% ---- measurement model (plant_sim.Batch._record): noise added in the same order and stream as Python
Tm = y(1) + 0.05*e(1);  TJm = y(2) + 0.05*e(2);  Tinm = y(3) + 0.05*e(3);
Q_J = P.F_J * P.rho_cp_J * (Tinm - TJm) / 60 / 1000;
P_st = P.P0_W * y(5)^3 + 0.02*P.P0_W*e(4);
pH = y(6) + 0.02*e(5);  m_ld = y(7) + 0.2*e(6);
u_logged = S.u;  if isnan(u_logged(1)), u_logged = [y(4); 0; 0]; end
meas = [Tm-273.15, TJm-273.15, Tinm-273.15, y(4)-273.15, Q_J, P_st, pH, m_ld, u_logged(2), u_logged(3)];
S.log.t(end+1,1) = t;  S.log.meas(end+1,:) = meas;  S.log.XE(end+1,1) = y(8);  S.log.T_true(end+1,1) = y(9);
% ---- causal estimator, every 0.5 min
if mod(k, 2) == 0
    xin = ([meas, S.st(:)'] - D.mu_x(:)') ./ D.sd_x(:)';
    [S.h, yn] = ftc_gru_step(xin(:), S.h, D.W);
    S.yhat = yn .* D.sd_y(:) + D.mu_y(:);
end
% ---- decisions every 5 min
if mod(k, 20) == 0
    if S.mode == 0
        S.u = [NaN; 0; 0];
        if t >= S.t_pid - 1e-9, S.stop = 1; end
    else
        if t >= 120 - 1e-9
            S.stop = 1;
        else
            if S.mode == 1                     % nominal observer, temperatures reset to measurements
                S.xn(13) = Tm;  S.xn(14) = TJm;
                x = S.xn;  q = [1; 1; 0; 0; 1];
            else                               % FT-NMPC: state and fault parameters from the estimator
                yh = S.yhat;
                x = [max(yh(1:11), 0) * S.TG0; yh(12) * P.x0(12); yh(13); TJm];
                q = [min(max(yh(17), 0.05), 2); min(max(yh(14), 0.05), 1.5); max(yh(18), 0); max(yh(19), 0); ...
                     min(max(yh(15), 0.05), 1)];
                S.est = [x(4)/(3*S.TG0); q(1); q(2); q(5); q(3); q(4)];
            end
            XE_now = x(4) / (3*S.TG0);
            nact = min(6, round((120 - t) / 5));
            [u, info, S.Uw] = ftc_nmpc(x, q, S.TG0, S.nb0, S.nA0, S.u_prev, nact, P, S.Uw);
            gain = info.XE_pred_end - XE_now;
            S.log.dec(end+1,:) = [t, XE_now, gain, u(1)-273.15, u(2), u(3), q', info.t_wall, info.iters, info.flag];
            S.log.solve(end+1,1) = info.t_wall;
            if S.mode == 3
                S.alarm = S.alarm || any(alarms(S.log.meas, P));  S.log.alarm = S.alarm;
                if S.alarm, delta = max(S.delta0, S.delta1); else, delta = S.delta0; end
                c1 = XE_now - delta >= 0.965 && gain < 1e-3;
                c2 = gain < 5e-4 && ~isnan(S.XE_prev) && XE_now - S.XE_prev < 5e-4;
                c3 = ~isnan(S.XE_prev) && ~isnan(S.XE_prev2) && XE_now < S.XE_prev && S.XE_prev < S.XE_prev2;
                cond = t > 0 && (c1 || c2);
                stopnow = (cond && S.cond_prev) || c3;
                S.cond_prev = cond;
            else
                stopnow = t > 0 && ((XE_now >= 0.965 && gain < 1e-3) || gain < 5e-4);
            end
            if stopnow
                S.stop = 1;
            else
                S.XE_prev2 = S.XE_prev;  S.XE_prev = XE_now;
                S.u = u;  S.u_prev = u;
                S.cat_dosed = S.cat_dosed + u(2)*5;  S.meoh_dosed = S.meoh_dosed + u(3)*5;
                S.log.cat_kg = S.cat_dosed * 138.205/1000;  S.log.meoh_kg = S.meoh_dosed * 32.042/1000;
                if S.mode == 1, S.xn = ftc_rk4(S.xn, u, q, P, 5, 40); end
            end
        end
    end
end
uo = S.u;  if isnan(uo(1)), uo(1) = 0; end
varargout = {uo, S.stop, S.est};
end

function a = alarms(M, P)
% model-free alarms on the logged measurements (columns as in meas): [load cell, TCU, stirrer]
a = false(1, 3);
if size(M, 1) < 10, return, end
dosed = [0; cumsum((M(2:end,9)*138.205 + M(2:end,10)*32.042) / 1000 * 0.25)];
mv = @(r) conv(r, ones(10,1)/10, 'valid');
a(1) = any(abs(mv(M(:,8) - (P.m0 + dosed))) > 0.32);
a(2) = any(abs(mv(M(:,4) - M(:,3))) > 0.25);
a(3) = any(abs(mv(M(:,6) / P.P0_W - 1)) > 0.03);
end

function x = ftc_rk4(x, u, q, P, dt, n)
h = dt / n;
for i = 1:n
    k1 = ftc_rhs(x, u, q, P); k2 = ftc_rhs(x + h/2*k1, u, q, P);
    k3 = ftc_rhs(x + h/2*k2, u, q, P); k4 = ftc_rhs(x + h*k3, u, q, P);
    x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end
end

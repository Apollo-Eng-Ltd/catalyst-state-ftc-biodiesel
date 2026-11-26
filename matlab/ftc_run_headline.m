%FTC_RUN_HEADLINE  Headline scenarios in Simulink under PID (35 min), nominal NMPC and FT-NMPC;
%   comparison with the Python pipeline (refs_headline.json, same recipe, noise sequence and faults).
%   Needs: ftc_sup_data.json, refs_headline.json, refs_v2.json, and build_ftc_sim (run first).
%   Controllers: PID-35, nominal NMPC, FT-NMPC with stop rule v1, FT-NMPC with stop rule v2.
assert(license('test', 'Optimization_Toolbox') == 1, 'fmincon (Optimization Toolbox) is needed for the NMPC');
D = jsondecode(fileread('ftc_sup_data.json'));
R = jsondecode(fileread('refs_headline.json'));
R2 = jsondecode(fileread('refs_v2.json'));          % Python FT-NMPC with stop rule v2 (controllers2.py)
mdl = 'ftc_biodiesel';  load_system(mdl);
cases = {'normal', 'degraded_oil', 'water_ingress', 'methanol_loss'};
modes = {'PID35', 'NMPC', 'FTNMPC', 'FTNMPCv2'};
res = struct([]);  r = 0;
fprintf('%-14s %-7s %9s %9s %7s %7s %8s %8s %9s\n', 'case', 'ctrl', 'XE MATLAB', 'XE Python', 't_stop', 't_Py', 'MeOH kg', 'solve s', 'RT factor');
for i = 1:numel(cases)
    cname = cases{i};  m = D.mag.(cname);
    f = struct('a_f', 1, 'eta_f', 1, 'FW_f', 0, 'kl_f', 0, 'phi_f', 1, 'bias_f', 0, 't_f', 0);
    switch cname
        case 'degraded_oil',  f.a_f = m;
        case 'water_ingress', f.FW_f = m / 100 * 600 * 1000 / 18.015 / 10;  f.t_f = 10;
        case 'methanol_loss', f.kl_f = m;  f.t_f = 10;
    end
    fn = fieldnames(f);
    for j = 1:numel(fn), assignin('base', fn{j}, f.(fn{j})); end
    for mo = 0:3
        assignin('base', 'ctrl_mode', mo);
        assignin('base', 'sup_cfg', struct('mode', mo, 'P', evalin('base', 'P'), 'D', D, ...
            'st', [4; 6; 1.088; 0.0002; 60; 53], 't_pid', 35));
        tic;  sim(mdl);  wall = toc;
        L = ftc_supervisor('log');
        if mo == 3, ref = R2.(cname); else, ref = R.refs.(cname).(modes{mo+1}); end
        r = r + 1;
        res(r).case = cname;  res(r).ctrl = modes{mo+1};
        res(r).XE = L.XE(end);  res(r).t_stop = L.t(end);  res(r).XE_py = ref.XE_stop;  res(r).t_py = ref.t_stop;
        res(r).meoh_kg = L.meoh_kg;  res(r).cat_kg = L.cat_kg;  res(r).wall = wall;
        res(r).solve_mean = mean([L.solve; NaN(isempty(L.solve))], 'omitnan');  res(r).solve_max = max([L.solve; 0]);
        res(r).t = L.t;  res(r).XE_traj = L.XE;  res(r).t_py_traj = ref.t;  res(r).XE_py_traj = ref.X_E;  res(r).dec = L.dec;
        fprintf('%-14s %-7s %9.4f %9.4f %7.1f %7.1f %8.1f %8.2f %9.0f\n', cname, modes{mo+1}, res(r).XE, res(r).XE_py, ...
            res(r).t_stop, res(r).t_py, res(r).meoh_kg, res(r).solve_mean, res(r).t_stop * 60 / wall);
    end
end
save('ftc_headline_results.mat', 'res');
% figure: conversion, MATLAB (lines) against Python (markers)
fig = figure('Color', 'w', 'Position', [100 100 1000 700]);
try, fig.Theme = 'light'; catch, end                 % light axes for print (R2025a+)
col = [0.45 0.45 0.45; 0.20 0.45 0.75; 0.85 0.40 0.10; 0.20 0.60 0.30];
for i = 1:numel(cases)
    subplot(2, 2, i); hold on; box on;
    for mo = 1:4
        k = find(strcmp({res.case}, cases{i}) & strcmp({res.ctrl}, modes{mo}));
        plot(res(k).t, 100*res(k).XE_traj, '-', 'Color', col(mo,:), 'LineWidth', 1.5, 'DisplayName', [modes{mo} ' Simulink']);
        plot(res(k).t_py_traj(1:5:end), 100*res(k).XE_py_traj(1:5:end), 'o', 'Color', col(mo,:), 'MarkerSize', 4, ...
            'DisplayName', [modes{mo} ' Python']);
    end
    yline(96.5, ':k', 'EN 14214', 'HandleVisibility', 'off');
    xlim([0 70]); ylim([80 100]); xlabel('time (min)'); ylabel('ester conversion (%)');
    title(strrep(cases{i}, '_', ' '));
    if i == 1, legend('Location', 'southeast', 'FontSize', 7); end
end
exportgraphics(fig, 'ftc_headline_simulink.png', 'Resolution', 200);
disp('saved ftc_headline_results.mat and ftc_headline_simulink.png');

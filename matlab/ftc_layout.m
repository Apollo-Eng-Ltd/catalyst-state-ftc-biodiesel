% Tidy the block layout of ftc_biodiesel for the paper figure (positions and wiring only).
mdl = 'ftc_biodiesel'; load_system(mdl); p = @(n, r) set_param([mdl '/' n], 'Position', r);
E = 'Estimates [X_E a_rel eta phi F_W k_loss]'; M = 'MVs [T_cmd u_cat F_A]';
p('T_sp', [20 60 80 90]); p('err', [120 65 140 85]); p('PI controller', [170 55 250 95]);
p('T_cmd PI', [290 65 310 85]); p('controller mode', [290 5 350 30]); p('select T_cmd', [380 50 410 110]);
p('clock', [20 150 50 180]); p('sample t', [80 150 110 180]); p('sample y', [80 210 110 240]);
p('estimator + NMPC', [150 140 270 250]); p('split u', [310 130 315 190]); p('stop rule', [300 220 330 250]);
p(E, [150 300 260 420]); p('u', [470 40 475 190]); p(M, [520 0 620 35]); p('log u', [520 75 580 95]);
for i = 1:5, n = {'a_rel','eta','F_W','k_loss','phi'}; p(['fault ' n{i}], [380 280+40*(i-1) 430 300+40*(i-1)]); end
p('q', [470 270 475 470]); p('reactor model', [530 120 640 220]); p('state x', [680 155 710 185]);
p('u held', [530 300 570 330]); p('sensor bias', [610 420 660 445]);
p('sensors', [760 270 880 390]); p('mass flow', [760 420 880 480]); p('outputs', [760 110 880 210]);
p('T measured', [930 280 960 300]); p('load cell mass', [930 435 960 465]);
lab = {'Conversion X_E (%)', 'Reactor T (C)', 'Jacket T (C)', 'Apparent pH'};
for i = 1:4, p(lab{i}, [1000 105+30*(i-1) 1090 125+30*(i-1)]); end
p('Conversion scope', [1140 40 1170 70]); p('Temperature scope', [1140 230 1170 280]); p('log XE', [1130 10 1190 30]);
set_param([mdl '/err'], 'ShowName', 'off'); set_param([mdl '/T_cmd PI'], 'ShowName', 'off');
h = find_system(mdl, 'FindAll', 'on', 'SearchDepth', 1, 'Type', 'line'); delete_line(h);
L = @(a, b) add_line(mdl, a, b, 'autorouting', 'smart');
L('T_sp/1', 'err/1'); L('err/1', 'PI controller/1'); L('PI controller/1', 'T_cmd PI/1'); L('T_sp/1', 'T_cmd PI/2');
L('clock/1', 'sample t/1'); L('sample t/1', 'estimator + NMPC/1'); L('sample y/1', 'estimator + NMPC/2');
L('estimator + NMPC/1', 'split u/1'); L('estimator + NMPC/2', 'stop rule/1'); L('estimator + NMPC/3', [E '/1']);
L('split u/1', 'select T_cmd/1'); L('controller mode/1', 'select T_cmd/2'); L('T_cmd PI/1', 'select T_cmd/3');
L('select T_cmd/1', 'u/1'); L('split u/2', 'u/2'); L('split u/3', 'u/3'); L('u/1', [M '/1']); L('u/1', 'log u/1');
for i = 1:5, L(['fault ' n{i} '/1'], ['q/' num2str(i)]); end
L('u/1', 'reactor model/2'); L('q/1', 'reactor model/3'); L('reactor model/1', 'state x/1'); L('state x/1', 'reactor model/1');
L('u/1', 'u held/1'); L('state x/1', 'sensors/1'); L('u held/1', 'sensors/2'); L('q/1', 'sensors/3');
L('sensor bias/1', 'sensors/4'); L('load cell mass/1', 'sensors/5');
L('state x/1', 'mass flow/1'); L('u/1', 'mass flow/2'); L('q/1', 'mass flow/3'); L('mass flow/1', 'load cell mass/1');
L('sensors/1', 'sample y/1'); L('sensors/1', 'outputs/1'); L('sensors/1', 'T measured/1'); L('T measured/1', 'err/2');
for i = 1:4, L(['outputs/' num2str(i)], [lab{i} '/1']); end
L('outputs/1', 'Conversion scope/1'); L('outputs/1', 'log XE/1'); L('outputs/2', 'Temperature scope/1'); L('outputs/3', 'Temperature scope/2');
set_param(mdl, 'ZoomFactor', 'FitSystem'); save_system(mdl); close_system(mdl); load_system(mdl);
t = fullfile(tempdir, 'ftcfig'); if ~isfolder(t), mkdir(t); end
print(['-s' mdl], '-dpng', '-r300', fullfile(t, 'Fig10_simulink_model.png')); print(['-s' mdl], '-dpdf', fullfile(t, 'Fig10_simulink_model.pdf'));
zip('Fig10_export.zip', {fullfile(t, 'Fig10_simulink_model.png'), fullfile(t, 'Fig10_simulink_model.pdf'), [mdl '.slx']});
z = dir('Fig10_export.zip'); fprintf('layout done, zip %d bytes\n', z.bytes);

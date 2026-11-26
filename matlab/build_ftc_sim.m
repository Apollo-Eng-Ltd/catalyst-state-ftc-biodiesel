%BUILD_FTC_SIM  Build the Simulink model ftc_biodiesel.slx entirely by script (reproducible).
%   Plant: MATLAB Function calling ftc_rhs + Integrator, load-cell mass integrator, fault injection.
%   Control: PI on the TCU setpoint (back-calculation anti-windup) or the supervisor (ftc_supervisor.m:
%   measurement noise, causal GRU estimator, NMPC every 5 min, stop rule), chosen by ctrl_mode
%   (0 PI, 1 nominal NMPC, 2 fault-tolerant NMPC). Time unit: minutes.
%   Run one batch: see ftc_run_headline.m.
P = ftc_params();  P = rmfield(P, 'names');
P.P0_W = 5 * 880 * 2^3 * 0.33^5;                              % stirrer power, W (faults.py)
P.m0 = 600 + P.x0(5)*32.042/1000 + P.x0(7)*138.205/1000;      % charge mass, kg
assignin('base', 'P', P);
defaults = struct('a_f', 1, 'eta_f', 1, 'FW_f', 0, 'kl_f', 0, 'phi_f', 1, 'bias_f', 0, 't_f', 0, 'ctrl_mode', 0);
fn = fieldnames(defaults);
for i = 1:numel(fn)
    if ~evalin('base', sprintf('exist(''%s'',''var'')', fn{i})), assignin('base', fn{i}, defaults.(fn{i})); end
end
mdl = 'ftc_biodiesel';
if bdIsLoaded(mdl), close_system(mdl, 0); end
if exist([mdl '.slx'], 'file'), delete([mdl '.slx']); end
new_system(mdl); open_system(mdl);
set_param(mdl, 'Solver', 'ode15s', 'StopTime', '120', 'MaxStep', '0.25', 'RelTol', '1e-7', 'AbsTol', '1e-9', ...
    'InitFcn', 'ftc_supervisor(''init'', sup_cfg);');
rt = sfroot;
B = @(lib, name, varargin) add_block(lib, [mdl '/' name], varargin{:});
setfun = @(name, code) set(rt.find('-isa', 'Stateflow.EMChart', 'Path', [mdl '/' name]), 'Script', code);

% PI loop
B('simulink/Sources/Constant', 'T_sp', 'Value', '333.15', 'Position', [30 60 90 90]);
B('simulink/Math Operations/Sum', 'err', 'Inputs', '+-', 'Position', [130 65 150 85]);
B('simulink/Continuous/PID Controller', 'PI controller', 'Controller', 'PI', 'P', '6', 'I', '6/8', ...
    'LimitOutput', 'on', 'UpperSaturationLimit', '353.15-333.15', 'LowerSaturationLimit', '298.15-333.15', ...
    'AntiWindupMode', 'back-calculation', 'Kb', '1/8', 'Position', [180 55 260 95]);
B('simulink/Math Operations/Sum', 'T_cmd PI', 'Inputs', '++', 'Position', [290 65 310 85]);

% supervisor (estimator + NMPC)
B('simulink/Sources/Clock', 'clock', 'Position', [30 140 60 170]);
B('simulink/Discrete/Zero-Order Hold', 'sample t', 'SampleTime', '0.25', 'Position', [90 140 120 170]);
B('simulink/Discrete/Zero-Order Hold', 'sample y', 'SampleTime', '0.25', 'Position', [90 200 120 230]);
B('simulink/User-Defined Functions/MATLAB Function', 'estimator + NMPC');
set_param([mdl '/estimator + NMPC'], 'Position', [160 130 280 240]);
setfun('estimator + NMPC', sprintf(['function [u, stop, est] = f(t, y)\n' ...
    '%% causal GRU estimator and NMPC every 5 min (ftc_supervisor.m)\n' ...
    'coder.extrinsic(''ftc_supervisor'');\n' ...
    'u = zeros(3,1); stop = 0; est = zeros(6,1);\n' ...
    '[u, stop, est] = ftc_supervisor(t, y);\n']));
d = datum(rt, mdl, 'estimator + NMPC', 'y'); d.Props.Array.Size = '9';
B('simulink/Sinks/Stop Simulation', 'stop rule', 'Position', [330 200 360 230]);
B('simulink/Signal Routing/Demux', 'split u', 'Outputs', '3', 'Position', [320 120 325 180]);
B('simulink/Sources/Constant', 'controller mode', 'Value', 'ctrl_mode', 'Position', [300 15 360 40]);
B('simulink/Signal Routing/Switch', 'select T_cmd', 'Criteria', 'u2 > Threshold', 'Threshold', '0.5', ...
    'Position', [400 55 430 105]);
B('simulink/Signal Routing/Mux', 'u', 'Inputs', '3', 'Position', [470 60 475 180]);

% faults
fnames = {'a_rel','eta','F_W','k_loss','phi'}; fin = {'1','1','0','0','1'}; ffin = {'a_f','eta_f','FW_f','kl_f','phi_f'};
for i = 1:5
    B('simulink/Sources/Step', ['fault ' fnames{i}], 'Time', 't_f', 'Before', fin{i}, 'After', ffin{i}, ...
        'SampleTime', '0', 'Position', [380 260+45*(i-1) 430 285+45*(i-1)]);
end
B('simulink/Signal Routing/Mux', 'q', 'Inputs', '5', 'Position', [470 250 475 480]);
B('simulink/Sources/Step', 'sensor bias', 'Time', 't_f', 'Before', '0', 'After', 'bias_f', ...
    'Position', [600 420 650 445]);

% reactor
B('simulink/User-Defined Functions/MATLAB Function', 'reactor model');
set_param([mdl '/reactor model'], 'Position', [530 110 640 210]);
setfun('reactor model', sprintf('function dx = f(x, u, q, P)\n%% plant right-hand side (ftc_rhs.m)\ndx = ftc_rhs(x, u, q, P);\n'));
d = datum(rt, mdl, 'reactor model', 'P'); d.Scope = 'Parameter';
d = datum(rt, mdl, 'reactor model', 'x'); d.Props.Array.Size = '14';
d = datum(rt, mdl, 'reactor model', 'u'); d.Props.Array.Size = '3';
d = datum(rt, mdl, 'reactor model', 'q'); d.Props.Array.Size = '5';
d = datum(rt, mdl, 'reactor model', 'dx'); d.Props.Array.Size = '14';
B('simulink/Continuous/Integrator', 'state x', 'InitialCondition', 'P.x0', 'Position', [680 145 710 175]);

% sensors
B('simulink/Discrete/Unit Delay', 'u held', 'InitialCondition', '[P.x0(13); 0; 0]', 'SampleTime', '0.25', ...
    'Position', [530 300 570 330]);
B('simulink/User-Defined Functions/MATLAB Function', 'sensors');
set_param([mdl '/sensors'], 'Position', [760 260 880 380]);
setfun('sensors', sprintf(['function y = f(x, ud, q, bias, m, P)\n' ...
    '%% true measured quantities (noise is added in the supervisor); ud = inputs of the last 0.25 min\n' ...
    'Tc = min(max(ud(1), 298.15), 353.15);\n' ...
    'Tin = x(14) + q(5) * (Tc - x(14));\n' ...
    'Vbp = (x(5) * 32.042/0.791 + x(4) * 292/0.875) / 1000;\n' ...
    'CB = (1/138.205) / (6*100/870*32.042/0.791/1000);\n' ...
    'pH = 12 + log10(max(x(7) + x(11), 1e-12) / Vbp / CB);\n' ...
    'y = [x(13) + bias; x(14); Tin; Tc; q(2); pH; P.m0 + m; x(4) / (3*P.x0(1)); x(13)];\n']));
d = datum(rt, mdl, 'sensors', 'P'); d.Scope = 'Parameter';
d = datum(rt, mdl, 'sensors', 'x'); d.Props.Array.Size = '14';
d = datum(rt, mdl, 'sensors', 'ud'); d.Props.Array.Size = '3';
d = datum(rt, mdl, 'sensors', 'q'); d.Props.Array.Size = '5';
d = datum(rt, mdl, 'sensors', 'y'); d.Props.Array.Size = '9';
B('simulink/User-Defined Functions/MATLAB Function', 'mass flow');
set_param([mdl '/mass flow'], 'Position', [760 400 880 460]);
setfun('mass flow', sprintf(['function dm = f(x, u, q)\n' ...
    '%% load-cell mass change, kg/min: methanol make-up minus loss, water ingress, catalyst top-up\n' ...
    'dm = (u(3) - q(4)*max(x(5),0)) * 32.042/1000 + q(3) * 18.015/1000 + u(2) * 138.205/1000;\n']));
d = datum(rt, mdl, 'mass flow', 'x'); d.Props.Array.Size = '14';
d = datum(rt, mdl, 'mass flow', 'u'); d.Props.Array.Size = '3';
d = datum(rt, mdl, 'mass flow', 'q'); d.Props.Array.Size = '5';
B('simulink/Continuous/Integrator', 'load cell mass', 'InitialCondition', '0', 'Position', [920 320 950 350]);
B('simulink/Signal Routing/Selector', 'T measured', 'InputPortWidth', '9', 'Indices', '1', 'Position', [920 270 950 290]);

% displays, scopes, logging
B('simulink/User-Defined Functions/MATLAB Function', 'outputs');
set_param([mdl '/outputs'], 'Position', [760 100 880 200]);
setfun('outputs', sprintf(['function [XE_pct, T_C, TJ_C, pH] = f(y)\n' ...
    'XE_pct = 100 * y(8);  T_C = y(9) - 273.15;  TJ_C = y(2) - 273.15;  pH = y(6);\n']));
d = datum(rt, mdl, 'outputs', 'y'); d.Props.Array.Size = '9';
labels = {'Conversion X_E (%)', 'Reactor T (C)', 'Jacket T (C)', 'Apparent pH'};
for i = 1:4
    B('simulink/Sinks/Display', labels{i}, 'Position', [1000 40+60*(i-1) 1100 70+60*(i-1)]);
end
B('simulink/Sinks/Display', 'Estimates [X_E a_rel eta phi F_W k_loss]', 'Position', [330 280 430 400]);
B('simulink/Sinks/Display', 'MVs [T_cmd u_cat F_A]', 'Position', [520 20 620 60]);
B('simulink/Sinks/Scope', 'Conversion scope', 'Position', [1150 40 1180 70]);
B('simulink/Sinks/Scope', 'Temperature scope', 'NumInputPorts', '2', 'Position', [1150 100 1180 150]);
B('simulink/Sinks/To Workspace', 'log XE', 'VariableName', 'XE_log', 'SaveFormat', 'Timeseries', 'Position', [1150 180 1210 200]);
B('simulink/Sinks/To Workspace', 'log u', 'VariableName', 'u_log', 'SaveFormat', 'Timeseries', 'Position', [520 70 580 90]);

% wiring
L = @(a, b) add_line(mdl, a, b, 'autorouting', 'smart');
L('T_sp/1', 'err/1'); L('err/1', 'PI controller/1'); L('PI controller/1', 'T_cmd PI/1'); L('T_sp/1', 'T_cmd PI/2');
L('clock/1', 'sample t/1'); L('sample t/1', 'estimator + NMPC/1'); L('sample y/1', 'estimator + NMPC/2');
L('estimator + NMPC/1', 'split u/1'); L('estimator + NMPC/2', 'stop rule/1');
L('estimator + NMPC/3', 'Estimates [X_E a_rel eta phi F_W k_loss]/1');
L('split u/1', 'select T_cmd/1'); L('controller mode/1', 'select T_cmd/2'); L('T_cmd PI/1', 'select T_cmd/3');
L('select T_cmd/1', 'u/1'); L('split u/2', 'u/2'); L('split u/3', 'u/3');
L('u/1', 'MVs [T_cmd u_cat F_A]/1'); L('u/1', 'log u/1');
for i = 1:5, L(['fault ' fnames{i} '/1'], ['q/' num2str(i)]); end
L('u/1', 'reactor model/2'); L('q/1', 'reactor model/3');
L('reactor model/1', 'state x/1'); L('state x/1', 'reactor model/1');
L('u/1', 'u held/1');
L('state x/1', 'sensors/1'); L('u held/1', 'sensors/2'); L('q/1', 'sensors/3');
L('sensor bias/1', 'sensors/4'); L('load cell mass/1', 'sensors/5');
L('state x/1', 'mass flow/1'); L('u/1', 'mass flow/2'); L('q/1', 'mass flow/3'); L('mass flow/1', 'load cell mass/1');
L('sensors/1', 'sample y/1'); L('sensors/1', 'outputs/1'); L('sensors/1', 'T measured/1'); L('T measured/1', 'err/2');
for i = 1:4, L(['outputs/' num2str(i)], [labels{i} '/1']); end
L('outputs/1', 'Conversion scope/1'); L('outputs/1', 'log XE/1');
L('outputs/2', 'Temperature scope/1'); L('outputs/3', 'Temperature scope/2');
set_param(mdl, 'ZoomFactor', 'FitSystem');
save_system(mdl);
fprintf('built %s.slx with %d blocks\n', mdl, numel(find_system(mdl, 'SearchDepth', 1)) - 1);

function d = datum(rt, mdl, name, var)
ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', [mdl '/' name]);
d = ch.find('-isa', 'Stateflow.Data', 'Name', var);
end

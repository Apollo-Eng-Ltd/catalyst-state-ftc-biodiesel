function P = ftc_params()
%FTC_PARAMS  Parameters of the pilot transesterification reactor (Phases 1-2, Python pipeline).
%   Kinetics: Noureddini and Zhu (1997), Tables 1-2 (k at 50 C, Ea in cal/mol).
%   Fitted to the 2013 yields (vegetable-oil group, hydroxide pathway, k_SE from Eze et al. 2015).
%   Pilot: 600 kg waste soybean oil (SWVO), 4 wt% K2CO3, MeOH:oil 6:1, start 53 C.
%   Units: mol, L, min, K, J.
P.k50    = [0.05 0.11 0.215 1.228 0.242 0.007];      % L mol^-1 min^-1 at 50 C
P.Ea_cal = [13145 9932 19860 14639 6421 9588];       % cal/mol
P.R_cal  = 1.9872;  P.T_ref = 323.15;
P.a      = 0.4824875377967684;   % activity factor (fit)
P.c_th   = 0.04860084968131397;  % inactive base threshold, mol/L (fit)
P.k_H    = 0.008388342934650512; % methoxide -> hydroxide, 1/min (fit)
P.k_SE   = 0.032;  P.Ea_SE = 62000;                  % hydroxide saponification (Eze et al. 2015)
P.GL_sat = 1e-3;   P.w_th = 2e-3;  P.k_N = 50;  P.k_S = 50;
P.dH     = -9.8e3;                                    % J per mol ester (Fjerbaek Sotoft et al. 2010)
P.C_REF  = 0.0365;
P.rho_cp = 1916.4842347185304;  P.UA = 99395.10872851178;      % J L^-1 K^-1, J min^-1 K^-1
P.V_J    = 207.07314318439956;  P.rho_cp_J = 4108.94;  P.F_J = 150;
P.Kc = 6;  P.tauI = 8;                                % PI on the TCU setpoint
P.TJin_lim = [298.15 353.15];
% initial state: TG DG MG E A GL Cat FFA W Soap OH (mol), V (L), T, T_J (K)
P.x0 = [686.4988558352403 0 0 0 4118.993135011442 0 173.65507760211278 ...
        11.635119238584107 6.661115736885928 0 0 823.3082118046866 326.15 326.15]';
P.names = {'TG','DG','MG','E','A','GL','Cat','FFA','W','Soap','OH','V','T','T_J'};
end

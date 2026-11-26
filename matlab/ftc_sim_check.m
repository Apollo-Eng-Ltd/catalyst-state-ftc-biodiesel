%FTC_SIM_CHECK  Run ftc_biodiesel.slx for the two reference batches and compare with Python.
ref_t = [0 5 10 20 30 40 50 60];
ref.normal       = [0 0.7772 0.8886 0.9480 0.9654 0.9701 0.9700 0.9681];
ref.degraded_oil = [0 0.5905 0.7675 0.8684 0.9077 0.9267 0.9363 0.9411];
runs = {'normal', 1.0; 'degraded_oil', 1 - 0.85*0.6};
worst = 0;
for k = 1:2
    assignin('base', 'a_f', runs{k,2}); assignin('base', 't_f', 0);
    tic; out = sim('ftc_biodiesel'); wall = toc;
    xe = interp1(out.XE_log.Time, squeeze(out.XE_log.Data), ref_t) / 100;
    d = 100 * (xe - ref.(runs{k,1}));  worst = max(worst, max(abs(d)));
    fprintf('%-13s max |dX_E| = %.4f pp; 60-min batch simulated in %.2f s (real-time factor %.0f)\n', ...
        runs{k,1}, max(abs(d)), wall, 3600 / wall);
end
assignin('base', 'a_f', 1); assignin('base', 't_f', 20);
fprintf('Simulink vs Python: worst %.4f pp\n', worst);

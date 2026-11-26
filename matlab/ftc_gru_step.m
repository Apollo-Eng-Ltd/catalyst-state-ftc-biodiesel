function [h, y] = ftc_gru_step(xin, h, W)
%FTC_GRU_STEP  One time step of the causal estimator (2-layer GRU, 96 units, tanh head), PyTorch gate order.
%   xin: normalised input (16x1); h: {h1, h2} hidden states; W: weights from ftc_sup_data.json.
%   y: normalised output (19x1). Port checked against PyTorch to 4e-15 (see matlab/reference).
L = {W.gru_weight_ih_l0, W.gru_weight_hh_l0, W.gru_bias_ih_l0, W.gru_bias_hh_l0; ...
     W.gru_weight_ih_l1, W.gru_weight_hh_l1, W.gru_bias_ih_l1, W.gru_bias_hh_l1};
inp = xin;
for l = 1:2
    gi = L{l,1} * inp + L{l,3};
    gh = L{l,2} * h{l} + L{l,4};
    r = 1 ./ (1 + exp(-(gi(1:96) + gh(1:96))));
    z = 1 ./ (1 + exp(-(gi(97:192) + gh(97:192))));
    n = tanh(gi(193:288) + r .* gh(193:288));
    h{l} = (1 - z) .* n + z .* h{l};
    inp = h{l};
end
y = W.head_2_weight * tanh(W.head_0_weight * inp + W.head_0_bias) + W.head_2_bias;
end

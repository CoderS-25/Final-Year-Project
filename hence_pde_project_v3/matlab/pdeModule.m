function [pi_out, mu_out, log_s_out, H_s] = pdeModule(currentSlice, H_prev, weights)
%PDEMODULE PDE module forward pass for one slice -- Eq. (4) exactly.
%
%   [pi_out, mu_out, log_s_out, H_s] = pdeModule(currentSlice, H_prev, weights)
%
%   IMPORTANT CORRECTION vs. an earlier draft of this module: the paper
%   defines exactly ONE auxiliary feature per slice,
%
%       H_s = G(MC(X_s), DSC(H_{s-1}))                              (4)
%
%   and this SAME H_s is used both (a) as the input to the mixed-logistic
%   estimator that predicts slice s's pixel distributions, and (b) as
%   the hidden state fed to DSC for slice s+1. There is no separate
%   "prediction pass" vs. "update pass" with different convolutions --
%   that was based on a different, earlier paper (SR-LVC) and does not
%   match HENCE. MC is a strictly causal masked conv, so MC(X_s)
%   computed once, vectorized over the whole slice, gives the exact
%   same per-pixel values it would if computed incrementally
%   pixel-by-pixel during real autoregressive decoding.
%
%   INPUT
%     currentSlice : H x W (slice X_s), read under teacher forcing
%     H_prev       : H x W x M (auxiliary feature H_{s-1}; M=16).
%                    Use zeros(H,W,M) for the first slice (s=1).
%     weights      : struct from initPDEWeights.m with fields
%         .maskedConv.W (7x7x1x16), .maskedConv.b (1x16)
%         .dsc.depthwise (5x5x16), .dsc.pointwise (16x16), .dsc.pointwise_bias (1x16)
%         .head.W1 (16x16), .head.b1 (1x16), .head.W2 (16x9), .head.b2 (1x9)
%
%   OUTPUT
%     pi_out, mu_out, log_s_out : H x W x 3 (K=3 mixture params for X_s)
%     H_s                       : H x W x 16, pass this in as H_prev for
%                                  the next slice

    mcFeat  = maskedConv(currentSlice, weights.maskedConv.W, weights.maskedConv.b);  % MC(X_s)
    dscFeat = dscBranch(H_prev, weights.dsc);                                        % DSC(H_{s-1})
    H_s     = gate(mcFeat, dscFeat);                                                 % Eq. 4

    [pi_out, mu_out, log_s_out] = estimatorHead(H_s, weights.head);
end

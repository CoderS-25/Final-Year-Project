function debug = pdeModule_debug(currentSlice, H_prev, weights)
%PDEMODULE_DEBUG
% Runs one HENCE PDE slice and returns all intermediate tensors.
%
% This is ONLY for Python <-> MATLAB verification.

    % ------------------------------------------------------------
    % 1. Masked convolution
    % ------------------------------------------------------------
    mcFeat = maskedConv( ...
        currentSlice, ...
        weights.maskedConv.W, ...
        weights.maskedConv.b);

    % ------------------------------------------------------------
    % 2. DSC branch
    % ------------------------------------------------------------
    dscFeat = dscBranch( ...
        H_prev, ...
        weights.dsc);

    % ------------------------------------------------------------
    % 3. Gate
    % ------------------------------------------------------------
    H_s = gate(mcFeat, dscFeat);

    % ------------------------------------------------------------
    % 4. First estimator 1x1 convolution
    % ------------------------------------------------------------
    hidden_pre = conv1x1( ...
        H_s, ...
        weights.head.W1, ...
        weights.head.b1);

    % ------------------------------------------------------------
    % 5. ReLU
    % ------------------------------------------------------------
    hidden = max(hidden_pre, 0);

    % ------------------------------------------------------------
    % 6. Second estimator 1x1 convolution
    % ------------------------------------------------------------
    raw = conv1x1( ...
        hidden, ...
        weights.head.W2, ...
        weights.head.b2);

    % ------------------------------------------------------------
    % 7. Split mixture parameters
    % ------------------------------------------------------------
    pi_raw    = raw(:, :, 1:3);
    mu        = raw(:, :, 4:6);
    log_s_raw = raw(:, :, 7:9);

    % ------------------------------------------------------------
    % 8. Pi clipping + normalization
    % ------------------------------------------------------------
    epsVal = 1e-6;

    pi_clipped = max( ...
        epsVal, ...
        min(1 - epsVal, pi_raw));

    pi = pi_clipped ./ sum(pi_clipped, 3);

    % ------------------------------------------------------------
    % 9. log_s clipping
    % ------------------------------------------------------------
    log_s = max(log_s_raw, -7);

    % ------------------------------------------------------------
    % Save everything
    % ------------------------------------------------------------

    debug.currentSlice = currentSlice;

    debug.mcFeat = mcFeat;
    debug.dscFeat = dscFeat;
    debug.H_s = H_s;

    debug.hidden_pre = hidden_pre;
    debug.hidden = hidden;

    debug.raw = raw;

    debug.pi_raw = pi_raw;
    debug.pi_clipped = pi_clipped;
    debug.mu = mu;
    debug.log_s_raw = log_s_raw;

    debug.pi = pi;
    debug.log_s = log_s;
end

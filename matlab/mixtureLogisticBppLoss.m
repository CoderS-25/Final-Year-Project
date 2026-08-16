function bpp = mixtureLogisticBppLoss(x, pi, mu, log_s, numBins, scaleFactorL)
%MIXTURELOGISTICBPPLOSS MATLAB equivalent of Python BPP loss.

    % x:
    %   H x W x S
    %
    % pi, mu, log_s:
    %   H x W x K x S
    %
    % K = 3

    [H, W, K, S] = size(mu);

    % Expand target x across the mixture dimension.
    % Python: x = x.expand_as(mu)
    x = reshape(x, H, W, 1, S);
    x = repmat(x, 1, 1, K, 1);

    % Quantization half-bin width
    halfBin = scaleFactorL / (numBins - 1);

    centered = x - mu;
    inv_s = exp(-log_s);

    plusIn = inv_s .* (centered + halfBin);
    minIn  = inv_s .* (centered - halfBin);

    % Logistic CDF
    cdfPlus = 1 ./ (1 + exp(-plusIn));
    cdfMin  = 1 ./ (1 + exp(-minIn));

    prob = cdfPlus - cdfMin;

    % Edge probabilities
    logProbEdgeLow  = plusIn - softplus(plusIn);
    logProbEdgeHigh = -softplus(minIn);

    % Middle probability
    logProbMid = log(max(prob, 1e-12));

    % Boundary masks
    isLow  = double(x <= -1 + 1e-3);
    isHigh = double(x >=  1 - 1e-3);
    isMid  = 1.0 - isLow - isHigh;

    logProb = ...
        isLow  .* logProbEdgeLow + ...
        isHigh .* logProbEdgeHigh + ...
        isMid  .* logProbMid;

    % Mixture weights
    logPi = log(max(pi, 1e-12));

    % log-sum-exp over K = 3 mixture components
    mixtureLogProb = logSumExp3(logPi + logProb);

    % Mean negative log likelihood
    nllNats = -mean(mixtureLogProb(:));

    % Convert nats to bits
    bpp = nllNats / log(2.0);
end


function y = softplus(x)
% Numerically stable softplus.

    y = max(x, 0) + log1p(exp(-abs(x)));
end


function y = logSumExp3(x)
% LogSumExp across mixture dimension.

    m = max(x, [], 3);

    y = m + log(sum(exp(x - m), 3));
end
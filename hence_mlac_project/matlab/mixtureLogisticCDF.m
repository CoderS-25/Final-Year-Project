function cdf_val = mixtureLogisticCDF(x, pi_k, mu_k, log_s_k)
%MIXTURELOGISTICCDF Mixture-of-Logistics CDF -- HENCE paper Eq. (11).
%
%   cdf_val = mixtureLogisticCDF(x, pi_k, mu_k, log_s_k)
%
%   Computes:
%       CDF(x) = sum_k  pi_k * sigmoid((x - mu_k) / exp(log_s_k))
%
%   This gives the probability that a pixel value is <= x under the
%   mixture distribution predicted by the Neural PDE for that pixel.
%
%   For the arithmetic coder we evaluate this at half-integer boundaries:
%       P(pixel == v) = CDF(v + 0.5) - CDF(v - 0.5)
%
%   INPUT
%     x        : scalar or array of pixel intensity values to evaluate at
%     pi_k     : 1 x K  mixture weights for ONE pixel (must sum to 1)
%     mu_k     : 1 x K  means for ONE pixel
%     log_s_k  : 1 x K  log-scales for ONE pixel
%
%   OUTPUT
%     cdf_val  : same shape as x, values in [0, 1]

    K = numel(pi_k);
    pi_k    = reshape(pi_k,    1, K);
    mu_k    = reshape(mu_k,    1, K);
    log_s_k = reshape(log_s_k, 1, K);

    x_col = x(:);  % flatten to column for broadcasting

    % Scale argument: (x - mu) / exp(log_s)
    % Shape: numel(x) x K
    scale = exp(log_s_k);                          % 1 x K
    arg   = (x_col - mu_k) ./ scale;              % N x K  (broadcast)

    % Sigmoid: sigma(t) = 1 / (1 + exp(-t))
    % Use numerically stable version to avoid overflow for large |arg|
    sig = 1 ./ (1 + exp(-arg));                    % N x K

    % Weighted sum over K components
    cdf_val = sum(pi_k .* sig, 2);                 % N x 1

    % Clamp to [0,1] to handle floating-point edge cases
    cdf_val = max(0, min(1, cdf_val));

    % Reshape back to match x's shape
    cdf_val = reshape(cdf_val, size(x));

end

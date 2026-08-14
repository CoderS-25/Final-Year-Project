function [pi_out, mu_out, log_s_out] = estimatorHead(H_s, headWeights)
%ESTIMATORHEAD Mixed-logistic estimator.
%
% Input:
%   H_s        : H x W x M
%   headWeights:
%       .W1 : M x M
%       .b1 : 1 x M
%       .W2 : M x 9
%       .b2 : 1 x 9
%
% Output:
%   pi_out    : H x W x 3
%   mu_out    : H x W x 3
%   log_s_out : H x W x 3

    % ---------------------------------------------------------------
    % First 1x1 convolution
    % ---------------------------------------------------------------
    hidden = conv1x1( ...
        H_s, ...
        headWeights.W1, ...
        headWeights.b1);

    % ReLU
    hidden = max(hidden, 0);

    % ---------------------------------------------------------------
    % Second 1x1 convolution
    % ---------------------------------------------------------------
    raw = conv1x1( ...
        hidden, ...
        headWeights.W2, ...
        headWeights.b2);

    % raw is H x W x 9
    %
    % Channels:
    %   1:3 -> mixture weights
    %   4:6 -> means
    %   7:9 -> log scales

    pi_raw    = raw(:, :, 1:3);
    mu_out    = raw(:, :, 4:6);
    log_s_raw = raw(:, :, 7:9);

    % ---------------------------------------------------------------
    % Mixture-weight normalization
    % ---------------------------------------------------------------
    epsVal = 1e-6;

    pi_clipped = max( ...
        epsVal, ...
        min(1 - epsVal, pi_raw));

    pi_out = pi_clipped ./ sum(pi_clipped, 3);

    % ---------------------------------------------------------------
    % Log-scale stability clamp
    % ---------------------------------------------------------------
    log_s_out = max(log_s_raw, -7);

end
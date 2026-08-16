function [pi_q, mu_q, log_s_q] = quantizeParameters(pi_in, mu_in, log_s_in, num_bits)
%QUANTIZEPARAMETERS Quantize mixture-of-logistics parameters to fixed-point.
%
%   [pi_q, mu_q, log_s_q] = quantizeParameters(pi_in, mu_in, log_s_in, num_bits)
%
%   The HENCE paper uses fixed-point quantization on the NN outputs before
%   feeding them into the arithmetic coder. This makes the CDF computation
%   numerically identical at encoder and decoder (critical for losslessness),
%   and mirrors the paper's hardware-friendly LUT design.
%
%   INPUT
%     pi_in    : H x W x K  mixture weights (sum to 1 per pixel)
%     mu_in    : H x W x K  means (in normalized [-1,1] range)
%     log_s_in : H x W x K  log-scales (clamped to >= -7 by PDE module)
%     num_bits : quantization precision (default: 8, giving 256 levels)
%
%   OUTPUT
%     pi_q    : H x W x K  quantized weights (still sum to 1, float64)
%     mu_q    : H x W x K  quantized means
%     log_s_q : H x W x K  quantized log-scales

    if nargin < 4 || isempty(num_bits)
        num_bits = 8;
    end

    levels = 2^num_bits - 1;  % e.g. 255 for 8-bit

    % --- Quantize pi (mixture weights, range [0,1]) ---
    pi_q_int = round(pi_in * levels);
    pi_q_int = max(0, min(levels, pi_q_int));
    % Re-normalize so weights still sum to 1 per pixel
    pi_sum = sum(pi_q_int, ndims(pi_in));
    pi_sum = max(pi_sum, 1);  % avoid divide-by-zero
    pi_q = pi_q_int ./ pi_sum;

    % --- Quantize mu (means, range [-1, 1]) ---
    % Map [-1,1] -> [0, levels], quantize, map back
    mu_shifted = (mu_in + 1) / 2;  % [0, 1]
    mu_int = round(mu_shifted * levels);
    mu_int = max(0, min(levels, mu_int));
    mu_q = (mu_int / levels) * 2 - 1;  % back to [-1, 1]

    % --- Quantize log_s (log-scales, range [-7, 0] typically) ---
    % Clamp to a safe range first (PDE module already clamps >= -7)
    LOG_S_MIN = -7.0;
    LOG_S_MAX =  2.0;
    log_s_clamped = max(LOG_S_MIN, min(LOG_S_MAX, log_s_in));
    % Map to [0, levels], quantize, map back
    log_s_shifted = (log_s_clamped - LOG_S_MIN) / (LOG_S_MAX - LOG_S_MIN);
    log_s_int = round(log_s_shifted * levels);
    log_s_int = max(0, min(levels, log_s_int));
    log_s_q = (log_s_int / levels) * (LOG_S_MAX - LOG_S_MIN) + LOG_S_MIN;

end

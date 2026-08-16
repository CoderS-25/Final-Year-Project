function [cdf_low, cdf_high] = getPixelInterval(sym, pi_k, mu_k, log_s_k, num_symbols, precision)
%GETPIXELINTERVAL Compute the arithmetic coding interval for ONE pixel.
%
%   [cdf_low, cdf_high] = getPixelInterval(sym, pi_k, mu_k, log_s_k,
%                                           num_symbols, precision)
%
%   Instead of building a full frequency table with num_symbols entries
%   (which is wasteful for 16-bit data), this function computes only the
%   two CDF values needed to encode/decode a single symbol:
%
%       cdf_low  = CDF(sym - 0.5)   (scaled to integer)
%       cdf_high = CDF(sym + 0.5)   (scaled to integer)
%
%   The arithmetic coder uses the interval [cdf_low, cdf_high) to encode.
%
%   INPUT
%     sym        : integer pixel value (0-indexed)
%     pi_k       : 1 x K  quantized mixture weights
%     mu_k       : 1 x K  quantized means (in [-1,1])
%     log_s_k    : 1 x K  quantized log-scales
%     num_symbols: alphabet size (256 for 8-bit, 65536 for 16-bit)
%     precision  : integer precision (default: 2^24 = 16777216)
%
%   OUTPUT
%     cdf_low  : uint64, cumulative frequency at (sym - 0.5)
%     cdf_high : uint64, cumulative frequency at (sym + 0.5)

    if nargin < 5 || isempty(num_symbols), num_symbols = 256; end
    if nargin < 6 || isempty(precision),   precision = 2^24;  end

    % Map integer boundaries to the [-1, 1] normalized domain
    low_boundary  = ((sym - 0.5) / (num_symbols - 1)) * 2 - 1;
    high_boundary = ((sym + 0.5) / (num_symbols - 1)) * 2 - 1;

    % Handle edge cases: first and last symbols
    if sym == 0
        cdf_low_f = 0.0;
    else
        cdf_low_f = mixtureLogisticCDF(low_boundary, pi_k, mu_k, log_s_k);
    end

    if sym == num_symbols - 1
        cdf_high_f = 1.0;
    else
        cdf_high_f = mixtureLogisticCDF(high_boundary, pi_k, mu_k, log_s_k);
    end

    % Scale to integer precision
    cdf_low  = uint64(floor(cdf_low_f  * precision));
    cdf_high = uint64(floor(cdf_high_f * precision));

    % Clamp to [0, precision]
    cdf_low  = min(cdf_low,  uint64(precision));
    cdf_high = min(cdf_high, uint64(precision));

end

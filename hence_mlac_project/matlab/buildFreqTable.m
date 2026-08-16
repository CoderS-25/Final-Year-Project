function [freq_table, cdf_table, exceptional_mask] = buildFreqTable(pi_k, mu_k, log_s_k, num_symbols, precision)
%BUILDFREQTABLE Build integer frequency/CDF table for one pixel.
%
%   [freq_table, cdf_table, exceptional_mask] =
%       buildFreqTable(pi_k, mu_k, log_s_k, num_symbols, precision)
%
%   Converts the continuous Mixture-of-Logistics CDF into a discrete
%   integer frequency table that the arithmetic coder can use.
%
%   For each symbol v in [0, num_symbols-1]:
%       freq(v) = round( precision * (CDF(v+0.5) - CDF(v-0.5)) )
%
%   INPUT
%     pi_k       : 1 x K  quantized mixture weights
%     mu_k       : 1 x K  quantized means
%     log_s_k    : 1 x K  quantized log-scales
%     num_symbols: number of discrete intensity levels (e.g. 256 for 8-bit,
%                  65536 for 16-bit). Default: 256.
%     precision  : total integer frequency budget (must be power of 2).
%                  Default: 65536 (16-bit precision, matches HENCE paper).
%
%   OUTPUT
%     freq_table      : 1 x num_symbols integer frequencies (sum = precision)
%     cdf_table       : 1 x (num_symbols+1) cumulative frequencies
%                       cdf_table(1)=0, cdf_table(end)=precision

    if nargin < 4 || isempty(num_symbols)
        num_symbols = 256;
    end
    if nargin < 5 || isempty(precision)
        precision = 65536;  % 2^16
    end

    % Evaluate boundaries: CDF at v-0.5 for v = 0,1,...,num_symbols
    boundaries = (0:num_symbols) - 0.5;           % 1 x (num_symbols+1)
    
    % Map boundaries to the [-1, 1] normalized domain used by the neural network
    boundaries_norm = (boundaries / (num_symbols - 1)) * 2 - 1;

    cdf_vals = mixtureLogisticCDF(boundaries_norm, pi_k, mu_k, log_s_k);

    % Convert to integer CDF scaled to [0, precision]
    cdf_int = round(cdf_vals * precision);
    cdf_int(1)   = 0;          % CDF(-0.5) must be exactly 0
    cdf_int(end) = precision;  % CDF(num_symbols-0.5) must be exactly precision

    % Ensure monotonicity (protect against floating-point noise)
    for v = 2:num_symbols+1
        if cdf_int(v) < cdf_int(v-1)
            cdf_int(v) = cdf_int(v-1);
        end
    end

    % Frequency = difference of consecutive CDF values
    freq_table = diff(cdf_int);   % 1 x num_symbols

    cdf_table = cdf_int;  % 1 x (num_symbols+1)

end

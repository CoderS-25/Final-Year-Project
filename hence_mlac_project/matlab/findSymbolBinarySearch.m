function sym = findSymbolBinarySearch(target, pi_k, mu_k, log_s_k, num_symbols, precision)
%FINDSYMBOLBINARYSEARCH Decoder helper: find the symbol for a given code value.
%
%   sym = findSymbolBinarySearch(target, pi_k, mu_k, log_s_k,
%                                 num_symbols, precision)
%
%   Uses binary search over [0, num_symbols-1] to find the symbol whose
%   CDF interval contains the target value. This avoids building a full
%   frequency table.
%
%   INPUT
%     target     : uint64, the scaled code value to look up
%     pi_k       : 1 x K  quantized mixture weights
%     mu_k       : 1 x K  quantized means
%     log_s_k    : 1 x K  quantized log-scales
%     num_symbols: alphabet size
%     precision  : integer precision
%
%   OUTPUT
%     sym : integer symbol (0-indexed) such that CDF(sym-0.5) <= target < CDF(sym+0.5)

    lo = 0;
    hi = num_symbols - 1;

    while lo < hi
        mid = floor((lo + hi) / 2);

        % Get CDF at (mid + 0.5) — the upper boundary of symbol mid
        mid_boundary = ((mid + 0.5) / (num_symbols - 1)) * 2 - 1;
        cdf_mid = uint64(floor(mixtureLogisticCDF(mid_boundary, pi_k, mu_k, log_s_k) * precision));

        if cdf_mid <= target
            lo = mid + 1;
        else
            hi = mid;
        end
    end

    sym = lo;

end

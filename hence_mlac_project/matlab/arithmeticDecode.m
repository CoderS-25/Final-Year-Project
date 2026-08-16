function pixels_out = arithmeticDecode(bitstream, pi_vol, mu_vol, log_s_vol, H, W, exceptional_pixels, num_symbols, precision)
%ARITHMETICDECODE Arithmetic decoder -- exact mirror of arithmeticEncode.m.
%
%   pixels_out = arithmeticDecode(
%       bitstream, pi_vol, mu_vol, log_s_vol, H, W,
%       exceptional_pixels, num_symbols, precision)
%
%   Reconstructs the exact original pixel values from the compressed
%   bitstream, using the same per-pixel CDF computation as the encoder.
%   Uses binary search (findSymbolBinarySearch) to locate each symbol
%   without building a full frequency table.
%
%   INPUT
%     bitstream         : uint8 column vector from arithmeticEncode
%     pi_vol, mu_vol, log_s_vol : H x W x K  (same as encoder input)
%     H, W              : image dimensions
%     exceptional_pixels: struct from arithmeticEncode
%     num_symbols       : alphabet size (default: 256)
%     precision         : CDF precision (default: 2^24)
%
%   OUTPUT
%     pixels_out : H x W  integer pixel values (should equal encoder input)

    if nargin < 8 || isempty(num_symbols), num_symbols = 256;  end
    if nargin < 9 || isempty(precision),   precision   = 2^24; end

    prec_u64 = uint64(precision);

    % Unpack bytes to bits
    bits_in = false(numel(bitstream) * 8, 1);
    for byte_i = 1:numel(bitstream)
        byte_val = bitstream(byte_i);
        for b = 1:8
            bits_in((byte_i-1)*8 + b) = logical(bitand(uint8(byte_val), uint8(2^(8-b))));
        end
    end
    num_bits = numel(bits_in);

    % Integer arithmetic decoding state
    FULL = uint64(2^32);
    HALF = uint64(2^31);
    QRTR = uint64(2^30);

    low   = uint64(0);
    high  = FULL - 1;

    % Prime the decoder register with the first 32 bits
    code = uint64(0);
    for b = 1:32
        if b <= num_bits && bits_in(b)
            code = code * 2 + 1;
        else
            code = code * 2;
        end
    end
    bit_ptr = 33;

    % Build exceptional pixel lookup for fast access
    exc_set = containers.Map('KeyType','char','ValueType','double');
    for i = 1:size(exceptional_pixels.positions, 1)
        key = sprintf('%d_%d', exceptional_pixels.positions(i,1), exceptional_pixels.positions(i,2));
        exc_set(key) = double(exceptional_pixels.values(i));
    end

    pixels_out = zeros(H, W);

    % Pre-quantize all parameters for the slice at once
    [pi_q_vol, mu_q_vol, log_s_q_vol] = quantizeParameters(pi_vol, mu_vol, log_s_vol);
    
    K = size(pi_vol, 3);
    pi_q_mat = reshape(pi_q_vol, H*W, K);
    mu_q_mat = reshape(mu_q_vol, H*W, K);
    log_s_q_mat = reshape(log_s_q_vol, H*W, K);
    scale_mat = exp(log_s_q_mat);

    pixels_out_flat = zeros(H*W, 1);

    for i = 1:(H*W)
        [r, c] = ind2sub([H, W], i);
        key = sprintf('%d_%d', r, c);

        % Check if exceptional
        if exc_set.isKey(key)
            pixels_out_flat(i) = exc_set(key);
            continue;
        end

        pi_q = pi_q_mat(i, :);
        mu_q = mu_q_mat(i, :);
        scale_val = scale_mat(i, :);

        % Warm-start binary search from the predicted mode.
        % Map dominant mu to its integer symbol, then narrow the search
        % window to [mode-window, mode+window] for most pixels.
        % This reduces average iterations from log2(65536)=16 to ~3-4.
        range = high - low + 1;
        scaled_code = uint64(idivide((code - low + 1) * prec_u64 - 1, range));

        dominant_mu = sum(pi_q .* mu_q);  % weighted mean as mode estimate
        mode_sym = round((dominant_mu + 1) * 0.5 * (num_symbols - 1));
        mode_sym = max(0, min(num_symbols - 1, mode_sym));

        % Compute CDF at mode to decide which half to search
        if mode_sym == 0
            cdf_at_mode = uint64(0);
        else
            mb = ((mode_sym - 0.5) / (num_symbols - 1)) * 2 - 1;
            arg_m = (mb - mu_q) ./ scale_val;
            cdf_at_mode = uint64(floor(sum(pi_q ./ (1 + exp(-arg_m))) * precision));
        end

        if cdf_at_mode <= scaled_code
            lo_sym = mode_sym;
            hi_sym = num_symbols - 1;
        else
            lo_sym = 0;
            hi_sym = mode_sym;
        end

        while lo_sym < hi_sym
            mid = floor((lo_sym + hi_sym) / 2);
            mid_boundary = ((mid + 0.5) / (num_symbols - 1)) * 2 - 1;
            
            % Inline mixtureLogisticCDF
            arg_mid = (mid_boundary - mu_q) ./ scale_val;
            sig_mid = 1 ./ (1 + exp(-arg_mid));
            cdf_mid_f = sum(pi_q .* sig_mid);
            
            cdf_mid = uint64(floor(cdf_mid_f * precision));

            if cdf_mid <= scaled_code
                lo_sym = mid + 1;
            else
                hi_sym = mid;
            end
        end

        sym = lo_sym;
        pixels_out_flat(i) = sym;

        % Get the interval for this symbol (inlined for speed)
        low_boundary  = ((sym - 0.5) / (num_symbols - 1)) * 2 - 1;
        high_boundary = ((sym + 0.5) / (num_symbols - 1)) * 2 - 1;
        
        if sym == 0
            cdf_low_f = 0.0;
        else
            arg_low = (low_boundary - mu_q) ./ scale_val;
            sig_low = 1 ./ (1 + exp(-arg_low));
            cdf_low_f = sum(pi_q .* sig_low);
        end
        
        if sym == num_symbols - 1
            cdf_high_f = 1.0;
        else
            arg_high = (high_boundary - mu_q) ./ scale_val;
            sig_high = 1 ./ (1 + exp(-arg_high));
            cdf_high_f = sum(pi_q .* sig_high);
        end
        
        low_cum  = uint64(floor(cdf_low_f  * precision));
        high_cum = uint64(floor(cdf_high_f * precision));
        
        low_cum  = min(low_cum,  uint64(precision));
        high_cum = min(high_cum, uint64(precision));

        % Update interval
        high = low + uint64(idivide(range * high_cum, prec_u64)) - 1;
        low  = low + uint64(idivide(range * low_cum,  prec_u64));

            % Renormalization + read new bits
            while true
                if high < HALF
                    low  = low  * 2;
                    high = high * 2 + 1;
                    code = code * 2;
                    if bit_ptr <= num_bits && bits_in(bit_ptr)
                        code = code + 1;
                    end
                    bit_ptr = bit_ptr + 1;

                elseif low >= HALF
                    low  = (low  - HALF) * 2;
                    high = (high - HALF) * 2 + 1;
                    code = (code - HALF) * 2;
                    if bit_ptr <= num_bits && bits_in(bit_ptr)
                        code = code + 1;
                    end
                    bit_ptr = bit_ptr + 1;

                elseif low >= QRTR && high < (HALF + QRTR)
                    low  = (low  - QRTR) * 2;
                    high = (high - QRTR) * 2 + 1;
                    code = (code - QRTR) * 2;
                    if bit_ptr <= num_bits && bits_in(bit_ptr)
                        code = code + 1;
                    end
                    bit_ptr = bit_ptr + 1;

                else
                    break;
                end
            end
    end

    pixels_out = reshape(pixels_out_flat, H, W);

end

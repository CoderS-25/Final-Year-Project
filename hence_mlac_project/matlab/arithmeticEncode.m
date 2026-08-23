function [bitstream, exceptional_pixels] = arithmeticEncode(pixels, pi_vol, mu_vol, log_s_vol, num_symbols, precision)
%ARITHMETICENCODE Arithmetic encoder using per-pixel mixture-logistic CDFs.
%
%   [bitstream, exceptional_pixels] = arithmeticEncode(
%       pixels, pi_vol, mu_vol, log_s_vol, num_symbols, precision)
%
%   Encodes a 2D slice of pixel values into a compact binary bitstream using
%   integer arithmetic coding with per-pixel probability intervals derived
%   from the HENCE Neural PDE outputs.
%
%   Uses getPixelInterval to compute CDF intervals directly per pixel,
%   avoiding the need to build a massive frequency table (critical for
%   16-bit data where num_symbols = 65536).
%
%   INPUT
%     pixels   : H x W  integer pixel values in [0, num_symbols-1]
%     pi_vol   : H x W x K  mixture weights for this slice
%     mu_vol   : H x W x K  mixture means for this slice
%     log_s_vol: H x W x K  mixture log-scales for this slice
%     num_symbols: symbol alphabet size (default: 256)
%     precision  : CDF precision budget (default: 2^24 = 16777216)
%
%   OUTPUT
%     bitstream         : uint8 column vector of compressed bytes
%     exceptional_pixels: struct with fields:
%         .positions  : Nx2 array of [row, col] for each exceptional pixel
%         .values     : Nx1 raw pixel values stored verbatim
%         .total_bits_used : number of bits in the bitstream

    if nargin < 5 || isempty(num_symbols), num_symbols = 256; end
    if nargin < 6 || isempty(precision),   precision   = 2^24; end

    [H, W] = size(pixels);
    pixels = double(pixels);

    % Integer arithmetic coding state (32-bit registers)
    FULL  = uint64(2^32);
    HALF  = uint64(2^31);
    QRTR  = uint64(2^30);

    low    = uint64(0);
    high   = FULL - 1;
    scale  = uint64(0);       % pending bits counter (E3 scaling)

    bits_out = false(1, H * W * 20);  % pre-allocate
    bit_idx  = 0;

    exceptional_pos = zeros(H*W, 2, 'uint32');
    exceptional_val = zeros(H*W, 1, 'uint16');
    exc_count = 0;

    prec_u64 = uint64(precision);

    % Pre-quantize all parameters for the slice at once
    [pi_q_vol, mu_q_vol, log_s_q_vol] = quantizeParameters(pi_vol, mu_vol, log_s_vol);
    
    K = size(pi_vol, 3);
    pi_q_mat = reshape(pi_q_vol, H*W, K);
    mu_q_mat = reshape(mu_q_vol, H*W, K);
    log_s_q_mat = reshape(log_s_q_vol, H*W, K);
    
    pixels_flat = double(pixels(:));
    
    % Vectorized CDF interval computation for all pixels
    low_boundary  = ((pixels_flat - 0.5) / (num_symbols - 1)) * 2 - 1;
    high_boundary = ((pixels_flat + 0.5) / (num_symbols - 1)) * 2 - 1;
    
    scale_mat = exp(log_s_q_mat);
    
    arg_low = (repmat(low_boundary, 1, K) - mu_q_mat) ./ scale_mat;
    sig_low = 1 ./ (1 + exp(-arg_low));
    cdf_low_f = sum(pi_q_mat .* sig_low, 2);
    
    arg_high = (repmat(high_boundary, 1, K) - mu_q_mat) ./ scale_mat;
    sig_high = 1 ./ (1 + exp(-arg_high));
    cdf_high_f = sum(pi_q_mat .* sig_high, 2);
    
    % Edge cases: only the absolute boundary symbols get the infinite-tail treatment.
    % Symbol 0 is the only symbol with no left neighbour, so its cdf_low = CDF(-inf) = 0.
    % Symbol 65535 is the only symbol with no right neighbour, so its cdf_high = CDF(+inf) = 1.
    % Applying this to wider ranges (e.g., 0-32) creates overlapping intervals and breaks losslessness.
    idx_0 = (pixels_flat == 0);
    cdf_low_f(idx_0) = 0.0;
    idx_max = (pixels_flat == num_symbols - 1);
    cdf_high_f(idx_max) = 1.0;
    
    cdf_low_all  = uint64(floor(cdf_low_f  * precision));
    cdf_high_all = uint64(floor(cdf_high_f * precision));
    
    cdf_low_all  = min(cdf_low_all,  uint64(precision));
    cdf_high_all = min(cdf_high_all, uint64(precision));

    for i = 1:(H*W)
        sym = pixels_flat(i);
        low_cum  = cdf_low_all(i);
        high_cum = cdf_high_all(i);

        % Check for degenerate interval (should be handled above, but just in case)
        if high_cum <= low_cum
            exc_count = exc_count + 1;
            [r, c] = ind2sub([H, W], i);
            exceptional_pos(exc_count, :) = [r, c];
            exceptional_val(exc_count)    = uint16(sym);
            continue;
        end

            % Arithmetic coding interval update
            range = high - low + 1;
            high  = low + uint64(idivide(range * high_cum, prec_u64)) - 1;
            low   = low + uint64(idivide(range * low_cum,  prec_u64));

            % Bit output + renormalization loop
            while true
                if high < HALF
                    bit_idx = bit_idx + 1;
                    bits_out(bit_idx) = false;
                    for i = 1:double(scale)
                        bit_idx = bit_idx + 1;
                        bits_out(bit_idx) = true;
                    end
                    scale = uint64(0);
                    low   = low  * 2;
                    high  = high * 2 + 1;

                elseif low >= HALF
                    bit_idx = bit_idx + 1;
                    bits_out(bit_idx) = true;
                    for i = 1:double(scale)
                        bit_idx = bit_idx + 1;
                        bits_out(bit_idx) = false;
                    end
                    scale = uint64(0);
                    low   = (low  - HALF) * 2;
                    high  = (high - HALF) * 2 + 1;

                elseif low >= QRTR && high < (HALF + QRTR)
                    scale = scale + 1;
                    low   = (low  - QRTR) * 2;
                    high  = (high - QRTR) * 2 + 1;

                else
                    break;
                end
            end
        end

    % Flush remaining bits
    scale = scale + 1;
    bit_idx = bit_idx + 1;
    if low < QRTR
        bits_out(bit_idx) = false;
        for i = 1:double(scale)
            bit_idx = bit_idx + 1;
            bits_out(bit_idx) = true;
        end
    else
        bits_out(bit_idx) = true;
        for i = 1:double(scale)
            bit_idx = bit_idx + 1;
            bits_out(bit_idx) = false;
        end
    end

    % Pack bits into bytes
    bits_out = bits_out(1:bit_idx);
    pad = mod(8 - mod(numel(bits_out), 8), 8);
    bits_out = [bits_out, false(1, pad)];
    bit_matrix = uint8(reshape(bits_out, 8, [])');
    powers = uint8([128 64 32 16 8 4 2 1]);
    bitstream = uint8(sum(bit_matrix .* repmat(powers, size(bit_matrix,1), 1), 2));

    % Trim exceptional arrays
    exceptional_pixels.positions = exceptional_pos(1:exc_count, :);
    exceptional_pixels.values    = exceptional_val(1:exc_count);
    exceptional_pixels.total_bits_used = bit_idx;

end

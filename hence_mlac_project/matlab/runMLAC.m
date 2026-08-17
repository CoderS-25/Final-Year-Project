function results = runMLAC(pde_mat_path, raw_volume_path, num_bits_pixel)
%RUNMLAC Master MLAC pipeline. Loads Track 1 outputs and runs full
%        encode -> decode -> losslessness check for each slice.
%
%   results = runMLAC(pde_mat_path, raw_volume_path, num_bits_pixel)
%
%   INPUT
%     pde_mat_path    : path to patient_XX_pde_out.mat (from Track 1)
%                       Must contain: pi_vol, mu_vol, log_s_vol
%                       Shape: H x W x K x numSlices
%     raw_volume_path : path to the original _norm.mat patient volume
%                       Used to get the real pixel values to compress.
%     num_bits_pixel  : 8 for 8-bit, 16 for 16-bit images (default: 8)
%
%   OUTPUT
%     results : struct with fields
%         .bpp             : bits per pixel achieved
%         .total_bits      : total compressed bits
%         .lossless        : true if all slices decoded perfectly
%         .exceptional_rate: fraction of pixels handled as exceptional

    if nargin < 3 || isempty(num_bits_pixel)
        num_bits_pixel = 8;
    end

    num_symbols = 2^num_bits_pixel;

    % ---------------------------------------------------------------
    % 1. Load Track 1 PDE outputs
    % ---------------------------------------------------------------
    fprintf('Loading PDE outputs from %s...\n', pde_mat_path);
    pde = load(pde_mat_path);

    pi_vol    = pde.pi_vol;    % H x W x K x S
    mu_vol    = pde.mu_vol;
    log_s_vol = pde.log_s_vol;

    [H, W, K, numSlices] = size(pi_vol);
    fprintf('Volume: %d x %d, K=%d, %d slices\n', H, W, K, numSlices);

    % ---------------------------------------------------------------
    % 2. Load original pixel values
    % ---------------------------------------------------------------
    fprintf('Loading raw MRI volume from %s...\n', raw_volume_path);
    raw_data = load(raw_volume_path);
    fields   = fieldnames(raw_data);
    volume   = double(raw_data.(fields{1}));  % H x W x S

    % [FIXED ALPHABET LOGIC] Evaluate raw 16-bit pixels directly over the full 65536 alphabet
    num_symbols = 65536;
    volume_int = uint16(volume);

    % ---------------------------------------------------------------
    % 3. Encode + Decode each slice
    % ---------------------------------------------------------------
    total_bits   = 0;
    total_pixels = 0;
    total_exc    = 0;
    all_lossless = true;

    % Enable parallel pool if not already running (optional, parfor will handle it)
    % Initialize arrays to hold parallel results
    slice_bits_all = zeros(numSlices, 1);
    slice_lossless_all = false(numSlices, 1);

    parfor s = 1:numSlices
        fprintf('  Slice %d/%d ...\n', s, numSlices);

        pi_s    = pi_vol(:, :, :, s);    % H x W x K
        mu_s    = mu_vol(:, :, :, s);
        log_s_s = log_s_vol(:, :, :, s);
        pixels_s = volume_int(:, :, s);  % H x W

        % Encode
        [bitstream, exc] = arithmeticEncode(pixels_s, pi_s, mu_s, log_s_s, num_symbols);

        slice_bits = exc.total_bits_used + numel(exc.values) * num_bits_pixel;
        slice_bits_all(s) = slice_bits;
        
        total_pixels = total_pixels + H * W;
        total_exc    = total_exc + numel(exc.values);

        % Decode
        pixels_recon = arithmeticDecode(bitstream, pi_s, mu_s, log_s_s, H, W, exc, num_symbols);

        % Losslessness check
        diff_map = abs(double(pixels_s) - double(pixels_recon));
        if max(diff_map(:)) == 0
            fprintf('  Slice %d LOSSLESS ✓  |  bits = %d\n', s, slice_bits);
            slice_lossless_all(s) = true;
        else
            fprintf('  Slice %d LOSSY ✗  |  max_error = %d\n', s, max(diff_map(:)));
        end
    end
    
    total_bits = sum(slice_bits_all);
    all_lossless = all(slice_lossless_all);

    % ---------------------------------------------------------------
    % 4. Summary
    % ---------------------------------------------------------------
    bpp = total_bits / total_pixels;
    exc_rate = total_exc / total_pixels;

    fprintf('\n========================================\n');
    fprintf('MLAC Results:\n');
    fprintf('  Total pixels   : %d\n', total_pixels);
    fprintf('  Total bits     : %d\n', total_bits);
    fprintf('  BPP achieved   : %.4f bpp\n', bpp);
    fprintf('  Baseline (raw) : %.1f bpp\n', double(num_bits_pixel));
    fprintf('  Compression    : %.2fx\n',  double(num_bits_pixel) / bpp);
    fprintf('  Exceptional    : %.2f%%\n', exc_rate * 100);
    fprintf('  Lossless       : %s\n', mat2str(all_lossless));
    fprintf('========================================\n');

    results.bpp              = bpp;
    results.total_bits       = total_bits;
    results.lossless         = all_lossless;
    results.exceptional_rate = exc_rate;
    results.compression_ratio = double(num_bits_pixel) / bpp;
end

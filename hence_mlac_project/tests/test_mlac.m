%TEST_MLAC Unit test for the MLAC pipeline on a tiny 4x4 synthetic image.
%
%   Run this BEFORE touching any real MRI data.
%   The test passes only when decoded output == original input (lossless).

fprintf('==============================================\n');
fprintf('MLAC Unit Test -- 4x4 Synthetic Image\n');
fprintf('==============================================\n');

% Add parent matlab folder to path
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'matlab'));

% ---------------------------------------------------------------
% 1. Create a tiny known image
% ---------------------------------------------------------------
H = 4; W = 4;
num_symbols = 256;

pixels = uint16([
    10,  50, 120, 200;
    30,  80, 140, 220;
    60, 100, 160, 240;
    90, 130, 180, 255
]);

fprintf('Original pixels:\n');
disp(double(pixels));

% ---------------------------------------------------------------
% 2. Create synthetic pi/mu/log_s
% ---------------------------------------------------------------
K = 3;
pi_vol    = ones(H, W, K) / K;
mu_vol    = zeros(H, W, K);
log_s_vol = -2 * ones(H, W, K);

for r = 1:H
    for c = 1:W
        pix_norm = (double(pixels(r,c)) / (num_symbols-1)) * 2 - 1;
        mu_vol(r, c, :) = pix_norm + [-0.1, 0, 0.1];
    end
end

% ---------------------------------------------------------------
% 3. Test 1: CDF sanity check
% ---------------------------------------------------------------
fprintf('\n--- Test 1: CDF sanity ---\n');
pi_k = [1/3, 1/3, 1/3];
mu_k = [0.0, 0.0, 0.0];
log_s_k = [-2, -2, -2];

cdf_at_minus1 = mixtureLogisticCDF(-1.5, pi_k, mu_k, log_s_k);
cdf_at_plus1  = mixtureLogisticCDF( 1.5, pi_k, mu_k, log_s_k);

assert(cdf_at_minus1 >= 0 && cdf_at_minus1 <= 0.1, 'CDF(-1.5) should be near 0');
assert(cdf_at_plus1  >= 0.9 && cdf_at_plus1 <= 1.0, 'CDF(+1.5) should be near 1');
fprintf('CDF(-1.5) = %.4f  (expected ~0)\n', cdf_at_minus1);
fprintf('CDF(+1.5) = %.4f  (expected ~1)\n', cdf_at_plus1);
fprintf('PASSED\n');

% ---------------------------------------------------------------
% 4. Test 2: Per-pixel interval sanity
% ---------------------------------------------------------------
fprintf('\n--- Test 2: Per-pixel interval sanity ---\n');
precision = 2^24;
[pi_q, mu_q, log_s_q] = quantizeParameters(pi_k, mu_k, log_s_k);
[cdf_lo, cdf_hi] = getPixelInterval(128, pi_q, mu_q, log_s_q, num_symbols, precision);
fprintf('Interval for sym=128: [%d, %d], width=%d\n', cdf_lo, cdf_hi, cdf_hi - cdf_lo);
assert(cdf_hi > cdf_lo, 'Interval must have positive width');
fprintf('PASSED\n');

% ---------------------------------------------------------------
% 5. Test 3: Full encode -> decode round-trip (THE MAIN TEST)
% ---------------------------------------------------------------
fprintf('\n--- Test 3: Encode -> Decode lossless round-trip ---\n');

[bitstream, exc] = arithmeticEncode(pixels, pi_vol, mu_vol, log_s_vol, num_symbols);

fprintf('Compressed to %d bytes (%d bits)\n', numel(bitstream), exc.total_bits_used);
fprintf('Exceptional pixels : %d / %d\n', numel(exc.values), H*W);

pixels_recon = arithmeticDecode(bitstream, pi_vol, mu_vol, log_s_vol, H, W, exc, num_symbols);

fprintf('\nDecoded pixels:\n');
disp(pixels_recon);

diff_map = abs(double(pixels) - double(pixels_recon));
max_error = max(diff_map(:));

if max_error == 0
    fprintf('LOSSLESS ROUND-TRIP PASSED! max_error = 0\n');
else
    fprintf('LOSSLESS TEST FAILED! max_error = %d\n', max_error);
    fprintf('Diff map:\n');
    disp(diff_map);
end

% ---------------------------------------------------------------
% 6. BPP check
% ---------------------------------------------------------------
fprintf('\n--- Test 4: BPP < 8 check ---\n');
raw_bits = H * W * 8;
num_bits_pixel = 8;
comp_bits = exc.total_bits_used + numel(exc.values) * num_bits_pixel;
bpp = comp_bits / (H * W);

fprintf('Raw bits      : %d  (%.2f bpp)\n', raw_bits, 8.0);
fprintf('Compressed    : %d  (%.4f bpp)\n', comp_bits, bpp);

if bpp < 8
    fprintf('BPP CHECK PASSED! %.4f < 8.0\n', bpp);
else
    fprintf('BPP >= 8 for this tiny test (normal for 4x4, check on real data)\n');
end

fprintf('\n==============================================\n');
if max_error == 0
    fprintf('ALL TESTS PASSED. MLAC pipeline is ready for Patient 27.\n');
else
    fprintf('TESTS FAILED. Fix encoder/decoder before proceeding.\n');
end
fprintf('==============================================\n');

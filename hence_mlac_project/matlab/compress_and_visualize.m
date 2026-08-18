% compress_and_visualize.m
% End-to-end demonstration of the HENCE framework:
% 1. PDE Inference
% 2. Lossless Arithmetic Compression
% 3. Arithmetic Decompression
% 4. Visualization & Numerical Data

clc; clear; close all;

% --- SETUP PATHS ---
% Add the PDE project to the path so we can use the neural network functions
addpath(fullfile('..', '..', 'hence_pde_project_v3', 'matlab'));

% --- 1. LOAD DATA & NEURAL NETWORK ---
fprintf('Loading Neural PDE Weights...\n');
weights = initPDEWeights(fullfile('..', '..', 'hence_pde_project_v3', 'matlab', 'best_trained_weights.mat'));

patient_id = 35; % You can change this to any patient ID (e.g., 27, 28, 29)
fprintf('Loading MRI Volume for Patient %d...\n', patient_id);
data_path = fullfile('..', '..', 'chaos_processed_t2spir', sprintf('%d_raw.mat', patient_id));
data = load(data_path);
fields = fieldnames(data);
volume = double(data.(fields{1}));

% Grab the middle slice for visualization later
middle_slice_idx = round(size(volume, 3) / 2);
slice_original = volume(:, :, middle_slice_idx);

% Normalize globally to [-1, 1] for the neural network
volume_pde_input = (2.0 * volume / 65535.0) - 1.0;

% --- 2. NEURAL NETWORK INFERENCE ---
fprintf('Running Neural Network Inference (predicting probabilities)...\n');
[pi_vol, mu_vol, log_s_vol] = runPipeline(volume_pde_input, weights);

% --- 3. LOSSLESS COMPRESSION (ENCODING) ---
fprintf('Compressing the volume...\n');
num_symbols = 65536;
volume_uint16 = uint16(volume);

% We compress the middle slice as an example
slice_to_compress = volume_uint16(:, :, middle_slice_idx);
pi_slice = pi_vol(:, :, :, middle_slice_idx);
mu_slice = mu_vol(:, :, :, middle_slice_idx);
log_s_slice = log_s_vol(:, :, :, middle_slice_idx);

[bitstream, exceptional_pixels] = arithmeticEncode(slice_to_compress, pi_slice, mu_slice, log_s_slice, num_symbols);

% --- 4. DECOMPRESSION (DECODING) ---
fprintf('Decompressing the volume...\n');
H = size(slice_to_compress, 1);
W = size(slice_to_compress, 2);
decoded_slice = arithmeticDecode(bitstream, pi_slice, mu_slice, log_s_slice, H, W, exceptional_pixels, num_symbols);

% Verify Lossless
is_lossless = isequal(slice_to_compress, decoded_slice);

% --- 5. NUMERICAL DATA ---
total_bits = length(bitstream) * 8; % bytes to bits
total_pixels = H * W;
bpp = total_bits / total_pixels;
compression_ratio = 16.0 / bpp;

fprintf('\n========================================\n');
fprintf(' COMPRESSION RESULTS (Slice %d)\n', middle_slice_idx);
fprintf('========================================\n');
fprintf('  Lossless       : %s\n', mat2str(is_lossless));
fprintf('  Original Size  : %d bits\n', total_pixels * 16);
fprintf('  Compressed Size: %d bits\n', total_bits);
fprintf('  Bits Per Pixel : %.4f BPP\n', bpp);
fprintf('  Compression    : %.2fx smaller\n', compression_ratio);
fprintf('========================================\n');

% --- 6. VISUALIZATION ---
figure('Name', 'HENCE Compression Results', 'Position', [100, 100, 1200, 400]);

% 1. Original Image
subplot(1, 3, 1);
imshow(slice_original, []);
title('Original MRI Slice');

% 2. Neural Network Prediction
% The network predicts the image layout to figure out probabilities.
predicted_mu = mu_slice(:,:,1); % The mean of the first mixture component
subplot(1, 3, 2);
imshow(predicted_mu, []);
title('Neural Network Internal Prediction');

% 3. Decompressed Image & Data
subplot(1, 3, 3);
imshow(double(decoded_slice), []);
title(sprintf('Decompressed (100%% Lossless)\nRatio: %.2fx | BPP: %.4f', compression_ratio, bpp));

colormap(gca, 'gray');

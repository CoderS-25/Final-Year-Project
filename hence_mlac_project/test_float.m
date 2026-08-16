addpath('matlab');
% Load the .npy file to compare with _norm.mat
% We need to use Python or read the npy format manually

% The .npy header is at least 10 bytes + magic
% Let's just check statistics of the .mat file vs what we expect

raw_path = 'C:\Users\user\Documents\Final Year Project\chaos_processed_t2spir\27_norm.mat';
raw = load(raw_path);
fields = fieldnames(raw);
vol_mat = double(raw.(fields{1}));  % H x W x S
fprintf('=== 27_norm.mat shape: [%s] ===\n', num2str(size(vol_mat)));

% The PDE output was generated on a specific dataset
% Let's check what the PDE distribution looks like for background vs tissue
pde_path = 'C:\Users\user\Documents\Final Year Project\hence_pde_project_v3\pde_outputs\patient_27_pde_out.mat';
pde = load(pde_path);

% Overall statistics of mu (middle component, most dominant)
mu2 = pde.mu_vol(:,:,2,:);
fprintf('\nPDE mu_k(2) stats:\n');
fprintf('  range: [%.4f, %.4f]\n', min(mu2(:)), max(mu2(:)));
fprintf('  mean: %.4f, median: %.4f\n', mean(mu2(:)), median(mu2(:)));

% Stats of actual pixels
fprintf('\nActual pixel stats:\n');
fprintf('  range: [%.4f, %.4f]\n', min(vol_mat(:)), max(vol_mat(:)));
fprintf('  mean: %.4f, median: %.4f\n', mean(vol_mat(:)), median(vol_mat(:)));

% For tissue pixels only (>-0.5 threshold)
tissue_mask = vol_mat > -0.5;
fprintf('\nTissue pixels (>-0.5): %d / %d (%.1f%%)\n', sum(tissue_mask(:)), numel(tissue_mask), 100*mean(tissue_mask(:)));

mu2_flat = mu2(:);
vol_flat = vol_mat(:);

% Compare PDE prediction vs actual for tissue pixels only
tissue_idx = vol_flat > -0.5;
tissue_err = abs(mu2_flat(tissue_idx) - vol_flat(tissue_idx));
fprintf('  Mean PDE error for tissue pixels: %.4f\n', mean(tissue_err));

% For background pixels
bg_idx = vol_flat <= -0.5;
bg_err = abs(mu2_flat(bg_idx) - vol_flat(bg_idx));
fprintf('  Mean PDE error for background pixels: %.4f\n', mean(bg_err));

% Check if the PDE predictions would actually compress tissue pixels
precision = 2^24;
num_symbols = 256;
tissue_pi = pde.pi_vol(64,130,2,15);
tissue_mu = pde.mu_vol(64,130,2,15);
tissue_ls = pde.log_s_vol(64,130,2,15);
tissue_val = vol_mat(64,130,15);

fprintf('\nTissue pixel (64,130,15): val=%.4f, mu=%.4f, diff=%.4f\n', tissue_val, tissue_mu, abs(tissue_val-tissue_mu));

% Map to 8-bit integer
sym_8bit = round((tissue_val - min(vol_mat(:))) / (max(vol_mat(:)) - min(vol_mat(:))) * (num_symbols-1));
fprintf('8-bit symbol: %d\n', sym_8bit);
sym_norm = (sym_8bit / (num_symbols-1)) * 2 - 1;
lo = (sym_norm - 0.5/(num_symbols-1)*2);
hi = (sym_norm + 0.5/(num_symbols-1)*2);
scale = exp(tissue_ls);
sig_lo = 1/(1+exp(-(lo - tissue_mu)/scale));
sig_hi = 1/(1+exp(-(hi - tissue_mu)/scale));
freq = round((sig_hi - sig_lo) * precision);
fprintf('8-bit CDF interval width (precision=2^24): %d\n', freq);

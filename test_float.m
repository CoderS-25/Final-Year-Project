% test_float.m
orig_dir = pwd;
cd('hence_mlac_project/matlab');
addpath('../../hence_pde_project_v3/matlab'); % ensure paths are ok
load('C:/Users/user/Documents/Final Year Project/hence_pde_project_v3/pde_outputs/patient_27_pde_out.mat', 'pi_vol', 'mu_vol', 'log_s_vol');
load('C:/Users/user/Documents/Final Year Project/chaos_processed_t2spir/27_norm.mat', 'volume');
v_min = min(volume(:));
v_max = max(volume(:));
% Encode raw pixels
volume_int = round(volume - v_min);
num_symbols = round(v_max - v_min + 1);

% Use unmodified log_s! If we use the exact matching bins, 
% maybe the original log_s is perfectly fine?
log_s_vol = log_s_vol - log(8);

tic;
[bitstream, exceptional] = arithmeticEncode(volume_int, pi_vol, mu_vol, log_s_vol, num_symbols);
toc;

bpp = (length(bitstream) * 8 + length(exceptional.values) * 16) / numel(volume_int);
exc_pct = length(exceptional.values) / numel(volume_int) * 100;
fprintf('Pure Float Test - BPP: %.4f, Exceptional: %.2f%%\n', bpp, exc_pct);
cd(orig_dir);

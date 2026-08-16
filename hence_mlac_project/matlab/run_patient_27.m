% RUN_PATIENT_27
% This script runs the entire MLAC compression and decompression pipeline
% on Patient 27 using the Neural PDE outputs from Track 1.

fprintf('==============================================\n');
fprintf('Running MLAC on Patient 27\n');
fprintf('==============================================\n');

% Paths to the data (using mfilename to ensure it works from any folder)
script_dir = fileparts(mfilename('fullpath'));
pde_mat_path = fullfile(script_dir, '..', '..', 'hence_pde_project_v3', 'pde_outputs', 'patient_27_pde_out.mat');
raw_volume_path = fullfile(script_dir, '..', '..', 'chaos_processed_t2spir', '27_norm.mat');

% 16-bit depth for MRI data
num_bits_pixel = 16; 

% Run the master MLAC script
results = runMLAC(pde_mat_path, raw_volume_path, num_bits_pixel);

fprintf('\nDone! Check the results above.\n');

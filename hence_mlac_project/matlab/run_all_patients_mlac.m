% RUN_ALL_PATIENTS_MLAC
% Runs the full HENCE MLAC pipeline on all 6 validation patients
% and reports per-patient BPP, compression ratio, and losslessness.
%
% Prerequisites:
%   1. PDE outputs for all patients must be regenerated with [-1,1] normalization.
%      Run run_all_patients.m in hence_pde_project_v3/matlab/ first.
%   2. Run this script from inside: hence_mlac_project/matlab/

script_dir = fileparts(mfilename('fullpath'));

pde_dir = fullfile(script_dir, '..', '..', 'hence_pde_project_v3', 'pde_outputs');
raw_dir = fullfile(script_dir, '..', '..', 'chaos_processed_t2spir');

% Validation patient IDs (same split as training)
patient_ids = [27, 28, 29, 30, 35, 40];
num_bits_pixel = 16;  % CHAOS T2SPIR is 16-bit

fprintf('=======================================================\n');
fprintf(' HENCE MLAC -- Full Validation Set Results\n');
fprintf('=======================================================\n');
fprintf('%-10s  %-10s  %-10s  %-10s  %-10s\n', ...
    'Patient', 'BPP', 'Ratio', 'Exc%', 'Lossless');
fprintf('%s\n', repmat('-', 1, 58));

% Storage for summary
bpp_list       = zeros(numel(patient_ids), 1);
ratio_list     = zeros(numel(patient_ids), 1);
lossless_list  = false(numel(patient_ids), 1);
exc_list       = zeros(numel(patient_ids), 1);

for k = 1:numel(patient_ids)
    pid = patient_ids(k);

    pde_path = fullfile(pde_dir, sprintf('patient_%d_pde_out.mat', pid));
    raw_path = fullfile(raw_dir,  sprintf('%d_raw.mat', pid));

    if ~exist(pde_path, 'file')
        fprintf('%-10d  SKIPPED (PDE output not found: %s)\n', pid, pde_path);
        continue;
    end
    if ~exist(raw_path, 'file')
        fprintf('%-10d  SKIPPED (raw volume not found: %s)\n', pid, raw_path);
        continue;
    end

    fprintf('\nRunning Patient %d...\n', pid);

    try
        results = runMLAC(pde_path, raw_path, num_bits_pixel);

        bpp_list(k)      = results.bpp;
        ratio_list(k)    = results.compression_ratio;
        lossless_list(k) = results.lossless;
        exc_list(k)      = results.exceptional_rate * 100;

        fprintf('%-10d  %-10.4f  %-10.2fx  %-10.2f  %-10s\n', ...
            pid, results.bpp, results.compression_ratio, ...
            results.exceptional_rate * 100, ...
            char('YES' * results.lossless + 'NO ' * ~results.lossless));
    catch ME
        fprintf('%-10d  ERROR: %s\n', pid, ME.message);
    end
end

% Summary
valid = bpp_list > 0;
fprintf('\n%s\n', repmat('=', 1, 58));
fprintf(' SUMMARY\n');
fprintf('%s\n', repmat('=', 1, 58));
fprintf('Patients processed  : %d / %d\n', sum(valid), numel(patient_ids));
fprintf('All lossless        : %s\n',  mat2str(all(lossless_list(valid))));
fprintf('Mean BPP            : %.4f bpp\n', mean(bpp_list(valid)));
fprintf('Std  BPP            : %.4f bpp\n', std(bpp_list(valid)));
fprintf('Min  BPP            : %.4f bpp  (Patient %d)\n', ...
    min(bpp_list(valid)), patient_ids(find(bpp_list == min(bpp_list(valid)), 1)));
fprintf('Max  BPP            : %.4f bpp  (Patient %d)\n', ...
    max(bpp_list(valid)), patient_ids(find(bpp_list == max(bpp_list(valid)), 1)));
fprintf('Mean compression    : %.2fx  vs raw 16-bit\n', mean(ratio_list(valid)));
fprintf('Mean exceptional %%  : %.3f%%\n', mean(exc_list(valid)));
fprintf('%s\n', repmat('=', 1, 58));

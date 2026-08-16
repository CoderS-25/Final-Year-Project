% run_all_patients.m
% This script runs the Neural PDE pipeline on an entire list of patients
% and saves the output matrices for Track 2 (Arithmetic Coding).

% 1. Setup
weights = initPDEWeights('best_trained_weights.mat');
data_dir = '../../chaos_processed_t2spir/';
output_dir = '../pde_outputs/';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 2. Select Patients
% You can put a single patient here like [30], or all validation patients!
patients_to_run = [27, 28, 29, 30, 35, 40]; 

fprintf('\nStarting batch processing for %d patients...\n', length(patients_to_run));

% 3. Loop through and run
for i = 1:length(patients_to_run)
    pid = patients_to_run(i);
    fprintf('\n=======================================\n');
    fprintf('Processing Patient %d...\n', pid);
    
    % Load volume
    in_path = fullfile(data_dir, sprintf('%d_norm.mat', pid));
    if ~exist(in_path, 'file')
        warning('Patient %d not found at %s. Skipping...', pid, in_path);
        continue;
    end
    
    data = load(in_path);
    fields = fieldnames(data);
    volume = double(data.(fields{1}));
    
    % Normalize
    volume = (volume - min(volume(:))) / (max(volume(:)) - min(volume(:))); 
    
    % Define output path
    out_path = fullfile(output_dir, sprintf('patient_%d_pde_out.mat', pid));
    
    % Run pipeline on the ENTIRE 3D volume and save automatically!
    % (This might take a minute or two per patient depending on their number of slices)
    runPipeline(volume, weights, out_path);
end

fprintf('\n=======================================\n');
fprintf('All patients processed! The outputs are saved in the "hence_pde_project_v3/pde_outputs" folder.\n');
fprintf('Track 2 can now use these .mat files for their Arithmetic Coding step.\n');

% run_HENCE.m
% Master orchestrator script to run the entire HENCE pipeline
% (PDE generation + MLAC compression) sequentially.

orig_dir = pwd;

fprintf('=======================================================\n');
fprintf(' Starting Full HENCE Pipeline (Track 1 & Track 2)\n');
fprintf('=======================================================\n');

try
    % Step 1: Run Neural PDE Pipeline (Track 1)
    fprintf('\n>>> STEP 1: Running Neural PDE (Track 1)...\n');
    cd('hence_pde_project_v3/matlab');
    run_all_patients;
    
    % Step 2: Run MLAC Pipeline (Track 2)
    fprintf('\n>>> STEP 2: Running MLAC (Track 2)...\n');
    cd(orig_dir);
    cd('hence_mlac_project/matlab');
    run_all_patients_mlac;
    
    % Return to original directory
    cd(orig_dir);
    fprintf('\n>>> HENCE PIPELINE COMPLETE!\n');
    
catch ME
    % Make sure we return to the root folder even if it crashes
    cd(orig_dir);
    rethrow(ME);
end

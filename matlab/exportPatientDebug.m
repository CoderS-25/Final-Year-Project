function exportPatientDebug(projectRoot, patientId, weightsPath, outPath)
%EXPORTPATIENTDEBUG Save all first-slice PDE stages for Python parity checks.
% This helper intentionally refuses to overwrite prior debug evidence.

    if nargin < 1 || isempty(projectRoot)
        error('exportPatientDebug: projectRoot is required.');
    end
    if nargin < 2 || isempty(patientId)
        patientId = 27;
    end
    if nargin < 3 || isempty(weightsPath)
        weightsPath = fullfile(projectRoot, 'best_trained_weights.mat');
    end
    if nargin < 4 || isempty(outPath)
        outPath = fullfile(projectRoot, ...
            sprintf('patient_%d_matlab_debug_stagewise.mat', patientId));
    end
    if exist(outPath, 'file') == 2
        error('exportPatientDebug: refusing to overwrite existing file: %s', outPath);
    end

    patientPath = fullfile(projectRoot, 'chaos_processed_t2spir', ...
        sprintf('%d_norm.mat', patientId));
    data = load(patientPath, 'volume');
    assert(isfield(data, 'volume'), 'exportPatientDebug: missing volume variable.');

    volume = double(data.volume);
    [height, width, ~] = size(volume);
    weights = initPDEWeights(weightsPath, 16);
    matlab_debug = pdeModule_debug(volume(:, :, 1), ...
        zeros(height, width, 16), weights);

    % Preserve names consumed by verify_patient.py, then include all
    % head stages with their descriptive pdeModule_debug field names.
    matlab_debug.MC = matlab_debug.mcFeat;
    matlab_debug.DSC = matlab_debug.dscFeat;
    matlab_debug.H = matlab_debug.H_s;
    save(outPath, '-struct', 'matlab_debug', '-v7');
    fprintf('exportPatientDebug: saved stagewise debug to %s\n', outPath);
end

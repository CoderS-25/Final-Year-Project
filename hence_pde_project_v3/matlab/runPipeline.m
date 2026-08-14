function [pi_vol, mu_vol, log_s_vol, debug] = runPipeline(volume, weights, outPath)
%RUNPIPELINE Run PDE module recurrently across an MRI volume.

[H, W, numSlices] = size(volume);

M = size(weights.dsc.pointwise, 1);
K = 3;

pi_vol    = zeros(H, W, K, numSlices);
mu_vol    = zeros(H, W, K, numSlices);
log_s_vol = zeros(H, W, K, numSlices);

% Debug storage
debug.mcFeat  = cell(numSlices, 1);
debug.dscFeat = cell(numSlices, 1);
debug.H_s     = cell(numSlices, 1);

% H_0
H_prev = zeros(H, W, M);

for s = 1:numSlices

    currentSlice = volume(:, :, s);

    % pdeModule exposes the four model outputs. This pipeline also returns
    % per-slice intermediates, so use the dedicated debug forward path.
    % Both functions implement the same paper-defined computation.
    debug_s = pdeModule_debug(currentSlice, H_prev, weights);
    pi_s    = debug_s.pi;
    mu_s    = debug_s.mu;
    log_s_s = debug_s.log_s;
    H_s     = debug_s.H_s;

    pi_vol(:, :, :, s)    = pi_s;
    mu_vol(:, :, :, s)    = mu_s;
    log_s_vol(:, :, :, s) = log_s_s;

    debug.mcFeat{s}  = debug_s.mcFeat;
    debug.dscFeat{s} = debug_s.dscFeat;
    debug.H_s{s}     = debug_s.H_s;

    H_prev = H_s;

end

assert(~any(isnan(pi_vol(:))), ...
    'runPipeline: NaN in pi_vol');

assert(~any(isnan(mu_vol(:))), ...
    'runPipeline: NaN in mu_vol');

assert(~any(isnan(log_s_vol(:))), ...
    'runPipeline: NaN in log_s_vol');

fprintf( ...
    'runPipeline: produced pi/mu/log_s volumes of size %d x %d x %d x %d\n', ...
    H, W, K, numSlices);

if nargin >= 3 && ~isempty(outPath)

    save(outPath, ...
        'pi_vol', ...
        'mu_vol', ...
        'log_s_vol', ...
        '-v7');

    fprintf( ...
        'runPipeline: saved output to %s\n', ...
        outPath);

end

end

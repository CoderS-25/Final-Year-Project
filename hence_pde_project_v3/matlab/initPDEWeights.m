function weights = initPDEWeights(weightsPath, M)
%INITPDEWEIGHTS Load trained_weights.mat (from train_pde.py) into MATLAB.
%
%   weights = initPDEWeights(weightsPath, M)
%
%   INPUT
%     weightsPath : path to trained_weights.mat exported by train_pde.py.
%                   Falls back to small random weights if not found, so
%                   the MATLAB pipeline is smoke-testable before training.
%     M           : hidden channel width (paper: M=16). Default: 16.
%
%   OUTPUT
%     weights : struct matching pdeModule.m's expected layout:
%         .maskedConv.W  (7x7x1xM), .maskedConv.b (1xM)
%         .dsc.depthwise (5x5xM), .dsc.pointwise (MxM), .dsc.pointwise_bias (1xM)
%         .head.W1 (MxM), .head.b1 (1xM), .head.W2 (Mx9), .head.b2 (1x9)
%
%   EXPECTED .mat CONTENTS (written by train_pde.py / scipy.io.savemat)
%     masked_conv_W, masked_conv_b,
%     dsc_depthwise, dsc_pointwise, dsc_pointwise_b,
%     head_W1, head_b1, head_W2, head_b2

    if nargin < 2 || isempty(M)
        M = 16;
    end

    if nargin >= 1 && ~isempty(weightsPath) && exist(weightsPath, 'file') == 2
        fprintf('initPDEWeights: loading trained weights from %s\n', weightsPath);
        raw = load(weightsPath);

        weights.maskedConv.W = double(raw.masked_conv_W);
        weights.maskedConv.b = double(raw.masked_conv_b);

        weights.dsc.depthwise      = double(raw.dsc_depthwise);
        weights.dsc.pointwise      = double(raw.dsc_pointwise);
        weights.dsc.pointwise_bias = double(raw.dsc_pointwise_b);

        weights.head.W1 = double(raw.head_W1);
        weights.head.b1 = double(raw.head_b1);
        weights.head.W2 = double(raw.head_W2);
        weights.head.b2 = double(raw.head_b2);

        validateWeightShapes(weights);
    else
        warning(['initPDEWeights: "%s" not found. Falling back to random-init ' ...
                 'weights (M=%d). Run train_pde.py first for real predictions.'], ...
                 weightsPath, M);
        weights = randomInitWeights(M);
    end
end

function weights = randomInitWeights(M)
    rng(0);
    scale = 0.05;
    K7 = 7; K5 = 5; K3 = 3;

    weights.maskedConv.W = scale * randn(K7, K7, 1, M);
    weights.maskedConv.b = zeros(1, M);

    weights.dsc.depthwise      = scale * randn(K5, K5, M);
    weights.dsc.pointwise      = scale * randn(M, M);
    weights.dsc.pointwise_bias = zeros(1, M);

    weights.head.W1 = scale * randn(M, M);
    weights.head.b1 = zeros(1, M);
    weights.head.W2 = scale * randn(M, 3 * K3);
    weights.head.b2 = zeros(1, 3 * K3);
end

function validateWeightShapes(weights)
    M = size(weights.dsc.pointwise, 1);
    assert(isequal(size(weights.maskedConv.W, [1 2 3]), [7 7 1]), 'maskedConv.W must be 7x7x1xM');
    assert(size(weights.maskedConv.W, 4) == M, 'maskedConv.W output channels must equal M');
    assert(isequal(size(weights.dsc.depthwise), [5 5 M]), 'dsc.depthwise must be 5x5xM');
    assert(isequal(size(weights.head.W1), [M M]), 'head.W1 must be MxM');
    assert(isequal(size(weights.head.W2), [M 9]), 'head.W2 must be Mx9 (K=3 mixture)');
end

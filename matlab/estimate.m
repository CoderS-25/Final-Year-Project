function predicted = estimate(context_vector)
%ESTIMATE PDE-based causal predictor.
%
% Interface:
%       predicted = estimate(context_vector)
%
% PDE model:
%
%       I_pred = N + W - NW
%
% This corresponds to the discrete mixed-derivative condition:
%
%       d^2 I / dx dy = 0
%
% with an edge-preserving correction based on the relative position
% of NW with respect to N and W.
%
% Input:
%   Whole image:
%       context_vector.N
%       context_vector.W
%       context_vector.NW
%       context_vector.NE
%
%   Single pixel:
%       [N W NW NE]
%
% Output:
%   Whole image -> predicted image
%   Single pixel -> predicted scalar

    if isstruct(context_vector)

        predicted = estimateVectorized(context_vector);

    else

        predicted = estimateScalar(context_vector);

    end

end


function predicted = estimateVectorized(ctx)

    N  = double(ctx.N);
    W  = double(ctx.W);
    NW = double(ctx.NW);

    % ------------------------------------------------------------
    % Basic discrete PDE / planar prediction
    % ------------------------------------------------------------

    predicted = N + W - NW;

    % ------------------------------------------------------------
    % Edge-preserving correction
    %
    % If NW is brighter than both N and W, the current pixel is
    % likely on a descending edge.
    %
    % If NW is darker than both N and W, the current pixel is
    % likely on an ascending edge.
    % ------------------------------------------------------------

    highEdge = NW >= max(N, W);
    lowEdge  = NW <= min(N, W);

    predicted(highEdge) = ...
        min(N(highEdge), W(highEdge));

    predicted(lowEdge) = ...
        max(N(lowEdge), W(lowEdge));

    % ------------------------------------------------------------
    % Final numerical safety check
    % ------------------------------------------------------------

    assert(all(isfinite(predicted(:))), ...
        'PDE predictor produced NaN or Inf.');

end


function predicted = estimateScalar(context_vector)

    assert(numel(context_vector) >= 3, ...
        'Context vector must contain at least [N W NW].');

    N  = double(context_vector(1));
    W  = double(context_vector(2));
    NW = double(context_vector(3));

    % ------------------------------------------------------------
    % Edge-preserving PDE predictor
    % ------------------------------------------------------------

    if NW >= max(N, W)

        predicted = min(N, W);

    elseif NW <= min(N, W)

        predicted = max(N, W);

    else

        predicted = N + W - NW;

    end

    assert(isfinite(predicted), ...
        'PDE predictor produced NaN or Inf.');

end
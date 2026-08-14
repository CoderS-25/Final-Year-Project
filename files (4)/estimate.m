function predicted = estimate(context_vector)
%ESTIMATE PDE-based causal predictor for lossless MRI compression.
%
%   predicted = estimate(context_vector)
%
%   Interface contract:
%       estimate(context_vector) -> predicted_value
%
%   METHODOLOGY
%   -----------
%   The core predictor is the discrete-harmonic / planar-diffusion
%   estimator:
%
%       predicted = N + W - NW
%
%   This is the exact solution of the discrete Laplace / zero mixed
%   second-derivative condition d^2I/(dx dy) = 0, i.e. it assumes the
%   local intensity surface is planar (a first-order PDE-consistent
%   estimate of how intensity "flows" from the causal neighborhood
%   into the current pixel). It is well suited to MRI slices, which
%   are smooth almost everywhere.
%
%   That planar estimate breaks down at edges (it can overshoot far
%   past both N and W). To keep the estimator PDE-consistent in flat
%   regions while remaining stable at edges, it is clamped using the
%   classic edge-preserving correction (as used in JPEG-LS's MED
%   predictor, itself an edge-aware discretization of the same
%   diffusion idea):
%
%       if   NW >= max(N, W):  predicted = min(N, W)
%       elif NW <= min(N, W):  predicted = max(N, W)
%       else:                  predicted = N + W - NW
%
%   INPUT
%   -----
%   context_vector can be either:
%     - a struct with fields N, W, NW (matrices, same size as the
%       image) -- the fully vectorized whole-image mode used by the
%       main pipeline, or
%     - a 1x4 (or longer) numeric vector [N, W, NW, NE] for a single
%       pixel, matching the literal getContext single-pixel output.
%
%   OUTPUT
%   ------
%   A matrix (whole-image mode) or scalar (single-pixel mode) of
%   predicted values, same class/size convention as the input.

    if isstruct(context_vector)
        predicted = estimateVectorized(context_vector);
    else
        predicted = estimateScalar(context_vector);
    end
end

function predicted = estimateVectorized(ctx)
    N  = ctx.N;
    W  = ctx.W;
    NW = ctx.NW;

    % Default: planar / discrete-harmonic prediction (fully vectorized).
    predicted = N + W - NW;

    % Edge-preserving clamp, applied with logical indexing -- still no
    % loops, per the performance constraint.
    maskHigh = NW >= max(N, W);   % local edge with NW as the "bright" corner
    maskLow  = NW <= min(N, W);   % local edge with NW as the "dark" corner

    predicted(maskHigh) = min(N(maskHigh), W(maskHigh));
    predicted(maskLow)  = max(N(maskLow),  W(maskLow));
end

function predicted = estimateScalar(context_vector)
    N  = context_vector(1);
    W  = context_vector(2);
    NW = context_vector(3);

    if NW >= max(N, W)
        predicted = min(N, W);
    elseif NW <= min(N, W)
        predicted = max(N, W);
    else
        predicted = N + W - NW;
    end
end

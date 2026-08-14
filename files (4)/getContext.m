function context = getContext(image, pixel_coords)
%GETCONTEXT Build causal neighborhood context for MRI predictive coding.
%
%   context = getContext(image, pixel_coords)
%
%   Interface contract:
%       getContext(image, pixel_coords) -> context_vector
%
%   Causal template (raster-scan order): Left (W), Top (N),
%   Top-Left (NW), Top-Right (NE). No pixel below or to the right of
%   the current pixel is ever used, since those pixels have not been
%   "transmitted" yet in raster-scan order.
%
%   USAGE MODES
%   -----------
%   1) Vectorized / whole-image mode (used internally by the pipeline
%      to satisfy the "no nested for-loops" performance constraint):
%
%           context = getContext(image, 'all')
%
%      Returns a struct with fields N, W, NW, NE, each the same size
%      as `image`, giving the causal neighbor value for every pixel
%      simultaneously.
%
%   2) Single-pixel mode (kept to satisfy the literal function
%      signature in the interface contract, e.g. for unit tests):
%
%           context = getContext(image, [row, col])
%
%      Returns a 1x4 vector [N, W, NW, NE] for that one pixel. Under
%      the hood this still uses the fully vectorized context maps
%      (built once) rather than any per-pixel loop logic.
%
%   BOUNDARY HANDLING
%   ------------------
%   The image is edge-replicated by one pixel on all sides before
%   shifting, so the first row/column (which lack a full causal
%   neighborhood) fall back to a sane neighbor value instead of
%   producing NaNs or artificial zero-bias.

    [N, W, NW, NE] = buildContextMaps(image);

    if nargin < 2 || (ischar(pixel_coords) && strcmpi(pixel_coords, 'all'))
        context.N  = N;
        context.W  = W;
        context.NW = NW;
        context.NE = NE;
    else
        r = pixel_coords(1);
        c = pixel_coords(2);
        context = [N(r, c), W(r, c), NW(r, c), NE(r, c)];
    end
end

function [N, W, NW, NE] = buildContextMaps(image)
%BUILDCONTEXTMAPS Fully vectorized causal-neighbor extraction.
%
%   Uses padarray + shifted-matrix indexing (no im2col needed for a
%   4-tap causal window, and no loops of any kind) to compute the
%   North, West, North-West and North-East neighbor maps for every
%   pixel in the image at once.

    img = double(image);           % avoid uint truncation of negatives downstream
    [rows, cols] = size(img);

    % Replicate-pad by 1 pixel on every side. Replicate (rather than
    % zero) padding avoids biasing the predictor toward zero at the
    % image boundary, which matters for MRI slices with nonzero
    % background intensity.
    padded = padarray(img, [1 1], 'replicate', 'both');

    % padded is (rows+2) x (cols+2); padded(2:rows+1, 2:cols+1) == img
    N  = padded(1:rows,   2:cols+1);   % row above
    W  = padded(2:rows+1, 1:cols);     % column to the left
    NW = padded(1:rows,   1:cols);     % diagonal up-left
    NE = padded(1:rows,   3:cols+2);   % diagonal up-right
end

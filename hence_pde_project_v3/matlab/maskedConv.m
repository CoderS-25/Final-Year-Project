function out = maskedConv(currentSlice, W, b)
%MASKEDCONV Causal masked KxK convolution -- the "MC" term of Eq. (4).
%
%   out = maskedConv(currentSlice, W, b)
%
%   Paper spec (Section IV-B.3, "Training Setting"): "a single masked
%   convolution layer with a kernel size of 7 and kernel number of 16."
%
%   INPUT
%     currentSlice : H x W (single-channel slice X_s)
%     W            : K x K x 1 x 16 raw (unmasked) weights. The causal
%                    mask is applied INSIDE this function, so the same
%                    raw weights work whether loaded from a trained
%                    PyTorch MaskedConv2d (which also masks internally)
%                    or randomly initialized here.
%     b            : 1 x 16 bias
%
%   OUTPUT
%     out : H x W x 16, i.e. MC(X_s)
%
%   MASK (PixelCNN "type A", strict causality): for a KxK kernel with
%   center at (c,c), c = ceil(K/2), a tap at position (r,col) is kept
%   only if it looks at a pixel that is strictly before the current
%   pixel in raster-scan order:
%       keep  <=>  r < c   OR   (r == c AND col < c)
%   This excludes the center tap itself (the current, not-yet-known
%   pixel) and everything below/right of it -- exactly the causal
%   neighborhood available during autoregressive decoding.

    K = size(W, 1);
    c = ceil(K / 2);
    [colGrid, rowGrid] = meshgrid(1:K, 1:K);
    mask = (rowGrid < c) | (rowGrid == c & colGrid < c);   % KxK logical, vectorized

    maskedW = W .* mask;   % broadcasts KxK mask across Cin=1, Cout=16

    currentSlice3D = reshape(double(currentSlice), size(currentSlice, 1), size(currentSlice, 2), 1);
    % CAUSALITY FIX: must use ZERO padding here, not 'replicate'. With
    % replicate padding, a boundary row/column pad is a full copy of the
    % nearest real row/column -- e.g. at pixel (1,1), the padded rows
    % above the image are literally a replicated copy of row 1 itself,
    % so kernel taps the causal mask considers "safely above" actually
    % read X(1,1)'s own value back in. That is a real self-information
    % leak at every image boundary, not merely a style choice. Zero
    % padding has no such leak: an out-of-bounds tap always contributes
    % exactly 0, regardless of the current pixel's value.
    out = convNCHW(currentSlice3D, maskedW, b, 'zero');
end

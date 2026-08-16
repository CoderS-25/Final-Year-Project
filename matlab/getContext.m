function context = getContext(image, pixel_coords)
%GETCONTEXT Build a causal neighborhood context for predictive coding.
%
% Interface:
%       context = getContext(image, pixel_coords)
%
% Causal neighbors:
%
%              N      NE
%
%       NW     X
%
%              W
%
% Raster-scan causality:
%   - N  = pixel above
%   - W  = pixel to the left
%   - NW = upper-left
%   - NE = upper-right
%
% IMPORTANT:
%   No context value is allowed to depend on the current pixel itself.
%   Therefore boundary positions use zero for unavailable neighbors.
%
% Modes:
%   getContext(image, 'all')
%       Returns struct containing N, W, NW, NE maps.
%
%   getContext(image, [row col])
%       Returns [N W NW NE] for one pixel.

    [N, W, NW, NE] = buildContextMaps(image);

    % ------------------------------------------------------------
    % Whole-image mode
    % ------------------------------------------------------------
    if nargin < 2 || ...
            (ischar(pixel_coords) && strcmpi(pixel_coords, 'all')) || ...
            (isstring(pixel_coords) && strcmpi(pixel_coords, "all"))

        context.N  = N;
        context.W  = W;
        context.NW = NW;
        context.NE = NE;

        return;
    end

    % ------------------------------------------------------------
    % Single-pixel mode
    % ------------------------------------------------------------
    r = pixel_coords(1);
    c = pixel_coords(2);

    [rows, cols] = size(image);

    assert(r >= 1 && r <= rows && ...
           c >= 1 && c <= cols, ...
           'Pixel coordinates are outside image bounds.');

    context = [ ...
        N(r,c), ...
        W(r,c), ...
        NW(r,c), ...
        NE(r,c) ...
    ];
end


function [N, W, NW, NE] = buildContextMaps(image)
%BUILDCONTEXTMAPS Fully vectorized causal context extraction.
%
% No nested loops.
%
% Missing causal neighbors at the image boundary are represented by 0.
% This guarantees that the current pixel never leaks into its own context.

    img = double(image);

    [rows, cols] = size(img);

    % ------------------------------------------------------------
    % Initialize missing neighbors to zero.
    % ------------------------------------------------------------

    N  = zeros(rows, cols);
    W  = zeros(rows, cols);
    NW = zeros(rows, cols);
    NE = zeros(rows, cols);

    % ------------------------------------------------------------
    % Valid North neighbors
    %
    % Pixel (r,c) gets img(r-1,c)
    % for r >= 2.
    % ------------------------------------------------------------

    if rows >= 2
        N(2:end, :) = img(1:end-1, :);
    end

    % ------------------------------------------------------------
    % Valid West neighbors
    %
    % Pixel (r,c) gets img(r,c-1)
    % for c >= 2.
    % ------------------------------------------------------------

    if cols >= 2
        W(:, 2:end) = img(:, 1:end-1);
    end

    % ------------------------------------------------------------
    % Valid North-West neighbors
    %
    % Pixel (r,c) gets img(r-1,c-1)
    % for r >= 2, c >= 2.
    % ------------------------------------------------------------

    if rows >= 2 && cols >= 2
        NW(2:end, 2:end) = img(1:end-1, 1:end-1);
    end

    % ------------------------------------------------------------
    % Valid North-East neighbors
    %
    % Pixel (r,c) gets img(r-1,c+1)
    % for r >= 2, c <= cols-1.
    %
    % NE is still causal because the entire previous row has already
    % been transmitted before the current row.
    % ------------------------------------------------------------

    if rows >= 2 && cols >= 2
        NE(2:end, 1:end-1) = img(1:end-1, 2:end);
    end
end
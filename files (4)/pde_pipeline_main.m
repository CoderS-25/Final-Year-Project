%% PDE_PIPELINE_MAIN
% Track 1 - PDE Module | Lossless MRI Image Compression Pipeline (Stage 1/2)
%
% Runs the full predictive-coding pipeline on sample "MRI-like" slices:
%   1) Build causal context windows            -> getContext(image, 'all')
%   2) PDE-based prediction                    -> estimate(context)
%   3) Residual computation                    -> actual - predicted
%   4) Visualization + histogram sanity check
%
% Entirely vectorized: no nested for-loops anywhere in the pipeline.

clear; clc; close all;

%% 1. Load / generate sample slices
% Real MRI DICOM data isn't available in this environment, so we use
% MATLAB's Shepp-Logan phantom (smooth, piecewise-constant-with-gradients
% intensity structure) as a stand-in "MRI slice", plus a couple of
% variants to exercise different intensity ranges and edge content.
% Swap in `dicomread(...)` here for real data -- the pipeline is
% agnostic to the image source.

sampleImages = {};
sampleNames  = {};

img1 = im2uint16(mat2gray(phantom('Modified Shepp-Logan', 256)));
sampleImages{end+1} = img1; sampleNames{end+1} = 'Shepp-Logan Phantom (256x256)';

% Smoothed variant: mimics real MRI's softer gradients / less abrupt edges
img2 = im2uint16(mat2gray(imgaussfilt(phantom(256), 2)));
sampleImages{end+1} = img2; sampleNames{end+1} = 'Smoothed Phantom (blurred, 256x256)';

% Small, noisy variant: stresses the edge-preserving clamp and boundary handling
rng(0);
img3base = im2uint16(mat2gray(phantom(128)));
noise    = uint16(5 * randn(size(img3base)));
img3     = img3base + noise; % small additive texture on top of smooth structure
sampleImages{end+1} = img3; sampleNames{end+1} = 'Noisy Phantom (128x128)';

%% 2. Run the pipeline on each sample and visualize
for k = 1:numel(sampleImages)
    actual_image = sampleImages{k};
    name = sampleNames{k};

    % --- Step 1: Feature extraction (vectorized causal context) ---
    context = getContext(actual_image, 'all');

    % --- Step 2: PDE-based prediction ---
    predicted_image = estimate(context);

    % --- Step 3: Residual computation ---
    % Use double precision explicitly so negative residuals are
    % preserved (not clipped by an unsigned integer type).
    residual_image = double(actual_image) - predicted_image;

    % --- Sanity checks ---
    assert(~any(isnan(residual_image(:))), ...
        'NaN detected in residual image for %s', name);
    fprintf('[%s] No NaNs. Residual mean = %.4f, std = %.4f\n', ...
        name, mean(residual_image(:)), std(residual_image(:)));

    % --- Step 4: Visualization ---
    figure('Name', name, 'Position', [100 100 1100 800]);

    subplot(2,2,1);
    imshow(actual_image, []);
    title('Original Image');

    subplot(2,2,2);
    imshow(predicted_image, []);
    title('PDE-Predicted Image');

    subplot(2,2,3);
    imshow(residual_image, []);
    title('Residual Image (actual - predicted)');

    subplot(2,2,4);
    histogram(residual_image(:), 100);
    title('Residual Histogram (should be a sharp spike at 0)');
    xlabel('Residual value');
    ylabel('Pixel count');
    xline(0, 'r--', 'LineWidth', 1.5);

    sgtitle(name);
end

fprintf('\nPipeline complete. Inspect the histograms: a sharp Laplace-like\n');
fprintf('spike centered at 0 indicates the PDE estimator is working well.\n');

%% PDE_PIPELINE_MAIN
%
% Track 1 - PDE Module
%
% Pipeline:
%
%       Image
%         |
%         v
%   getContext()
%         |
%         v
%   estimate()
%         |
%         v
%   predicted image
%         |
%         v
% actual - predicted
%         |
%         v
% residual image
%

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf('TRACK 1 - PDE PREDICTIVE CODING PIPELINE\n');
fprintf('============================================================\n');


%% ============================================================
% 1. Generate sample images
% =============================================================

sampleImages = {};
sampleNames  = {};

% -------------------------------------------------------------
% Sample 1 - Shepp-Logan
% -------------------------------------------------------------

img1 = im2uint16( ...
    mat2gray( ...
        phantom('Modified Shepp-Logan',256)));

sampleImages{end+1} = img1;
sampleNames{end+1}  = 'Shepp-Logan';


% -------------------------------------------------------------
% Sample 2 - Smoothed MRI-like image
% -------------------------------------------------------------

img2 = im2uint16( ...
    mat2gray( ...
        imgaussfilt(phantom(256),2)));

sampleImages{end+1} = img2;
sampleNames{end+1}  = 'Smoothed Phantom';


% -------------------------------------------------------------
% Sample 3 - Noisy image
% -------------------------------------------------------------

rng(0);

img3base = im2uint16( ...
    mat2gray(phantom(128)));

noise = 5 * randn(size(img3base));

img3 = double(img3base) + noise;

img3 = uint16( ...
    max(0,min(double(intmax('uint16')),img3)));

sampleImages{end+1} = img3;
sampleNames{end+1}  = 'Noisy Phantom';


%% ============================================================
% 2. Run pipeline
% ============================================================

for k = 1:numel(sampleImages)

    actual_image = sampleImages{k};
    name = sampleNames{k};

    fprintf('\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('%s\n',name);
    fprintf('------------------------------------------------------------\n');

    % ----------------------------------------------------------
    % Stage A - Context extraction
    % ----------------------------------------------------------

    context = getContext(actual_image,'all');

    % ----------------------------------------------------------
    % Stage B - PDE prediction
    % ----------------------------------------------------------

    predicted_image = estimate(context);

    % ----------------------------------------------------------
    % Stage C - Residual
    % ----------------------------------------------------------

    actual_double = double(actual_image);

    residual_image = ...
        actual_double - predicted_image;

    % ----------------------------------------------------------
    % Safety checks
    % ----------------------------------------------------------

    assert(isequal(size(predicted_image), ...
                   size(actual_image)), ...
        'Prediction size mismatch.');

    assert(isequal(size(residual_image), ...
                   size(actual_image)), ...
        'Residual size mismatch.');

    assert(all(isfinite(predicted_image(:))), ...
        'Prediction contains NaN/Inf.');

    assert(all(isfinite(residual_image(:))), ...
        'Residual contains NaN/Inf.');

    % ----------------------------------------------------------
    % Statistics
    % ----------------------------------------------------------

    residual_mean = mean(residual_image(:));
    residual_std  = std(residual_image(:));

    abs_residual = abs(residual_image(:));

    mae = mean(abs_residual);

    zero_fraction = ...
        mean(abs_residual <= 1);

    fprintf('Prediction mean      : %.4f\n', ...
        mean(predicted_image(:)));

    fprintf('Prediction std       : %.4f\n', ...
        std(predicted_image(:)));

    fprintf('Residual mean        : %.4f\n', ...
        residual_mean);

    fprintf('Residual std         : %.4f\n', ...
        residual_std);

    fprintf('Residual MAE         : %.4f\n', ...
        mae);

    fprintf('|Residual| <= 1      : %.2f %%\n', ...
        100*zero_fraction);

    % ----------------------------------------------------------
    % Visualize
    % ----------------------------------------------------------

    figure( ...
        'Name',name, ...
        'Position',[100 100 1200 800]);

    subplot(2,2,1);

    imshow(actual_image,[]);

    title('Actual Image');


    subplot(2,2,2);

    imshow(predicted_image,[]);

    title('PDE Prediction');


    subplot(2,2,3);

    imshow(residual_image,[]);

    title('Residual = Actual - Prediction');


    subplot(2,2,4);

    histogram(residual_image(:),100);

    xlabel('Residual');
    ylabel('Pixel count');

    title('Residual Distribution');

    xline(0,'--','LineWidth',1.5);

    sgtitle(name);

end


fprintf('\n');
fprintf('============================================================\n');
fprintf('TRACK 1 PIPELINE COMPLETE\n');
fprintf('============================================================\n');
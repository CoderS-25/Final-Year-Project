%% TEST_PDE_MODULE
% Unit tests / sanity checks for the PDE predictive-coding module
% (Section 3.4 of the PRD: Unit Testing and Edge Cases).
%
% Run with: test_pde_module   (or click Run in the MATLAB editor)
% Prints PASS/FAIL for each check; throws an error on first failure
% so it can also be wired into a CI script via `assert`.

clear; clc;
fprintf('Running PDE module unit tests...\n\n');
nPassed = 0; nTotal =   0;

%% Test 1: No NaNs on a tiny image (heavy boundary influence)
nTotal = nTotal + 1;
try
    smallImg = uint16([10 20 30; 40 50 60; 70 80 90]);
    ctx = getContext(smallImg, 'all');
    pred = estimate(ctx);
    resid = double(smallImg) - pred;
    assert(~any(isnan(pred(:))), 'Predicted image contains NaN');
    assert(~any(isnan(resid(:))), 'Residual image contains NaN');
    fprintf('[PASS] Test 1: No NaNs on tiny 3x3 image (boundary-heavy)\n');
    nPassed = nPassed + 1;
catch ME
    fprintf('[FAIL] Test 1: %s\n', ME.message);
end

%% Test 2: First row / first column boundary handling produces finite,
%          reasonable values (not zero-biased, not out of range)
nTotal = nTotal + 1;
try
    img = uint16(magic(6)) * 100;
    ctx = getContext(img, 'all');
    pred = estimate(ctx);
    firstRow = pred(1, :);
    firstCol = pred(:, 1);
    assert(all(isfinite(firstRow)), 'First row prediction not finite');
    assert(all(isfinite(firstCol)), 'First column prediction not finite');
    % With replicate padding, top-left corner's N/W/NW all equal the
    % pixel itself, so its prediction should exactly equal the pixel.
    assert(pred(1,1) == double(img(1,1)), ...
        'Top-left corner prediction should equal itself under replicate padding');
    fprintf('[PASS] Test 2: First row/column boundary handling is sane\n');
    nPassed = nPassed + 1;
catch ME
    fprintf('[FAIL] Test 2: %s\n', ME.message);
end

%% Test 3: Negative residuals are preserved (not truncated to 0)
nTotal = nTotal + 1;
try
    % Construct a case where actual < predicted, forcing a negative residual.
    img = uint16([100 100 100; 100 100 100; 100 5 100]); % sharp local dip
    ctx = getContext(img, 'all');
    pred = estimate(ctx);
    resid = double(img) - pred;
    assert(any(resid(:) < 0), ...
        'Expected at least one negative residual, but none found');
    fprintf('[PASS] Test 3: Negative residuals preserved (no uint truncation)\n');
    nPassed = nPassed + 1;
catch ME
    fprintf('[FAIL] Test 3: %s\n', ME.message);
end

%% Test 4: Perfectly flat image -> zero residual everywhere
nTotal = nTotal + 1;
try
    img = uint16(ones(10, 10) * 500);
    ctx = getContext(img, 'all');
    pred = estimate(ctx);
    resid = double(img) - pred;
    assert(all(resid(:) == 0), 'Flat image should yield exactly zero residuals');
    fprintf('[PASS] Test 4: Flat image yields zero residuals everywhere\n');
    nPassed = nPassed + 1;
catch ME
    fprintf('[FAIL] Test 4: %s\n', ME.message);
end

%% Test 5: Perfectly planar (linear gradient) image -> zero residual
%          away from the edge-clamp regions (validates the PDE math)
nTotal = nTotal + 1;
try
    [X, Y] = meshgrid(1:20, 1:20);
    img = uint16(X * 3 + Y * 2 + 50); % I(x,y) = 3x + 2y + 50 is planar
    ctx = getContext(img, 'all');
    pred = estimate(ctx);
    resid = double(img) - pred;
    % Interior (row>1, col>1) should be exactly predicted since
    % N + W - NW is exact for a planar surface and the edge-clamp
    % conditions won't trigger away from the boundary for a monotone
    % gradient like this one.
    interiorResid = resid(2:end, 2:end);
    assert(max(abs(interiorResid(:))) < 1e-9, ...
        'Planar image should be predicted exactly in the interior');
    fprintf('[PASS] Test 5: Planar gradient predicted exactly (validates PDE math)\n');
    nPassed = nPassed + 1;
catch ME
    fprintf('[FAIL] Test 5: %s\n', ME.message);
end

%% Test 6: Single-pixel mode of getContext matches whole-image mode
nTotal = nTotal + 1;
try
    img = uint16(reshape(1:36, 6, 6)');
    ctxAll = getContext(img, 'all');
    r = 4; c = 5;
    ctxPixel = getContext(img, [r, c]);
    expected = [ctxAll.N(r,c), ctxAll.W(r,c), ctxAll.NW(r,c), ctxAll.NE(r,c)];
    assert(isequal(ctxPixel, expected), ...
        'Single-pixel context does not match whole-image context maps');
    fprintf('[PASS] Test 6: Single-pixel getContext matches vectorized maps\n');
    nPassed = nPassed + 1;
catch ME
    fprintf('[FAIL] Test 6: %s\n', ME.message);
end

%% Test 7: Two real-scale sample "MRI slices" run end-to-end without error
nTotal = nTotal + 1;
try
    samples = {phantom(64), imgaussfilt(phantom(64), 1.5)};
    for i = 1:numel(samples)
        im = im2uint16(mat2gray(samples{i}));
        ctx = getContext(im, 'all');
        pred = estimate(ctx);
        resid = double(im) - pred;
        assert(~any(isnan(resid(:))), 'NaN found in sample %d residual', i);
        assert(isequal(size(pred), size(im)), 'Predicted image size mismatch');
    end
    fprintf('[PASS] Test 7: End-to-end run on 2 sample MRI-like slices\n');
    nPassed = nPassed + 1;
catch ME
    fprintf('[FAIL] Test 7: %s\n', ME.message);
end

%% Summary
fprintf('\n%d / %d tests passed.\n', nPassed, nTotal);
if nPassed == nTotal
    fprintf('ALL TESTS PASSED.\n');
else
    error('Some PDE module tests failed. See output above.');
end

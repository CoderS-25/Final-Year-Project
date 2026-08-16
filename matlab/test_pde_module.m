%% TEST_PDE_MODULE
%
% Track 1 - PDE module unit tests.
%
% Tests:
%   1. No NaNs
%   2. Causal boundary handling
%   3. Negative residual preservation
%   4. Flat image prediction
%   5. Planar image prediction
%   6. Single-pixel / vectorized consistency
%   7. MRI-like end-to-end test
%
% Run:
%
%       test_pde_module
%

clear;
clc;

fprintf('\n');
fprintf('============================================================\n');
fprintf('TRACK 1 - PDE MODULE UNIT TESTS\n');
fprintf('============================================================\n\n');

nPassed = 0;
nTotal  = 0;


%% ============================================================
% TEST 1 - No NaNs on tiny image
% =============================================================

nTotal = nTotal + 1;

try

    img = uint16([
        10 20 30;
        40 50 60;
        70 80 90
    ]);

    ctx = getContext(img, 'all');

    pred = estimate(ctx);

    residual = double(img) - pred;

    assert(all(isfinite(ctx.N(:))));
    assert(all(isfinite(ctx.W(:))));
    assert(all(isfinite(ctx.NW(:))));
    assert(all(isfinite(ctx.NE(:))));

    assert(all(isfinite(pred(:))));
    assert(all(isfinite(residual(:))));

    fprintf('[PASS] Test 1 - No NaN/Inf values\n');

    nPassed = nPassed + 1;

catch ME

    fprintf('[FAIL] Test 1 - %s\n', ME.message);

end


%% ============================================================
% TEST 2 - Boundary causality
% =============================================================

nTotal = nTotal + 1;

try

    img = uint16([
        10 20 30;
        40 50 60;
        70 80 90
    ]);

    ctx = getContext(img, 'all');

    % ----------------------------------------------------------
    % Top-left pixel has NO causal neighbors.
    % Therefore all four context values must be zero.
    % ----------------------------------------------------------

    assert(ctx.N(1,1)  == 0);
    assert(ctx.W(1,1)  == 0);
    assert(ctx.NW(1,1) == 0);
    assert(ctx.NE(1,1) == 0);

    % ----------------------------------------------------------
    % First row:
    % N/NW unavailable.
    % W is available after the first column.
    % ----------------------------------------------------------

    assert(ctx.N(1,2)  == 0);
    assert(ctx.NW(1,2) == 0);
    assert(ctx.W(1,2)  == double(img(1,1)));

    % ----------------------------------------------------------
    % First column:
    % W/NW unavailable.
    % N is available after first row.
    % ----------------------------------------------------------

    assert(ctx.W(2,1)  == 0);
    assert(ctx.NW(2,1) == 0);
    assert(ctx.N(2,1)  == double(img(1,1)));

    % ----------------------------------------------------------
    % Interior pixel.
    % ----------------------------------------------------------

    assert(ctx.N(2,2)  == double(img(1,2)));
    assert(ctx.W(2,2)  == double(img(2,1)));
    assert(ctx.NW(2,2) == double(img(1,1)));
    assert(ctx.NE(2,2) == double(img(1,3)));

    fprintf('[PASS] Test 2 - Boundary handling is causal\n');

    nPassed = nPassed + 1;

catch ME

    fprintf('[FAIL] Test 2 - %s\n', ME.message);

end


%% ============================================================
% TEST 3 - Negative residuals preserved
% =============================================================

nTotal = nTotal + 1;

try

    img = uint16([
        100 100 100;
        100 100 100;
        100   5 100
    ]);

    ctx = getContext(img, 'all');

    pred = estimate(ctx);

    % IMPORTANT:
    % Cast actual image to double before subtraction.
    residual = double(img) - pred;

    assert(any(residual(:) < 0), ...
        'Negative residual was not preserved.');

    fprintf('[PASS] Test 3 - Negative residuals preserved\n');

    nPassed = nPassed + 1;

catch ME

    fprintf('[FAIL] Test 3 - %s\n', ME.message);

end


%% ============================================================
% TEST 4 - Flat image
% =============================================================

nTotal = nTotal + 1;

try

    img = uint16(ones(20,20) * 500);

    ctx = getContext(img, 'all');

    pred = estimate(ctx);

    residual = double(img) - pred;

    % Interior should be exactly zero residual.
    interior = residual(2:end, 2:end);

    assert(all(interior(:) == 0), ...
        'Flat image should have zero interior residual.');

    fprintf('[PASS] Test 4 - Flat image predicted exactly in interior\n');

    nPassed = nPassed + 1;

catch ME

    fprintf('[FAIL] Test 4 - %s\n', ME.message);

end


%% ============================================================
% TEST 5 - Planar image
% =============================================================

nTotal = nTotal + 1;

try

    [X,Y] = meshgrid(1:30,1:30);

    img = uint16(3*X + 2*Y + 50);

    ctx = getContext(img, 'all');

    pred = estimate(ctx);

    residual = double(img) - pred;

    % Interior must be exactly reconstructed.
    interior = residual(2:end,2:end);

    assert(max(abs(interior(:))) < 1e-9, ...
        'Planar surface was not predicted exactly.');

    fprintf('[PASS] Test 5 - Planar PDE prediction exact\n');

    nPassed = nPassed + 1;

catch ME

    fprintf('[FAIL] Test 5 - %s\n', ME.message);

end


%% ============================================================
% TEST 6 - Single-pixel mode matches vectorized mode
% =============================================================

nTotal = nTotal + 1;

try

    img = uint16(reshape(1:36,6,6)');

    ctxAll = getContext(img, 'all');

    testPositions = [
        1 1;
        1 4;
        4 1;
        3 4;
        6 6
    ];

    for k = 1:size(testPositions,1)

        r = testPositions(k,1);
        c = testPositions(k,2);

        ctxPixel = getContext(img,[r c]);

        expected = [
            ctxAll.N(r,c), ...
            ctxAll.W(r,c), ...
            ctxAll.NW(r,c), ...
            ctxAll.NE(r,c)
        ];

        assert(isequal(ctxPixel,expected), ...
            'Single-pixel context mismatch.');

    end

    fprintf('[PASS] Test 6 - Single-pixel/vectorized context match\n');

    nPassed = nPassed + 1;

catch ME

    fprintf('[FAIL] Test 6 - %s\n', ME.message);

end


%% ============================================================
% TEST 7 - Two MRI-like images
% =============================================================

nTotal = nTotal + 1;

try

    samples = {
        phantom(128), ...
        imgaussfilt(phantom(128),2)
    };

    for k = 1:numel(samples)

        img = im2uint16(mat2gray(samples{k}));

        ctx = getContext(img,'all');

        pred = estimate(ctx);

        residual = double(img) - pred;

        assert(isequal(size(pred),size(img)));

        assert(isequal(size(residual),size(img)));

        assert(all(isfinite(pred(:))));

        assert(all(isfinite(residual(:))));

    end

    fprintf('[PASS] Test 7 - MRI-like images run end-to-end\n');

    nPassed = nPassed + 1;

catch ME

    fprintf('[FAIL] Test 7 - %s\n', ME.message);

end


%% ============================================================
% SUMMARY
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('RESULT: %d / %d TESTS PASSED\n',nPassed,nTotal);
fprintf('============================================================\n');

if nPassed == nTotal

    fprintf('ALL TRACK 1 PDE TESTS PASSED.\n');

else

    error( ...
        'Track 1 failed: %d / %d tests failed.', ...
        nTotal-nPassed, nTotal);

end
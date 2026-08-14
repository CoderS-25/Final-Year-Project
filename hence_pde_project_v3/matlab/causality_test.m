pkg load image;
addpath('/home/claude/hence_v3/matlab');

rng_state = 2;
rand('seed', rng_state);
H = 12; W = 12; M = 16;

weights = initPDEWeights('', M);   % random-init fallback (no trained file needed for this test)
h_prev = zeros(H, W, M);

base = rand(H, W) * 2 - 1;

testPositions = [1 1; 1 W; H 1; H W; 1 6; 6 1; 6 6; H 6];   % 1-indexed MATLAB equivalents of the python (0-indexed) test set

[pi0, mu0, log_s0, h_s0] = pdeModule(base, h_prev, weights);

allPass = true;
for i = 1:size(testPositions, 1)
    r = testPositions(i, 1);
    c = testPositions(i, 2);
    perturbed = base;
    perturbed(r, c) = perturbed(r, c) + 100.0;

    [pi1, mu1, log_s1, h_s1] = pdeModule(perturbed, h_prev, weights);

    d = max(abs(squeeze(mu1(r, c, :)) - squeeze(mu0(r, c, :))));
    if d < 1e-6
        status = 'PASS';
    else
        status = 'FAIL';
        allPass = false;
    end
    printf('pixel (%d,%d): max|delta mu| at that pixel = %.8f  [%s]\n', r-1, c-1, d, status);
end

if allPass
    disp('ALL CAUSALITY TESTS PASSED');
else
    disp('CAUSALITY LEAK DETECTED');
end

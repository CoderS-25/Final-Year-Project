function features = dscBranch(hPrev, dscWeights)
%DSCBRANCH Depthwise-separable conv on the previous slice's hidden state.
%
%   features = dscBranch(hPrev, dscWeights)
%
%   Implements the "DSC" term of Eq. (4): DSC(H_{s-1}). Per Fig. 3(a) of
%   the paper, the DSC layer is: 5x5 depthwise conv -> ReLU -> 1x1
%   pointwise conv (M -> M channels, M=16). This module does NOT expand
%   channels (unlike the earlier draft's incorrect 1->3C design) --
%   HENCE uses a single Hardtanh gate, so both branches feeding the gate
%   must already be the same M-channel width.
%
%   INPUT
%     hPrev      : H x W x M  (auxiliary feature H_{s-1}; M=16)
%     dscWeights : struct with fields
%         .depthwise      : 5 x 5 x M (one 5x5 filter per channel)
%         .pointwise      : M x M     (Cin x Cout, for conv1x1)
%         .pointwise_bias : 1 x M
%
%   OUTPUT
%     features : H x W x M, i.e. DSC(H_{s-1})

    [H, Wd, M] = size(hPrev);
    padded = padarray(hPrev, [2 2], 'replicate', 'both');

    depthwiseOut = zeros(H, Wd, M);
    for m = 1:M
        kernel = rot90(dscWeights.depthwise(:, :, m), 2);  % correlation -> convolution
        depthwiseOut(:, :, m) = conv2(padded(:, :, m), kernel, 'valid');
    end

    depthwiseOut = max(depthwiseOut, 0);   % ReLU (Fig. 3a: DW Conv -> ReLU -> PW Conv)

    features = conv1x1(depthwiseOut, dscWeights.pointwise, dscWeights.pointwise_bias);
end

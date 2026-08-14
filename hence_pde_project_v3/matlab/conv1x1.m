function out = conv1x1(featureMap, W, b)
%CONV1X1 Pointwise (1x1) convolution as a per-pixel matrix multiply.
%
%   out = conv1x1(featureMap, W, b)
%
%   INPUT
%     featureMap : H x W x Cin
%     W          : Cin x Cout weight matrix
%     b          : 1 x Cout bias vector (pass [] for no bias)
%
%   OUTPUT
%     out : H x W x Cout

    [H, Wd, Cin] = size(featureMap);
    flat = reshape(featureMap, H * Wd, Cin);
    outFlat = flat * W;
    if nargin >= 3 && ~isempty(b)
        outFlat = outFlat + b;
    end
    Cout = size(W, 2);
    out = reshape(outFlat, H, Wd, Cout);
end

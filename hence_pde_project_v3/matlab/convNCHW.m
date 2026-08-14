function out = convNCHW(input, W, b, padMode)
%CONVNCHW Generic multi-in/multi-out-channel 2D convolution.
%
%   out = convNCHW(input, W, b, padMode)
%
%   INPUT
%     input   : H x W x Cin
%     W       : K x K x Cin x Cout   (cross-correlation convention, i.e.
%               identical to a PyTorch Conv2d weight tensor once
%               permuted -- see pde_model.py's export_weights_to_mat)
%     b       : 1 x Cout (bias) or [] for no bias
%     padMode : padarray mode, e.g. 'replicate'
%
%   OUTPUT
%     out : H x W x Cout
%
%   NOTE: MATLAB's conv2 performs true convolution (kernel flipped
%   180deg); PyTorch's Conv2d performs cross-correlation (no flip). We
%   flip each 2D kernel slice with rot90(...,2) before calling conv2 so
%   that, given the SAME (unflipped) weight values as trained in
%   PyTorch, both languages produce identical output. This keeps the
%   exported .mat weights a straight reshape of the PyTorch tensor with
%   no separate flip step needed at export time.
%
%   Loops over Cin x Cout (a small, fixed number of channels -- e.g.
%   16x16 -- never over pixels), so this remains fully vectorized in the
%   spatial dimensions that matter for the "no nested pixel loops"
%   performance constraint.

    [H, Wd, Cin] = size(input);
    K = size(W, 1);
    pad = floor(K / 2);
    Cout = size(W, 4);

    padded = padarray(input, [pad pad], padPadArrayArg(padMode), 'both');
    out = zeros(H, Wd, Cout);

    for co = 1:Cout
        acc = zeros(H, Wd);
        for ci = 1:Cin
            kernel = rot90(W(:, :, ci, co), 2);   % correlation -> convolution
            acc = acc + conv2(padded(:, :, ci), kernel, 'valid');
        end
        out(:, :, co) = acc;
    end

    if nargin >= 3 && ~isempty(b)
        out = out + reshape(b, 1, 1, Cout);
    end
end

function arg = padPadArrayArg(padMode)
%PADPADARRAYARG Translate our 'zero' shorthand into padarray's numeric
%   pad-value form; pass any other mode string straight through.
    if strcmp(padMode, 'zero')
        arg = 0;
    else
        arg = padMode;
    end
end

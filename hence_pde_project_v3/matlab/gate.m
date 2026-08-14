function H_s = gate(mcFeat, dscFeat)
%GATE The paper's non-linear gate G -- Eq. (5).
%
%   H_s = gate(mcFeat, dscFeat)
%
%       G(a, b) = Hardtanh(a + b) =  1        if a+b > 1
%                                   -1        if a+b < -1
%                                    a+b      otherwise
%
%   INPUT
%     mcFeat, dscFeat : H x W x M, same shape (M=16)
%
%   OUTPUT
%     H_s : H x W x M -- the auxiliary feature for slice s (Eq. 4),
%           used BOTH to predict slice s's pixel distributions AND as
%           the hidden state handed to dscBranch.m for slice s+1.

    assert(isequal(size(mcFeat), size(dscFeat)), ...
        'gate: mcFeat and dscFeat must have matching shapes');
    H_s = max(-1, min(1, mcFeat + dscFeat));   % Hardtanh
end

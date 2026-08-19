"""
pde_model.py

PyTorch replica of the HENCE paper's PDE module (Chen, Chen, Zhang, Chen,
Luo, Yu -- IEEE Access 2024, "HENCE: Hardware End-to-End Neural
Conditional Entropy Encoder for Lossless 3D Medical Image Compression").

Implements Eq. (4)-(5) exactly:
    H_s = G(MC(X_s), DSC(H_{s-1})),   G(a,b) = Hardtanh(a+b)

and the mixed-logistic estimator (two 1x1 convs, K=3 mixture, Eq. 14's
clip-normalize for pi instead of softmax).

    MATLAB                  PyTorch
    -----------------------------------------
    maskedConv.m        <-> MaskedConv7x7   (Eq. 4, "MC")
    dscBranch.m          <-> DSCBranch       (Eq. 4, "DSC")
    gate.m                <-> Gate            (Eq. 5, "G")
    estimatorHead.m      <-> EstimatorHead   (mixed-logistic estimator)
    pdeModule.m           <-> PDEModule       (single-pass, one H_s/slice)

CORRECTED vs. an earlier draft: there is exactly ONE auxiliary feature
H_s per slice, used both to predict that slice's pixels and as next
slice's hidden state -- no separate "prediction"/"update" convs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

M = 16   # hidden/auxiliary feature channels (paper: "kernel number of 16")
K = 3    # mixture components (paper: Section III, "we set K = 3")
MASKED_CONV_KERNEL = 7   # paper: "masked convolution layer with a kernel size of 7"
DSC_KERNEL = 5            # paper: "kernel size of depth separable convolution to 5"
MIN_LOG_SCALE = -14.0
PI_CLIP_EPS = 1e-6


class MaskedConv7x7(nn.Module):
    """Causal masked KxK conv -- the "MC" term of Eq. (4).

    PixelCNN "type A" mask: excludes the center tap (current, unknown
    pixel) and everything below/right of it in raster-scan order, so
    the output at (i,j) depends only on pixels strictly before (i,j).
    """

    def __init__(self, k: int = MASKED_CONV_KERNEL, out_channels: int = M):
        super().__init__()
        # padding=0 here deliberately: forward() does the padding manually
        # (zero-padding, see the causality note there), so this Conv2d's
        # own padding/padding_mode settings are never used for spatial
        # padding -- it exists only to hold the learnable weight/bias.
        self.conv = nn.Conv2d(1, out_channels, kernel_size=k, padding=0, bias=True)
        mask = torch.zeros(k, k)
        center = k // 2  # 0-indexed center row/col
        for r in range(k):
            for c in range(k):
                if r < center or (r == center and c < center):
                    mask[r, c] = 1.0
        self.register_buffer("mask", mask.view(1, 1, k, k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.conv.weight * self.mask
        pad = self.conv.kernel_size[0] // 2
        # CAUSALITY FIX: zero padding, NOT replicate. With replicate
        # padding, the pad rows/cols above/left of the image are full
        # copies of the nearest real row/col -- e.g. at pixel (1,1) (row
        # 0, col 0 in 0-indexed terms) the rows padded "above" the image
        # are literally row 0 replicated, so causal-mask taps the mask
        # keeps as "strictly above" actually read pixel (1,1)'s own
        # value back in at the boundary. Zero padding removes this leak:
        # out-of-bounds taps always contribute exactly 0.
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
        return F.conv2d(x_padded, masked_weight, self.conv.bias)


class DSCBranch(nn.Module):
    """Depthwise-separable conv on the previous slice's hidden state --
    the "DSC" term of Eq. (4). Per Fig. 3(a): 5x5 depthwise -> ReLU ->
    1x1 pointwise, M -> M channels (no channel expansion -- HENCE's
    single Hardtanh gate needs both branches at the same width)."""

    def __init__(self, channels: int = M, k: int = DSC_KERNEL):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=k, padding=k // 2, groups=channels,
            bias=False, padding_mode="replicate",
        )
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, h_prev: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(h_prev)
        x = self.relu(x)
        x = self.pointwise(x)
        return x


class Gate(nn.Module):
    """G(a,b) = Hardtanh(a+b) -- Eq. (5). The ONLY gate in HENCE (Table 1:
    16-channel tanh, one gate -- vs. SR-LVC's 48-channel GRU, two gates)."""

    def __init__(self):
        super().__init__()
        self.hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)

    def forward(self, mc_feat: torch.Tensor, dsc_feat: torch.Tensor) -> torch.Tensor:
        return self.hardtanh(mc_feat + dsc_feat)


class EstimatorHead(nn.Module):
    """Mixed-logistic estimator: two 1x1 convs (ReLU between), K=3
    mixture -> 9 output channels [pi(3), mu(3), log_s(3)].

    PI NORMALIZATION -- Eq. (14): the paper explicitly replaces Softmax
    (hardware-unfriendly exp()) with clip-then-normalize:
        pi_j = clip(pi_raw_j, 0, 1) / sum_k clip(pi_raw_k, 0, 1)
    This is intentionally NOT softmax -- keep it this way.
    """

    def __init__(self, channels: int = M, k: int = K):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # Keep this non-in-place so verification can retain the true
        # pre-ReLU estimator tensor without changing model semantics.
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, 3 * k, kernel_size=1, bias=True)
        self.k = k

    def forward(self, h_s: torch.Tensor, return_debug: bool = False):
        hidden_pre = self.conv1(h_s)
        hidden = self.relu(hidden_pre)
        raw = self.conv2(hidden)  # (B, 9, H, W)

        pi_raw = raw[:, 0:self.k, :, :]
        mu = raw[:, self.k:2 * self.k, :, :]
        log_s_raw = raw[:, 2 * self.k:3 * self.k, :, :]

        pi_clipped = torch.clamp(pi_raw, PI_CLIP_EPS, 1.0 - PI_CLIP_EPS)  # Eq. 14's C(.)
        pi = pi_clipped / pi_clipped.sum(dim=1, keepdim=True)

        log_s = torch.clamp(log_s_raw, min=MIN_LOG_SCALE)

        if return_debug:
            return pi, mu, log_s, {
                "hidden_pre": hidden_pre,
                "hidden": hidden,
                "raw": raw,
                "pi_raw": pi_raw,
                "pi_clipped": pi_clipped,
                "log_s_raw": log_s_raw,
            }

        return pi, mu, log_s


class PDEModule(nn.Module):

    def __init__(self, channels=M, k=K):
        super().__init__()

        self.masked_conv = MaskedConv7x7(
            MASKED_CONV_KERNEL,
            channels
        )

        self.dsc_branch = DSCBranch(
            channels,
            DSC_KERNEL
        )

        self.gate = Gate()

        self.estimator_head = EstimatorHead(
            channels,
            k
        )

    def forward(
        self,
        x,
        h_prev,
        return_debug=False
    ):

        mc_feat = self.masked_conv(x)

        dsc_feat = self.dsc_branch(h_prev)

        h_s = self.gate(
            mc_feat,
            dsc_feat
        )

        if return_debug:
            pi, mu, log_s, head_debug = self.estimator_head(
                h_s,
                return_debug=True,
            )

            debug = {
                "mc_feat": mc_feat,
                "dsc_feat": dsc_feat,
                "h_s": h_s,
                "pi": pi,
                "mu": mu,
                "log_s": log_s,
                **head_debug,
            }

            return (
                pi,
                mu,
                log_s,
                h_s,
                debug
            )

        pi, mu, log_s = self.estimator_head(h_s)

        return (
            pi,
            mu,
            log_s,
            h_s
        )

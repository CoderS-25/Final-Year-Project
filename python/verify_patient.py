# ============================================================
# HENCE PDE - SINGLE PATIENT PYTHON VERIFICATION
#
# Purpose:
#   1. Load one normalized patient volume
#   2. Load MATLAB-exported trained weights
#   3. Run the Python PDE module slice-by-slice
#   4. Calculate BPP
#   5. Save Python intermediate tensors
#   6. Compare Python Slice 1 against MATLAB debug output
#
# Patient: 27
# ============================================================

import os
import sys
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat, savemat


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = r"C:\Users\user\Documents\Final Year Project"

# Python project
PYTHON_ROOT = os.path.join(
    PROJECT_ROOT,
    "hence_pde_project_v3",
    "python"
)

# Add Python project to import path
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)


# ------------------------------------------------------------
# Patient data
# ------------------------------------------------------------

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "chaos_processed_t2spir"
)

PATIENT_ID = 27

PATIENT_FILE = os.path.join(
    DATA_ROOT,
    f"{PATIENT_ID}_norm.mat"
)


# ------------------------------------------------------------
# MATLAB exported weights
# ------------------------------------------------------------

WEIGHTS_FILE = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "chaos_paper_16bit",
    "best_weights.mat"
)


# ------------------------------------------------------------
# Python output
# ------------------------------------------------------------

PYTHON_OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    f"patient_{PATIENT_ID}_python_bpp_stagewise_tf32off_paperloss.mat"
)


# ------------------------------------------------------------
# MATLAB debug file
#
# This is the file you created in MATLAB using:
#
# save('patient_27_matlab_debug.mat', ...
#      '-struct', 'matlab_debug', '-v7.3');
# ------------------------------------------------------------

MATLAB_DEBUG_FILE = os.path.join(
    PROJECT_ROOT,
    f"patient_{PATIENT_ID}_matlab_debug_stagewise.mat"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

BIT_DEPTH = 16
NUM_BINS = 65536
SCALE_L = 8.0

CHANNELS = 16
NUM_MIXTURES = 3


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# MATLAB is a double-precision numerical reference. CUDA TensorFloat-32
# rounds float32 convolution inputs/multiplications and first diverges in
# the estimator's 1x1 convolution. Disable TF32 for this parity script;
# this does not change the HENCE architecture, parameters, or equations.
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


# ============================================================
# IMPORT PDE MODEL
# ============================================================

from pde_model import PDEModule


# ============================================================
# PRINT HEADER
# ============================================================

def print_header():

    print()
    print("=" * 60)
    print("HENCE PDE - SINGLE PATIENT PYTHON VERIFICATION")
    print("=" * 60)

    print()
    print(f"Patient      : {PATIENT_ID}")
    print(f"Device       : {DEVICE}")
    print(f"Data folder  : {DATA_ROOT}")
    print(f"Weights      : {WEIGHTS_FILE}")
    print(f"Bit depth    : {BIT_DEPTH}")
    print(f"Num bins     : {NUM_BINS}")
    print(f"Scale L      : {SCALE_L}")

    print()


# ============================================================
# LOAD PATIENT VOLUME
# ============================================================

def load_patient_volume():

    print("Loading patient volume:")
    print(PATIENT_FILE)

    if not os.path.isfile(PATIENT_FILE):

        raise FileNotFoundError(
            "\nPatient file not found:\n"
            + PATIENT_FILE
            + "\n\n"
            "Check that the file is actually named:\n"
            f"{PATIENT_ID}_norm.mat"
        )

    mat = loadmat(PATIENT_FILE)

    # --------------------------------------------------------
    # Preferred variable name
    # --------------------------------------------------------

    if "volume" in mat:

        volume = mat["volume"]

        print()
        print("Using MAT variable: volume")

    else:

        candidates = []

        for key, value in mat.items():

            if key.startswith("__"):
                continue

            if isinstance(value, np.ndarray):

                if value.ndim == 3:

                    candidates.append(
                        (key, value)
                    )

        if len(candidates) == 0:

            raise RuntimeError(
                "Could not find a 3-D volume in MAT file."
            )

        key, volume = candidates[0]

        print()
        print(
            f"MAT variable 'volume' not found."
        )

        print(
            f"Using first 3-D variable instead: {key}"
        )

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    volume = np.asarray(
        volume,
        dtype=np.float32
    )

    print()
    print("Original volume:")
    print(f"Shape : {volume.shape}")
    print(f"Min   : {volume.min():.6f}")
    print(f"Max   : {volume.max():.6f}")

    # --------------------------------------------------------
    # Make sure volume is H x W x S
    #
    # Expected:
    #
    #     256 x 256 x 30
    #
    # --------------------------------------------------------

    if volume.ndim != 3:

        raise RuntimeError(
            f"Expected 3-D volume, got shape {volume.shape}"
        )

    # We expect the slice dimension to be the smallest
    # dimension for this dataset.
    #
    # Example:
    #     (256,256,30)
    #
    # If MATLAB accidentally stored:
    #     (30,256,256)
    #
    # transpose it.

    if (
        volume.shape[0] < volume.shape[1]
        and volume.shape[0] < volume.shape[2]
    ):

        print()
        print(
            "Detected slice dimension at axis 0."
        )

        volume = np.transpose(
            volume,
            (1, 2, 0)
        )

        print(
            f"Transposed volume to H x W x S: "
            f"{volume.shape}"
        )

    elif (
        volume.shape[1] < volume.shape[0]
        and volume.shape[1] < volume.shape[2]
    ):

        print()
        print(
            "Detected slice dimension at axis 1."
        )

        volume = np.transpose(
            volume,
            (0, 2, 1)
        )

        print(
            f"Transposed volume to H x W x S: "
            f"{volume.shape}"
        )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    vmin = float(volume.min())
    vmax = float(volume.max())

    if vmin >= -1.0001 and vmax <= 1.0001:

        print()
        print(
            "Volume is already normalized to [-1,1]."
        )

    else:

        print()
        print(
            "Volume is not normalized to [-1,1]."
        )

        print(
            "Applying min-max normalization."
        )

        if vmax == vmin:

            raise RuntimeError(
                "Volume has zero dynamic range."
            )

        volume = (
            2.0
            * (volume - vmin)
            / (vmax - vmin)
            - 1.0
        )

    print()
    print("Normalized volume:")
    print(f"Min   : {volume.min():.6f}")
    print(f"Max   : {volume.max():.6f}")

    H, W, S = volume.shape

    print()
    print(
        f"Volume shape : {H} x {W} x {S}"
    )

    return volume


# ============================================================
# PRINT MODEL PARAMETERS
# ============================================================

def print_model_parameters(model):

    print()
    print("PDE model parameters:")

    for name, param in model.named_parameters():

        print(
            f"{name:<40}"
            f"{tuple(param.shape)}"
        )


# ============================================================
# LOAD MATLAB WEIGHTS
# ============================================================

def load_weights_into_model(
    model,
    mat_path,
    device
):

    print()
    print("Loading MATLAB-exported weights:")
    print(mat_path)

    if not os.path.isfile(mat_path):

        raise FileNotFoundError(
            "\nMATLAB weight file not found:\n"
            + mat_path
        )

    mat = loadmat(mat_path)

    required = [
        "masked_conv_W",
        "masked_conv_b",
        "dsc_depthwise",
        "dsc_pointwise",
        "dsc_pointwise_b",
        "head_W1",
        "head_b1",
        "head_W2",
        "head_b2",
    ]

    for key in required:

        if key not in mat:

            raise KeyError(
                f"Missing MATLAB weight: {key}"
            )

    # --------------------------------------------------------
    # Extract MATLAB arrays
    # --------------------------------------------------------

    masked_conv_W = np.asarray(
        mat["masked_conv_W"],
        dtype=np.float32
    )

    masked_conv_b = np.asarray(
        mat["masked_conv_b"],
        dtype=np.float32
    ).reshape(-1)

    dsc_depthwise = np.asarray(
        mat["dsc_depthwise"],
        dtype=np.float32
    )

    dsc_pointwise = np.asarray(
        mat["dsc_pointwise"],
        dtype=np.float32
    )

    dsc_pointwise_b = np.asarray(
        mat["dsc_pointwise_b"],
        dtype=np.float32
    ).reshape(-1)

    head_W1 = np.asarray(
        mat["head_W1"],
        dtype=np.float32
    )

    head_b1 = np.asarray(
        mat["head_b1"],
        dtype=np.float32
    ).reshape(-1)

    head_W2 = np.asarray(
        mat["head_W2"],
        dtype=np.float32
    )

    head_b2 = np.asarray(
        mat["head_b2"],
        dtype=np.float32
    ).reshape(-1)

    # --------------------------------------------------------
    # Print MATLAB shapes
    # --------------------------------------------------------

    print()
    print("MATLAB weight shapes:")

    print(
        f"masked_conv_W      : "
        f"{masked_conv_W.shape}"
    )

    print(
        f"masked_conv_b      : "
        f"{masked_conv_b.shape}"
    )

    print(
        f"dsc_depthwise      : "
        f"{dsc_depthwise.shape}"
    )

    print(
        f"dsc_pointwise      : "
        f"{dsc_pointwise.shape}"
    )

    print(
        f"dsc_pointwise_b    : "
        f"{dsc_pointwise_b.shape}"
    )

    print(
        f"head_W1            : "
        f"{head_W1.shape}"
    )

    print(
        f"head_b1            : "
        f"{head_b1.shape}"
    )

    print(
        f"head_W2            : "
        f"{head_W2.shape}"
    )

    print(
        f"head_b2            : "
        f"{head_b2.shape}"
    )

    # ========================================================
    # CONVERT MATLAB -> PYTORCH
    # ========================================================

    # --------------------------------------------------------
    # masked convolution
    #
    # MATLAB:
    #
    #     (7,7,1,16)
    #
    # PyTorch:
    #
    #     (16,1,7,7)
    # --------------------------------------------------------

    masked_conv_W_pt = np.transpose(
        masked_conv_W,
        (3, 2, 0, 1)
    )

    # --------------------------------------------------------
    # depthwise
    #
    # MATLAB:
    #
    #     (5,5,16)
    #
    # PyTorch:
    #
    #     (16,1,5,5)
    # --------------------------------------------------------

    dsc_depthwise_pt = np.transpose(
        dsc_depthwise,
        (2, 0, 1)
    )[:, np.newaxis, :, :]

    # --------------------------------------------------------
    # pointwise
    #
    # MATLAB:
    #
    #     (Cin,Cout)
    #
    # PyTorch:
    #
    #     (Cout,Cin,1,1)
    # --------------------------------------------------------

    dsc_pointwise_pt = np.transpose(
        dsc_pointwise,
        (1, 0)
    )[:, :, np.newaxis, np.newaxis]

    # --------------------------------------------------------
    # estimator head W1
    # --------------------------------------------------------

    head_W1_pt = np.transpose(
        head_W1,
        (1, 0)
    )[:, :, np.newaxis, np.newaxis]

    # --------------------------------------------------------
    # estimator head W2
    #
    # MATLAB:
    #
    #     (16,9)
    #
    # PyTorch:
    #
    #     (9,16,1,1)
    # --------------------------------------------------------

    head_W2_pt = np.transpose(
        head_W2,
        (1, 0)
    )[:, :, np.newaxis, np.newaxis]

    # --------------------------------------------------------
    # Biases
    # --------------------------------------------------------

    masked_conv_b_pt = masked_conv_b

    dsc_pointwise_b_pt = dsc_pointwise_b

    head_b1_pt = head_b1

    head_b2_pt = head_b2

    # ========================================================
    # CONVERT TO TORCH
    # ========================================================

    converted = {

        "masked_conv.conv.weight":
            torch.from_numpy(
                masked_conv_W_pt
            ),

        "masked_conv.conv.bias":
            torch.from_numpy(
                masked_conv_b_pt
            ),

        "dsc_branch.depthwise.weight":
            torch.from_numpy(
                dsc_depthwise_pt
            ),

        "dsc_branch.pointwise.weight":
            torch.from_numpy(
                dsc_pointwise_pt
            ),

        "dsc_branch.pointwise.bias":
            torch.from_numpy(
                dsc_pointwise_b_pt
            ),

        "estimator_head.conv1.weight":
            torch.from_numpy(
                head_W1_pt
            ),

        "estimator_head.conv1.bias":
            torch.from_numpy(
                head_b1_pt
            ),

        "estimator_head.conv2.weight":
            torch.from_numpy(
                head_W2_pt
            ),

        "estimator_head.conv2.bias":
            torch.from_numpy(
                head_b2_pt
            ),
    }

    # ========================================================
    # CHECK AGAINST MODEL
    # ========================================================

    state_dict = model.state_dict()

    print()
    print(
        "PyTorch target / converted weight shapes:"
    )

    for name, value in converted.items():

        if name not in state_dict:

            raise KeyError(
                f"Model does not contain parameter: {name}"
            )

        expected = tuple(
            state_dict[name].shape
        )

        actual = tuple(
            value.shape
        )

        print(
            f"{name:<40}"
            f"{actual!s:<22}"
            f"expected {expected}"
        )

        if actual != expected:

            raise RuntimeError(
                "\nShape mismatch for "
                f"{name}: "
                f"{actual} != {expected}"
            )

    # ========================================================
    # LOAD
    # ========================================================

    state_dict.update(converted)

    model.load_state_dict(
        state_dict
    )

    model.to(device)

    model.eval()

    print()
    print(
        "Successfully loaded MATLAB-exported weights."
    )


# ============================================================
# MIXED LOGISTIC BPP
# ============================================================

def mixture_logistic_bpp_loss(
    x,
    pi,
    mu,
    log_s,
    num_bins=256,
    scale_factor=1.0
):

    """
    x:
        [N,1,H,W]

    pi:
        [N,3,H,W]

    mu:
        [N,3,H,W]

    log_s:
        [N,3,H,W]

    Returns:
        scalar bits-per-pixel
    """

    # Match train_pde.py and MATLAB's mixtureLogisticBppLoss.m.  These
    # are the paper's discretized-logistic bin edges, including stable
    # handling of the normalized-domain endpoints x = -1 and x = 1.
    x = x.expand_as(mu)
    half_bin = scale_factor / (num_bins - 1)
    centered = x - mu
    inv_s = torch.exp(-log_s)

    plus_in = inv_s * (centered + half_bin)
    min_in = inv_s * (centered - half_bin)
    prob = torch.sigmoid(plus_in) - torch.sigmoid(min_in)

    log_prob_edge_low = plus_in - F.softplus(plus_in)
    log_prob_edge_high = -F.softplus(min_in)
    log_prob_mid = torch.log(torch.clamp(prob, min=1e-12))

    is_low = (x <= -1 + 1e-3).float()
    is_high = (x >= 1 - 1e-3).float()
    is_mid = 1.0 - is_low - is_high
    log_prob = (
        is_low * log_prob_edge_low
        + is_high * log_prob_edge_high
        + is_mid * log_prob_mid
    )

    log_pi = torch.log(torch.clamp(pi, min=1e-12))
    mixture_log_prob = torch.logsumexp(log_pi + log_prob, dim=1)
    return -mixture_log_prob.mean() / np.log(2.0)


# ============================================================
# RUN ONE PATIENT
# ============================================================

@torch.no_grad()
def run_patient(
    model,
    volume
):

    H, W, S = volume.shape

    # --------------------------------------------------------
    # Hidden state
    #
    # MATLAB:
    #
    # H0 = zeros(H,W,16)
    #
    # PyTorch:
    #
    # [1,16,H,W]
    # --------------------------------------------------------

    h_prev = torch.zeros(
        1,
        CHANNELS,
        H,
        W,
        dtype=torch.float32,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    pi_all = np.zeros(
        (H, W, NUM_MIXTURES, S),
        dtype=np.float32
    )

    mu_all = np.zeros(
        (H, W, NUM_MIXTURES, S),
        dtype=np.float32
    )

    log_s_all = np.zeros(
        (H, W, NUM_MIXTURES, S),
        dtype=np.float32
    )

    mc_all = np.zeros(
        (H, W, CHANNELS, S),
        dtype=np.float32
    )

    dsc_all = np.zeros(
        (H, W, CHANNELS, S),
        dtype=np.float32
    )

    h_all = np.zeros(
        (H, W, CHANNELS, S),
        dtype=np.float32
    )

    slice_bpp = np.zeros(
        S,
        dtype=np.float64
    )

    slice1_head_debug = {}

    # --------------------------------------------------------
    # Slice loop
    # --------------------------------------------------------

    for s in range(S):

        # ----------------------------------------------------
        # MATLAB:
        #
        # slice = volume(:,:,s)
        #
        # PyTorch:
        #
        # [1,1,H,W]
        # ----------------------------------------------------

        current_slice = torch.from_numpy(
            volume[:, :, s]
        ).float()

        current_slice = current_slice.to(
            DEVICE
        )

        current_slice = current_slice.unsqueeze(
            0
        ).unsqueeze(
            0
        )

        # ----------------------------------------------------
        # PDE forward
        # ----------------------------------------------------

        result = model(
            current_slice,
            h_prev,
            return_debug=True
        )

        pi, mu, log_s, h_s, debug = result

        # ----------------------------------------------------
        # BPP
        # ----------------------------------------------------

        bpp = mixture_logistic_bpp_loss(
            current_slice,
            pi,
            mu,
            log_s,
            NUM_BINS,
            SCALE_L
        )

        slice_bpp[s] = float(
            bpp.item()
        )

        # ----------------------------------------------------
        # Convert outputs
        #
        # PyTorch:
        #
        # [1,C,H,W]
        #
        # MATLAB:
        #
        # [H,W,C]
        # ----------------------------------------------------

        pi_np = (
            pi[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        mu_np = (
            mu[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        log_s_np = (
            log_s[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        mc_np = (
            debug["mc_feat"][0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        dsc_np = (
            debug["dsc_feat"][0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        h_np = (
            h_s[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        if s == 0:
            for name in (
                "hidden_pre",
                "hidden",
                "raw",
                "pi_raw",
                "pi_clipped",
                "log_s_raw",
            ):
                slice1_head_debug[name] = (
                    debug[name][0]
                    .permute(1, 2, 0)
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        pi_all[:, :, :, s] = pi_np
        mu_all[:, :, :, s] = mu_np
        log_s_all[:, :, :, s] = log_s_np

        mc_all[:, :, :, s] = mc_np
        dsc_all[:, :, :, s] = dsc_np
        h_all[:, :, :, s] = h_np

        # ----------------------------------------------------
        # Hidden state recurrence
        #
        # IMPORTANT:
        #
        # H_s becomes H_prev for slice s+1.
        # ----------------------------------------------------

        h_prev = h_s

        print(
            f"Slice {s + 1:02d}/{S} | "
            f"BPP = {slice_bpp[s]:.6f}"
        )

    # --------------------------------------------------------
    # Average BPP
    # --------------------------------------------------------

    mean_bpp = float(
        np.mean(slice_bpp)
    )

    return (
        slice_bpp,
        mean_bpp,
        pi_all,
        mu_all,
        log_s_all,
        mc_all,
        dsc_all,
        h_all,
        slice1_head_debug,
    )


# ============================================================
# PYTHON NUMERICAL SANITY CHECKS
# ============================================================

def sanity_checks(
    pi,
    mu,
    log_s,
    H
):

    print()
    print("Numerical sanity checks:")

    print(
        f"NaN in pi    : "
        f"{np.isnan(pi).any()}"
    )

    print(
        f"NaN in mu    : "
        f"{np.isnan(mu).any()}"
    )

    print(
        f"NaN in log_s : "
        f"{np.isnan(log_s).any()}"
    )

    print(
        f"NaN in H     : "
        f"{np.isnan(H).any()}"
    )

    # --------------------------------------------------------
    # Mixture sum
    # --------------------------------------------------------

    pi_sum = np.sum(
        pi,
        axis=2
    )

    print()
    print("Mixture-weight check:")

    print(
        f"Minimum sum(pi) : "
        f"{pi_sum.min()}"
    )

    print(
        f"Maximum sum(pi) : "
        f"{pi_sum.max()}"
    )


# ============================================================
# FIRST PIXEL DEBUG
# ============================================================

def print_python_slice1_debug(
    mc,
    dsc,
    H,
    pi,
    mu,
    log_s
):

    print()
    print("=" * 44)
    print("PYTHON SLICE 1 FIRST PIXEL")
    print("=" * 44)

    print()
    print("MC(1,1,:) =")
    print(
        mc[0, 0, :, 0]
    )

    print()
    print("DSC(1,1,:) =")
    print(
        dsc[0, 0, :, 0]
    )

    print()
    print("H1(1,1,:) =")
    print(
        H[0, 0, :, 0]
    )

    print()
    print("PI(1,1,:) =")
    print(
        pi[0, 0, :, 0]
    )

    print()
    print("MU(1,1,:) =")
    print(
        mu[0, 0, :, 0]
    )

    print()
    print("LOG_S(1,1,:) =")
    print(
        log_s[0, 0, :, 0]
    )


# ============================================================
# COMPARE ARRAY
# ============================================================

def compare_array(
    name,
    matlab_array,
    python_array
):

    matlab_array = np.asarray(
        matlab_array,
        dtype=np.float64
    )

    python_array = np.asarray(
        python_array,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Check shapes
    # --------------------------------------------------------

    print()
    print(f"{name} comparison:")

    print(
        f"MATLAB shape : "
        f"{matlab_array.shape}"
    )

    print(
        f"Python shape : "
        f"{python_array.shape}"
    )

    if matlab_array.shape != python_array.shape:

        print(
            "SHAPE MISMATCH"
        )

        return

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    diff = np.abs(
        matlab_array
        - python_array
    )

    print(
        f"Maximum = "
        f"{diff.max():.12e}"
    )

    print(
        f"Mean    = "
        f"{diff.mean():.12e}"
    )

    print(
        f"RMSE    = "
        f"{np.sqrt(np.mean(diff ** 2)):.12e}"
    )

    # --------------------------------------------------------
    # Percentage of nearly identical values
    # --------------------------------------------------------

    close = np.isclose(
        matlab_array,
        python_array,
        rtol=1e-5,
        atol=1e-5
    )

    percentage = (
        100.0
        * np.mean(close)
    )

    print(
        f"Within 1e-5 = "
        f"{percentage:.4f}%"
    )


# ============================================================
# MATLAB vs PYTHON COMPARISON
# ============================================================

def compare_matlab_debug(
    mc,
    dsc,
    H,
    pi,
    mu,
    log_s,
    python_head_debug,
):

    print()
    print("=" * 60)
    print("MATLAB vs PYTHON - SLICE 1 COMPARISON")
    print("=" * 60)

    if not os.path.isfile(
        MATLAB_DEBUG_FILE
    ):

        print()
        print(
            "MATLAB debug file not found:"
        )

        print(
            MATLAB_DEBUG_FILE
        )

        print()
        print(
            "Skipping MATLAB/Python comparison."
        )

        return

    print()
    print(
        "Loading MATLAB debug file:"
    )

    print(
        MATLAB_DEBUG_FILE
    )

    mat = loadmat(
        MATLAB_DEBUG_FILE
    )

    required = [
        "MC",
        "DSC",
        "H",
        "pi",
        "mu",
        "log_s",
    ]

    missing = [
        key
        for key in required
        if key not in mat
    ]

    if missing:

        raise RuntimeError(
            "MATLAB debug file is missing: "
            + str(missing)
        )

    # --------------------------------------------------------
    # MATLAB arrays
    #
    # The file contains:
    #
    # MC  = H x W x 16
    # DSC = H x W x 16
    # H   = H x W x 16
    # pi  = H x W x 3
    # mu  = H x W x 3
    # log_s = H x W x 3
    # --------------------------------------------------------

    matlab_mc = mat["MC"]
    matlab_dsc = mat["DSC"]
    matlab_H = mat["H"]

    matlab_pi = mat["pi"]
    matlab_mu = mat["mu"]
    matlab_log_s = mat["log_s"]

    # --------------------------------------------------------
    # Python slice 1
    # --------------------------------------------------------

    python_mc = mc[:, :, :, 0]
    python_dsc = dsc[:, :, :, 0]
    python_H = H[:, :, :, 0]

    python_pi = pi[:, :, :, 0]
    python_mu = mu[:, :, :, 0]
    python_log_s = log_s[:, :, :, 0]

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    compare_array(
        "MC",
        matlab_mc,
        python_mc
    )

    compare_array(
        "DSC",
        matlab_dsc,
        python_dsc
    )

    compare_array(
        "H",
        matlab_H,
        python_H
    )

    compare_array(
        "PI",
        matlab_pi,
        python_pi
    )

    compare_array(
        "MU",
        matlab_mu,
        python_mu
    )

    compare_array(
        "LOG_S",
        matlab_log_s,
        python_log_s
    )

    stage_names = (
        "hidden_pre",
        "hidden",
        "raw",
        "pi_raw",
        "pi_clipped",
        "log_s_raw",
    )

    print()
    print("ESTIMATOR HEAD - STAGEWISE COMPARISON")

    for name in stage_names:
        if name not in mat:
            raise RuntimeError(
                f"MATLAB debug file is missing estimator stage: {name}"
            )
        compare_array(
            name,
            mat[name],
            python_head_debug[name],
        )

    # --------------------------------------------------------
    # First pixel comparison
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FIRST PIXEL - MATLAB vs PYTHON")
    print("=" * 60)

    print()
    print("MATLAB MC(1,1,:)")
    print(
        matlab_mc[0, 0, :]
    )

    print()
    print("Python MC(1,1,:)")
    print(
        python_mc[0, 0, :]
    )

    print()
    print("MATLAB DSC(1,1,:)")
    print(
        matlab_dsc[0, 0, :]
    )

    print()
    print("Python DSC(1,1,:)")
    print(
        python_dsc[0, 0, :]
    )

    print()
    print("MATLAB H(1,1,:)")
    print(
        matlab_H[0, 0, :]
    )

    print()
    print("Python H(1,1,:)")
    print(
        python_H[0, 0, :]
    )

    print()
    print("MATLAB PI(1,1,:)")
    print(
        matlab_pi[0, 0, :]
    )

    print()
    print("Python PI(1,1,:)")
    print(
        python_pi[0, 0, :]
    )

    print()
    print("MATLAB MU(1,1,:)")
    print(
        matlab_mu[0, 0, :]
    )

    print()
    print("Python MU(1,1,:)")
    print(
        python_mu[0, 0, :]
    )

    print()
    print("MATLAB LOG_S(1,1,:)")
    print(
        matlab_log_s[0, 0, :]
    )

    print()
    print("Python LOG_S(1,1,:)")
    print(
        python_log_s[0, 0, :]
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    volume,
    slice_bpp,
    mean_bpp,
    pi,
    mu,
    log_s,
    mc,
    dsc,
    H,
    slice1_head_debug,
):

    output = {

        "patient_id":
            np.array(
                [[PATIENT_ID]],
                dtype=np.int32
            ),

        "slice_bpp_python":
            slice_bpp,

        "mean_bpp_python":
            np.array(
                [[mean_bpp]],
                dtype=np.float64
            ),

        "pi_python":
            pi,

        "mu_python":
            mu,

        "log_s_python":
            log_s,

        "MC_python":
            mc,

        "DSC_python":
            dsc,

        "H_python":
            H,

        "volume_python":
            volume,

        "weight_file_sha256": np.array([
            hashlib.sha256(
                open(WEIGHTS_FILE, "rb").read()
            ).hexdigest()
        ]),
    }

    for name, value in slice1_head_debug.items():
        output[f"{name}_python"] = value

    savemat(
        PYTHON_OUTPUT_FILE,
        output,
        do_compression=False
    )

    print()
    print(
        "Saved Python verification results to:"
    )

    print(
        PYTHON_OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print_header()

    # --------------------------------------------------------
    # Load volume
    # --------------------------------------------------------

    volume = load_patient_volume()

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print("Creating PDE model...")

    model = PDEModule(
        channels=CHANNELS,
        k=NUM_MIXTURES
    )

    model = model.to(
        DEVICE
    )

    print_model_parameters(
        model
    )

    # --------------------------------------------------------
    # Load MATLAB weights
    # --------------------------------------------------------

    load_weights_into_model(
        model,
        WEIGHTS_FILE,
        DEVICE
    )

    # --------------------------------------------------------
    # Run patient
    # --------------------------------------------------------

    (
        slice_bpp,
        mean_bpp,
        pi,
        mu,
        log_s,
        mc,
        dsc,
        H,
        slice1_head_debug,
    ) = run_patient(
        model,
        volume
    )

    # --------------------------------------------------------
    # Average BPP
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        f"Python Patient {PATIENT_ID} "
        f"BPP = {mean_bpp:.4f}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Intermediate shapes
    # --------------------------------------------------------

    print()
    print("Intermediate parameter shapes:")

    print(
        f"pi_python    : "
        f"{pi.shape}"
    )

    print(
        f"mu_python    : "
        f"{mu.shape}"
    )

    print(
        f"log_s_python : "
        f"{log_s.shape}"
    )

    print(
        f"MC           : "
        f"{mc.shape}"
    )

    print(
        f"DSC          : "
        f"{dsc.shape}"
    )

    print(
        f"H            : "
        f"{H.shape}"
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    sanity_checks(
        pi,
        mu,
        log_s,
        H
    )

    # --------------------------------------------------------
    # First pixel
    # --------------------------------------------------------

    print_python_slice1_debug(
        mc,
        dsc,
        H,
        pi,
        mu,
        log_s,
    )

    # --------------------------------------------------------
    # MATLAB comparison
    # --------------------------------------------------------

    compare_matlab_debug(
        mc,
        dsc,
        H,
        pi,
        mu,
        log_s,
        slice1_head_debug,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        volume,
        slice_bpp,
        mean_bpp,
        pi,
        mu,
        log_s,
        mc,
        dsc,
        H,
        slice1_head_debug,
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print(
        "Verification complete."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

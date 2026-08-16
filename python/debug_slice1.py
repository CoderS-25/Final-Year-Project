import os
import sys
import numpy as np
import torch
from scipy.io import loadmat

# ------------------------------------------------------------
# Import your existing model
# ------------------------------------------------------------
from pde_model import PDEModule


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE = r"C:\Users\user\Documents\Final Year Project"

DATA_DIR = os.path.join(
    BASE,
    "chaos_processed_t2spir"
)

WEIGHTS_PATH = os.path.join(
    BASE,
    "best_trained_weights.mat"
)

PATIENT_ID = 27

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("============================================")
print("PYTHON SLICE 1 INTERMEDIATE DEBUG")
print("============================================")
print("Device :", device)


# ------------------------------------------------------------
# Load patient volume
# ------------------------------------------------------------
mat_path = os.path.join(
    DATA_DIR,
    f"{PATIENT_ID}_norm.mat"
)

mat = loadmat(mat_path)

# Find the actual volume variable.
if "volume" in mat:
    volume = mat["volume"]
else:
    candidates = [
        v for k, v in mat.items()
        if not k.startswith("__")
        and isinstance(v, np.ndarray)
        and v.ndim == 3
    ]

    if not candidates:
        raise RuntimeError(
            "Could not find a 3D volume in the MAT file."
        )

    volume = candidates[0]

volume = np.asarray(volume, dtype=np.float32)

print("Volume shape:", volume.shape)
print("Volume min  :", volume.min())
print("Volume max  :", volume.max())


# ------------------------------------------------------------
# Dataset format
# ------------------------------------------------------------
# MATLAB volume is:
#       H x W x S
#
# PyTorch model expects:
#       B x C x H x W
#
# for one slice.
# ------------------------------------------------------------

slice1 = volume[:, :, 0]

x = torch.from_numpy(slice1)
x = x.unsqueeze(0).unsqueeze(0)
x = x.to(device)

print("Slice tensor:", tuple(x.shape))


# ------------------------------------------------------------
# Hidden state
# ------------------------------------------------------------
M = 16

H, W = slice1.shape

h_prev = torch.zeros(
    1,
    M,
    H,
    W,
    device=device
)


# ------------------------------------------------------------
# Load MATLAB-exported weights
# ------------------------------------------------------------
raw = loadmat(WEIGHTS_PATH)

print("\nLoading weights:")
print(WEIGHTS_PATH)


# ------------------------------------------------------------
# Create model
# ------------------------------------------------------------
model = PDEModule(
    channels=M,
    k=3
).to(device)

model.eval()


# ------------------------------------------------------------
# Load exported weights
#
# IMPORTANT:
# Use the SAME loading function that verify_patient.py
# already uses if your project has one.
# ------------------------------------------------------------

# If your verify_patient.py already contains a function such as
# load_matlab_weights(model, WEIGHTS_PATH), use that function here.
#
# Otherwise, STOP here rather than guessing the exact loader.


print("\nModel created.")
print("Now run the same MATLAB-weight loading code used by")
print("verify_patient.py.")
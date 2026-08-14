import torch
import scipy.io as sio
import numpy as np

from pde_model import MaskedConv7x7


print("=" * 60)
print("MC ISOLATION TEST")
print("=" * 60)

mat_path = r"C:\Users\user\Documents\Final Year Project\patient_27_matlab_debug_stagewise.mat"

data = sio.loadmat(mat_path)

print("\nMATLAB debug variables:")
for k, v in data.items():
    if not k.startswith("__"):
        print(k, np.shape(v))


# ------------------------------------------------------------
# Load MATLAB MC
# ------------------------------------------------------------

mc_mat = data["MC"]

print("\nMATLAB MC:")
print("shape:", mc_mat.shape)
print("first pixel:")
print(mc_mat[0, 0, :])


# ------------------------------------------------------------
# Load patient volume
# ------------------------------------------------------------

vol_data = sio.loadmat(
    r"C:\Users\user\Documents\Final Year Project\chaos_processed_t2spir\27_norm.mat"
)

x = vol_data["volume"].astype(np.float32)

print("\nVolume:")
print("shape:", x.shape)
print("min:", x.min())
print("max:", x.max())


# MATLAB volume is H x W x S
# Take slice 1
slice1 = x[:, :, 0]

# PyTorch = B C H W
x_t = torch.from_numpy(slice1).unsqueeze(0).unsqueeze(0)

print("\nPyTorch input:")
print("shape:", x_t.shape)
print("first pixel:", x_t[0, 0, 0, 0].item())


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = MaskedConv7x7()
model.eval()


# ------------------------------------------------------------
# Load MATLAB weights
# ------------------------------------------------------------

weights = sio.loadmat(
    r"C:\Users\user\Documents\Final Year Project\experiments\chaos_paper_16bit\best_weights.mat"
)

W = weights["masked_conv_W"]
b = weights["masked_conv_b"].reshape(-1)

print("\nMATLAB weight shape:", W.shape)
print("MATLAB bias shape:", b.shape)


# MATLAB:
# H x W x input_channels x output_channels
#
# PyTorch:
# output_channels x input_channels x H x W

W_pt = np.transpose(W, (3, 2, 0, 1))

print("Converted weight shape:", W_pt.shape)

with torch.no_grad():
    model.conv.weight.copy_(torch.from_numpy(W_pt.astype(np.float32)))
    model.conv.bias.copy_(torch.from_numpy(b.astype(np.float32)))


# ------------------------------------------------------------
# Show raw weights
# ------------------------------------------------------------

print("\nFirst output channel weights:")
print(model.conv.weight[0, 0])

print("\nPyTorch loaded bias:")
print(model.conv.bias.detach().cpu().numpy())

print("\nMATLAB bias:")
print(b)

print("\nMATLAB MC first pixel:")
print(mc_mat[0, 0, :])

print("\nPython MC first pixel:")
print(mc_py[0, 0, :])

print("\nMATLAB MC first pixel - MATLAB bias:")
print(mc_mat[0, 0, :] - b)

print("\nPython MC first pixel - Python bias:")
print(mc_py[0, 0, :] - model.conv.bias.detach().cpu().numpy())

# ------------------------------------------------------------
# Apply MC
# ------------------------------------------------------------

with torch.no_grad():
    mc_py = model(x_t)

mc_py = mc_py[0].permute(1, 2, 0).numpy()

print("\nPython MC:")
print("shape:", mc_py.shape)
print("first pixel:")
print(mc_py[0, 0, :])


# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

diff = np.abs(mc_mat[:, :, :] - mc_py)

print("\nComparison:")
print("max :", diff.max())
print("mean:", diff.mean())
print("rmse:", np.sqrt(np.mean(diff ** 2)))

print(
    "within 1e-5:",
    np.mean(diff <= 1e-5) * 100,
    "%"
)


# ------------------------------------------------------------
# Compare interior only
# ------------------------------------------------------------

margin = 4

mc_mat_inner = mc_mat[
    margin:-margin,
    margin:-margin,
    :
]

mc_py_inner = mc_py[
    margin:-margin,
    margin:-margin,
    :
]

diff_inner = np.abs(mc_mat_inner - mc_py_inner)

print("\nInterior-only comparison:")
print("max :", diff_inner.max())
print("mean:", diff_inner.mean())
print("rmse:", np.sqrt(np.mean(diff_inner ** 2)))

print(
    "within 1e-5:",
    np.mean(diff_inner <= 1e-5) * 100,
    "%"
)
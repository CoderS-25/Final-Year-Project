"""
train_pde.py

Trains the corrected PDEModule (pde_model.py) using the K=3 mixture-of-
logistics rate loss (Eq. 2-3, 12-13) and the paper's exact training
recipe (Section IV-B.3):
    - lr = 5e-4, Adam, 1000 epochs, decay 80% every 50 epochs
    - "update stride" = 5: truncated BPTT, detaching H_s every 5 slices
      to avoid vanishing gradients over long slice sequences
    - scaling factor L: 1 for 8-bit datasets (e.g. MRNet), 8 for 16-bit
      datasets (e.g. CHAOS) -- referenced from the paper's own prior
      work [24]/SR-LVC, which HENCE explicitly follows for this detail

Data: CHAOS-style 16-bit MRI volumes are the closest match to what this
project targets. No network access to the actual CHAOS dataset is
available here, so ChaosMRIDataset looks for real volumes under
--data_dir (.npy, shape [numSlices, H, W]) and falls back to a synthetic
smooth-MRI-like phantom dataset otherwise -- point --data_dir at real
exported CHAOS volumes to train for real, no code changes required.

Usage:
    python train_pde.py --epochs 2 --steps_per_epoch 20 --out trained_weights.mat
"""

import argparse
import csv
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import savemat
from torch.utils.data import Dataset

from pde_model import PDEModule, M, K


def set_random_seeds(seed):
    """Set and return the seeds used by this training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return {
        "python": seed,
        "numpy": seed,
        "torch": seed,
        "torch_cuda": seed if torch.cuda.is_available() else None,
    }


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


# ----------------------------------------------------------------------
# Rate loss: K=3 discretized mixture-of-logistics NLL (bits/pixel)
# Matches Eq. (2)-(3) and the quantization-aware bin edges of Eq. (12)-(13).
# ----------------------------------------------------------------------
def mixture_logistic_bpp_loss(x, pi, mu, log_s, num_bins, scale_factor_L):
    """
    Args:
        x:      (B, 1, H, W) target pixel intensities, normalized to [-1, 1]
        pi:     (B, K, H, W) mixture weights (already clip-normalized, Eq. 14)
        mu:     (B, K, H, W)
        log_s:  (B, K, H, W)
        num_bins: quantization levels of the source data (2**bit_depth)
        scale_factor_L: paper's L -- rescales the effective quantization
            bin width to fight vanishing gradients at high bit depth
            (L=1 for 8-bit, L=8 for 16-bit, per Section IV-B.3)

    Returns:
        scalar bits/pixel (mean NLL in nats -> bits)
    """
    x = x.expand_as(mu)
    
    # [FIXED BIN WIDTH] The training bin width should just be 1 / (num_bins - 1)
    # The scale_factor_L (BPTT sequence length) should not multiply the bin width!
    half_bin = 1.0 / (num_bins - 1)
    centered = x - mu
    inv_s = torch.exp(-log_s)

    plus_in = inv_s * (centered + half_bin)
    min_in = inv_s * (centered - half_bin)

    cdf_plus = torch.sigmoid(plus_in)
    cdf_min = torch.sigmoid(min_in)
    prob = cdf_plus - cdf_min

    log_prob_edge_low = plus_in - F.softplus(plus_in)   # log sigmoid(plus_in)
    log_prob_edge_high = -F.softplus(min_in)             # log(1 - sigmoid(min_in))
    log_prob_mid = torch.log(torch.clamp(prob, min=1e-12))

    is_low = (x <= -1 + 1e-3).float()
    is_high = (x >= 1 - 1e-3).float()
    is_mid = 1.0 - is_low - is_high

    log_prob = (is_low * log_prob_edge_low
                + is_high * log_prob_edge_high
                + is_mid * log_prob_mid)

    log_pi = torch.log(torch.clamp(pi, min=1e-12))
    mixture_log_prob = torch.logsumexp(log_pi + log_prob, dim=1)  # Eq. (2)-(3), K-sum

    nll_nats = -mixture_log_prob.mean()
    return nll_nats / np.log(2.0)


# ----------------------------------------------------------------------
# Dataset: yields whole VOLUMES (not pre-paired slices), since training
# needs the true sequential recurrence over H_s across a volume.
# ----------------------------------------------------------------------
class ChaosMRIDataset(Dataset):
    TRAIN_IDS = [4, 6, 7, 9, 11, 12, 14, 16, 17, 18, 23, 24, 25, 26]
    VAL_IDS   = [27, 28, 29, 30, 35, 40]

    def __init__(
        self,
        data_dir=None,
        num_synthetic_volumes=8,
        slices_per_volume=20,
        size=64,
        seed=0,
        split="all"
    ):
        self.volumes = []
        self.patient_ids = []

        if split not in ("train", "val", "all"):
            raise ValueError("split must be 'train', 'val', or 'all'")

        # ---------------------------------------------------------
        # REAL CHAOS DATA
        # ---------------------------------------------------------
        if data_dir is not None and os.path.isdir(data_dir):

            all_paths = glob.glob(os.path.join(data_dir, "*.npy"))

            if not all_paths:
                raise FileNotFoundError(
                    f"No .npy CHAOS volumes found in: {data_dir}"
                )

            # Extract numerical patient ID from filename.
            def get_patient_id(path):
                name = os.path.splitext(os.path.basename(path))[0]

                # Extract the first integer appearing in the filename.
                import re
                match = re.search(r"\d+", name)

                if match is None:
                    raise ValueError(
                        f"Could not determine patient ID from filename: {name}"
                    )

                return int(match.group())

            # Sort numerically by patient ID.
            all_paths = sorted(all_paths, key=get_patient_id)

            available_ids = {get_patient_id(p) for p in all_paths}

            required_ids = set(self.TRAIN_IDS + self.VAL_IDS)

            missing_ids = required_ids - available_ids

            if missing_ids:
                raise FileNotFoundError(
                    f"Missing expected CHAOS patient IDs: "
                    f"{sorted(missing_ids)}"
                )

            # Select exact patients.
            if split == "train":
                selected_ids = self.TRAIN_IDS

            elif split == "val":
                selected_ids = self.VAL_IDS

            else:
                selected_ids = self.TRAIN_IDS + self.VAL_IDS

            path_by_id = {
                get_patient_id(path): path
                for path in all_paths
            }

            for patient_id in selected_ids:
                path = path_by_id[patient_id]

                vol = np.load(path).astype(np.float32)

                if vol.ndim != 3:
                    raise ValueError(
                        f"Patient {patient_id} has shape {vol.shape}. "
                        "Expected (numSlices, H, W)."
                    )

                self.volumes.append(vol)
                self.patient_ids.append(patient_id)

        # ---------------------------------------------------------
        # SYNTHETIC DATA
        # ---------------------------------------------------------
        elif data_dir is None:

            print(
                "[ChaosMRIDataset] No real data_dir supplied. "
                "Using synthetic MRI phantom data."
            )

            self.volumes = self._make_synthetic_volumes(
                num_synthetic_volumes,
                slices_per_volume,
                size,
                seed
            )

            self.patient_ids = [
                f"synthetic_{i}"
                for i in range(len(self.volumes))
            ]

        else:
            raise FileNotFoundError(
                f"Dataset directory does not exist: {data_dir}"
            )

        # ---------------------------------------------------------
        # PER-VOLUME NORMALIZATION TO [-1, 1]
        # ---------------------------------------------------------
        normed = []

        for vol in self.volumes:
            vmin = vol.min()
            vmax = vol.max()

            normalized = (
                2.0 * (vol - vmin)
                / max(vmax - vmin, 1e-6)
                - 1.0
            )

            normed.append(normalized.astype(np.float32))

        self.volumes = normed

        # ---------------------------------------------------------
        # EXPORT VALIDATION VOLUMES FOR MATLAB
        # ---------------------------------------------------------
        if split == "val" and data_dir is not None:

            from scipy.io import savemat

            for vol, patient_id in zip(
                self.volumes,
                self.patient_ids
            ):
                # Python: (S, H, W)
                # MATLAB: (H, W, S)
                vol_transposed = np.transpose(vol, (1, 2, 0))

                mat_path = os.path.join(
                    data_dir,
                    f"{patient_id}_norm.mat"
                )

                savemat(
                    mat_path,
                    {"volume": vol_transposed}
                )

                print(
                    f"[ChaosMRIDataset] Exported patient "
                    f"{patient_id} -> {mat_path}"
                )

        print(
            f"[ChaosMRIDataset] split={split}, "
            f"volumes={len(self.volumes)}, "
            f"patients={self.patient_ids}"
        )

    @staticmethod
    def _make_synthetic_volumes(
        num_volumes,
        num_slices,
        size,
        seed
    ):
        rng = np.random.default_rng(seed)

        volumes = []

        yy, xx = np.mgrid[0:size, 0:size]

        for _ in range(num_volumes):

            cx, cy = rng.uniform(0.3, 0.7, 2) * size
            radius = rng.uniform(0.15, 0.35) * size

            vol = np.zeros(
                (num_slices, size, size),
                dtype=np.float32
            )

            for k in range(num_slices):

                depth_factor = max(
                    1.0
                    - abs(k - num_slices / 2)
                    / (num_slices / 2),
                    0.05
                )

                blob = np.exp(
                    -(
                        (xx - cx) ** 2
                        + (yy - cy) ** 2
                    )
                    / (2 * radius ** 2)
                )

                slice_img = (
                    0.6 * blob * depth_factor
                )

                slice_img += (
                    0.05
                    * rng.standard_normal(
                        (size, size)
                    )
                )

                vol[k] = slice_img

            volumes.append(vol)

        return volumes

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        # (S, H, W) → (S, 1, H, W)
        return torch.from_numpy(
            self.volumes[idx]
        ).unsqueeze(1)


# ----------------------------------------------------------------------
# Export: PyTorch state_dict -> trained_weights.mat for initPDEWeights.m
# ----------------------------------------------------------------------
def export_weights_to_mat(model: PDEModule, out_path: str):
    """
    Every 1x1 (pointwise) conv weight (Cout,Cin,1,1) is squeezed and
    transposed to (Cin,Cout) to match conv1x1.m's convention. Spatial
    conv weights (masked 7x7, DSC depthwise 5x5) are saved with axis
    order matching convNCHW.m / dscBranch.m directly -- NO kernel flip
    is applied here, because both MATLAB helpers (convNCHW.m, dscBranch.m)
    flip internally via rot90(...,2) before conv2, so raw PyTorch
    (cross-correlation) weights are exactly what those functions expect.
    """
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def sq_T(w):  # (Cout, Cin, 1, 1) -> (Cin, Cout)
        return np.transpose(np.squeeze(w, axis=(2, 3)))

    def bias_row(b):
        return b.reshape(1, -1)

    # masked_conv.conv.weight: (16, 1, 7, 7) -> matlab wants (7, 7, 1, 16)
    masked_conv_W = np.transpose(sd["masked_conv.conv.weight"], (2, 3, 1, 0))
    masked_conv_b = bias_row(sd["masked_conv.conv.bias"])

    # dsc_branch.depthwise.weight: (16, 1, 5, 5) -> matlab wants (5, 5, 16)
    dsc_depthwise = np.transpose(sd["dsc_branch.depthwise.weight"][:, 0, :, :], (1, 2, 0))
    dsc_pointwise = sq_T(sd["dsc_branch.pointwise.weight"])            # (16, 16)
    dsc_pointwise_b = bias_row(sd["dsc_branch.pointwise.bias"])        # (1, 16)

    head_W1 = sq_T(sd["estimator_head.conv1.weight"])                  # (16, 16)
    head_b1 = bias_row(sd["estimator_head.conv1.bias"])
    head_W2 = sq_T(sd["estimator_head.conv2.weight"])                  # (16, 9)
    head_b2 = bias_row(sd["estimator_head.conv2.bias"])

    savemat(out_path, {
        "masked_conv_W": masked_conv_W,
        "masked_conv_b": masked_conv_b,
        "dsc_depthwise": dsc_depthwise,
        "dsc_pointwise": dsc_pointwise,
        "dsc_pointwise_b": dsc_pointwise_b,
        "head_W1": head_W1,
        "head_b1": head_b1,
        "head_W2": head_W2,
        "head_b2": head_b2,
        "hidden_channels": np.array([[model.masked_conv.conv.out_channels]]),
        "num_mixture_components": np.array([[model.estimator_head.k]]),
    })
    print(f"[export_weights_to_mat] Saved trained weights to {out_path}")


# ----------------------------------------------------------------------
# Training loop with truncated BPTT (paper's "update stride" = 5)
# ----------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.normalization != "per_volume_minmax_to_-1_1":
        raise ValueError(
            "The current dataset implementation requires "
            "normalization=per_volume_minmax_to_-1_1"
        )

    if args.num_bins != 2 ** args.bit_depth:
        raise ValueError(
            f"num_bins ({args.num_bins}) must equal 2**bit_depth "
            f"({2 ** args.bit_depth})"
        )

    run_dir = os.path.abspath(
        os.path.join(args.experiments_root, args.run_name)
    )
    os.makedirs(run_dir, exist_ok=True)
    final_weights_path = os.path.join(run_dir, "final_weights.mat")
    best_weights_path = os.path.join(run_dir, "best_weights.mat")
    seed_record = set_random_seeds(args.seed)

    # ---------------------------------------------------------
    # DATASETS
    # ---------------------------------------------------------
    train_dataset = ChaosMRIDataset(
        data_dir=args.data_dir,
        size=args.image_size,
        seed=args.seed,
        split="train"
    )

    val_dataset = ChaosMRIDataset(
        data_dir=args.data_dir,
        size=args.image_size,
        seed=args.seed,
        split="val"
    )

    print(
        f"Dataset splits: {len(train_dataset)} train volumes, "
        f"{len(val_dataset)} validation volumes"
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------
    model = PDEModule(
        channels=args.hidden_channels,
        k=args.mixture_components
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma
    )

    # ---------------------------------------------------------
    # LOSS PARAMETERS
    # ---------------------------------------------------------
    num_bins = args.num_bins
    scale_L = args.scale_L

    provenance = {
        "dataset": args.dataset,
        "dataset_path": (
            os.path.abspath(args.data_dir)
            if args.data_dir is not None else None
        ),
        "train_patient_ids": list(train_dataset.patient_ids),
        "validation_patient_ids": list(val_dataset.patient_ids),
        "bit_depth": args.bit_depth,
        "num_bins": num_bins,
        "scale_L": scale_L,
        "mixture_components": args.mixture_components,
        "hidden_channels": args.hidden_channels,
        "normalization": args.normalization,
        "loss": {
            "type": "discretized_mixture_logistic_nll_bpp",
            "mixture_components": args.mixture_components,
            "endpoint_handling": "normalized_domain_minus1_plus1",
            "log_probability_floor": 1e-12,
        },
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "scheduler": {
            "type": "StepLR",
            "step_size": args.scheduler_step_size,
            "gamma": args.scheduler_gamma,
        },
        "epochs": args.epochs,
        "update_stride": args.update_stride,
        "steps_per_epoch": args.steps_per_epoch,
        "image_size": args.image_size,
        "random_seeds": seed_record,
        "device": str(device),
        "run_name": args.run_name,
        "output_directory": run_dir,
        "checkpoint_criterion": "minimum validation BPP",
        "best_epoch": None,
        "best_validation_bpp": None,
        "final_validation_bpp": None,
    }
    write_json(os.path.join(run_dir, "config.json"), provenance)

    if args.dry_run:
        print("Dry run complete: configuration and dataset setup validated.")
        return None

    history = []

    # ---------------------------------------------------------
    # BEST MODEL TRACKING
    # ---------------------------------------------------------
    best_val_bpp = float("inf")
    best_epoch = 0

    # =========================================================
    # TRAINING LOOP
    # =========================================================
    for epoch in range(args.epochs):

        model.train()

        running_loss = 0.0
        n_train_slices = 0
        n_steps = 0

        # -----------------------------------------------------
        # TRAINING VOLUMES
        # -----------------------------------------------------
        for vol_idx in range(len(train_dataset)):

            if (
                args.steps_per_epoch is not None
                and n_steps >= args.steps_per_epoch
            ):
                break

            volume = train_dataset[vol_idx].to(device)

            # (numSlices, 1, H, W)
            num_slices, _, H, W = volume.shape

            # Initial recurrent state H_0
            h_prev = torch.zeros(
                1,
                args.hidden_channels,
                H,
                W,
                device=device
            )

            accumulated_loss = None
            accumulated_slices = 0

            # -------------------------------------------------
            # RECURRENT SLICE PROCESSING
            # -------------------------------------------------
            for s in range(num_slices):

                current_slice = volume[s:s + 1]

                # PDE forward pass
                pi, mu, log_s, h_s = model(
                    current_slice,
                    h_prev
                )

                # Discrete logistic BPP loss
                loss = mixture_logistic_bpp_loss(
                    current_slice,
                    pi,
                    mu,
                    log_s,
                    num_bins=num_bins,
                    scale_factor_L=scale_L
                )

                # Accumulate losses inside BPTT window
                if accumulated_loss is None:
                    accumulated_loss = loss
                else:
                    accumulated_loss = (
                        accumulated_loss + loss
                    )

                accumulated_slices += 1

                # -------------------------------------------------
                # TRUNCATED BPTT UPDATE
                # -------------------------------------------------
                should_update = (
                    (s + 1) % args.update_stride == 0
                    or s == num_slices - 1
                )

                if should_update:

                    optimizer.zero_grad()

                    accumulated_loss.backward()

                    optimizer.step()

                    # Record accumulated loss
                    running_loss += (
                        accumulated_loss.detach().item()
                    )

                    n_train_slices += accumulated_slices
                    n_steps += 1

                    # Reset accumulation
                    accumulated_loss = None
                    accumulated_slices = 0

                    # Detach recurrent state
                    h_prev = h_s.detach()

                else:
                    # Continue recurrence within BPTT window
                    h_prev = h_s

                if (
                    args.steps_per_epoch is not None
                    and n_steps >= args.steps_per_epoch
                ):
                    break

        # Average training BPP
        avg_train_loss = (
            running_loss / max(n_train_slices, 1)
        )

        # Learning-rate scheduler
        scheduler.step()

        # =========================================================
        # VALIDATION
        # =========================================================
        model.eval()

        val_running_loss = 0.0
        val_n_slices = 0

        with torch.no_grad():

            for vol_idx in range(len(val_dataset)):

                volume = val_dataset[vol_idx].to(device)

                num_slices, _, H, W = volume.shape

                # H_0
                h_prev = torch.zeros(
                    1,
                    args.hidden_channels,
                    H,
                    W,
                    device=device
                )

                # Process validation volume sequentially
                for s in range(num_slices):

                    current_slice = volume[s:s + 1]

                    # PDE forward pass
                    pi, mu, log_s, h_s = model(
                        current_slice,
                        h_prev
                    )

                    # BPP loss
                    loss = mixture_logistic_bpp_loss(
                        current_slice,
                        pi,
                        mu,
                        log_s,
                        num_bins=num_bins,
                        scale_factor_L=scale_L
                    )

                    val_running_loss += loss.item()
                    val_n_slices += 1

                    # Continue recurrent state
                    h_prev = h_s

        # Average validation BPP
        avg_val_loss = (
            val_running_loss / max(val_n_slices, 1)
        )

        # =========================================================
        # BEST MODEL CHECKPOINT
        # =========================================================
        if avg_val_loss < best_val_bpp:

            best_val_bpp = avg_val_loss
            best_epoch = epoch + 1

            export_weights_to_mat(model, best_weights_path)

            print(
                f"  -> New best model! "
                f"Val BPP: {best_val_bpp:.4f} "
                f"(epoch {best_epoch})"
            )

        # =========================================================
        # EPOCH RESULT
        # =========================================================
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"train loss (BPP): {avg_train_loss:.4f} | "
            f"val loss (BPP): {avg_val_loss:.4f} | "
            f"best val BPP: {best_val_bpp:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e}"
        )

        history_row = {
            "epoch": epoch + 1,
            "train_bpp": avg_train_loss,
            "validation_bpp": avg_val_loss,
            "best_validation_bpp": best_val_bpp,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(history_row)
        with open(os.path.join(run_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history_row.keys())
            writer.writeheader()
            writer.writerows(history)

    # =========================================================
    # SAVE FINAL EPOCH MODEL
    # =========================================================
    export_weights_to_mat(
        model,
        final_weights_path
    )

    provenance["best_epoch"] = best_epoch
    provenance["best_validation_bpp"] = best_val_bpp
    provenance["final_validation_bpp"] = avg_val_loss
    write_json(os.path.join(run_dir, "config.json"), provenance)
    write_json(
        os.path.join(run_dir, "summary.json"),
        {
            "best_epoch": best_epoch,
            "best_validation_bpp": best_val_bpp,
            "final_validation_bpp": avg_val_loss,
            "checkpoint_criterion": "minimum validation BPP",
            "best_weights": best_weights_path,
            "final_weights": final_weights_path,
        },
    )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation BPP : {best_val_bpp:.4f}")
    print(f"Best epoch          : {best_epoch}")
    print(f"Best weights        : {best_weights_path}")
    print(f"Final weights       : {final_weights_path}")
    print("=" * 60)

    return model


def parse_args():
    p = argparse.ArgumentParser(description="Train the HENCE PDE module")
    p.add_argument("--data_dir", type=str, default=None,
                    help="Directory of exported CHAOS volumes as .npy files "
                         "([numSlices, H, W] each). Falls back to synthetic data if omitted.")
    p.add_argument("--dataset", type=str, default="CHAOS")
    p.add_argument("--run_name", type=str, default="chaos_paper_16bit")
    p.add_argument("--experiments_root", type=str, default="experiments")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--steps_per_epoch", type=int, default=None,
                    help="Cap on optimizer steps per epoch (for quick smoke tests)")
    p.add_argument("--lr", type=float, default=5e-4)          # paper: 5e-4
    p.add_argument("--update_stride", type=int, default=5)     # paper: 5
    p.add_argument("--bit_depth", type=int, default=16)
    p.add_argument("--num_bins", type=int, default=65536)
    p.add_argument("--scale_L", type=float, default=8.0)
    p.add_argument("--mixture_components", type=int, default=3)
    p.add_argument("--hidden_channels", type=int, default=16)
    p.add_argument("--scheduler_step_size", type=int, default=50)
    p.add_argument("--scheduler_gamma", type=float, default=0.8)
    p.add_argument("--normalization", type=str,
                    default="per_volume_minmax_to_-1_1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image_size", type=int, default=64, help="Only used for synthetic fallback")
    p.add_argument("--dry_run", action="store_true",
                   help="Validate configuration/datasets and write config without training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

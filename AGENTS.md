# AGENTS.md

## Project

This is a final-year research project implementing and verifying the HENCE PDE image-compression method. The base paper is located at `reference/HENCE_PDE_base_paper.pdf`.

## Primary Rule

Preserve the algorithm described in the paper. Do not introduce alternative architectures, losses, normalization methods, or tensor conventions without explicit evidence.

## Before Editing

For any bug:

1. Inspect relevant MATLAB code.
2. Inspect corresponding Python code.
3. Inspect the paper if necessary.
4. Reproduce the problem.
5. Identify the first point of divergence.
6. Explain the root cause.
7. Make the smallest appropriate fix.
8. Rerun the relevant tests.

## MATLAB/Python Verification

MATLAB is treated as the reference implementation where it is already verified against the paper. Python must reproduce MATLAB behavior numerically.

For comparisons, report shape, maximum absolute difference, mean absolute difference, RMSE, and percentage within tolerance. Use approximately `1e-5` as the initial float32 comparison tolerance unless the numerical method requires otherwise.

## Current Known Status

MC, DSC, and H already match MATLAB closely. Do not modify them without evidence.

The estimator must be debugged stage-by-stage:

```text
H → hidden_pre → hidden → raw → pi_raw → mu → log_s_raw → pi → log_s
```

## Editing Rules

Do not rewrite the entire project, delete working code, change unrelated files, change training parameters merely to improve results, change bit depth without confirming training configuration, replace paper equations with approximations, or hide failing tests.

Prefer small changes, explicit diagnostics, reproducible tests, clear comments, and preserved debug scripts.

## Validation

After any code change:

1. Run the relevant unit or debug test.
2. Run MATLAB/Python numerical comparison where applicable.
3. Run `verify_patient.py`.
4. Inspect BPP, NaN/Inf, mixture-weight sums, and tensor shapes.
5. Report what changed and why.

Do not claim success unless the relevant validation was actually run.

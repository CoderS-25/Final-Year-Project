# Track 1 PDE Module: Verification & Improvement Plan

This document outlines the systematic plan to thoroughly verify, improve, and benchmark your PDE module so that it is bulletproof for your project viva.

## Phase 1: Verification & Viva Preparation

*   **Code & Toolbox Review:** Ensure all functions run efficiently without requiring obscure toolboxes, ensuring your code runs perfectly on the grader's machine.
*   **Test Validation:** We have already fixed the mathematical logic in Test 5 (changing the monotone gradient to a saddle gradient). We will formalize why this was necessary so you can explain it.
*   **Viva Defense Prep:** Document the exact mathematical justification for the $N + W - NW$ formula (discrete Laplace condition) and the MED edge-preserving clamp so you are ready to answer tough questions from your professors.

## Phase 2: Improvement & Evaluation

*   **The "NE" Neighbor Dilemma:** Your `getContext.m` extracts the North-East (`NE`) neighbor, but `estimate.m` completely ignores it. We will investigate if modifying the PDE formula to include `NE` improves MRI prediction (e.g., using a 4-tap predictor like CALIC instead of the 3-tap JPEG-LS predictor).
*   **Performance Metrics (MAE/MSE):** We will update the pipeline to calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), and Entropy. These are the standard scientific metrics you must include in your project report.
*   **Predictor Benchmarking:** We will write a benchmark script to compare your PDE predictor against simpler predictors (e.g., $N$ only, or $\frac{N+W}{2}$). This will *prove* to your professors why the PDE approach was worth the effort.
*   **Real MRI Data:** The current code uses the Shepp-Logan phantom (a fake, simulated MRI). We will adapt the code to process a real DICOM MRI scan to validate real-world performance.

## User Review Required
Please review the plan above. Click **Proceed** if you are ready to execute this plan. Once approved, I will begin writing the benchmarking scripts and generating the Viva defense points!

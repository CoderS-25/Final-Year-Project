# Product Requirements Document (PRD)
## Module: Track 1 - PDE (Partial Differential Equation) Module
**Project Context:** Lossless MRI Image Compression Pipeline (Stage 1 of 2)
**Target Language:** MATLAB

---

### 1. High-Level Objective
The goal of this module is to perform predictive coding on MRI images. It must predict the value of each pixel based on its causal neighborhood (pixels already processed). It will then subtract the predicted value from the actual value to generate a "residual" (error) image. Because MRI images are smooth, these residuals will be clustered around zero, making them highly compressible by the downstream MLAC (Arithmetic Coding) module.

### 2. Interface Contracts
To guarantee compatibility with other team members, the following function signatures must be strictly adhered to:
*   `getContext(image, pixel_coords) -> context_vector`
*   `estimate(context_vector) -> predicted_value`

*(Note for AI generation: While the contract implies pixel-by-pixel processing, the actual MATLAB implementation under the hood must be vectorized to process the entire image at once to meet performance requirements).*

### 3. Functional Requirements

#### 3.1. Feature Extraction (Data Preparation)
*   **Goal:** Build causal neighborhood context windows for every pixel in the image. 
*   **Causal Template:** The neighborhood must only include pixels that would have already been transmitted in a raster-scan order (e.g., Left, Top, Top-Left, Top-Right).
*   **Performance Constraint:** **STRICTLY NO NESTED FOR-LOOPS.** You must vectorize this step.
*   **Technical Approach:** Use MATLAB's `im2col` (Image to Column) function or shifted-matrix tricks (like `padarray` combined with matrix shifting) to extract all neighborhoods simultaneously.

#### 3.2. PDE-Based Estimator (Prediction Math)
*   **Goal:** Implement a mathematical prediction rule that consumes the context windows generated in step 3.1 and outputs a predicted pixel value.
*   **Methodology:** The prediction must use a Partial Differential Equation (PDE) based interpolation or diffusion estimator. (e.g., predicting the flow of intensity from the neighboring pixels to the center pixel).
*   **Input:** The vectorized context matrices/vectors.
*   **Output:** A matrix of predicted pixel values matching the dimensions of the original image.

#### 3.3. Residual Computation
*   **Goal:** Calculate the difference between the actual image and the predicted image.
*   **Formula:** `residual_image = actual_image - predicted_image`
*   **Validation Check:** Plot a histogram of the `residual_image`. The distribution must be a sharp spike centered at `0` (Laplace-like distribution). If the distribution is wide or flat, the PDE estimator is failing.

#### 3.4. Unit Testing and Edge Cases
*   **Boundary Handling:** The code must robustly handle the first row and first column of the image, which do not have a complete causal neighborhood. (e.g., using zero-padding, symmetric padding, or fallback prediction rules).
*   **Sanity Checks:** 
    *   Confirm no `NaN` (Not a Number) outputs occur on the edges.
    *   Test on 2-3 sample MRI slice matrices to verify performance and accuracy.
    *   Ensure the data types (e.g., `uint16`, `double`) are managed properly so negative residuals are not truncated to zero.

---

### 4. Instructions for AI Code Generation (Prompting Guide)
*When pasting this into another AI, append the following instructions:*

> "Act as an expert MATLAB engineer specializing in image processing. Based on the PRD above, write the complete MATLAB code for Track 1. Prioritize extreme vectorization (avoiding loops entirely) using `padarray` and matrix shifting or `im2col`. Include the functions `getContext` and `estimate`, a main script to run the pipeline on a sample image, and a visualization block that shows the original image, the predicted image, the residual image, and a histogram of the residuals to prove it is sharply peaked at zero."

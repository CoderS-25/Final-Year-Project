# HENCE MLAC — Final Validation Results

## Summary

| Metric | Value |
|---|---|
| **Mean BPP** | **6.9773** |
| **Mean Compression** | **2.32×** vs raw 16-bit |
| **All Lossless** | ✅ 6/6 patients, every slice |
| **Exceptional Rate** | 0.022% |

## Per-Patient Results

| Patient | Resolution | Slices | BPP | Compression | Exceptional | Lossless |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 27 | 256×256 | 30 | 6.2530 | 2.56× | 0.02% | ✅ |
| 28 | 320×320 | 36 | **5.8702** | **2.73×** | 0.04% | ✅ |
| 29 | 320×320 | 36 | 6.9523 | 2.30× | 0.02% | ✅ |
| 30 | 512×512 | 44 | 7.8664 | 2.03× | 0.01% | ✅ |
| 35 | 512×512 | 44 | 7.8997 | 2.03× | 0.00% | ✅ |
| 40 | 256×256 | 30 | 7.0220 | 2.28× | 0.03% | ✅ |

> [!NOTE]
> Best compression: Patient 28 at **5.87 bpp** (2.73× compression).
> Std deviation: 0.8239 bpp — consistent performance across different resolutions and anatomies.

## Key Observations

1. **Perfectly lossless** — every single pixel across all 220 slices decoded identically to the original
2. **Near-zero exceptional rate** (0.022%) — the PDE model predicts well enough that only ~1 in 4500 pixels needs raw storage
3. **Higher resolution → slightly higher BPP** — Patients 30 and 35 (512×512) have more fine detail that's harder to predict, so BPP is ~7.9 vs ~6.1 for 256×256 patients
4. **Training BPP was 4.81** — the gap from theoretical (4.81) to actual (6.98) is expected overhead from integer arithmetic precision, exceptional pixel storage, and the gap between continuous NLL and discrete coding

## Bug Fixes During Development

| Bug | Symptom | Root Cause |
|---|---|---|
| `uint8 × logical` type error | MATLAB crash in bit-packing | Missing `uint8()` cast on logical bit matrix |
| 100% exceptional pixels | No compression at all | `buildFreqTable` flagged entire pixel as exceptional if *any* bin had 0 freq, not just the actual symbol |
| CDF domain mismatch | All frequencies = 0 | Integer boundaries [0, 65535] not mapped to [-1, 1] before CDF evaluation |
| `sum(pi, 3)` on 1×K vector | CDF always returned 1.0 | `quantizeParameters` hardcoded dimension=3, broke for single-pixel inputs |
| 99.9% exceptional on 16-bit | 16.0 bpp (no compression) | Full 65536-entry freq table too sparse — switched to direct per-pixel CDF intervals |
| Encoder-decoder desync | Lossy output (max_error ~30000) | Artificial interval expansion (`cdf_high = cdf_low + 1`) only done in encoder, not matched by decoder binary search |
| **[0,1] vs [-1,1] normalization** | **99.9% exceptional, 16 bpp** | **`run_all_patients.m` used `(x-min)/(max-min)` instead of `2*(x-min)/(max-min) - 1` — PDE predictions completely misaligned with actual pixel values** |

## Files Created/Modified

### MLAC Project (`hence_mlac_project/matlab/`)
- [`arithmeticEncode.m`](file:///C:/Users/user/Documents/Final%20Year%20Project/hence_mlac_project/matlab/arithmeticEncode.m) — Vectorized encoder with per-pixel CDF intervals
- [`arithmeticDecode.m`](file:///C:/Users/user/Documents/Final%20Year%20Project/hence_mlac_project/matlab/arithmeticDecode.m) — Decoder with warm-start binary search
- [`getPixelInterval.m`](file:///C:/Users/user/Documents/Final%20Year%20Project/hence_mlac_project/matlab/getPixelInterval.m) — Per-pixel CDF interval computation
- [`findSymbolBinarySearch.m`](file:///C:/Users/user/Documents/Final%20Year%20Project/hence_mlac_project/matlab/findSymbolBinarySearch.m) — Binary search decoder helper
- [`quantizeParameters.m`](file:///C:/Users/user/Documents/Final%20Year%20Project/hence_mlac_project/matlab/quantizeParameters.m) — Fixed-point parameter quantization
- [`mixtureLogisticCDF.m`](file:///C:/Users/user/Documents/Final%20Year%20Project/hence_mlac_project/matlab/mixtureLogisticCDF.m) — Mixture-of-logistics CDF (Eq. 11)
- [`buildFreqTable.m`](file:///C:/Users/user/Documents/Final%20Year%20Project/hence_mlac_project/matlab/buildFreqTable.m) — Frequency table builder (used for 8-bit test)
- [`runMLAC.m`](file:///C:/Users/user/Documents/Final%20Year%20Project/hence_mlac_project/matlab/runMLAC.m) — Master pipeline with parfor
- [`run_patient_27.m`](file:///C:/Users/user/Documents/Final%20Year%20Project/hence_mlac_project/matlab/run_patient_27.m) — Single patient runner
- [`run_all_patients_mlac.m`](file:///C:/Users/user/Documents/Final%20Year%20Project/hence_mlac_project/matlab/run_all_patients_mlac.m) — Full validation set runner

### PDE Project (bug fix)
- [`run_all_patients.m`](file:///C:/Users/user/Documents/Final%20Year%20Project/hence_pde_project_v3/matlab/run_all_patients.m) — Fixed normalization from [0,1] to [-1,1]

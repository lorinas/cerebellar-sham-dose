# Cerebellar Sham Dose

This repository contains a reproducible Python pipeline for estimating cerebellar exposure and circuit-level response from commercial sham TMS.

## Hypothesis

The hypothesis is that if sham TMS produces a nonzero cerebellar E-field, and that field is large enough to perturb cerebellar circuit output, then sham stimulation may not be a biologically neutral control condition.

To test this hypothesis, we combined individualized E-field modeling, published coil-specific sham-field measurements, waveform filtering, and a validated virtual model of the cerebellum to estimate whether sham TMS exposure can produce a Purkinje-cell population response comparable to active cerebellar tDCS.

The analysis is structured around five main questions:

1. What is the cerebellar GM E-field exposure during active/sham TMS and active tDCS?
2. What is the acute PC population response to sham exposure?
3. At what intensity does sham TMS reach active-tDCS response?
4. Does anatomy affect when sham TMS reaches active tDCS?
5. Are these patterns reproduced across clinical cohorts?

## Pipeline overview

The analysis follows this sequence:

```text
individualized cerebellar E-field simulations and ROI extraction
→ waveform-filtered effective dose
→ propagation of dose through mossy-fiber drive in the cerebellar virtual model
→ subject-level sham vs active-tDCS analysis
→ anatomy and cohort-level analyses
```

The main comparison is within subject:

```text
sham TMS response
vs.
the same subject's active-tDCS response
```

A sham condition is considered to reach active tDCS when the modeled Purkinje-cell response from sham TMS is greater than or equal to that subject's modeled active-tDCS response.

## Sham E-field modeling

Sham TMS was not modeled as a zero-field condition.

Different commercial sham coils have different residual E-field distributions. MagVenture sham resembles its active counterpart, with a sham/active ratio near the coil center of approximately 7%. In contrast, Magstim sham differs spatially from active Magstim stimulation: the field under the coil center is relatively small, while the residual sham peak is displaced away from the center and reaches approximately 26% of the active peak in the published measurements.

We modeled sham E-field distributions using measured active and sham coil-field maps from Smith and Peterchev. Their dataset provides coil-centered x/y/z measurement points and E-field vector components for active and sham Magstim and MagVenture configurations.

For each coordinate, we converted the measured vector components into E-field magnitude and calculated a local sham/active correction factor:

```text
local sham/active factor = measured sham |E| / measured active |E|
```

We then used this factor together with each subject's active E-field simulation to estimate how much of the active field remained during sham stimulation. Each subject's cerebellar GM ROI elements were mapped into coil-centered space, the empirical sham/active factor was interpolated at each element's position, and that factor was multiplied by the subject-specific active E-field magnitude:

```text
subject-specific sham field
=
subject-specific active field
×
empirical sham/active correction factor
```

This preserves both coil-specific sham behavior and individual anatomy.

For the MagVenture sham + electrodes condition, we modeled the magnetic sham field and added the electrode-related contribution following the Smith and Peterchev sham+electrode estimate.

## Cerebellar virtual model

Circuit-level responses were estimated using the Lorenzi cerebellar mean-field model. This model represents the main cerebellar cortical populations, including granule cells, Golgi cells, molecular-layer interneurons, and Purkinje cells.

In this pipeline, the E-field-derived dose was mapped onto the model's external mossy-fiber drive variable. The primary endpoint was the acute Purkinje-cell population response.

The model output should be interpreted as a computational circuit-level prediction, not as a direct physiological recording.

## Main outputs

The pipeline generates:

```text
E-field tables
target anatomy tables
waveform-filtered dose tables
gamma calibration tables
mean-field prediction tables
subject-level sham-vs-tDCS tables
anatomy and statistical model tables
```

## Repository structure

```text
config/
  paths_template.yaml              template for local paths
  cohort_metadata_template.csv     template for subject metadata

resources/
  waveforms/                       digitized TMS waveforms used for waveform filtering
  external/
    smith_peterchev_2018/
      README.md                    instructions for external coil-measurement source data

scripts/
  00_check_metadata.py
  01_run_simnibs_extract_efield.py
  01b_extract_target_anatomy.py
  01c_collect_step1_outputs.py
  02_compute_membrane_filtered_dose.py
  02b_scan_gamma_inputs.py
  02c_build_magstim_empirical_kernel.py
  02d_extract_magstim_kernel_efield.py
  02e_build_step1_empirical_kernel_tables.py
  02f_extract_magventure_kernel_efield.py
  03_run_meanfield_predictions.py
  04_build_subject_level_table.py
  05_run_primary_statistics.py
  05b_make_report_ready_tables.py
  06_make_main_figures.py
  07_make_supplementary_figures.py
  08_run_discrete_time_survival_models.py

src/shamdose/
  reusable Python modules used by the scripts
```

## Local configuration

Users should create a private local configuration file:

```text
config/paths_local.yaml
```

from:

```text
config/paths_template.yaml
```

Users should also provide their own subject metadata file based on:

```text
config/cohort_metadata_template.csv
```

## External data required

The empirical coil-field correction requires the Smith and Peterchev active/sham coil measurement data.

Place the raw spreadsheet locally under:

```text
resources/external/smith_peterchev_2018/raw/
```

Then run the preprocessing and kernel scripts to generate local processed files.

The cerebellar mean-field model code must also be available locally and referenced in `config/paths_local.yaml`.

## Basic script order

A typical full run follows this order.

### 1. Check metadata

```bash
python scripts/00_check_metadata.py
```

### 2. Extract E-field distributions

```bash
python scripts/01_run_simnibs_extract_efield.py --cohort HC
python scripts/01_run_simnibs_extract_efield.py --cohort SZ
python scripts/01_run_simnibs_extract_efield.py --cohort CUD

python scripts/01c_collect_step1_outputs.py --cohort HC
python scripts/01c_collect_step1_outputs.py --cohort SZ
python scripts/01c_collect_step1_outputs.py --cohort CUD
```

### 3. Extract target anatomy

```bash
python scripts/01b_extract_target_anatomy.py --cohort HC
python scripts/01b_extract_target_anatomy.py --cohort SZ
python scripts/01b_extract_target_anatomy.py --cohort CUD
```

### 4. Build empirical coil kernels

```bash
python scripts/02c_build_magstim_empirical_kernel.py
```

### 5. Extract empirical sham E-field rows

```bash
python scripts/02d_extract_magstim_kernel_efield.py --cohort HC
python scripts/02d_extract_magstim_kernel_efield.py --cohort SZ
python scripts/02d_extract_magstim_kernel_efield.py --cohort CUD

python scripts/02f_extract_magventure_kernel_efield.py --cohort HC
python scripts/02f_extract_magventure_kernel_efield.py --cohort SZ
python scripts/02f_extract_magventure_kernel_efield.py --cohort CUD
```

### 6. Build empirical-kernel Step 1 tables

```bash
python scripts/02e_build_step1_empirical_kernel_tables.py --cohort ALL --promote
```

### 7. Compute waveform-filtered dose and gamma calibration

```bash
python scripts/02_compute_membrane_filtered_dose.py --cohort ALL
python scripts/02b_scan_gamma_inputs.py --cohort ALL
```

### 8. Run cerebellar mean-field predictions

```bash
python scripts/03_run_meanfield_predictions.py --cohort HC --n-workers 2
python scripts/03_run_meanfield_predictions.py --cohort SZ --n-workers 2
python scripts/03_run_meanfield_predictions.py --cohort CUD --n-workers 2
```

### 9. Build analysis tables and statistics

```bash
python scripts/04_build_subject_level_table.py
python scripts/05_run_primary_statistics.py
python scripts/08_run_discrete_time_survival_models.py
python scripts/05b_make_report_ready_tables.py
```

### 10. Make figures

```bash
python scripts/06_make_main_figures.py --cohort HC
python scripts/06_make_main_figures.py --cohort SZ
python scripts/06_make_main_figures.py --cohort CUD

python scripts/07_make_supplementary_figures.py --figure all
```

Some steps require the SimNIBS Python environment and access to local head-model folders. Downstream statistical and plotting steps run from generated CSV tables.

## Key references

Smith JE, Peterchev AV. Electric Field Measurement of Two Commercial Active/Sham Coils for Transcranial Magnetic Stimulation. Journal of Neural Engineering.

Lorenzi RM et al. A multi-layer mean-field model of the cerebellum embedding microstructure and population-specific dynamics. PLOS Computational Biology.

## Results

Stay tuned for results.

## License

No license has been selected yet. Reuse permissions should be determined before distributing this code as an official lab release.

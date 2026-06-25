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

A sham condition is considered to reach active tDCS when the modeled Purkinje-cell response from sham TMS is greater than or equal to that subject's modeled active-tDCS response.

## Sham E-field modeling

Sham TMS was not modeled as a zero-field condition.

We modeled sham E-field distributions using measured active and sham coil-field maps from Smith and Peterchev. Their dataset provides coil-centered x/y/z measurement points and E-field vector components for active and sham Magstim and MagVenture configurations.

## Cerebellar virtual model

Circuit-level responses were estimated using the Lorenzi cerebellar mean-field model. This model represents the main cerebellar cortical populations, including granule cells, Golgi cells, molecular-layer interneurons, and Purkinje cells.

In this pipeline, the E-field-derived dose was mapped onto the model's external mossy-fiber drive variable. The primary endpoint was the acute Purkinje-cell population response.

The model output should be interpreted as a computational circuit-level prediction, not as a direct physiological recording.

## Key references

Smith JE, Peterchev AV. Electric Field Measurement of Two Commercial Active/Sham Coils for Transcranial Magnetic Stimulation. Journal of Neural Engineering.

Lorenzi RM et al. A multi-layer mean-field model of the cerebellum embedding microstructure and population-specific dynamics. PLOS Computational Biology.

## Results

Stay tuned for results.

## License

No license has been selected yet. Reuse permissions should be determined before distributing this code as an official lab release.

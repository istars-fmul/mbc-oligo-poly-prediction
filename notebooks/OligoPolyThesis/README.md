# Thesis code: oligometastatic vs polymetastatic breast cancer

## Overview

This folder contains the notebooks for the MSc thesis *Predictive Modelling and
Stratification of Metastatic Breast Cancer Patients* (Instituto Superior Técnico, October 2026).
The thesis asks whether clinical variables, somatic mutations (WES) and RNA-seq expression from
the primary tumour can distinguish patients with oligometastatic (OMD) from polymetastatic (PMD)
disease at first metastatic diagnosis, using nested cross-validation in a small,
high-dimensional cohort.

The notebooks are the record of the analyses reported in the thesis and keep their displayed
outputs. They depend on data and saved results under `/mnt/data3`, so not every analysis can be
rerun from this directory alone. The thesis code-availability statement cites
<https://github.com/istars-fmul/mbc-oligo-poly-prediction>.

## Scope

| Location | Contents |
|---|---|
| `notebooks/OligoPolyThesis/` | Analysis notebooks (this folder) |
| `scripts/OligoPolyThesis/` | Shared utility module, representation-generation scripts and small configuration files |
| `/mnt/data3/MCPP-Breast/Machine-Learning-Models-poly_vs_oligo/` | Final results and saved models (outside git), referred to below as `RESULTS/` |

Preprocessed inputs live under `/mnt/data3/MCPP-Breast/` (`Clinical-Data/`, `Mutation-Data/`,
`Transcriptomic-Data/Preprocessed/`).

## Terminology

- **Common cohort (N=110)**: the shared cohort (43 OMD, 67 PMD) with complete clinical, mutation
  and transcriptomic data, used with identical frozen outer folds for the primary modality
  comparison.
- **Full availability**: modality-specific cohorts using all eligible patients (clinical N=149,
  mutation N=112, transcriptomics N=139). These are complementary analyses.
- **Eight model-prioritised genes**: H3C14, KIAA2012, MYEOV, TMPRSS4, ZNF549, GDAP1, RUNDC3B and
  ZNF205.

Some internal result paths and filenames keep historical names: `primary_110` (common-cohort
results), `harmonised_main8` and `main8` (results for the eight model-prioritised genes). They are
retained for compatibility and provenance and do not denote separate analyses.

## Analysis map

Notebook names are shortened to their date prefix. Result directories are relative to `RESULTS/`
unless an absolute path is given.

| Thesis analysis | Notebook | Supporting script(s) | Main result directory |
|---|---|---|---|
| Clinical QC and preprocessing (§3.2) | `20260401-qc_…`, `20260401-preprocessing_…` | | `/mnt/data3/MCPP-Breast/Clinical-Data/` |
| Mutation QC and Phi/Fisher redundancy (§3.3) | `20260505_Mutation_QC_…` | `mutation_phi_fisher_analysis.py` | `/mnt/data3/MCPP-Breast/Mutation-Data/` |
| TPM preprocessing (§3.1.2) | `20260707-preprocessing-tpm-…` | | `/mnt/data3/MCPP-Breast/Transcriptomic-Data/Preprocessed/` |
| Clinical model (§4.2.1) | `20260601-modeling-clinical-data` | | `Common-Cohort-Modality-Comparison/`; full availability in `Clinical-Data/` |
| Mutation model (§4.2.2) | `20260601-modeling-mutation` | | `Common-Cohort-Modality-Comparison/`; full availability in `Mutation-Data/` |
| Clinical + mutation (§4.2.3) | `20260702-modeling-clinical-mutation-…` | | `Common-Cohort-Modality-Comparison/` |
| Transcriptomic model and representation benchmark, N=139 (§4.2.4, §4.4.1–4.4.2) | `20260710-representation-learning-tpm-…` | `run_tybalt.py`, `run_multiplier.R`, `run_genomic_super_signature.R` | `Transcriptomics-Representation-Learning/`; N=110 results in `Common-Cohort-Modality-Comparison/` |
| METABRIC VAE / HVAE (§4.4.3) | `20260729-…-METABRIC-IntegrativeVAE-…` | | `Transcriptomics-Representation-Learning/METABRIC-IntegrativeVAE/` |
| moBRCA-net-inspired attention (§4.4.3) | `20260730-…-moBRCAnet-attention-…` | | `Transcriptomics-BC-Specific-Representations/moBRCA-net-Attention/` |
| Rostami-inspired autoencoder (§4.4.3) | `20260731-…-rostami-autoencoder-…` | | `Transcriptomics-BC-Specific-Representations/Rostami-Autoencoder/` |
| Clinical + transcriptomics (§4.2.4) | `20260929-modeling-clinical-transcriptomics-…` | | `Common-Cohort-Modality-Comparison/` |
| Mutation + transcriptomics (§4.2.4) | `20260929-modeling-mutation-transcriptomics-…` | | `Common-Cohort-Modality-Comparison/` |
| Three-modality model (§4.3) | `20260825-modeling-clinical-mutation-transcriptomics` | | `Common-Cohort-Modality-Comparison/` |
| Common-cohort summary (§4.1, Table 4.1, Figure 4.1, Tables A.1–A.2) | `20260913-summary-of-model-results-oligopoly` | `utils_ml_oligopoly.py` (see below) | `Common-Cohort-Modality-Comparison/` |
| Common-cohort interpretation and selection of the eight genes (§3.10, Table 4.18) | `20260924-common-cohort-model-interpretation-oligopoly` | | `Common-Cohort-Modality-Comparison/primary_110/interpretation/`, `…/reconstruction/` |
| Expression differences (§4.6.1, Figures 4.15–4.16) | `20260924-mcpp-model-prioritized-gene-expression-…` | | `MCPP-Biological-Characterization/common-110/harmonised_main8/` |
| Expression–SHAP relationships (Figure 4.17) | `20261002-expression-shap-survival-context-…` | | `MCPP-Biological-Characterization/common-110/harmonised_main8/` |
| Overall survival (§4.6.3, Figures 4.18–4.19, Table 4.20) | `20260924-mcpp-metastatic-overall-survival` | | `MCPP-Biological-Characterization/common-110/survival/` |
| TCGA-BRCA PFI and DSS (§4.6.4, Table 4.21, Figure 4.20) | `20260922-tcga-brca-prognostic-relevance-…` | | `TCGA-BRCA-Prognostic-Relevance/main8/` |

## Suggested reading / execution order

This follows the scientific dependencies rather than the filename dates.

1. **QC and preprocessing**: clinical (`20260401` QC, then preprocessing), mutation (`20260505`,
   with `mutation_phi_fisher_analysis.py` run between saving the filtered matrix and loading the
   pairwise results) and TPM (`20260707`). The three modalities are independent of each other.
2. **Representation input generation**: `run_genomic_super_signature.R`, `run_multiplier.R` and
   `run_tybalt.py`, which need the preprocessed TPM matrix.
3. **Full-availability and representation analyses**: `20260601` clinical, `20260601` mutation
   and `20260710` transcriptomics, then the METABRIC-based representations `20260729`, `20260730`
   and `20260731`. These notebooks are independent of each other.
4. **Primary N=110 common-cohort comparison**: the seven input configurations, reported in the
   N=110 sections of the single-modality notebooks, in `20260702` (clinical + mutation), the two
   `20260929` pairwise notebooks, `20260825` (three modalities) and the `20260913` summary.
5. **Interpretation**: `20260924-common-cohort-model-interpretation-oligopoly`, which also defines
   the eight model-prioritised genes.
6. **Biological and prognostic characterisation**: `20260924` expression, `20261002`
   expression–SHAP, `20260924` overall survival and `20260922` TCGA. They all use the eight genes
   from step 5 but are otherwise independent of each other.

## Primary N=110 common-cohort comparison

The thesis's primary seven-configuration comparison (clinical, mutation, transcriptomics, the
three pairwise combinations and all three modalities) is stored under
`RESULTS/Common-Cohort-Modality-Comparison/`. It was produced by five functions in
`scripts/OligoPolyThesis/utils_ml_oligopoly.py`, called in this order:

1. `load_common_sources`
2. `initialise_common_output`
3. `prepare_training_partitions` (writes the training-partition cache used when a unit is computed)
4. `run_common_comparison`
5. `summarise_common_results`

No notebook calls these functions. The notebooks read the completed outputs, mainly
through `load_primary110`, which checks the saved results against the locked input files,
patient set, folds and completed-unit records.

The frozen outer-fold assignments come from the prediction file of an earlier three-modality
analysis,
`Clinical-Mutation-Transcriptomics-Combined/Reviewed/gene_level/clinical_mutation_transcriptomics_nested_cv_predictions_reviewed.csv`.
This file is retained solely as the frozen-fold source, and `load_primary110` checks its hash.
`common_cohort_protocol.json` records the protocol, input files and their hashes, and
`common_cohort_completion_report.md` records the validation of the completed run.

## Scripts

All scripts are in `scripts/OligoPolyThesis/`.

- `utils_ml_oligopoly.py`: the central thesis utility module, imported by all notebooks.
- `mutation_phi_fisher_analysis.py`: pairwise Phi/Fisher redundancy analysis for the mutation QC
  step, run outside the notebook because of the number of feature pairs.
- `run_tybalt.py`: aligns the preprocessed TPM matrix to the Tybalt input genes.
- `run_multiplier.R`: projects the cohort into the pretrained MultiPLIER latent space.
- `run_genomic_super_signature.R`: computes GenomicSuperSignature RAV scores.

Supporting files:

- `common_cohort_mutation_identifier_groups.json`: Ensembl/Entrez identifiers that were merged into a
  single gene during mutation QC. `gene_symbols()` uses it to label mutation features with gene
  symbols.
- `gene_plot_display_layouts.json`: layout coordinates used to relabel preserved SHAP images
  with gene symbols.

## Results

Code and notebooks are in git. Large results and saved models are kept outside git under
`/mnt/data3/MCPP-Breast/Machine-Learning-Models-poly_vs_oligo/`. The notebooks keep their
displayed outputs, so figures and tables can be read without rerunning anything. Earlier runtime
folders, caches and figure-export directories have been removed and are not required inputs;
writer cells recreate figure directories when they are run. The fitted common-cohort
interpretation pipelines are not retained; the interpretation notebook reads the saved coefficient,
SHAP and importance outputs.

## Supplementary FLAML analyses

The clinical (`20260601`), mutation (`20260601`), transcriptomic (`20260710`) and
METABRIC-representation (`20260729`, `20260730`, `20260731`) notebooks contain FLAML (AutoML)
comparisons. These are supplementary methodological/model-selection analyses retained in the
repository; they are not reported in the submitted thesis. All FLAML flags (`RUN_FLAML`, `RUN_FULL_COHORT_FLAML`) are intentionally `False`.
FLAML must be installed for `utils_ml_oligopoly.py` to import.

## Reproducibility notes

- Splits, samplers and models use fixed seeds (42), and the common-cohort analyses share frozen
  outer folds.
- Most notebooks have `RUN_*` flags that switch between refitting and loading saved results.
  Check them before using *Run All*. At present the following notebooks contain flags set to
  `True` that can trigger source preparation, refitting or new attribution runs:
  - `20260729` (METABRIC VAE/HVAE: source preparation, Integrated Gradients)
  - `20260730` (moBRCA-net: source preparation, nested CV)
  - `20260731` (Rostami: source preparation, nested CV)

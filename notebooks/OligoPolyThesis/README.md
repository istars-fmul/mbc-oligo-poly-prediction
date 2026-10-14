# Thesis code: oligometastatic vs polymetastatic breast cancer

## Overview

This folder contains the notebooks for the MSc thesis *Predictive Modelling and Stratification of Metastatic Breast Cancer Patients* (Instituto Superior Técnico, October 2026).

The thesis investigates whether clinical variables, somatic mutations from whole-exome sequencing (WES), and RNA-seq expression from the primary tumour can distinguish patients with oligometastatic (OMD) from polymetastatic (PMD) disease at first metastatic diagnosis.

The analyses include modality-specific modelling, multimodal integration, transcriptomic representation learning, model interpretation, and subsequent biological and prognostic characterisation.

The notebooks retain their displayed outputs. Large datasets, saved model outputs and analysis results are stored on the iSTARS server and are not included in this Git repository.

Throughout this README, `RESULTS/` refers to the main thesis results directory on the iSTARS server.

## Scope

| Location | Contents |
|---|---|
| `notebooks/OligoPolyThesis/` | Analysis notebooks for the MSc thesis |
| `scripts/OligoPolyThesis/` | Shared utility module, representation-generation scripts and supporting configuration files |
| `RESULTS/` | Saved thesis model outputs and analysis results on the iSTARS server |

Preprocessed clinical, mutation and transcriptomic inputs are also stored on the iSTARS server.

## Terminology

- **Common cohort (N=110)**: the shared cohort of 110 patients (43 OMD and 67 PMD) with complete clinical, mutation and transcriptomic data. The primary modality comparison uses identical frozen outer folds across all configurations.
- **Full availability**: modality-specific analyses using all eligible patients: clinical N=149, mutation N=112 and transcriptomics N=139.
- **Eight model-prioritised genes**: H3C14, KIAA2012, MYEOV, TMPRSS4, ZNF549, GDAP1, RUNDC3B and ZNF205.

Some internal result paths and filenames retain historical names, including `primary_110`, `harmonised_main8` and `main8`. These are kept for compatibility with the saved analysis outputs and do not represent separate analyses.

## Analysis map

Notebook names are shortened below for readability.

| Thesis analysis | Notebook | Supporting script(s) | Main result location |
|---|---|---|---|
| Clinical QC and preprocessing (3.2) | `20260401-qc_…`, `20260401-preprocessing_…` | — | iSTARS server |
| Mutation QC and Phi/Fisher redundancy (3.3) | `20260505_Mutation_QC_…` | `mutation_phi_fisher_analysis.py` | iSTARS server |
| TPM preprocessing (3.1.2) | `20260707-preprocessing-tpm-…` | — | iSTARS server |
| Clinical model (4.2.1) | `20260601-modeling-clinical-data` | — | `RESULTS/Common-Cohort-Modality-Comparison/` and full-availability outputs |
| Mutation model (4.2.2) | `20260601-modeling-mutation` | — | `RESULTS/Common-Cohort-Modality-Comparison/` and full-availability outputs |
| Clinical + mutation (4.2.3) | `20260702-modeling-clinical-mutation-…` | — | `RESULTS/Common-Cohort-Modality-Comparison/` |
| Transcriptomic model and representation benchmark (4.2.4, 4.4.1–4.4.2) | `20260710-representation-learning-tpm-…` | `run_tybalt.py`, `run_multiplier.R`, `run_genomic_super_signature.R` | `RESULTS/Transcriptomics-Representation-Learning/`; N=110 results in `RESULTS/Common-Cohort-Modality-Comparison/` |
| METABRIC VAE/HVAE (4.4.3) | `20260729-…-METABRIC-IntegrativeVAE-…` | — | `RESULTS/Transcriptomics-Representation-Learning/METABRIC-IntegrativeVAE/` |
| moBRCA-net-inspired attention (4.4.3) | `20260730-…-moBRCAnet-attention-…` | — | `RESULTS/Transcriptomics-BC-Specific-Representations/moBRCA-net-Attention/` |
| Rostami-inspired autoencoder (4.4.3) | `20260731-…-rostami-autoencoder-…` | — | `RESULTS/Transcriptomics-BC-Specific-Representations/Rostami-Autoencoder/` |
| Clinical + transcriptomics (4.2.4) | `20260929-modeling-clinical-transcriptomics-…` | — | `RESULTS/Common-Cohort-Modality-Comparison/` |
| Mutation + transcriptomics (4.2.4) | `20260929-modeling-mutation-transcriptomics-…` | — | `RESULTS/Common-Cohort-Modality-Comparison/` |
| Three-modality model (4.3) | `20260825-modeling-clinical-mutation-transcriptomics` | — | `RESULTS/Common-Cohort-Modality-Comparison/` |
| Common-cohort summary (4.1, Table 4.1, Figure 4.1, Tables A.1–A.2) | `20260913-summary-of-model-results-oligopoly` | `utils_ml_oligopoly.py` | `RESULTS/Common-Cohort-Modality-Comparison/` |
| Common-cohort interpretation and selection of the eight genes (3.10, Table 4.18) | `20260924-common-cohort-model-interpretation-oligopoly` | — | `RESULTS/Common-Cohort-Modality-Comparison/primary_110/interpretation/` and `RESULTS/Common-Cohort-Modality-Comparison/reconstruction/` |
| Expression differences (4.6.1, Figures 4.15–4.16) | `20260924-mcpp-model-prioritized-gene-expression-…` | — | `RESULTS/MCPP-Biological-Characterization/common-110/harmonised_main8/` |
| Expression–SHAP relationships (Figure 4.17) | `20261002-expression-shap-survival-context-…` | — | `RESULTS/MCPP-Biological-Characterization/common-110/harmonised_main8/` |
| Overall survival (4.6.3, Figures 4.18–4.19, Table 4.20) | `20260924-mcpp-metastatic-overall-survival` | — | `RESULTS/MCPP-Biological-Characterization/common-110/survival/` |
| TCGA-BRCA PFI and DSS (4.6.4, Table 4.21, Figure 4.20) | `20260922-tcga-brca-prognostic-relevance-…` | — | `RESULTS/TCGA-BRCA-Prognostic-Relevance/main8/` |

## Suggested reading / execution order

This follows the scientific dependencies rather than the filename dates.

1. **QC and preprocessing**: clinical (`20260401` QC, then preprocessing), mutation (`20260505`, with `mutation_phi_fisher_analysis.py` used for the pairwise Phi/Fisher analysis), and TPM (`20260707`).
2. **Representation input generation**: `run_genomic_super_signature.R`, `run_multiplier.R` and `run_tybalt.py`.
3. **Full-availability and representation analyses**: clinical, mutation and transcriptomic modelling, followed by the METABRIC-based representations (`20260729`, `20260730`, `20260731`).
4. **Primary N=110 common-cohort comparison**: the seven single- and multi-modality input configurations.
5. **Interpretation**: `20260924-common-cohort-model-interpretation-oligopoly`, which also defines the eight model-prioritised genes.
6. **Biological and prognostic characterisation**: expression, expression–SHAP relationships, metastatic overall survival and TCGA-BRCA analyses.

## Primary N=110 common-cohort comparison

The thesis's primary comparison evaluates seven input configurations:

1. clinical;
2. mutation;
3. transcriptomics;
4. clinical + mutation;
5. clinical + transcriptomics;
6. mutation + transcriptomics;
7. clinical + mutation + transcriptomics.

The completed results are stored under:

    RESULTS/Common-Cohort-Modality-Comparison/

They were produced using functions in:

    scripts/OligoPolyThesis/utils_ml_oligopoly.py

The main generation steps are:

    load_common_sources
    initialise_common_output
    prepare_training_partitions
    run_common_comparison
    summarise_common_results

The notebooks primarily read the completed outputs through `load_primary110`, which checks the saved results against the locked input files, patient set, fold assignments and completed-unit records.

The frozen outer-fold assignments originate from an earlier three-modality prediction file retained on the iSTARS server as the frozen-fold source.

`common_cohort_protocol.json` records the analysis protocol and locked input files, while `common_cohort_completion_report.md` records validation of the completed comparison.

## Scripts

All thesis-specific scripts are in `scripts/OligoPolyThesis/`.

- `utils_ml_oligopoly.py`: shared utility module used throughout the thesis notebooks.
- `mutation_phi_fisher_analysis.py`: pairwise Phi/Fisher redundancy analysis used during mutation QC.
- `run_tybalt.py`: prepares the institutional transcriptomic data for the pretrained Tybalt representation.
- `run_multiplier.R`: projects the cohort into the pretrained MultiPLIER latent space.
- `run_genomic_super_signature.R`: computes GenomicSuperSignature RAV scores.

Supporting files:

- `common_cohort_mutation_identifier_groups.json`: identifier information used to label merged mutation features with gene symbols.
- `gene_plot_display_layouts.json`: display coordinates used when relabelling selected SHAP figures.

## Results

The code and notebooks are stored in Git.

Large datasets, saved model outputs and analysis results are stored on the iSTARS server and are not included in this repository.

The notebooks retain their displayed outputs, allowing the main figures and tables to be inspected without rerunning the analyses.

The retained reconstruction outputs contain fold-specific selected-feature information used in the thesis tables.

For the common-cohort interpretation analyses, the fitted pipelines are not retained. The interpretation notebook instead reads the saved coefficients, feature importances and SHAP outputs.

## Supplementary FLAML analyses

The clinical (`20260601`), mutation (`20260601`), transcriptomic (`20260710`) and METABRIC-representation (`20260729`, `20260730`, `20260731`) notebooks contain supplementary FLAML (AutoML) comparisons.

These are methodological/model-selection analyses and are not reported in the submitted thesis.

The FLAML execution flags, including `RUN_FLAML` and `RUN_FULL_COHORT_FLAML`, are intentionally set to `False`.

FLAML must be installed for `utils_ml_oligopoly.py` to import.

## Software and environments

The analyses were run on the iSTARS server using two environments.

**Python** (notebooks and the Python scripts): Python 3.12.3 in the virtual environment `~/proj-mcpp-breast/.venv`.

```bash
source ~/proj-mcpp-breast/.venv/bin/activate
```

When rerunning notebooks in VS Code or Jupyter on iSTARS, select the kernel **"Python (.venv proj-mcpp-breast)"**; some notebooks retain historical kernel metadata that may show a different name.

**R** (`run_multiplier.R`, `run_genomic_super_signature.R`): the conda environment `mcpp-r`, with R 4.4.3 and Bioconductor 3.20.

```bash
conda activate mcpp-r
```

Both R scripts install missing packages automatically. To reproduce the thesis results, use the recorded `mcpp-r` environment rather than relying on this fallback, which would install the current package versions.

| Software / package | Version |
|---|---:|
| Python | 3.12.3 |
| NumPy | 2.4.4 |
| pandas | 2.3.3 |
| SciPy | 1.17.1 |
| scikit-learn | 1.7.2 |
| XGBoost | 2.1.4 |
| LightGBM | 4.7.0 |
| FLAML | 2.6.0 |
| imbalanced-learn | 0.14.2 |
| SHAP | 0.52.0 |
| lifelines | 0.30.3 |
| statsmodels | 0.14.6 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| matplotlib | 3.10.8 |
| R | 4.4.3 |
| Bioconductor | 3.20 |
| GenomicSuperSignature | 1.14.0 |
| PLIER | 0.99.0 |

The complete environment specifications are stored in `environments/OligoPolyThesis/`:

- `python/requirements.txt`: direct Python dependencies of the thesis code;
- `python/requirements-freeze.txt`: complete package list of the Python environment;
- `R/environment.yml`: conda specification of `mcpp-r`;
- `R/environment-explicit.txt`: exact conda package builds;
- `R/sessionInfo.txt`: R session information after loading the thesis R packages.

Some resources are stored separately on the iSTARS server, including the pretrained MultiPLIER model and GenomicSuperSignature RAV model used by the R scripts, and the pretrained Tybalt models used by `run_tybalt.py`. Rerunning the full workflow also requires access to the server-side datasets and saved results.

## Reproducibility notes

- Randomised splits, samplers and models use fixed seeds where applicable, with seed 42 used throughout the main analyses.
- The primary N=110 common-cohort analyses use the same frozen outer folds across all seven input configurations.
- Most notebooks contain `RUN_*` flags that determine whether saved results are loaded or analyses are recomputed.
- Check these flags before using **Run All**.
- Some representation-learning notebooks contain flags that can trigger source preparation, model fitting or attribution calculations.
- Large saved results and restricted patient-level data are not distributed through Git.
- Earlier temporary runtime directories, caches and duplicate figure-export directories are not required for reading the retained thesis results.
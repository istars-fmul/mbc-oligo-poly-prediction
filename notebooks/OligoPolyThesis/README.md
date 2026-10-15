# Thesis code: oligometastatic vs polymetastatic breast cancer

## Overview

This folder contains the notebooks for the MSc thesis *Predictive Modelling and Stratification of Metastatic Breast Cancer Patients* (Instituto Superior Técnico, October 2026).

The thesis investigates whether clinical variables, somatic mutations from whole-exome sequencing (WES), and RNA-seq expression from the primary tumour can distinguish patients with oligometastatic (OMD) from polymetastatic (PMD) disease at first metastatic diagnosis.
The analyses include modality-specific modelling, multimodal integration, transcriptomic representation learning, model interpretation, and subsequent biological and prognostic characterisation.

The notebooks retain their displayed outputs, whereas large datasets, saved model outputs and analysis results are stored on the iSTARS server and are not made available to the public.

## Scope

| Location | Directory | Contents |
|---|---|---|
| Repository (Public) | `mbc-oligo-poly-prediction/environments/OligoPolyThesis/` | Complete specifications of the environments used (Python and R) |
| Repository (Public) | `mbc-oligo-poly-prediction/notebooks/OligoPolyThesis/` | Analysis notebooks for the MSc thesis |
| Repository (Public) | `mbc-oligo-poly-prediction/scripts/OligoPolyThesis/` | Shared utility module, representation-generation scripts and supporting configuration files |
| iSTARS server (private) | `/mnt/data3/MCPP-Breast/` | Multimodal input and preprocessed data |
| iSTARS server (private) | `/mnt/data3/MCPP-Breast/Machine-Learning-Models-poly_vs_oligo/` | Saved thesis model outputs and analysis results |

Throughout this README, `DATA/` refers to `/mnt/data3/MCPP-Breast/` and `RESULTS/` to `/mnt/data3/MCPP-Breast/Machine-Learning-Models-poly_vs_oligo/`. For example, `RESULTS/figures/` means `/mnt/data3/MCPP-Breast/Machine-Learning-Models-poly_vs_oligo/figures/`.

## Terminology

- **Common cohort (N=110)**: the shared cohort of 110 patients (43 OMD and 67 PMD) with complete clinical, mutation and transcriptomic data, for direct and fair comparison between data modalities and models.
- **Full availability**: modality-specific analyses using all eligible patients: clinical N=149, mutation N=112 and transcriptomics N=139.
- **Eight model-prioritised genes**: _H3C14_, _KIAA2012_, _MYEOV_, _TMPRSS4_, _ZNF549_, _GDAP1_, _RUNDC3B_ and _ZNF205_.

## Analysis map

Notebook names are shortened below for readability.

| Thesis analysis | Notebook | Supporting script(s) | Main result location |
|---|---|---|---|
| Clinical QC and preprocessing (3.2) | `20260401-qc_…`, `20260401-preprocessing_…` | — | `DATA/Clinical-Data/` |
| Mutation QC and Phi/Fisher redundancy (3.3) | `20260505_Mutation_QC_…` | `mutation_phi_fisher_analysis.py` | `DATA/Mutation-Data/` |
| TPM preprocessing (3.1.2) | `20260707-preprocessing-tpm-…` | — | `DATA/Transcriptomic-Data` |
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

The notebooks primarily read the completed outputs through `load_primary110`, which checks the saved results against the locked input files, patient set, fold assignments and completed-unit records. The frozen outer-fold assignments originate from an earlier three-modality prediction file retained on the iSTARS server as the frozen-fold source.

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

The code and notebooks are stored in the Git repository, whereas large datasets, saved model outputs and analysis results are only stored on the iSTARS server.

The notebooks retain their displayed outputs, allowing the main figures and tables, including the ones presented on the thesis document, to be inspected without rerunning the analyses.
Saved outputs contain fold-specific selected-feature information used in the thesis tables which are not included on the notebooks.

For the common-cohort interpretation analyses, the fitted pipelines are not retained. The interpretation notebook instead reads the saved coefficients, feature importances and SHAP outputs.

## Supplementary FLAML analyses

The clinical (`20260601`), mutation (`20260601`), transcriptomic (`20260710`) and METABRIC-representation (`20260729`, `20260730`, `20260731`) notebooks contain supplementary FLAML (AutoML) comparisons. These are methodological/model-selection analyses and are not reported in the submitted thesis since the performance was lower than the pipeline reported.

## Software and environments

The analyses were run on the iSTARS server using two environments, their complete specifications are stored in `environments/OligoPolyThesis/`.

**Python** (notebooks and the Python scripts): Python 3.12.3. Use either a conda environment (`mcpp-python`) or a virtual environment (`.venv`).

*Option 1 – conda*

```bash
conda env create -f environments/OligoPolyThesis/Python/environment.yml
conda activate mcpp-python
```

*Option 2 – venv* (requires Python 3.12.3 installed on the system)

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r environments/OligoPolyThesis/Python/requirements.txt
```

To reproduce the exact thesis environment, install from `environments/OligoPolyThesis/Python/requirements-freeze.txt` instead.
If you use Jupyter, register the environment as a kernel (optional; VS Code detects it automatically):

```bash
python -m ipykernel install --user --name mcpp-python
```

**R** (`run_multiplier.R`, `run_genomic_super_signature.R`): R 4.4.3 with Bioconductor 3.20, in the conda environment `mcpp-r`.

*Option 1 – environment file (any platform)*

```bash
conda env create -f environments/OligoPolyThesis/R/environment.yml
conda activate mcpp-r
```

*Option 2 – exact package builds (Linux x86-64 only)*

```bash
conda create --name mcpp-r --file environments/OligoPolyThesis/R/environment-explicit.txt
conda activate mcpp-r
```

*Then, for both options*, install readr 2.2.0 from CRAN (not available through conda):

```bash
Rscript -e 'install.packages("remotes", repos = "https://cloud.r-project.org"); remotes::install_version("readr", version = "2.2.0", repos = "https://cloud.r-project.org")'
```
Package versions loaded during the thesis runs are recorded in `environments/OligoPolyThesis/R/sessionInfo.txt`.

Both R scripts install missing packages automatically, to reproduce the exact thesis results, use the recorded `mcpp-r` environment rather than relying on this fallback, which would install the current package versions.

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

## Reproducibility notes

- Randomised splits, samplers and models use fixed seeds where applicable, with seed 42 used throughout the main analyses.
- Most notebooks contain `RUN_*` flags that determine whether saved results are loaded or analyses are recomputed (Check these flags before using **Run All**.)
- Large saved results and the multimodal data (clinical, transcriptomic and mutation) are not publicly available. Rerunning the full workflow requires access to the server-side datasets and saved results.

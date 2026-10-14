# Oligometastatic vs Polymetastatic Breast Cancer

This repository contains code and supporting resources for research on the distinction between oligometastatic (OMD) and polymetastatic (PMD) breast cancer.

The project investigates whether clinical, genomic and transcriptomic information from the primary tumour can help characterise and predict metastatic disease burden at first metastatic diagnosis. The work includes data preprocessing, predictive modelling, multimodal integration, transcriptomic representation learning, model interpretation, and biological and prognostic analyses.

The code developed specifically for Marta Pimentel's MSc thesis, *Predictive Modelling and Stratification of Metastatic Breast Cancer Patients*, is organised in the `OligoPolyThesis` directories:

- `notebooks/OligoPolyThesis/` — Jupyter notebooks containing the thesis analyses;
- `scripts/OligoPolyThesis/` — shared utilities and supporting scripts used by those notebooks.

A more detailed description of the thesis workflow, notebooks and analysis outputs is available in `notebooks/OligoPolyThesis/README.md`.

Large datasets, saved model outputs and analysis results are stored separately on the iSTARS server and are not included in this Git repository.

## Directory structure

    ├── Makefile           <- Makefile with project commands
    ├── README.md          <- Top-level project documentation
    │
    ├── docker             <- Docker and VM configuration
    │
    ├── logs               <- Log files
    │
    ├── notebooks          <- Jupyter notebooks
    │   └── OligoPolyThesis/
    │                      <- MSc thesis notebooks
    │
    ├── outputs            <- Project outputs stored in the repository, when applicable
    │
    ├── params.yaml        <- Configuration parameters
    │
    ├── pipelines          <- Project pipelines
    │
    ├── resources          <- Additional project resources
    │
    ├── scripts            <- Project scripts
    │   └── OligoPolyThesis/
    │                      <- MSc thesis utilities and supporting scripts
    │
    ├── src                <- Shared project source code
    │
    └── tests              <- Test modules

## Prerequisites

Project-specific requirements depend on the analysis being run.

The MSc thesis notebooks primarily use Python and Jupyter, with some transcriptomic representation-learning steps also requiring R/Bioconductor packages or external pretrained models.

## Install dependencies

Dependencies vary across the different analyses and software environments used in the project.

For the MSc thesis code, see `notebooks/OligoPolyThesis/README.md`.

## Usage

For the MSc thesis analyses, start from `notebooks/OligoPolyThesis/README.md`.

This document describes the main notebooks, supporting scripts, recommended execution order and corresponding saved results.

## Credentials

Restricted project data and saved analysis results require authorised access to the iSTARS server.

No credentials, passwords or access tokens should be stored in this repository.

## Contact

For questions about the workflow or analysis code, open an issue in this repository.

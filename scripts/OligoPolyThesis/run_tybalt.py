# -----------------------------------------------------------------------------
# Prepare Tybalt-aligned input for MCPP-Breast transcriptomics data
#
# Aligns the preprocessed institutional TPM matrix to the ordered 5,000-gene
# input space of the pretrained Tybalt encoders (expression features mapping
# to the same gene symbol are averaged; Tybalt genes absent from the cohort
# are filled with zero). The encoders are not applied here: scaling and
# projection happen inside the nested-CV pipelines of
# 20260710-representation-learning-tpm-oligopolypatients.ipynb.
#
# Usage:  python run_tybalt.py [base_dir]    (default: /mnt/data3/MCPP-Breast)
# Reads:  <base_dir>/Transcriptomic-Data/Preprocessed/ and ~/external/tybalt
# Writes: CSV files in <base_dir>/Machine-Learning-Models-poly_vs_oligo/
#         Transcriptomics-Representation-Learning/Tybalt/ (aligned input,
#         gene-overlap summary and run summary)
# -----------------------------------------------------------------------------

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

if len(sys.argv) >= 2:
    base_dir = Path(sys.argv[1]).expanduser()
else:
    base_dir = Path("/mnt/data3/MCPP-Breast")

tybalt_repo_dir = Path("~/external/tybalt").expanduser()

print("Base directory:")
print(base_dir)

print("\nTybalt repository directory:")
print(tybalt_repo_dir)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

transcriptomics_preprocessed_dir = (
    base_dir
    / "Transcriptomic-Data"
    / "Preprocessed"
)

preprocessed_transcriptomics_path = (
    transcriptomics_preprocessed_dir
    / "transcriptomics_oligo_poly_log2_tpm_preprocessed.csv"
)

feature_mapping_path = (
    transcriptomics_preprocessed_dir
    / "transcriptomics_feature_name_mapping.csv"
)

tybalt_training_input_path = (
    tybalt_repo_dir
    / "data"
    / "pancan_scaled_zeroone_rnaseq.tsv.gz"
)

tybalt_models = {
    "onehidden": (
        tybalt_repo_dir
        / "models"
        / "encoder_onehidden_vae.hdf5"
    ),
    "twohidden100": (
        tybalt_repo_dir
        / "models"
        / "encoder_twohidden100_vae.hdf5"
    ),
    "twohidden300": (
        tybalt_repo_dir
        / "models"
        / "encoder_twohidden300_vae.hdf5"
    ),
}

output_dir = (
    base_dir
    / "Machine-Learning-Models-poly_vs_oligo"
    / "Transcriptomics-Representation-Learning"
    / "Tybalt"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

from pathlib import Path
import sys
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from scripts.OligoPolyThesis.utils_ml_oligopoly import clean_gene_symbol


# -----------------------------------------------------------------------------
# Load institutional data
# -----------------------------------------------------------------------------

print("\nLoading preprocessed transcriptomics matrix...")

transcriptomics_df = pd.read_csv(
    preprocessed_transcriptomics_path
)

print(
    "Transcriptomics matrix shape:",
    transcriptomics_df.shape,
)

print("\nLoading feature mapping...")

feature_mapping_df = pd.read_csv(
    feature_mapping_path
)

print(
    "Feature mapping shape:",
    feature_mapping_df.shape,
)


# -----------------------------------------------------------------------------
# Load Tybalt gene order
# -----------------------------------------------------------------------------

print("\nLoading Tybalt training gene list...")

tybalt_header_df = pd.read_csv(
    tybalt_training_input_path,
    sep="\t",
    nrows=0,
    index_col=0,
)

tybalt_genes = list(
    tybalt_header_df.columns
)

print(
    "Number of genes expected by Tybalt:",
    len(tybalt_genes),
)

if len(tybalt_genes) != 5000:
    raise ValueError(
        "Expected 5000 Tybalt input genes, "
        f"found {len(tybalt_genes)}."
    )


# -----------------------------------------------------------------------------
# Check pretrained encoder files
# -----------------------------------------------------------------------------

for model_name, model_path in tybalt_models.items():
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing Tybalt model {model_name}: "
            f"{model_path}"
        )

print("\nAll three pretrained Tybalt encoders were found.")


# -----------------------------------------------------------------------------
# Build sample-by-gene institutional expression matrix
# -----------------------------------------------------------------------------

feature_cols = [
    col
    for col in transcriptomics_df.columns
    if col.startswith("transcriptomics__")
]

if len(feature_cols) == 0:
    raise ValueError(
        "No transcriptomics feature columns found."
    )

if "sample_id_normalised" not in transcriptomics_df.columns:
    raise ValueError(
        "Missing sample_id_normalised."
    )

required_mapping_cols = [
    "model_feature_name",
    "gene_name",
]

missing_mapping_cols = [
    col
    for col in required_mapping_cols
    if col not in feature_mapping_df.columns
]

if missing_mapping_cols:
    raise ValueError(
        "Missing feature-mapping columns: "
        + ", ".join(missing_mapping_cols)
    )

feature_to_gene_df = (
    feature_mapping_df[
        [
            "model_feature_name",
            "gene_name",
        ]
    ]
    .copy()
)

feature_to_gene_df["gene_name"] = (
    feature_to_gene_df["gene_name"]
    .map(clean_gene_symbol)
)

feature_to_gene_df = feature_to_gene_df[
    feature_to_gene_df[
        "model_feature_name"
    ].isin(feature_cols)
    &
    feature_to_gene_df[
        "gene_name"
    ].notna()
].copy()

print(
    "\nNumber of transcriptomics features "
    "with gene symbols:",
    feature_to_gene_df.shape[0],
)

expression_df = transcriptomics_df[
    ["sample_id_normalised"]
    + feature_to_gene_df[
        "model_feature_name"
    ].tolist()
].copy()

expression_df = expression_df.set_index(
    "sample_id_normalised"
)

expression_df.columns = (
    feature_to_gene_df[
        "gene_name"
    ].tolist()
)

# Average multiple Ensembl features mapping to the same gene symbol.
expression_gene_symbol_df = (
    expression_df
    .T
    .groupby(level=0)
    .mean()
    .T
)

print(
    "Expression matrix with gene symbols:",
    expression_gene_symbol_df.shape,
)


# -----------------------------------------------------------------------------
# Align institutional data to the exact 5000-gene Tybalt input space
# -----------------------------------------------------------------------------

common_genes = [
    gene
    for gene in tybalt_genes
    if gene in expression_gene_symbol_df.columns
]

missing_tybalt_genes = [
    gene
    for gene in tybalt_genes
    if gene not in expression_gene_symbol_df.columns
]

print("\nGene overlap with Tybalt:")
print("Tybalt input genes:", len(tybalt_genes))
print("Common genes:", len(common_genes))
print("Missing Tybalt genes:", len(missing_tybalt_genes))
print(
    "Fraction available:",
    len(common_genes) / len(tybalt_genes),
)

if len(common_genes) < 1000:
    raise ValueError(
        "Low overlap with Tybalt input genes. "
        "Check gene identifiers."
    )

tybalt_input_df = pd.DataFrame(
    0.0,
    index=expression_gene_symbol_df.index,
    columns=tybalt_genes,
)

tybalt_input_df.loc[
    :,
    common_genes,
] = expression_gene_symbol_df.loc[
    :,
    common_genes,
]

if tybalt_input_df.shape != (
    transcriptomics_df.shape[0],
    5000,
):
    raise ValueError(
        "Unexpected aligned Tybalt input shape: "
        f"{tybalt_input_df.shape}"
    )

if tybalt_input_df.isna().any().any():
    raise ValueError(
        "NaN values found in Tybalt-aligned input."
    )

if not np.isfinite(
    tybalt_input_df.values
).all():
    raise ValueError(
        "Non-finite values found in Tybalt-aligned input."
    )


# -----------------------------------------------------------------------------
# Save unscaled aligned input
# -----------------------------------------------------------------------------

tybalt_aligned_input_path = (
    output_dir
    / "transcriptomics_tybalt_aligned_input.csv"
)

(
    tybalt_input_df
    .reset_index()
    .to_csv(
        tybalt_aligned_input_path,
        index=False,
    )
)

print("\nSaved unscaled Tybalt-aligned input:")
print(tybalt_aligned_input_path)

print(
    "Aligned input shape:",
    tybalt_input_df.shape,
)


# -----------------------------------------------------------------------------
# Save overlap summary
# -----------------------------------------------------------------------------

gene_overlap_summary_df = pd.DataFrame(
    {
        "n_tybalt_input_genes": [
            len(tybalt_genes)
        ],
        "n_expression_genes_with_symbols": [
            expression_gene_symbol_df.shape[1]
        ],
        "n_common_genes": [
            len(common_genes)
        ],
        "n_missing_tybalt_genes": [
            len(missing_tybalt_genes)
        ],
        "fraction_tybalt_genes_available": [
            len(common_genes)
            / len(tybalt_genes)
        ],
    }
)

gene_overlap_summary_path = (
    output_dir
    / "transcriptomics_tybalt_gene_overlap_summary.csv"
)

gene_overlap_summary_df.to_csv(
    gene_overlap_summary_path,
    index=False,
)


# -----------------------------------------------------------------------------
# Save run summary
# -----------------------------------------------------------------------------

run_summary_df = pd.DataFrame(
    [
        {
            "n_samples": tybalt_input_df.shape[0],
            "n_tybalt_input_genes": tybalt_input_df.shape[1],
            "n_common_genes": len(common_genes),
            "n_missing_tybalt_genes": len(missing_tybalt_genes),
            "input_scaling": (
                "none; fold-specific MinMax scaling "
                "is performed inside nested-CV pipeline"
            ),
            "onehidden_model_path": str(
                tybalt_models["onehidden"]
            ),
            "twohidden100_model_path": str(
                tybalt_models["twohidden100"]
            ),
            "twohidden300_model_path": str(
                tybalt_models["twohidden300"]
            ),
            "aligned_input_path": str(
                tybalt_aligned_input_path
            ),
        }
    ]
)

run_summary_path = (
    output_dir
    / "transcriptomics_tybalt_run_summary.csv"
)

run_summary_df.to_csv(
    run_summary_path,
    index=False,
)

print("\nGene overlap summary saved to:")
print(gene_overlap_summary_path)

print("\nRun summary saved to:")
print(run_summary_path)

print("\nTybalt input preparation completed successfully.")
# -----------------------------------------------------------------------------
# Run MultiPLIER projection for MCPP-Breast transcriptomics data
# -----------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
    base_dir <- path.expand(args[1])
} else {
    base_dir <- path.expand("~/MCPP-Breast")
}

message("Base directory:")
message(base_dir)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

multiplier_repo_dir <- file.path(
    base_dir,
    "Machine-Learning-Models-poly_vs_oligo",
    "External-Data",
    "multi-plier"
)

multiplier_model_path <- file.path(
    base_dir,
    "Machine-Learning-Models-poly_vs_oligo",
    "External-Data",
    "multi-plier-models",
    "recount_PLIER_model.RDS"
)

plier_util_path <- file.path(
    multiplier_repo_dir,
    "util",
    "plier_util.R"
)

transcriptomics_preprocessed_dir <- file.path(
    base_dir,
    "Transcriptomic-Data",
    "Preprocessed"
)

preprocessed_transcriptomics_path <- file.path(
    transcriptomics_preprocessed_dir,
    "transcriptomics_oligo_poly_log2_tpm_preprocessed.csv"
)

feature_mapping_path <- file.path(
    transcriptomics_preprocessed_dir,
    "transcriptomics_feature_name_mapping.csv"
)

output_dir <- file.path(
    base_dir,
    "Machine-Learning-Models-poly_vs_oligo",
    "Transcriptomics-Representation-Learning",
    "MultiPLIER"
)

dir.create(
    output_dir,
    recursive = TRUE,
    showWarnings = FALSE
)

scores_output_path <- file.path(
    output_dir,
    "transcriptomics_multiplier_scores.csv"
)

gene_overlap_output_path <- file.path(
    output_dir,
    "transcriptomics_multiplier_gene_overlap_summary.csv"
)

run_summary_output_path <- file.path(
    output_dir,
    "transcriptomics_multiplier_run_summary.csv"
)

# -----------------------------------------------------------------------------
# Check required files
# -----------------------------------------------------------------------------

if (!file.exists(multiplier_model_path)) {
    stop(
        paste(
            "MultiPLIER model file not found:",
            multiplier_model_path
        )
    )
}

if (!file.exists(plier_util_path)) {
    stop(
        paste(
            "MultiPLIER utility file not found:",
            plier_util_path
        )
    )
}

if (!file.exists(preprocessed_transcriptomics_path)) {
    stop(
        paste(
            "Preprocessed transcriptomics file not found:",
            preprocessed_transcriptomics_path
        )
    )
}

if (!file.exists(feature_mapping_path)) {
    stop(
        paste(
            "Feature mapping file not found:",
            feature_mapping_path
        )
    )
}

# -----------------------------------------------------------------------------
# Load packages
# -----------------------------------------------------------------------------

required_packages <- c(
    "readr",
    "dplyr",
    "tibble",
    "stringr"
)

for (pkg in required_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(
            pkg,
            repos = "https://cloud.r-project.org"
        )
    }
}

library(readr)
library(dplyr)
library(tibble)
library(stringr)

# -----------------------------------------------------------------------------
# Load MultiPLIER utilities
# -----------------------------------------------------------------------------

source(plier_util_path)

if (!exists("GetNewDataB")) {
    stop(
        "GetNewDataB was not loaded from plier_util.R"
    )
}

message("\nGetNewDataB successfully loaded.")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

clean_gene_symbol <- function(x) {
    x <- as.character(x)
    x <- str_trim(x)
    x[x == ""] <- NA_character_
    x
}

# -----------------------------------------------------------------------------
# Load input data
# -----------------------------------------------------------------------------

message("\nLoading transcriptomics matrix...")

transcriptomics_df <- read_csv(
    preprocessed_transcriptomics_path,
    show_col_types = FALSE
)

message("Transcriptomics dimensions:")
print(dim(transcriptomics_df))

message("\nLoading feature mapping...")

feature_mapping_df <- read_csv(
    feature_mapping_path,
    show_col_types = FALSE
)

message("Feature mapping dimensions:")
print(dim(feature_mapping_df))

# -----------------------------------------------------------------------------
# Load pretrained recount2 MultiPLIER model
# -----------------------------------------------------------------------------

message("\nLoading pretrained recount2 MultiPLIER model...")

multiplier_model <- readRDS(
    multiplier_model_path
)

message("Pretrained MultiPLIER model loaded.")

if (is.null(multiplier_model$Z)) {
    stop(
        "The loaded MultiPLIER object does not contain a Z matrix."
    )
}

message("\nMultiPLIER Z dimensions:")
print(dim(multiplier_model$Z))

# -----------------------------------------------------------------------------
# Identify feature columns
# -----------------------------------------------------------------------------

feature_cols <- grep(
    "^transcriptomics__",
    colnames(transcriptomics_df),
    value = TRUE
)

if (length(feature_cols) == 0) {
    stop(
        "No transcriptomics feature columns found."
    )
}

required_metadata_cols <- c(
    "sample_id_normalised"
)

missing_metadata_cols <- setdiff(
    required_metadata_cols,
    colnames(transcriptomics_df)
)

if (length(missing_metadata_cols) > 0) {
    stop(
        paste(
            "Missing required metadata columns:",
            paste(
                missing_metadata_cols,
                collapse = ", "
            )
        )
    )
}

required_mapping_cols <- c(
    "model_feature_name",
    "ensembl_gene_id",
    "gene_name"
)

missing_mapping_cols <- setdiff(
    required_mapping_cols,
    colnames(feature_mapping_df)
)

if (length(missing_mapping_cols) > 0) {
    stop(
        paste(
            "Missing required columns in feature mapping:",
            paste(
                missing_mapping_cols,
                collapse = ", "
            )
        )
    )
}

# -----------------------------------------------------------------------------
# Build expression matrix: genes x samples
# -----------------------------------------------------------------------------

sample_ids <- transcriptomics_df$sample_id_normalised

expression_samples_by_features <- transcriptomics_df %>%
    select(
        all_of(feature_cols)
    )

expression_features_by_samples <- t(
    as.matrix(
        expression_samples_by_features
    )
)

colnames(
    expression_features_by_samples
) <- sample_ids

rownames(
    expression_features_by_samples
) <- feature_cols

message("\nFeature-level expression matrix:")
print(dim(expression_features_by_samples))

# -----------------------------------------------------------------------------
# Map model feature names to gene symbols
# -----------------------------------------------------------------------------

feature_to_symbol_df <- feature_mapping_df %>%
    select(
        model_feature_name,
        ensembl_gene_id,
        gene_name
    ) %>%
    mutate(
        gene_name = clean_gene_symbol(
            gene_name
        )
    ) %>%
    filter(
        model_feature_name %in%
            rownames(
                expression_features_by_samples
            ),
        !is.na(gene_name)
    )

if (nrow(feature_to_symbol_df) == 0) {
    stop(
        "No transcriptomics features could be mapped to gene symbols."
    )
}

message(
    "\nNumber of transcriptomics features with gene symbols: ",
    nrow(feature_to_symbol_df)
)

expression_with_symbols <- expression_features_by_samples[
    feature_to_symbol_df$model_feature_name,
    ,
    drop = FALSE
]

rownames(
    expression_with_symbols
) <- feature_to_symbol_df$gene_name

# If multiple Ensembl IDs map to the same gene symbol,
# average their expression values.
expression_symbol_df <- as.data.frame(
    expression_with_symbols
) %>%
    rownames_to_column(
        "gene_symbol"
    ) %>%
    group_by(
        gene_symbol
    ) %>%
    summarise(
        across(
            everything(),
            ~ mean(.x, na.rm = TRUE)
        ),
        .groups = "drop"
    )

expression_symbol_matrix <- expression_symbol_df %>%
    column_to_rownames(
        "gene_symbol"
    ) %>%
    as.matrix()

message("\nExpression matrix with gene symbols:")
print(dim(expression_symbol_matrix))

# -----------------------------------------------------------------------------
# Align institutional expression data with MultiPLIER model genes
# -----------------------------------------------------------------------------

multiplier_genes <- rownames(
    multiplier_model$Z
)

if (is.null(multiplier_genes)) {
    stop(
        "The MultiPLIER Z matrix does not contain gene names."
    )
}

common_genes <- intersect(
    rownames(
        expression_symbol_matrix
    ),
    multiplier_genes
)

message("\nGene overlap:")
message(
    "Expression genes with symbols: ",
    nrow(expression_symbol_matrix)
)

message(
    "MultiPLIER model genes: ",
    length(multiplier_genes)
)

message(
    "Common genes: ",
    length(common_genes)
)

message(
    "Fraction of institutional genes represented in MultiPLIER: ",
    round(
        length(common_genes) /
            nrow(expression_symbol_matrix),
        4
    )
)

message(
    "Fraction of MultiPLIER genes available: ",
    round(
        length(common_genes) /
            length(multiplier_genes),
        4
    )
)

if (length(common_genes) < 100) {
    stop(
        "Very low gene overlap with the MultiPLIER model. Check gene identifiers."
    )
}

# -----------------------------------------------------------------------------
# Prepare expression matrix for projection
# -----------------------------------------------------------------------------

expression_common <- expression_symbol_matrix[
    common_genes,
    ,
    drop = FALSE
]

message("\nExpression matrix before variance filtering:")
print(dim(expression_common))

zero_variance_genes <- apply(
    expression_common,
    1,
    var,
    na.rm = TRUE
) == 0

n_zero_variance_genes <- sum(
    zero_variance_genes
)

message(
    "Zero-variance genes removed: ",
    n_zero_variance_genes
)

if (n_zero_variance_genes > 0) {
    expression_common <- expression_common[
        !zero_variance_genes,
        ,
        drop = FALSE
    ]
}

if (anyNA(expression_common)) {
    stop(
        "Missing values detected in the expression matrix used for MultiPLIER projection."
    )
}

# Published applications of GetNewDataB add a small positive constant
# to avoid exact zero values in the projected expression matrix.
expression_common <- expression_common + 0.01

message("\nExpression matrix used for projection:")
print(dim(expression_common))

# -----------------------------------------------------------------------------
# Save gene-overlap summary
# -----------------------------------------------------------------------------

gene_overlap_summary_df <- tibble(
    model = "pretrained_recount2_MultiPLIER",
    n_expression_genes_with_symbols = nrow(
        expression_symbol_matrix
    ),
    n_multiplier_model_genes = length(
        multiplier_genes
    ),
    n_common_genes_before_variance_filter = length(
        common_genes
    ),
    n_zero_variance_genes_removed = n_zero_variance_genes,
    n_genes_used_for_projection = nrow(
        expression_common
    ),
    fraction_expression_genes_in_multiplier = (
        length(common_genes) /
            nrow(expression_symbol_matrix)
    ),
    fraction_multiplier_genes_available = (
        length(common_genes) /
            length(multiplier_genes)
    )
)

write_csv(
    gene_overlap_summary_df,
    gene_overlap_output_path
)

message("\nGene overlap summary:")
print(gene_overlap_summary_df)

# -----------------------------------------------------------------------------
# Project samples into pretrained MultiPLIER latent-variable space
# -----------------------------------------------------------------------------

message(
    "\nProjecting institutional samples into the pretrained MultiPLIER latent space..."
)

multiplier_scores_matrix <- GetNewDataB(
    exprs.mat = expression_common,
    plier.model = multiplier_model
)

if (is.null(multiplier_scores_matrix)) {
    stop(
        "GetNewDataB returned NULL."
    )
}

message("\nProjected MultiPLIER scores dimensions:")
print(dim(multiplier_scores_matrix))

# Expected orientation:
# latent variables x samples.
if (
    ncol(multiplier_scores_matrix) !=
    ncol(expression_common)
) {
    stop(
        paste0(
            "Unexpected MultiPLIER score dimensions. ",
            "Expected number of columns to equal number of samples."
        )
    )
}

# -----------------------------------------------------------------------------
# Assign safe latent-variable names
# -----------------------------------------------------------------------------

lv_names <- rownames(
    multiplier_scores_matrix
)

if (is.null(lv_names)) {
    lv_names <- paste0(
        "LV",
        seq_len(
            nrow(multiplier_scores_matrix)
        )
    )
}

safe_lv_names <- lv_names %>%
    str_replace_all(
        "[^A-Za-z0-9]+",
        "_"
    ) %>%
    str_replace_all(
        "_+",
        "_"
    ) %>%
    str_replace_all(
        "^_|_$",
        ""
    )

safe_lv_names <- paste0(
    "multiplier__",
    safe_lv_names
)

rownames(
    multiplier_scores_matrix
) <- safe_lv_names

# -----------------------------------------------------------------------------
# Convert scores to samples x latent variables
# -----------------------------------------------------------------------------

scores_df <- as.data.frame(
    t(multiplier_scores_matrix)
) %>%
    rownames_to_column(
        "sample_id_normalised"
    )

message("\nScores dataframe dimensions:")
print(dim(scores_df))

message(
    "Number of latent variables: ",
    ncol(scores_df) - 1
)

message("\nFirst latent-variable columns:")
print(
    head(
        colnames(scores_df)[-1],
        10
    )
)

if (
    !identical(
        as.character(scores_df$sample_id_normalised),
        as.character(colnames(expression_common))
    )
) {
    warning(
        "Sample order in projected scores differs from the expression matrix. Verify IDs before modelling."
    )
}

if (anyNA(scores_df)) {
    stop(
        "Missing values detected in projected MultiPLIER scores."
    )
}

# -----------------------------------------------------------------------------
# Save sample-level MultiPLIER scores
# -----------------------------------------------------------------------------

write_csv(
    scores_df,
    scores_output_path
)

message("\nScores saved to:")
message(scores_output_path)

# -----------------------------------------------------------------------------
# Save run summary
# -----------------------------------------------------------------------------

run_summary_df <- tibble(
    model = "pretrained_recount2_MultiPLIER",
    model_path = multiplier_model_path,
    plier_util_path = plier_util_path,
    n_samples = ncol(expression_common),
    n_expression_genes_with_symbols = nrow(
        expression_symbol_matrix
    ),
    n_multiplier_model_genes = length(
        multiplier_genes
    ),
    n_common_genes_before_variance_filter = length(
        common_genes
    ),
    n_zero_variance_genes_removed = n_zero_variance_genes,
    n_genes_used_for_projection = nrow(
        expression_common
    ),
    n_latent_variables = nrow(
        multiplier_scores_matrix
    ),
    scores_output_path = scores_output_path,
    gene_overlap_output_path = gene_overlap_output_path
)

write_csv(
    run_summary_df,
    run_summary_output_path
)

# -----------------------------------------------------------------------------
# Final messages
# -----------------------------------------------------------------------------

message("\nDone.")

message("\nPretrained MultiPLIER model:")
message(multiplier_model_path)

message("\nScores saved to:")
message(scores_output_path)

message("\nGene overlap summary saved to:")
message(gene_overlap_output_path)

message("\nRun summary saved to:")
message(run_summary_output_path)
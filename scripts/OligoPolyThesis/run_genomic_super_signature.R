#!/usr/bin/env Rscript

# =============================================================================
# Run GenomicSuperSignature on preprocessed transcriptomics data
# =============================================================================

# -----------------------------------------------------------------------------
# Parse command-line arguments
# -----------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
    base_dir <- path.expand(args[1])
} else {
    base_dir <- path.expand("~/MCPP-Breast")
}

message("Base directory:")
message(base_dir)

if (!dir.exists(base_dir)) {
    stop(
        paste0(
            "Base directory does not exist: ",
            base_dir
        )
    )
}

# -----------------------------------------------------------------------------
# Helper: install/load packages
# -----------------------------------------------------------------------------

install_if_missing_cran <- function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        message(paste0("Installing CRAN package: ", pkg))
        install.packages(pkg, repos = "https://cloud.r-project.org")
    }
}

install_if_missing_bioc <- function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        if (!requireNamespace("BiocManager", quietly = TRUE)) {
            install.packages(
                "BiocManager",
                repos = "https://cloud.r-project.org"
            )
        }

        message(
            paste0(
                "Installing Bioconductor package: ",
                pkg
            )
        )

        BiocManager::install(
            pkg,
            ask = FALSE,
            update = FALSE
        )
    }
}

install_if_missing_cran("readr")
install_if_missing_cran("dplyr")
install_if_missing_cran("tibble")
install_if_missing_cran("stringr")

install_if_missing_bioc("GenomicSuperSignature")

suppressPackageStartupMessages({
    library(readr)
    library(dplyr)
    library(tibble)
    library(stringr)
    library(GenomicSuperSignature)
})

# -----------------------------------------------------------------------------
# User-defined parameters
# -----------------------------------------------------------------------------

rav_model_name <- "C2"
top_n_validation_report <- 50

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

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
    "GenomicSuperSignature"
)

dir.create(
    output_dir,
    recursive = TRUE,
    showWarnings = FALSE
)

message("\nResolved paths:")
message(
    paste(
        "transcriptomics_preprocessed_dir:",
        transcriptomics_preprocessed_dir
    )
)
message(
    paste(
        "preprocessed_transcriptomics_path:",
        preprocessed_transcriptomics_path
    )
)
message(
    paste(
        "feature_mapping_path:",
        feature_mapping_path
    )
)
message(
    paste(
        "output_dir:",
        output_dir
    )
)

if (!dir.exists(transcriptomics_preprocessed_dir)) {
    stop(
        paste0(
            "Transcriptomics preprocessed directory does not exist: ",
            transcriptomics_preprocessed_dir
        )
    )
}

if (!file.exists(preprocessed_transcriptomics_path)) {
    stop(
        paste0(
            "Preprocessed transcriptomics file does not exist: ",
            preprocessed_transcriptomics_path
        )
    )
}

if (!file.exists(feature_mapping_path)) {
    stop(
        paste0(
            "Feature mapping file does not exist: ",
            feature_mapping_path
        )
    )
}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

find_first_existing_column <- function(
    df,
    candidate_columns,
    object_name
) {
    existing <- candidate_columns[
        candidate_columns %in% colnames(df)
    ]

    if (length(existing) == 0) {
        stop(
            paste0(
                "Could not find any of these columns in ",
                object_name,
                ": ",
                paste(
                    candidate_columns,
                    collapse = ", "
                )
            )
        )
    }

    return(existing[1])
}

clean_ensembl_id <- function(x) {
    x <- as.character(x)

    x <- str_replace(
        x,
        "^transcriptomics__",
        ""
    )

    x <- str_replace(
        x,
        "\\.[0-9]+$",
        ""
    )

    return(x)
}

normalise_rav_name <- function(x) {
    x <- as.character(x)

    ifelse(
        startsWith(x, "RAV"),
        x,
        paste0("RAV", x)
    )
}


make_unique_non_missing_gene_symbols <- function(
    mapping_df,
    feature_col,
    ensembl_col,
    symbol_col
) {
    out <- mapping_df %>%
        mutate(
            transcriptomics_feature = as.character(
                .data[[feature_col]]
            ),
            ensembl_gene_id = clean_ensembl_id(
                .data[[ensembl_col]]
            ),
            gene_symbol = as.character(
                .data[[symbol_col]]
            )
        ) %>%
        mutate(
            gene_symbol = na_if(
                gene_symbol,
                ""
            ),
            gene_symbol = na_if(
                gene_symbol,
                "NA"
            )
        ) %>%
        filter(
            !is.na(gene_symbol)
        ) %>%
        distinct(
            transcriptomics_feature,
            ensembl_gene_id,
            gene_symbol
        )

    return(out)
}

extract_rav_genes <- function(RAVmodel) {

    possible_genes <- NULL

    try({
        possible_genes <- rownames(RAVmodel)
    }, silent = TRUE)

    if (
        !is.null(possible_genes) &&
        length(possible_genes) > 0
    ) {
        return(possible_genes)
    }

    try({
        possible_genes <- rownames(
            RAVmodel@metadata$geneSets
        )
    }, silent = TRUE)

    if (
        !is.null(possible_genes) &&
        length(possible_genes) > 0
    ) {
        return(possible_genes)
    }

    try({
        possible_genes <- rownames(
            RAVmodel@data
        )
    }, silent = TRUE)

    if (
        !is.null(possible_genes) &&
        length(possible_genes) > 0
    ) {
        return(possible_genes)
    }

    stop(
        "Could not extract gene names from the RAVmodel."
    )
}

# -----------------------------------------------------------------------------
# Load input data
# -----------------------------------------------------------------------------

message(
    "\nLoading preprocessed transcriptomics matrix:"
)

message(
    preprocessed_transcriptomics_path
)

transcriptomics_df <- read_csv(
    preprocessed_transcriptomics_path,
    show_col_types = FALSE
)

message(
    "Input transcriptomics dataframe dimensions:"
)

message(
    paste(
        dim(transcriptomics_df),
        collapse = " x "
    )
)

message(
    "\nLoading feature mapping:"
)

message(
    feature_mapping_path
)

feature_mapping_df <- read_csv(
    feature_mapping_path,
    show_col_types = FALSE
)

message(
    "Feature mapping dimensions:"
)

message(
    paste(
        dim(feature_mapping_df),
        collapse = " x "
    )
)

# -----------------------------------------------------------------------------
# Identify required columns
# -----------------------------------------------------------------------------

sample_id_col <- find_first_existing_column(
    transcriptomics_df,
    c(
        "sample_id_normalised",
        "sample_id",
        "sample_id_original"
    ),
    "transcriptomics_df"
)

target_col <- find_first_existing_column(
    transcriptomics_df,
    c("Polymetastatic"),
    "transcriptomics_df"
)

feature_cols <- colnames(
    transcriptomics_df
)[
    startsWith(
        colnames(transcriptomics_df),
        "transcriptomics__"
    )
]

if (length(feature_cols) == 0) {
    stop(
        "No transcriptomics feature columns found."
    )
}

mapping_feature_col <- find_first_existing_column(
    feature_mapping_df,
    c(
        "model_feature_name",
        "transcriptomics_feature",
        "feature",
        "feature_name",
        "feature_col",
        "transcriptomics_feature_name"
    ),
    "feature_mapping_df"
)

mapping_ensembl_col <- find_first_existing_column(
    feature_mapping_df,
    c(
        "ensembl_gene_id",
        "gene_id",
        "ensembl_id"
    ),
    "feature_mapping_df"
)

mapping_symbol_col <- find_first_existing_column(
    feature_mapping_df,
    c(
        "gene_symbol",
        "external_gene_name",
        "gene_name",
        "symbol"
    ),
    "feature_mapping_df"
)

message("\nDetected columns:")
message(
    paste(
        "sample_id_col:",
        sample_id_col
    )
)
message(
    paste(
        "target_col:",
        target_col
    )
)
message(
    paste(
        "mapping_feature_col:",
        mapping_feature_col
    )
)
message(
    paste(
        "mapping_ensembl_col:",
        mapping_ensembl_col
    )
)
message(
    paste(
        "mapping_symbol_col:",
        mapping_symbol_col
    )
)

# -----------------------------------------------------------------------------
# Prepare expression matrix
# -----------------------------------------------------------------------------

metadata_df <- transcriptomics_df %>%
    select(
        all_of(
            c(
                sample_id_col,
                target_col
            )
        )
    )

expression_samples_by_features <- transcriptomics_df %>%
    select(
        all_of(feature_cols)
    )

expression_samples_by_features <- as.data.frame(
    expression_samples_by_features
)

rownames(expression_samples_by_features) <- transcriptomics_df[[sample_id_col]]

# -----------------------------------------------------------------------------
# Prepare feature mapping
# -----------------------------------------------------------------------------

mapping_clean <- make_unique_non_missing_gene_symbols(
    mapping_df = feature_mapping_df,
    feature_col = mapping_feature_col,
    ensembl_col = mapping_ensembl_col,
    symbol_col = mapping_symbol_col
)

mapping_clean <- mapping_clean %>%
    filter(
        transcriptomics_feature %in% feature_cols
    )

if (nrow(mapping_clean) == 0) {
    stop(
        "No feature mapping rows matched transcriptomics features."
    )
}

features_with_symbols <- intersect(
    colnames(
        expression_samples_by_features
    ),
    mapping_clean$transcriptomics_feature
)

expression_samples_by_features <- (
    expression_samples_by_features[
        ,
        features_with_symbols,
        drop = FALSE
    ]
)

mapping_clean <- mapping_clean %>%
    filter(
        transcriptomics_feature %in% features_with_symbols
    )

message(
    "\nFeature mapping summary:"
)

message(
    paste(
        "Number of transcriptomics feature columns:",
        length(feature_cols)
    )
)

message(
    paste(
        "Number of features with gene symbols:",
        length(features_with_symbols)
    )
)

# -----------------------------------------------------------------------------
# Convert to genes x samples using gene symbols
# -----------------------------------------------------------------------------

expression_features_by_samples <- t(
    as.matrix(
        expression_samples_by_features
    )
)

feature_to_symbol <- mapping_clean$gene_symbol

names(
    feature_to_symbol
) <- mapping_clean$transcriptomics_feature

gene_symbols <- feature_to_symbol[
    rownames(
        expression_features_by_samples
    )
]

rownames(
    expression_features_by_samples
) <- gene_symbols

keep_genes <- (
    !is.na(
        rownames(
            expression_features_by_samples
        )
    )
    &
    rownames(
        expression_features_by_samples
    ) != ""
)

expression_features_by_samples <- (
    expression_features_by_samples[
        keep_genes,
        ,
        drop = FALSE
    ]
)

expression_gene_symbol_df <- (
    as.data.frame(
        expression_features_by_samples
    )
    %>%
    rownames_to_column(
        "gene_symbol"
    )
    %>%
    group_by(
        gene_symbol
    )
    %>%
    summarise(
        across(
            everything(),
            ~ mean(
                .x,
                na.rm = TRUE
            )
        ),
        .groups = "drop"
    )
)

expression_gene_symbol_by_sample <- (
    expression_gene_symbol_df
    %>%
    column_to_rownames(
        "gene_symbol"
    )
    %>%
    as.matrix()
)

gene_symbol_counts <- table(
    rownames(
        expression_features_by_samples
    )
)

duplicated_gene_symbol_summary <- tibble(
    gene_symbol = names(
        gene_symbol_counts
    ),
    n_features = as.integer(
        gene_symbol_counts
    )
) %>%
    arrange(
        desc(n_features)
    )

duplicated_gene_symbol_summary_path <- file.path(
    output_dir,
    "transcriptomics_gss_duplicated_gene_symbol_summary.csv"
)

write_csv(
    duplicated_gene_symbol_summary,
    duplicated_gene_symbol_summary_path
)

message(
    "\nFinal gene-symbol expression matrix dimensions:"
)

message(
    paste(
        dim(
            expression_gene_symbol_by_sample
        ),
        collapse = " x "
    )
)

# -----------------------------------------------------------------------------
# Load GenomicSuperSignature model
# -----------------------------------------------------------------------------

message(
    "\nLoading GenomicSuperSignature RAVmodel:"
)

message(
    rav_model_name
)

RAVmodel <- getModel(
    rav_model_name,
    load = TRUE
)

message(
    "RAVmodel loaded."
)

# -----------------------------------------------------------------------------
# Restrict to genes available in RAVmodel
# -----------------------------------------------------------------------------

rav_genes <- extract_rav_genes(
    RAVmodel
)

common_genes <- intersect(
    rownames(
        expression_gene_symbol_by_sample
    ),
    rav_genes
)

if (
    length(common_genes) < 1000
) {
    warning(
        paste0(
            "Only ",
            length(common_genes),
            " genes overlap with the RAVmodel."
        )
    )
}

expression_gss_input <- (
    expression_gene_symbol_by_sample[
        common_genes,
        ,
        drop = FALSE
    ]
)

gene_overlap_summary <- tibble(
    n_expression_genes_with_symbols = nrow(
        expression_gene_symbol_by_sample
    ),
    n_ravmodel_genes = length(
        rav_genes
    ),
    n_common_genes = length(
        common_genes
    ),
    fraction_expression_genes_in_ravmodel = (
        length(common_genes) /
        nrow(
            expression_gene_symbol_by_sample
        )
    ),
    fraction_ravmodel_genes_available = (
        length(common_genes) /
        length(rav_genes)
    )
)

gene_overlap_summary_path <- file.path(
    output_dir,
    "transcriptomics_gss_gene_overlap_summary.csv"
)

write_csv(
    gene_overlap_summary,
    gene_overlap_summary_path
)

message(
    "\nGene overlap summary:"
)

print(
    gene_overlap_summary
)

# -----------------------------------------------------------------------------
# Save GSS input matrix
# -----------------------------------------------------------------------------

gss_input_matrix_path <- file.path(
    output_dir,
    "transcriptomics_gss_input_matrix_gene_symbols_by_samples.csv"
)

gss_input_df <- (
    as.data.frame(
        expression_gss_input
    )
    %>%
    rownames_to_column(
        "gene_symbol"
    )
)

write_csv(
    gss_input_df,
    gss_input_matrix_path
)

# -----------------------------------------------------------------------------
# Validate dataset against RAVmodel
# -----------------------------------------------------------------------------

message("\nRunning validate()...")

validation_results <- validate(
    expression_gss_input,
    RAVmodel
)

validation_results_rds_path <- file.path(
    output_dir,
    "transcriptomics_genomic_super_signature_validation_results.rds"
)

validation_results_txt_path <- file.path(
    output_dir,
    "transcriptomics_genomic_super_signature_validation_results.txt"
)

validation_results_csv_path <- file.path(
    output_dir,
    "transcriptomics_genomic_super_signature_validation_results.csv"
)

# Save the complete native R object
saveRDS(
    validation_results,
    validation_results_rds_path
)

# Save a human-readable representation for inspection
capture.output(
    {
        print(validation_results)
        str(validation_results)
    },
    file = validation_results_txt_path
)

# Save only the scalar validation fields required for QC
required_validation_cols <- c(
    "score",
    "PC",
    "sw",
    "cl_size",
    "cl_num"
)

missing_validation_cols <- setdiff(
    required_validation_cols,
    colnames(validation_results)
)

if (length(missing_validation_cols) > 0) {
    stop(
        paste0(
            "validate() output is missing expected columns: ",
            paste(missing_validation_cols, collapse = ", ")
        )
    )
}

validation_results_csv_df <- data.frame(
    RAV = rownames(validation_results),
    score = as.numeric(validation_results[, "score"]),
    PC = as.integer(validation_results[, "PC"]),
    sw = as.numeric(validation_results[, "sw"]),
    cl_size = as.integer(validation_results[, "cl_size"]),
    cl_num = as.integer(validation_results[, "cl_num"]),
    stringsAsFactors = FALSE
)

if (nrow(validation_results_csv_df) != nrow(validation_results)) {
    stop(
        "Validation CSV row count does not match validate() output."
    )
}

write_csv(
    validation_results_csv_df,
    validation_results_csv_path
)

message("Validation results saved as RDS, TXT, and CSV.")
message(validation_results_rds_path)
message(validation_results_txt_path)
message(validation_results_csv_path)

message("Validation results preview:")
print(head(validation_results_csv_df))

# -----------------------------------------------------------------------------
# Select top validated RAVs for QC only
# -----------------------------------------------------------------------------

message(
    "\nSelecting top validated RAVs for QC..."
)

validated_top <- validatedSignatures(
    validation_results,
    RAVmodel,
    num.out = top_n_validation_report,
    scoreCutoff = 0,
    filterMessage = FALSE
)

validated_top_rds_path <- file.path(
    output_dir,
    "transcriptomics_genomic_super_signature_validated_ravs_top.rds"
)

validated_top_txt_path <- file.path(
    output_dir,
    "transcriptomics_genomic_super_signature_validated_ravs_top.txt"
)

validated_top_path <- file.path(
    output_dir,
    "transcriptomics_genomic_super_signature_validated_ravs_top.csv"
)

saveRDS(
    validated_top,
    validated_top_rds_path
)

capture.output(
    {
        print(
            validated_top
        )

        str(
            validated_top
        )
    },
    file = validated_top_txt_path
)

validated_top_simple_df <- tibble(
    RAV = rownames(
        as.data.frame(
            validated_top
        )
    )
)

write_csv(
    validated_top_simple_df,
    validated_top_path
)

selected_ravs <- normalise_rav_name(
    validated_top_simple_df$RAV
)

message(
    "Top validated RAVs for QC:"
)

print(
    selected_ravs
)

# -----------------------------------------------------------------------------
# Calculate sample-level RAV scores
# -----------------------------------------------------------------------------

message(
    "\nRunning calculateScore()..."
)

score_object <- calculateScore(
    expression_gss_input,
    RAVmodel,
    rescale.after = FALSE
)

if (is.list(score_object)) {
    scores_matrix <- score_object[[1]]
} else {
    scores_matrix <- score_object
}

scores_df <- (
    as.data.frame(
        scores_matrix
    )
    %>%
    rownames_to_column(
        sample_id_col
    )
)

score_cols <- setdiff(
    colnames(scores_df),
    sample_id_col
)

normalised_score_cols <- normalise_rav_name(
    score_cols
)

colnames(scores_df)[
    colnames(scores_df) %in% score_cols
] <- normalised_score_cols

score_cols <- setdiff(
    colnames(scores_df),
    sample_id_col
)

scores_df <- scores_df %>%
    rename_with(
        .cols = all_of(
            score_cols
        ),
        .fn = ~ paste0(
            "gss__",
            .x
        )
    )

scores_all_path <- file.path(
    output_dir,
    "transcriptomics_genomic_super_signature_scores_all.csv"
)

write_csv(
    scores_df,
    scores_all_path
)

message(
    "All GSS score matrix dimensions:"
)

message(
    paste(
        dim(scores_df),
        collapse = " x "
    )
)

# -----------------------------------------------------------------------------
# Save run summary
# -----------------------------------------------------------------------------

run_summary <- tibble(
    base_dir = base_dir,
    rav_model_name = rav_model_name,
    n_ravs_reported_for_validation = top_n_validation_report,
    n_samples = ncol(
        expression_gss_input
    ),
    n_input_genes = nrow(
        expression_gss_input
    ),
    n_available_gss_features = length(
        score_cols
    ),
    preprocessed_transcriptomics_path = (
        preprocessed_transcriptomics_path
    ),
    feature_mapping_path = (
        feature_mapping_path
    ),
    output_dir = output_dir
)

run_summary_path <- file.path(
    output_dir,
    "transcriptomics_genomic_super_signature_run_summary.csv"
)

write_csv(
    run_summary,
    run_summary_path
)

# -----------------------------------------------------------------------------
# Final messages
# -----------------------------------------------------------------------------

message(
    "\nSaved outputs:"
)

message(
    validation_results_rds_path
)

message(
    validation_results_txt_path
)

message(
    validation_results_csv_path
)

message(
    validated_top_path
)

message(
    scores_all_path
)

message(
    gene_overlap_summary_path
)

message(
    duplicated_gene_symbol_summary_path
)

message(
    gss_input_matrix_path
)

message(
    run_summary_path
)

message(
    "\nGenomicSuperSignature script completed successfully."
)
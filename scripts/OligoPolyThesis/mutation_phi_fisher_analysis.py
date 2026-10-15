"""
Mutation Phi/Fisher pairwise association analysis.

This script computes pairwise associations between binary mutation features in
the quality-controlled mutation dataset. For each pair of mutation features, it
builds a 2x2 contingency table, calculates the Phi coefficient, and applies
Fisher's exact test.

Because the number of possible feature pairs can be very large, the analysis is
processed in chunks and parallelized using joblib. To reduce output size, only
feature pairs with an absolute Phi coefficient above a predefined threshold are
saved.

The resulting chunk files can then be combined and used in the mutation
preprocessing workflow to identify highly correlated or redundant mutation
features before downstream machine learning modeling.
"""

from pathlib import Path
import sys
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from scripts.OligoPolyThesis.utils_ml_oligopoly import get_2x2_table, phi_coefficient_from_table, compute_pair, process_chunk, chunked_iterable

import os
import math
import pandas as pd
import numpy as np

from itertools import combinations, islice
from scipy.stats import fisher_exact
from joblib import Parallel, delayed

# =========================
# Configuração
# =========================
INPUT_PATH = "/mnt/data3/MCPP-Breast/Mutation-Data/mutation_qc_filtered.csv"
OUTPUT_DIR = "/mnt/data3/MCPP-Breast/Mutation-Data"
OUTPUT_PREFIX = "20260813-mutation_phi_fisher"

COMBINED_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"{OUTPUT_PREFIX}_results_combined.csv"
)

DELETE_CHUNK_FILES_AFTER_MERGE = True

N_JOBS = 2              # usar poucos CPUs para não ocupar o servidor todo
CHUNK_SIZE = 50000      # número de pares por chunk
PHI_SAVE_THRESHOLD = 0.5  # só guardar pares com |phi| >= este valor

# =========================
# Funções auxiliares
# =========================


# =========================
# Main
# =========================
def main():
    print("Loading filtered mutation dataset...")
    df = pd.read_csv(INPUT_PATH)

    feature_cols = [
        col for col in df.columns
        if col not in ["sample_id", "mutation_source", "n_mutated_genes"]
    ]

    print(f"Number of feature columns: {len(feature_cols)}")

    total_pairs = len(feature_cols) * (len(feature_cols) - 1) // 2
    print(f"Number of pairs to analyse: {total_pairs}")
    print(f"Running with n_jobs={N_JOBS}, chunk_size={CHUNK_SIZE}, phi_save_threshold={PHI_SAVE_THRESHOLD}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pairs_iter = combinations(feature_cols, 2)

    chunk_idx = 0
    saved_rows_total = 0
    saved_chunk_files = []

    for pairs_chunk in chunked_iterable(pairs_iter, CHUNK_SIZE):
        chunk_idx += 1
        print(f"Processing chunk {chunk_idx} with {len(pairs_chunk)} pairs...")

        # divide o chunk por N_JOBS subchunks
        subchunk_size = max(1, len(pairs_chunk) // N_JOBS)
        subchunks = [pairs_chunk[i:i + subchunk_size] for i in range(0, len(pairs_chunk), subchunk_size)]

        parallel_results = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(process_chunk)(df, subchunk, PHI_SAVE_THRESHOLD) for subchunk in subchunks
        )

        flat_results = [item for sublist in parallel_results for item in sublist]

        chunk_output = os.path.join(
            OUTPUT_DIR,
            f"{OUTPUT_PREFIX}_chunk_{chunk_idx:04d}.csv"
        )

        if flat_results:
            chunk_df = pd.DataFrame(flat_results)
            chunk_df.to_csv(chunk_output, index=False)
            saved_rows_total += len(chunk_df)
            saved_chunk_files.append(chunk_output)
            print(f"Saved chunk {chunk_idx} with {len(chunk_df)} rows to {chunk_output}")
        else:
            print(f"Chunk {chunk_idx} produced no rows above |phi| >= {PHI_SAVE_THRESHOLD}")

    print("Finished all chunks.")
    print(f"Total saved rows: {saved_rows_total}")

    if saved_chunk_files:
        print("Merging chunk files into a single combined results file...")

        combined_df = pd.concat(
            (pd.read_csv(chunk_file) for chunk_file in saved_chunk_files),
            ignore_index=True
        )

        combined_df.to_csv(COMBINED_OUTPUT_PATH, index=False)

        print("Combined Phi/Fisher results saved to:")
        print(COMBINED_OUTPUT_PATH)
        print(f"Combined dataframe shape: {combined_df.shape}")

        if DELETE_CHUNK_FILES_AFTER_MERGE:
                print("Deleting individual chunk files...")

                for chunk_file in saved_chunk_files:
                    os.remove(chunk_file)

                print("Individual chunk files deleted.")

        else:
            print("No chunk files were saved, so no combined file was created.")


if __name__ == "__main__":
    main()
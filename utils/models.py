import pandas as pd
import numpy as np
from utils import plot
from typing import List, Optional, Literal
from scipy.spatial.distance import cosine, euclidean, cityblock


# Finn ut hvordan du må gjøre det med når det er training og når det er test, om du må tenke på forskjellen mellom fit_transform og transform bl.a.
class SequenceSimilarity:
    def __init__(
        self,
        normalization_method: Literal[
            "z-score", "min-max", "robust", "mean", "none"
        ] = "z-score",
        similarity_method: Literal[
            "pearson", "cosine", "euclidean", "manhattan"
        ] = "pearson",
        aggregation_method: Literal[
            "mean", "mean-ignore-0", "median", "max", "min"
        ] = "mean",
    ):
        self.normalization_method = normalization_method
        self.similarity_method = similarity_method
        self.aggregation_method = aggregation_method
        self.sequence_cols: List[str] = []
        self.id_col: str = ""
        self.similarity_matrix: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        sequence_cols: list[str],
        id_col: str,
        n_unique_ids: int = 0,
    ) -> "SequenceSimilarity":
        self._validate_inputs(df, sequence_cols, id_col, n_unique_ids)
        self.sequence_cols = sequence_cols
        self.id_col = id_col

        prepared_df = self._prepare_data(df, n_unique_ids)
        col_similarity_matrices = self._compute_column_similarities(prepared_df)
        self.similarity_matrix = self._compute_similarity_matrix(
            col_similarity_matrices
        )
        self._is_fitted = True

        return self

    def get_similarity(self, entity_id: str) -> Optional[pd.Series]:
        self._check_if_fitted()

        if entity_id not in self.similarity_matrix.index:
            return None
        return self.similarity_matrix.loc[entity_id]

    def get_top_similar(
        self, entity_id: str, n: int = 5, exclude_self: bool = True
    ) -> Optional[pd.Series]:
        similarities = self.get_similarity(entity_id)
        if similarities is None:
            return None

        if exclude_self and entity_id in similarities.index:
            similarities = similarities.drop(entity_id)

        return similarities.nlargest(n)

    def get_n_largest_correlations_dict(
        self, n: int = 5, exclude_self: bool = True
    ) -> dict[str, pd.Series]:
        self._check_if_fitted()
        top_corrs = {}
        for col in self.similarity_matrix.columns:
            series = self.similarity_matrix[col].copy()

            if exclude_self and col in series.index:
                series = series.drop(col, errors="ignore")

            top_corrs[col] = series.reindex(
                series.abs().sort_values(ascending=False).index
            )[:n]

        return top_corrs

    def get_capped_largest_correlations_dict(
        self, cap_value: float = 0.5, exclude_self: bool = True
    ):
        self._check_if_fitted()
        capped_corrs = {}
        for col in self.similarity_matrix.columns:
            series = self.similarity_matrix[col].copy()

            if exclude_self and col in series.index:
                series = series.drop(col, errors="ignore")

            filtered = series[series.abs() >= cap_value]
            filtered = filtered.reindex(
                filtered.abs().sort_values(ascending=False).index
            )
            if not filtered.empty:
                capped_corrs[col] = filtered
        return capped_corrs

    def plot_similarities(
        self, title: str = "Sequence similarities", drop_zero: bool = True
    ) -> None:
        self._check_if_fitted()
        plot.heatmap_plot(self.similarity_matrix, title=title, drop_0=drop_zero)

    def get_processed_ids(self):
        self._check_if_fitted()
        return self.similarity_matrix.index

    def _validate_inputs(
        self, df: pd.DataFrame, sequence_cols: List[str], id_col: str, n_unique_ids: int
    ) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        if not sequence_cols:
            raise ValueError("sequence_cols must contain at least one column")

        if id_col not in df.columns:
            raise KeyError(f"id_col '{id_col}' not found in DataFrame")

        if n_unique_ids < 0:
            raise ValueError("n_unique_ids must be non-negative")

    def _check_if_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "This SequenceSimilarity instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

    def _prepare_data(self, df: pd.DataFrame, n_unique_ids: int) -> pd.DataFrame:
        df_copy = df.copy()
        if n_unique_ids > 0:
            unique_ids = df_copy[self.id_col].drop_duplicates()
            n_samples = min(n_unique_ids, len(unique_ids))
            random_ids = unique_ids.sample(n=n_samples, random_state=41)
            df_copy = df_copy[df_copy[self.id_col].isin(random_ids)]
        return df_copy

    def _normalize_sequence(self, sequence: np.ndarray):
        method = self.normalization_method

        if method == "z-score":
            mean, std = sequence.mean(), sequence.std()
            return (
                np.zeros_like(sequence, dtype=float)
                if (std == 0 or pd.isna(std))
                else (sequence - mean) / std
            )

        elif method == "min-max":
            min_val, max_val = np.min(sequence), np.max(sequence)
            return (
                np.zeros_like(sequence, dtype=float)
                if max_val == min_val
                else (sequence - min_val) / (max_val - min_val)
            )

        elif method == "robust":
            median = np.median(sequence)
            iqr = np.percentile(sequence, 75) - np.percentile(sequence, 25)
            return (
                np.zeros_like(sequence, dtype=float)
                if iqr == 0
                else (sequence - median) / iqr
            )

        elif method == "mean":
            mean = np.mean(sequence)
            return np.zeros_like(sequence) if mean == 0 else sequence / mean

        elif method == "none":
            return sequence

        else:
            raise ValueError(
                f"Unknown normalization method: {self.normalization_method}"
            )

    def _compute_column_similarities(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        col_similarity_matrices = []

        for col in self.sequence_cols:
            normalized_sequences = []
            for entity_id in df[self.id_col].unique():
                entity_df = df[df[self.id_col] == entity_id]
                sequence = entity_df[col].values
                normalized_sequence = self._normalize_sequence(sequence)
                normalized_sequences.append((entity_id, normalized_sequence))
            similarity_matrix = self._compute_pairwise_similarity(normalized_sequences)
            col_similarity_matrices.append(similarity_matrix)
        return col_similarity_matrices

    def _compute_pairwise_similarity(self, sequences: List[tuple]) -> pd.DataFrame:
        n = len(sequences)
        sim_matrix = np.zeros((n, n))
        ids = [entity_id for entity_id, _ in sequences]

        for i in range(n):
            for j in range(i, n):
                seq_i = sequences[i][1]
                seq_j = sequences[j][1]
                sim_score = self._calculate_sequence_similarity(seq_i, seq_j)
                sim_matrix[i, j] = sim_score
                sim_matrix[j, i] = sim_score

        return pd.DataFrame(sim_matrix, index=ids, columns=ids)

    def _calculate_sequence_similarity(
        self, seq_i: np.ndarray, seq_j: np.ndarray
    ) -> float:
        method = self.similarity_method
        max_length = min(len(seq_i), len(seq_j))
        seq_i, seq_j = seq_i[:max_length], seq_j[:max_length]

        if max_length == 0:
            return 0.0

        if method == "pearson":
            if np.std(seq_i) == 0 or np.std(seq_j) == 0:
                return 0.0
            pearson_corr = np.corrcoef(seq_i, seq_j)[0, 1]
            if np.isnan(pearson_corr):
                return 0.0
            return float(pearson_corr)
        elif method == "cosine":
            sim = 1 - cosine(seq_i, seq_j)
            return 0.0 if np.isnan(sim) else sim
        elif method == "euclidean":
            dist = euclidean(seq_i, seq_j)
            return 1 / (1 + dist)
        elif method == "manhattan":
            dist = cityblock(seq_i, seq_j)
            return 1 / (1 + dist)
        else:
            raise ValueError(f"Unknown similarity method: {self.similarity_method}")

    def _compute_similarity_matrix(self, similarity_matrices: List[pd.DataFrame]):
        method = self.aggregation_method
        if method == "mean":
            return sum(similarity_matrices) / len(similarity_matrices)

        elif method == "mean-ignore-0":
            stacked = pd.concat([m.stack() for m in similarity_matrices], axis=1)
            stacked.replace(0.0, np.nan, inplace=True)
            mean_series = stacked.mean(axis=1, skipna=True)
            mean_df = mean_series.unstack()
            return mean_df.fillna(0)

        elif method == "median":
            return (
                pd.concat([m.stack() for m in similarity_matrices], axis=1)
                .median(axis=1)
                .unstack()
            )

        elif method == "max":
            return (
                pd.concat([m.stack() for m in similarity_matrices], axis=1)
                .max(axis=1)
                .unstack()
            )

        elif method == "min":
            return (
                pd.concat([m.stack() for m in similarity_matrices], axis=1)
                .min(axis=1)
                .unstack()
            )

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

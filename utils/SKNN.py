import numpy as np
from typing import List
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Tuple
from scipy.spatial.distance import cosine


Array2D = np.ndarray


class SequenceKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        k: int = 5,
        threshold_length: int = 24,
        threshold_value: float = 0.05,
        verbose: bool = True,
    ):
        self.k = k
        self.threshold_length = threshold_length
        self.threshold_value = threshold_value
        self.tau = -threshold_length / np.log(threshold_value)
        self.verbose = verbose

        self.sequences_: List[Array2D] = []
        self.y_: np.ndarray = np.array([])

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray, id_col: str
    ) -> "SequenceKNeighborsClassifier":
        self.id_col = id_col
        groups = X.groupby(id_col)
        self.sequences_ = [g.drop(columns=[id_col]).to_numpy()[::-1] for _, g in groups]
        ids: List[str] = list(groups.groups.keys())

        if isinstance(y, np.ndarray):
            y_series = pd.Series(y, index=X.index)
        else:
            y_series = y

        self.y_ = np.array([int(y_series.loc[id]) for id in ids])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        groups = X.groupby(self.id_col)
        sequences: List[Array2D] = [
            g.drop(columns=[self.id_col]).to_numpy()[::-1] for _, g in groups
        ]

        prob_list: List[List[float]] = []
        total = len(sequences)

        for idx_seq, seq in enumerate(sequences, start=1):
            similarities = np.array(
                [self._similarity(seq, train_seq) for train_seq in self.sequences_]
            )
            idx = np.argsort(similarities)[::-1][: self.k]
            p1 = float(np.mean(self.y_[idx]))
            prob_list.append([1.0 - p1, p1])
            if self.verbose:
                print(f"Processed {idx_seq}/{total} sequences", end="\r")

        return np.array(prob_list)

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        prob = self.predict_proba(X)
        return (prob[:, 1] >= 0.5).astype(int)

    def _exponential_decay(
        self, seq1_len: int, seq2_len: int
    ) -> Tuple[np.ndarray, float]:
        min_len, max_len = min(seq1_len, seq2_len), max(seq1_len, seq2_len)
        y_values = np.exp(-np.arange(max_len) / self.tau)
        y_normalized = y_values / y_values.sum()
        return y_normalized[:min_len], y_normalized[min_len:].sum()

    def _similarity(self, seq1: Array2D, seq2: Array2D) -> float:
        weights, _ = self._exponential_decay(len(seq1), len(seq2))
        s: float = 0.0
        for i, w in enumerate(weights):
            v1 = seq1[i]
            v2 = seq2[i]

            cos_sim = 1 - cosine(v1, v2)
            s += w * cos_sim
        return s

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import json

classifiers = {
    # Tree-baserte modeller
    "rf": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "gb": lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "ada": lambda: AdaBoostClassifier(n_estimators=100, random_state=42),
    "dt": lambda: DecisionTreeClassifier(random_state=42),
    # Modeller som bør scales
    "lr": lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    ),
    "svc": lambda: Pipeline(
        [("scaler", StandardScaler()), ("clf", SVC(random_state=42))]
    ),
    "knn": lambda: Pipeline(
        [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]
    ),
}


def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find: {file_path}")

    with open(file_path, "r") as f:
        return json.load(f)


def build_model_from_config(
    step_type: str, model_code: str, file_path="results/BestParams.json"
):
    config = load_json(file_path)

    # Get params from json-dict
    if model_code == "regressor":
        params = config.get("regressor", {})
    else:
        params = config[step_type][model_code]
    params = {k: v for k, v in params.items() if k != "cv_score"}

    # Build base model
    if model_code not in classifiers and model_code != "regressor":
        raise ValueError(f"Unknown model_code: {model_code}")

    if model_code == "regressor":
        return GradientBoostingRegressor(**params)
    model = classifiers[model_code]()
    model.set_params(**params)
    return model


def get_metrics(y_pred: np.ndarray, y_test: np.ndarray) -> dict:
    within_01 = np.abs(y_test - y_pred) <= 0.01
    accuracy_within_01 = np.mean(within_01)

    within_10 = np.abs(y_test - y_pred) <= 0.1
    accuracy_within_10 = np.mean(within_10)

    full_miss = np.abs(y_test - y_pred) == 1
    full_miss_share = np.mean(full_miss)

    preds_at_edges = np.mean((y_pred == 0) | (y_pred == 1))
    truth_at_edges = np.mean((y_test == 0) | (y_test == 1))
    at_edge_diff = abs(preds_at_edges - truth_at_edges)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    test_mean = np.mean(y_test)
    pred_mean = np.mean(y_pred)
    mean_diff = abs(test_mean - pred_mean)

    return {
        "Accuracy within 1%": accuracy_within_01,
        "Accuracy within 10%": accuracy_within_10,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Test mean": test_mean,
        "Pred mean": pred_mean,
        "Mean difference": mean_diff,
        "Full miss": full_miss_share,
        "Predictions at edge": preds_at_edges,
        "Truth at edge": truth_at_edges,
        "At edge difference": at_edge_diff,
    }


def print_metrics(y_pred: np.ndarray, y_test: np.ndarray) -> None:
    metrics = get_metrics(y_pred, y_test)

    print("=" * 50)
    print("MODEL METRICS")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:30s} {value:.4f}")
        else:
            print(f"{key:30s} {value}")
    print("=" * 50)


def get_bin_frequencies(
    y_pred: np.ndarray, y_test: np.ndarray, bin_size: float = 0.1
) -> list:
    bin_analysis = []

    # Exactly 0
    true_mask = y_test == 0
    pred_mask = y_pred == 0
    true_count = np.sum(true_mask)
    pred_count = np.sum(pred_mask)
    bin_analysis.append(
        {
            "range": "0",
            "true_count": int(true_count),
            "pred_count": int(pred_count),
            "true_mean": 0.0,
            "pred_mean": 0.0,
        }
    )

    bin_edges = np.arange(0.01, 1.0, bin_size)
    for i in range(len(bin_edges)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i] + bin_size - 0.01 if i < len(bin_edges) - 1 else 0.99

        true_mask = (y_test >= bin_start) & (y_test <= bin_end)
        pred_mask = (y_pred >= bin_start) & (y_pred <= bin_end)

        true_count = np.sum(true_mask)
        pred_count = np.sum(pred_mask)
        true_mean = np.mean(y_test[true_mask]) if true_count > 0 else 0
        pred_mean = np.mean(y_pred[pred_mask]) if pred_count > 0 else 0

        bin_analysis.append(
            {
                "range": f"[{bin_start:.2f}, {bin_end:.2f}]",
                "true_count": int(true_count),
                "pred_count": int(pred_count),
                "true_mean": float(true_mean),
                "pred_mean": float(pred_mean),
            }
        )

    # Exactly 1
    true_mask = y_test == 1
    pred_mask = y_pred == 1
    true_count = np.sum(true_mask)
    pred_count = np.sum(pred_mask)
    bin_analysis.append(
        {
            "range": "1",
            "true_count": int(true_count),
            "pred_count": int(pred_count),
            "true_mean": 1.0,
            "pred_mean": 1.0,
        }
    )

    return bin_analysis


def print_bin_frequencies(
    y_pred: np.ndarray, y_test: np.ndarray, bin_size: float = 0.1
) -> None:
    """
    Print bin frequency analysis in a clean tabular format.

    Args:
        y_pred: Predicted values
        y_test: True values
        bin_size: Size of each bin (default 0.1)
    """
    bin_analysis = get_bin_frequencies(y_pred, y_test, bin_size)

    print("=" * 80)
    print("BIN FREQUENCY ANALYSIS")
    print("=" * 80)
    print(
        f"{'Range':15s} {'True Count':>12s} {'Pred Count':>12s} {'True Mean':>12s} {'Pred Mean':>12s}"
    )
    print("-" * 80)

    for bin_info in bin_analysis:
        print(
            f"{bin_info['range']:15s} "
            f"{bin_info['true_count']:12d} "
            f"{bin_info['pred_count']:12d} "
            f"{bin_info['true_mean']:12.4f} "
            f"{bin_info['pred_mean']:12.4f}"
        )

    print("=" * 80)


def plot_bins(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    bins: int = 100,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
    figsize: tuple = (12, 6),
) -> None:
    # Determine x-axis range
    if x_min is None:
        x_min = min(y_test.min(), y_pred.min())
    if x_max is None:
        x_max = max(y_test.max(), y_pred.max())

    bins_array = np.linspace(x_min, x_max, bins)
    true_counts, _ = np.histogram(y_test, bins=bins_array)
    pred_counts, _ = np.histogram(y_pred, bins=bins_array)
    bin_width = bins_array[1] - bins_array[0]
    bar_width = bin_width * 0.45
    bin_centers = (bins_array[:-1] + bins_array[1:]) / 2

    plt.figure(figsize=figsize)
    plt.bar(bin_centers - bar_width / 2, true_counts, width=bar_width, label="True LGD")
    plt.bar(
        bin_centers + bar_width / 2, pred_counts, width=bar_width, label="Predicted LGD"
    )
    plt.xlabel("LGD")
    plt.ylabel("Frequency")
    plt.title("True LGD vs Predicted LGD")
    plt.xlim(x_min, x_max)
    if y_min is not None or y_max is not None:
        plt.ylim(y_min, y_max)

    plt.legend()
    plt.grid(True)
    plt.show()


import math
import matplotlib.pyplot as plt


def plot_top_n_by_columns(
    df: pd.DataFrame,
    col_tuples,
    n: int = 10,
    cols_per_row: int = 3,
):
    label_cols = ["Classifier equal 0", "Classifier equal 1", "Tie breaker"]
    num_plots = len(col_tuples)

    # Compute subplot grid
    rows = math.ceil(num_plots / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(6 * cols_per_row, 5 * rows))

    # Flatten axes to 1D list
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (col, ascending) in enumerate(col_tuples):
        ax = axes[idx]

        # --- Data Prep ---
        sorted_df = df.sort_values(col, ascending=ascending)
        top = sorted_df.drop_duplicates(subset=[col] + label_cols).head(n)

        labels = top[label_cols].apply(
            lambda row: f"{row[label_cols[0]]}, {row[label_cols[1]]}, {row[label_cols[2]]}",
            axis=1,
        )
        values = top[col]

        # Reverse order for better readable barh
        labels = labels[::-1]
        values = values[::-1]

        vmin = top[col].min()
        vmax = top[col].max()
        margin = (vmax - vmin) * 0.05

        # --- Plot ---
        ax.barh(labels, values)
        ax.set_xlabel(col)
        limit = "Bottom" if ascending else "Top"
        order = "ascending" if ascending else "descending"
        number = min(n, len(top))
        ax.set_title(f"{limit} {number} by '{col}' ({order})")
        ax.set_xlim(vmin - margin, vmax + margin)

    # Hide empty axes if grid not full
    for empty_ax in axes[num_plots:]:
        empty_ax.set_visible(False)

    plt.tight_layout()
    plt.show()

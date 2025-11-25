from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


def plot_top_n_by_columns(
    df: pd.DataFrame,
    col_tuples,
    n: int = 10,
    label_cols=["Classifier equal 0", "Classifier equal 1", "Tie breaker"],
):
    for col, ascending in col_tuples:
        sorted_df = df.sort_values(col, ascending=ascending)
        top = sorted_df.drop_duplicates(subset=col).head(n)

        labels = top[label_cols].apply(
            lambda row: f"{row[label_cols[0]]}, "
            f"{row[label_cols[1]]}, "
            f"{row[label_cols[2]]}",
            axis=1,
        )
        values = top[col]

        labels = labels[::-1]
        values = values[::-1]

        vmin = top[col].min()
        vmax = top[col].max()

        # Plot
        plt.figure(figsize=(12, 6))
        plt.barh(labels, values)
        plt.xlabel(col)
        plt.ylabel("Classifiers")
        plt.title(f"Top {n} unique values by '{col}' (ascending={ascending})")

        margin = (vmax - vmin) * 0.05
        plt.xlim(vmin - margin, vmax + margin)

        plt.tight_layout()
        plt.show()

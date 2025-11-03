import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix as conf_matrix
from sklearn.metrics import roc_curve, auc
import math

figsize = (14, 4)


def pairplot(df, columns=[], figsize=figsize):
    if len(columns) < 1:
        return
    vis_frame = df[columns]
    plt.figure(figsize=figsize)
    sns.pairplot(vis_frame)
    plt.show()


def heatmap_plot(df, columns=[], figsize=figsize, drop_0=False, title="Heatmap"):
    if len(columns) < 1:
        vis_frame = df
    else:
        vis_frame = df[columns]
    if drop_0:
        vis_frame = vis_frame.loc[:, (vis_frame != 0).any()]
        vis_frame = vis_frame.loc[(vis_frame != 0).any(axis=1), :]
    plt.figure(figsize=figsize)
    sns.heatmap(vis_frame, annot=True, cmap="coolwarm", center=0)
    plt.title(title)
    plt.show()


def correlation_matrix(df: pd.DataFrame, columns=[], figsize=figsize):
    if len(columns) < 1:
        vis_frame = df
    else:
        vis_frame = df[columns]
    corr = vis_frame.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.show()


def multi_feature_histogram(
    df: pd.DataFrame, columns=[], figsize=figsize, bins=20, column_width=3
):
    if len(columns) < 1:
        return
    vis_frame = df[columns]
    layout = (math.ceil(len(columns) / column_width), column_width)
    vis_frame.hist(bins, figsize=figsize, layout=layout)
    plt.tight_layout()
    plt.show()


def confusion_matrix(
    x_dir,
    y_dir,
    x_feature_title="Predicted",
    y_feature_title="Actual",
    title="Confusion matrix",
):
    cm = conf_matrix(y_dir, x_dir)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"{x_feature_title} 0", f"{x_feature_title} 1"],
        yticklabels=[f"{y_feature_title} 0", f"{y_feature_title} 1"],
    )
    plt.xlabel(x_feature_title)
    plt.ylabel(y_feature_title)
    plt.title(title)
    plt.show()


def roc_curve(y_true, probs, figsize=figsize):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(figsize))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.show()


def multi_roc_curve(y_true, probs_list=[], probs_labels=[], figsize=figsize):
    if len(probs_list) < 1 or len(probs_list) != len(probs_labels):
        return
    plt.figure(figsize=(figsize))

    for i in range(len(probs_list)):
        probs = probs_list[i]
        label = probs_labels[i]
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def linear_pred_plot(y_true, preds, figsize=figsize):
    plt.figure(figsize=figsize)
    plt.scatter(y_true, preds, alpha=0.6, edgecolors="k")
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    plt.set_xlabel("Actual values", fontsize=11)
    plt.set_ylabel("Predicted values", fontsize=11)
    plt.set_title("Predictions vs. Actual", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def residual_plot(y_true, preds, figsize=figsize):
    residuals = y_true - preds
    plt.figure(figsize=figsize)
    plt.scatter(preds, residuals, alpha=0.6, edgecolors="k")
    plt.axhline(y=0, color="r", linestyle="--", lw=2)
    plt.set_xlabel("Predicted values", fontsize=11)
    plt.set_ylabel("Residuals", fontsize=11)
    plt.set_title("Residual plot", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.show()


def multi_linear_step_plot(dfs, value_col: str, id_col: str):
    plt.figure(figsize=figsize)
    for df in dfs:
        plt.plot(
            df.index, df[value_col], marker="o", label=f"{id_col}: {df[id_col][0]}"
        )
    plt.xlabel("Step")
    plt.ylabel(value_col)
    plt.title(f"{value_col} per {id_col}")
    plt.legend()
    plt.grid(True)
    plt.show()

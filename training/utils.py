import numpy as np
import matplotlib.pyplot as plt


def plot_model_comparison(metrics: dict, title):
    metrics_to_plot = [
        "auc",
        "ndcg@10",
        "ndcg@20",
        "precision@10",
        "precision@20",
        "recall@10",
        "recall@20"
    ]

    labels = [
        "AUC",
        "NDCG@10",
        "NDCG@20",
        "Precision@10",
        "Precision@20",
        "Recall@10",
        "Recall@20"
    ]

    models = list(metrics.keys())
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 6))
    colours = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    bars_list = []
    for i, model_name in enumerate(models):
        values = [metrics[model_name][metric] for metric in metrics_to_plot]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model_name,
            color=colours[i % len(colours)],
            alpha=0.9
        )
        bars_list.append(bars)

    # Add values above bars
    for bars in bars_list:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.008,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333"
            )

    ax.set_title(
        title,
        fontsize=17,
        fontweight="bold",
        pad=18,
        alpha=0.8
    )
    ax.set_ylabel("Validation Score", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.3,
    )
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1)
    ax.legend(
        frameon=False,
        ncol=len(models),
        loc="upper right",
        fontsize=13,
        bbox_to_anchor=(1, 0.95)
    )

    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    has_val_loss = bool(history.get("val_loss"))
    has_auc = bool(history.get("auc"))
    has_ndcg = bool(history.get("ndcg@10"))
    n_plots = 1 + has_auc + has_ndcg

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]
    plot_idx = 0

    ax = axes[plot_idx]
    ax.plot(epochs, history["train_loss"], marker="o", label="Train")

    if has_val_loss:
        ax.plot(epochs, history["val_loss"], marker="o", label="Validation")

    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    plot_idx += 1

    if has_auc:
        ax = axes[plot_idx]
        ax.plot(epochs, history["auc"], marker="o", color="tab:green")
        ax.set_title("Validation AUC")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.grid(alpha=0.3)

        plot_idx += 1

    if has_ndcg:
        ax = axes[plot_idx]
        ax.plot(epochs, history["ndcg@10"], marker="o", label="NDCG@10")

        if "ndcg@20" in history and history["ndcg@20"]:
            ax.plot(epochs, history["ndcg@20"], marker="o", label="NDCG@20")

        ax.set_title("Ranking Quality")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NDCG")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

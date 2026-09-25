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
    width = 0.34
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AUC
    axes[1].plot(epochs, history["auc"], marker="o", color="tab:green")
    axes[1].set_title("Validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].grid(alpha=0.3)

    # NDCG
    axes[2].plot(epochs, history["ndcg@10"], marker="o", label="NDCG@10")
    axes[2].plot(epochs, history["ndcg@20"], marker="o", label="NDCG@20")
    axes[2].set_title("Ranking quality")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("NDCG")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


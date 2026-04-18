from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def save_sentiment_distribution(sentiment_summary: pd.DataFrame, brand_summary: pd.DataFrame, path: Path) -> None:
    brand_order = brand_summary["brand"].tolist()
    pivot = (
        sentiment_summary.pivot(index="brand", columns="sentiment_label", values="pct_reviews")
        .fillna(0)
        .reindex(brand_order)
    )
    means = brand_summary.set_index("brand").reindex(brand_order)["mean_compound"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    pivot[["positive", "neutral", "negative"]].plot(
        kind="bar",
        stacked=True,
        ax=axes[0],
        color=["#2ecc71", "#95a5a6", "#e74c3c"],
    )
    axes[0].set_title("Sentiment Breakdown (%)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Percentage of Reviews")
    axes[0].legend(title="Sentiment")
    axes[0].tick_params(axis="x", rotation=25)

    mean_colors = ["#d90429", "#2563eb", "#63b3de", "#7b3294"]
    means.plot(kind="bar", ax=axes[1], color=mean_colors)
    axes[1].set_title("Mean Compound Sentiment Score")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Compound Score (-1 to +1)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylim(means.min() - 0.08, 0.03)
    for patch, value in zip(axes[1].patches, means.values):
        y_pos = value / 2 if value < 0 else value + 0.01
        axes[1].annotate(
            f"{value:.3f}",
            (patch.get_x() + patch.get_width() / 2, y_pos),
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": "black", "edgecolor": "white", "alpha": 0.18, "pad": 0.2},
        )

    fig.suptitle("Figure 1 - Sentiment Distribution by Brand", y=1.02, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_sentiment_vs_rating(df: pd.DataFrame, path: Path) -> None:
    brands = [brand for brand in df["brand"].dropna().unique().tolist()]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=False, sharey=True)
    colors = ["#d90429", "#111111", "#63b3de", "#7b3294"]
    for ax, brand, color in zip(axes.flat, brands, colors):
        subset = df.loc[df["brand"] == brand].copy()
        sns.regplot(
            data=subset,
            x="compound",
            y="reviewer_rating",
            scatter_kws={"alpha": 0.45, "s": 18, "color": color},
            line_kws={"color": "black"},
            ax=ax,
        )
        corr = subset["compound"].corr(subset["reviewer_rating"])
        ax.set_title(f"{brand} (r = {corr:.3f})")
        ax.set_xlabel("Compound Sentiment Score")
        ax.set_ylabel("Star Rating")
        ax.set_yticks([1, 2, 3])
    fig.suptitle("Figure 2 - Sentiment Score vs Star Rating by Brand", y=1.02, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_topic_heatmap(topic_terms: pd.DataFrame, path: Path) -> None:
    label_col = "publication_label" if "publication_label" in topic_terms.columns else "topic_label"
    heatmap_data = topic_terms.groupby(["term", label_col], as_index=False)["rank"].min()
    heatmap_data = heatmap_data.pivot(index="term", columns=label_col, values="rank")
    heatmap_data = heatmap_data.sort_index()
    fig, ax = plt.subplots(figsize=(18, 12))
    sns.heatmap(heatmap_data, cmap="YlOrRd_r", ax=ax, cbar_kws={"label": "Rank"})
    ax.set_title("Figure 3 - LDA Topic-Word Heatmap", fontsize=20, fontweight="bold", pad=16)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Term")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_topic_distribution(topic_distribution: pd.DataFrame, path: Path) -> None:
    label_col = "publication_label" if "publication_label" in topic_distribution.columns else "dominant_topic_label"
    brands = [brand for brand in topic_distribution["brand"].dropna().unique().tolist()]
    fig, axes = plt.subplots(len(brands), 1, figsize=(16, 4.5 * len(brands)), sharex=False)
    if len(brands) == 1:
        axes = [axes]
    palette = ["#0b7285", "#f08c00", "#c2255c", "#6741d9", "#2b8a3e", "#495057"]
    for ax, brand in zip(axes, brands):
        subset = topic_distribution.loc[topic_distribution["brand"] == brand].sort_values("pct_reviews", ascending=True)
        total_n = int(subset["n_reviews"].sum())
        wrapped_labels = [textwrap.fill(str(label), width=28) for label in subset[label_col]]
        colors = palette[: len(subset)]
        ax.barh(wrapped_labels, subset["pct_reviews"], color=colors)
        ax.set_title(f"{brand} (n = {total_n:,})", fontweight="bold", fontsize=15)
        ax.set_xlabel("Percent of Reviews")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=11, pad=10)
        ax.set_xlim(0, max(subset["pct_reviews"].max() + 10, 35))
        for i, value in enumerate(subset["pct_reviews"]):
            ax.text(value + 0.8, i, f"{value:.1f}%", va="center", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.suptitle("Figure 4 - LDA Topic Distribution per Brand", y=0.995, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_aspect_distribution(aspect_distribution: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.barplot(data=aspect_distribution, x="brand", y="pct_reviews", hue="aspect_top_category", ax=ax)
    ax.set_title("Figure 5 - Dominant Aspect Distribution by Brand", fontsize=20, fontweight="bold", pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("Percentage of Reviews")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Dominant Aspect", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_monthly_sentiment(monthly: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.lineplot(data=monthly, x="review_month", y="mean_compound", hue="brand", marker="o", ax=ax)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_title("Figure 6 - Monthly Mean Sentiment Trend by Brand", fontsize=20, fontweight="bold", pad=16)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Compound Score")
    ax.legend(title="Brand")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_emotion_prevalence(emotion_prevalence: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.barplot(data=emotion_prevalence, x="brand", y="pct_reviews", hue="emotion", ax=ax)
    ax.set_title("Figure 7 - Emotion Marker Prevalence by Brand", fontsize=20, fontweight="bold", pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("Percentage of Reviews")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_coherence_comparison(coherence_df: pd.DataFrame, path: Path) -> None:
    plot_df = coherence_df.loc[coherence_df["topic_id"] != "overall"].copy()
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.barplot(data=plot_df, x="topic_label", y="coherence_cv", hue="model", ax=ax)
    ax.set_title("Figure 8 - Topic Coherence Comparison", fontsize=20, fontweight="bold", pad=16)
    ax.set_xlabel("Topic")
    ax.set_ylabel("c_v Coherence")
    ax.tick_params(axis="x", rotation=40)
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

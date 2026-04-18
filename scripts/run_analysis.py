from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fast_fashion_analysis.analysis import (  # noqa: E402
    apply_publication_labels,
    aspect_prevalence,
    bertopic_distribution_by_brand,
    build_bertopic_construct_mapping_table,
    build_construct_mapping_table,
    build_lda_topics,
    build_bertopic_topics,
    build_study2_alignment_table,
    compute_coherence_scores,
    describe_sample,
    emotion_prevalence,
    infer_bertopic_publication_label,
    monthly_sentiment,
    score_dictionary_categories,
    score_sentiment,
    top_terms_by_brand,
    topic_distribution_by_brand,
)
from fast_fashion_analysis.io import ensure_directories, load_config, load_reviews, save_table, save_workbook  # noqa: E402
from fast_fashion_analysis.preprocess import prepare_reviews  # noqa: E402
from fast_fashion_analysis.visuals import (  # noqa: E402
    save_aspect_distribution,
    save_coherence_comparison,
    save_emotion_prevalence,
    save_monthly_sentiment,
    save_sentiment_distribution,
    save_sentiment_vs_rating,
    save_topic_distribution,
    save_topic_heatmap,
    set_theme,
)

PUBLIC_COLUMNS = [
    "brand",
    "address",
    "reviewer_country",
    "reviewer_published_date",
    "review_title",
    "review_text",
    "reviewer_rating",
    "compound",
    "pos",
    "neg",
    "neu",
    "sentiment_label",
    "aspect_top_category",
    "emotion_top_category",
    "dominant_topic_id",
    "dominant_topic_label",
    "dominant_topic_publication_label",
    "bertopic_topic_id",
    "bertopic_topic_label",
    "bertopic_topic_publication_label",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast fashion review analysis.")
    parser.add_argument("--config", default="config/default.yml", help="Path to config file.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    ensure_directories(PROJECT_ROOT)
    set_theme()

    raw_reviews = load_reviews(config)
    print("Loaded raw reviews.")
    clean_reviews = prepare_reviews(raw_reviews, config)
    print("Prepared filtered review sample.")
    scored_reviews = score_sentiment(clean_reviews, config)
    scored_reviews = score_dictionary_categories(scored_reviews, config["aspects"], prefix="aspect")
    scored_reviews = score_dictionary_categories(scored_reviews, config["emotions"], prefix="emotion")
    scored_reviews, topic_terms, _ = build_lda_topics(scored_reviews, config)
    scored_reviews, bertopic_terms, bertopic_info = build_bertopic_topics(scored_reviews, config)
    scored_reviews = apply_publication_labels(
        scored_reviews,
        id_column="dominant_topic_id",
        raw_label_column="dominant_topic_label",
        label_map=config["publication_labels"]["lda"],
        output_column="dominant_topic_publication_label",
    )
    scored_reviews = apply_publication_labels(
        scored_reviews,
        id_column="bertopic_topic_id",
        raw_label_column="bertopic_topic_label",
        label_map={},
        output_column="bertopic_topic_publication_label",
    )
    scored_reviews["bertopic_topic_publication_label"] = scored_reviews["bertopic_topic_label"].map(infer_bertopic_publication_label)
    topic_terms = apply_publication_labels(
        topic_terms,
        id_column="topic_id",
        raw_label_column="topic_label",
        label_map=config["publication_labels"]["lda"],
        output_column="publication_label",
    )
    bertopic_terms = apply_publication_labels(
        bertopic_terms,
        id_column="topic_id",
        raw_label_column="topic_label",
        label_map={},
        output_column="publication_label",
    )
    bertopic_terms["publication_label"] = bertopic_terms["topic_label"].map(infer_bertopic_publication_label)
    bertopic_info = apply_publication_labels(
        bertopic_info,
        id_column="bertopic_topic_id",
        raw_label_column="Name",
        label_map={},
        output_column="publication_label",
    )
    bertopic_info["publication_label"] = bertopic_info["Name"].map(infer_bertopic_publication_label)
    lda_coherence = compute_coherence_scores(scored_reviews, topic_terms, model_name="LDA")
    bertopic_coherence = compute_coherence_scores(
        scored_reviews.loc[scored_reviews["bertopic_topic_id"] != -1],
        bertopic_terms,
        model_name="BERTopic",
    )
    coherence_scores = pd.concat([lda_coherence, bertopic_coherence], ignore_index=True)
    print("Computed sentiment, dictionary scores, LDA, BERTopic, and coherence diagnostics.")

    summaries = describe_sample(scored_reviews)
    monthly = monthly_sentiment(scored_reviews)
    topic_distribution = topic_distribution_by_brand(scored_reviews)
    bertopic_distribution = bertopic_distribution_by_brand(scored_reviews)
    topic_distribution = apply_publication_labels(
        topic_distribution,
        id_column="dominant_topic_label",
        raw_label_column="dominant_topic_label",
        label_map={v: config["publication_labels"]["lda"].get(k, v) for k, v in topic_terms[["topic_id", "topic_label"]].drop_duplicates().itertuples(index=False)},
        output_column="publication_label",
    )
    bertopic_distribution = apply_publication_labels(
        bertopic_distribution,
        id_column="bertopic_topic_label",
        raw_label_column="bertopic_topic_label",
        label_map={},
        output_column="publication_label",
    )
    bertopic_distribution["publication_label"] = bertopic_distribution["bertopic_topic_label"].map(infer_bertopic_publication_label)
    aspect_distribution = aspect_prevalence(scored_reviews)
    emotion_distribution = emotion_prevalence(scored_reviews)
    term_summary = top_terms_by_brand(scored_reviews)
    construct_mapping = pd.concat(
        [
            build_construct_mapping_table(config, "lda", config["publication_labels"]["lda"]),
            build_bertopic_construct_mapping_table(sorted(bertopic_distribution["publication_label"].dropna().unique().tolist())),
        ],
        ignore_index=True,
    )
    study2_alignment = build_study2_alignment_table()

    save_table(scored_reviews, PROJECT_ROOT / "data/processed/reviews_1_3_scored.csv")
    public_reviews = scored_reviews[PUBLIC_COLUMNS].copy()
    public_reviews_path = PROJECT_ROOT / "outputs/tables/study1_scored_reviews_public.csv"
    save_table(public_reviews, public_reviews_path)
    with zipfile.ZipFile(PROJECT_ROOT / "outputs/tables/study1_scored_reviews_public.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(public_reviews_path, arcname="study1_scored_reviews_public.csv")
    save_table(topic_terms, PROJECT_ROOT / "outputs/tables/topic_terms.csv")
    save_table(bertopic_terms, PROJECT_ROOT / "outputs/tables/bertopic_terms.csv")
    save_table(bertopic_info, PROJECT_ROOT / "outputs/tables/bertopic_info.csv")
    save_table(coherence_scores, PROJECT_ROOT / "outputs/tables/topic_coherence.csv")
    save_table(topic_distribution, PROJECT_ROOT / "outputs/tables/topic_distribution_by_brand.csv")
    save_table(bertopic_distribution, PROJECT_ROOT / "outputs/tables/bertopic_distribution_by_brand.csv")
    save_table(construct_mapping, PROJECT_ROOT / "outputs/tables/construct_mapping.csv")
    save_table(study2_alignment, PROJECT_ROOT / "outputs/tables/study2_alignment.csv")
    save_table(aspect_distribution, PROJECT_ROOT / "outputs/tables/aspect_distribution_by_brand.csv")
    save_table(emotion_distribution, PROJECT_ROOT / "outputs/tables/emotion_prevalence_by_brand.csv")
    save_table(monthly, PROJECT_ROOT / "outputs/tables/monthly_sentiment_by_brand.csv")
    save_table(term_summary, PROJECT_ROOT / "outputs/tables/top_terms_by_brand.csv")
    for name, table in summaries.items():
        save_table(table, PROJECT_ROOT / f"outputs/tables/{name}.csv")
    workbook_sheets = {
        "Scored Reviews": scored_reviews,
        "Brand Summary": summaries["brand_summary"],
        "Rating Counts": summaries["rating_counts"],
        "Sentiment Summary": summaries["sentiment_summary"],
        "LDA Topic Terms": topic_terms,
        "LDA Topic Distribution": topic_distribution,
        "BERTopic Terms": bertopic_terms,
        "BERTopic Info": bertopic_info,
        "BERTopic Distribution": bertopic_distribution,
        "Construct Mapping": construct_mapping,
        "Study2 Alignment": study2_alignment,
        "Aspect Distribution": aspect_distribution,
        "Emotion Prevalence": emotion_distribution,
        "Monthly Sentiment": monthly,
        "Topic Coherence": coherence_scores,
        "Top Terms by Brand": term_summary,
    }
    save_workbook(workbook_sheets, PROJECT_ROOT / "outputs/tables/study1_analysis_workbook.xlsx")
    print("Saved processed tables.")

    save_sentiment_distribution(
        summaries["sentiment_summary"],
        summaries["brand_summary"],
        PROJECT_ROOT / "outputs/figures/fig1_sentiment_distribution.png",
    )
    print("Saved figure 1.")
    save_sentiment_vs_rating(scored_reviews, PROJECT_ROOT / "outputs/figures/fig2_sentiment_vs_rating.png")
    print("Saved figure 2.")
    save_topic_heatmap(topic_terms, PROJECT_ROOT / "outputs/figures/fig3_lda_heatmap.png")
    print("Saved figure 3.")
    save_topic_distribution(topic_distribution, PROJECT_ROOT / "outputs/figures/fig4_topic_distribution.png")
    print("Saved figure 4.")
    save_aspect_distribution(aspect_distribution, PROJECT_ROOT / "outputs/figures/fig5_aspect_distribution.png")
    print("Saved figure 5.")
    save_monthly_sentiment(monthly, PROJECT_ROOT / "outputs/figures/fig6_sentiment_trend.png")
    print("Saved figure 6.")
    save_emotion_prevalence(emotion_distribution, PROJECT_ROOT / "outputs/figures/fig7_emotion_prevalence.png")
    print("Saved figure 7.")
    save_coherence_comparison(coherence_scores, PROJECT_ROOT / "outputs/figures/fig8_topic_coherence_comparison.png")
    print("Saved figure 8.")

    print("Analysis complete.")
    print(f"Processed reviews: {len(scored_reviews):,}")
    print("Outputs written to:")
    print(f"  {PROJECT_ROOT / 'outputs/figures'}")
    print(f"  {PROJECT_ROOT / 'outputs/tables'}")


if __name__ == "__main__":
    main()

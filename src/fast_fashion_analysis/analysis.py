from __future__ import annotations

from collections import Counter
import re

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.porter import PorterStemmer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

STEMMER = PorterStemmer()

TOKEN_NORMALIZATION = {
    "deliveri": "delivery",
    "deliver": "delivery",
    "shipment": "shipping",
    "ship": "shipping",
    "order": "order",
    "parcel": "parcel",
    "packag": "parcel",
    "receiv": "receive",
    "reciev": "receive",
    "return": "return",
    "refund": "refund",
    "cancel": "cancel",
    "qualiti": "quality",
    "product": "product",
    "item": "item",
    "size": "size",
    "servic": "service",
    "custom": "customer",
    "store": "store",
    "staff": "staff",
    "manag": "manager",
    "gift": "gift",
    "card": "card",
    "wrong": "wrong",
    "contact": "contact",
    "reply": "reply",
    "account": "account",
    "experi": "experience",
    "terribl": "terrible",
    "horribl": "horrible",
    "bad": "bad",
    "hate": "hate",
}

CUSTOM_STOPWORDS = {
    "zara",
    "shein",
    "hm",
    "hampm",
    "urban",
    "outfitters",
    "brand",
    "company",
    "clothes",
    "wa",
    "ha",
    "thei",
    "thi",
    "that",
    "with",
    "from",
    "have",
    "still",
    "would",
    "could",
    "also",
    "dai",
    "said",
    "told",
    "ask",
    "asked",
    "just",
    "like",
    "bui",
    "buy",
    "compani",
    "experience",
    "bad",
    "worst",
    "terrible",
    "horrible",
    "poor",
    "time",
    "good",
    "realli",
    "look",
    "went",
    "monei",
}


def normalize_token(token: str) -> str:
    token = token.lower()
    if not token.isalpha() or len(token) <= 2:
        return ""
    stem = STEMMER.stem(token)
    return TOKEN_NORMALIZATION.get(stem, stem)


def normalized_analyzer(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z]+", str(text).lower()):
        if token in ENGLISH_STOP_WORDS:
            continue
        normalized = normalize_token(token)
        if normalized and normalized not in CUSTOM_STOPWORDS and normalized not in ENGLISH_STOP_WORDS:
            tokens.append(normalized)
    return tokens


def score_sentiment(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    scored = df.copy()
    sentiments = pd.DataFrame(scored["review_full_text"].map(analyzer.polarity_scores).tolist(), index=scored.index)
    scored[["neg", "neu", "pos", "compound"]] = sentiments[["neg", "neu", "pos", "compound"]]

    positive_threshold = config["sentiment"]["positive_threshold"]
    negative_threshold = config["sentiment"]["negative_threshold"]

    scored["sentiment_label"] = np.select(
        [
            scored["compound"] >= positive_threshold,
            scored["compound"] <= negative_threshold,
        ],
        [
            "positive",
            "negative",
        ],
        default="neutral",
    )
    return scored


def _keyword_count(text: str, keywords: list[str]) -> int:
    lowered = f" {text.lower()} "
    return sum(lowered.count(keyword.lower()) for keyword in keywords)


def score_dictionary_categories(df: pd.DataFrame, category_map: dict, prefix: str) -> pd.DataFrame:
    scored = df.copy()
    for category, keywords in category_map.items():
        column = f"{prefix}_{category}"
        scored[column] = scored["review_full_text"].map(lambda text: _keyword_count(text, keywords))
    score_columns = [f"{prefix}_{category}" for category in category_map]
    scored[f"{prefix}_top_category"] = scored[score_columns].idxmax(axis=1).str.replace(f"{prefix}_", "", regex=False)
    no_match = scored[score_columns].sum(axis=1).eq(0)
    scored.loc[no_match, f"{prefix}_top_category"] = "unclassified"
    return scored


def build_lda_topics(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vectorizer = CountVectorizer(
        analyzer=normalized_analyzer,
        max_features=config["analysis"]["max_features"],
        min_df=5,
        max_df=0.85,
    )
    matrix = vectorizer.fit_transform(df["review_full_text"])
    lda = LatentDirichletAllocation(
        n_components=config["analysis"]["lda_topics"],
        random_state=config["analysis"]["random_state"],
        learning_method="batch",
    )
    topic_weights = lda.fit_transform(matrix)

    feature_names = np.array(vectorizer.get_feature_names_out())
    top_n = config["analysis"]["lda_top_words"]
    topic_rows = []
    topic_labels = {}

    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-top_n:][::-1]
        top_terms = feature_names[top_indices].tolist()
        topic_label = " | ".join(top_terms[:3])
        topic_labels[topic_idx] = topic_label
        for rank, term in enumerate(top_terms, start=1):
            topic_rows.append(
                {
                    "topic_id": topic_idx,
                    "topic_label": topic_label,
                    "rank": rank,
                    "term": term,
                    "weight": float(topic[top_indices[rank - 1]]),
                }
            )

    doc_topics = pd.DataFrame(topic_weights, columns=[f"topic_{i}" for i in range(topic_weights.shape[1])])
    dominant_topic_id = doc_topics.to_numpy().argmax(axis=1)
    doc_topics["dominant_topic_id"] = dominant_topic_id
    doc_topics["dominant_topic_label"] = [topic_labels[idx] for idx in dominant_topic_id]

    topic_terms = pd.DataFrame(topic_rows)
    return df.join(doc_topics), topic_terms, doc_topics


def _tokenize_documents(df: pd.DataFrame) -> list[list[str]]:
    tokenized = []
    for text in df["review_full_text"].astype(str):
        tokens = normalized_analyzer(text)
        tokenized.append(tokens)
    return tokenized


def _topic_word_lists(topic_terms: pd.DataFrame, topic_col: str = "topic_id") -> list[list[str]]:
    grouped = topic_terms.sort_values([topic_col, "rank"]).groupby(topic_col)["term"].apply(list)
    return grouped.tolist()


def compute_coherence_scores(df: pd.DataFrame, topic_terms: pd.DataFrame, model_name: str, topic_col: str = "topic_id") -> pd.DataFrame:
    tokenized_docs = _tokenize_documents(df)
    dictionary = Dictionary(tokenized_docs)
    topic_lists = _topic_word_lists(topic_terms, topic_col=topic_col)
    coherence_model = CoherenceModel(
        topics=topic_lists,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v",
    )
    per_topic = coherence_model.get_coherence_per_topic()
    summary = []
    for (topic_id, topic_label), coherence in zip(
        topic_terms[[topic_col, "topic_label"]].drop_duplicates().sort_values(topic_col).itertuples(index=False),
        per_topic,
    ):
        summary.append(
            {
                "model": model_name,
                "topic_id": topic_id,
                "topic_label": topic_label,
                "coherence_cv": float(coherence),
            }
        )
    summary.append(
        {
            "model": model_name,
            "topic_id": "overall",
            "topic_label": "overall",
            "coherence_cv": float(coherence_model.get_coherence()),
        }
    )
    return pd.DataFrame(summary)


def build_bertopic_topics(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from bertopic import BERTopic
    from bertopic.backend._sklearn import SklearnEmbedder

    vectorizer = TfidfVectorizer(
        analyzer=normalized_analyzer,
        max_features=config["analysis"]["max_features"],
        min_df=5,
        max_df=0.85,
    )
    embedder_pipeline = make_pipeline(
        vectorizer,
        TruncatedSVD(
            n_components=config["analysis"]["bertopic_embedding_dim"],
            random_state=config["analysis"]["random_state"],
        ),
    )
    topic_vectorizer = CountVectorizer(
        analyzer=normalized_analyzer,
        max_features=config["analysis"]["max_features"],
        min_df=1,
        max_df=1.0,
    )
    topic_model = BERTopic(
        embedding_model=SklearnEmbedder(embedder_pipeline),
        vectorizer_model=topic_vectorizer,
        min_topic_size=config["analysis"]["bertopic_min_topic_size"],
        top_n_words=config["analysis"]["bertopic_top_words"],
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(df["review_full_text"].tolist())
    topic_info = topic_model.get_topic_info()

    topic_label_map = {}
    rows = []
    for topic_id in topic_info["Topic"].tolist():
        terms = topic_model.get_topic(topic_id)
        if not terms:
            continue
        topic_label = " | ".join([term for term, _ in terms[:3]])
        topic_label_map[topic_id] = topic_label
        for rank, (term, weight) in enumerate(terms[: config["analysis"]["bertopic_top_words"]], start=1):
            rows.append(
                {
                    "topic_id": topic_id,
                    "topic_label": topic_label,
                    "rank": rank,
                    "term": term,
                    "weight": float(weight),
                }
            )

    assignments = pd.DataFrame(
        {
            "bertopic_topic_id": topics,
            "bertopic_topic_label": [topic_label_map.get(topic_id, "outlier") for topic_id in topics],
        },
        index=df.index,
    )
    topic_terms = pd.DataFrame(rows)
    topic_info = topic_info.rename(columns={"Topic": "bertopic_topic_id", "Count": "n_reviews"})
    return df.join(assignments), topic_terms, topic_info


def apply_publication_labels(
    df: pd.DataFrame,
    id_column: str,
    raw_label_column: str,
    label_map: dict,
    output_column: str,
) -> pd.DataFrame:
    labeled = df.copy()
    labeled[output_column] = labeled[id_column].map(label_map).fillna(labeled[raw_label_column])
    return labeled


def build_construct_mapping_table(config: dict, model_name: str, label_map: dict) -> pd.DataFrame:
    rows = []
    for topic_id, publication_label in label_map.items():
        mapping = config["construct_mapping"][model_name].get(topic_id, {})
        rows.append(
            {
                "model": model_name.upper() if model_name == "lda" else "BERTopic",
                "topic_id": topic_id,
                "publication_label": publication_label,
                "primary_construct": mapping.get("primary_construct", ""),
                "interpretation": mapping.get("interpretation", ""),
            }
        )
    return pd.DataFrame(rows)


def infer_bertopic_publication_label(raw_label: str) -> str:
    label = str(raw_label).lower()
    if "shipped" in label or "order online" in label or "ship" in label:
        return "Shipping Delays & Online Fulfilment"
    if "manager" in label or "receipt" in label or "worst customer" in label or "queue" in label:
        return "In-Store Staff, Policy & Queue Problems"
    if "evri" in label or "returned items" in label or "account" in label or "shein" in label:
        return "SHEIN Logistics, Replies & Account Issues"
    if "order cancelled" in label or "cancelled order" in label:
        return "Order Cancellation & Fulfilment Breakdown"
    if "quality" in label or "cheap" in label or "jeans" in label or "pair" in label:
        return "Poor Quality & Cheap Products"
    if "day delivery" in label or "hermes" in label or "paid day" in label:
        return "Premium Delivery & Courier Failure"
    if "didnt receive" in label or "receive order" in label or "did receive" in label or "received order" in label or "haven received" in label:
        return "Non-Delivery & Missing Parcel"
    if "gift" in label or "gift card" in label or "credit" in label:
        return "Gift Card & Store Credit Issues"
    if "wrong item" in label or "wrong size" in label or "sent wrong" in label:
        return "Wrong Item & Wrong Size"
    if "terrible customer" in label or "terrible service" in label:
        return "Severe Service Failure"
    if "bad experience" in label or "service bad" in label or "really bad" in label:
        return "Bad Experience & Negative Engagement"
    if "absolutely" in label or "clothes" in label:
        return "Mixed Negative Complaints"
    return raw_label


def build_bertopic_construct_mapping_table(publication_labels: list[str]) -> pd.DataFrame:
    mapping = {
        "Mixed Negative Complaints": ("Negative Past Experience", "The outlier cluster captures diffuse but repeated negative complaints that often reflect accumulated poor experiences."),
        "Shipping Delays & Online Fulfilment": ("Low Service Quality", "Shipping and fulfilment delays indicate operational low service quality in online ordering."),
        "In-Store Staff, Policy & Queue Problems": ("Low Service Quality", "Store manager, receipt, and policy issues reflect in-store service quality breakdowns."),
        "SHEIN Logistics, Replies & Account Issues": ("Negative Past Experience", "Repeated SHEIN-specific logistics and reply failures indicate accumulated negative past experiences."),
        "Order Cancellation & Fulfilment Breakdown": ("Negative Past Experience", "Order cancellation problems reflect unresolved service failures that accumulate into negative experience."),
        "Poor Quality & Cheap Products": ("Low Service Quality", "Poor product quality and cheapness reflect core product-related service failure."),
        "Premium Delivery & Courier Failure": ("Low Service Quality", "Premium delivery and courier failure indicate unmet service promises and logistics breakdown."),
        "Non-Delivery & Missing Parcel": ("Negative Past Experience", "Non-delivery and missing parcels often reflect repeated unresolved failures and heightened consumer memory of harm."),
        "Gift Card & Store Credit Issues": ("Low Service Quality", "Gift-card and store-credit disputes indicate payment and restitution service failures."),
        "Wrong Item & Wrong Size": ("Low Service Quality", "Wrong-item and wrong-size complaints reflect fulfilment and product-quality failures."),
        "Severe Service Failure": ("Brand Hate", "Highly severe complaint language signals strong aversive emotion approaching brand hate."),
        "Bad Experience & Negative Engagement": ("Brand Hate", "Explicitly hostile bad-experience narratives indicate an emergent aversive response consistent with brand hate."),
    }
    rows = []
    for label in publication_labels:
        construct, interpretation = mapping.get(label, ("", ""))
        rows.append(
            {
                "model": "BERTopic",
                "topic_id": "",
                "publication_label": label,
                "primary_construct": construct,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def build_study2_alignment_table() -> pd.DataFrame:
    rows = [
        {
            "study2_construct": "Low Service Quality",
            "study1_role": "Primary exploratory anchor",
            "how_study1_supports_it": "Study 1 identifies recurring delivery, refund, payment, fulfilment, product-quality, and frontline service failures as the dominant complaint structure.",
        },
        {
            "study2_construct": "Negative Past Experience",
            "study1_role": "Exploratory precursor",
            "how_study1_supports_it": "Study 1 surfaces repeated, unresolved, and cumulative complaint narratives, especially around non-delivery, failed recovery, and repeated contact attempts.",
        },
        {
            "study2_construct": "Brand Hate",
            "study1_role": "Emergent cue",
            "how_study1_supports_it": "Study 1 captures hostile, severe, and aversive complaint language in several BERTopic clusters, indicating escalation beyond simple dissatisfaction.",
        },
        {
            "study2_construct": "Brand Switching",
            "study1_role": "Downstream outcome",
            "how_study1_supports_it": "Study 1 provides contextual evidence for exit intentions and brand rejection, but the formal test of brand switching is reserved for Study 2.",
        },
    ]
    return pd.DataFrame(rows)


def describe_sample(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rating_counts = (
        df.groupby(["brand", "reviewer_rating"], observed=False)
        .size()
        .reset_index(name="n_reviews")
        .sort_values(["brand", "reviewer_rating"])
    )
    brand_summary = (
        df.groupby("brand", observed=False)
        .agg(
            n_reviews=("review_full_text", "size"),
            mean_tokens=("token_count", "mean"),
            median_tokens=("token_count", "median"),
            mean_compound=("compound", "mean"),
        )
        .reset_index()
    )
    sentiment_summary = (
        df.groupby(["brand", "sentiment_label"], observed=False)
        .size()
        .reset_index(name="n_reviews")
    )
    sentiment_summary["pct_reviews"] = sentiment_summary.groupby("brand", observed=False)["n_reviews"].transform(
        lambda s: 100 * s / s.sum()
    )
    return {
        "rating_counts": rating_counts,
        "brand_summary": brand_summary,
        "sentiment_summary": sentiment_summary,
    }


def monthly_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.dropna(subset=["review_date"])
        .groupby(["brand", "review_month"], observed=False)
        .agg(
            mean_compound=("compound", "mean"),
            n_reviews=("review_full_text", "size"),
        )
        .reset_index()
    )
    monthly["review_month"] = pd.to_datetime(monthly["review_month"])
    return monthly.sort_values(["brand", "review_month"])


def topic_distribution_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        df.groupby(["brand", "dominant_topic_label"], observed=False)
        .size()
        .reset_index(name="n_reviews")
    )
    distribution["pct_reviews"] = distribution.groupby("brand", observed=False)["n_reviews"].transform(
        lambda s: 100 * s / s.sum()
    )
    return distribution.sort_values(["brand", "pct_reviews"], ascending=[True, False])


def bertopic_distribution_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        df.loc[df["bertopic_topic_id"] != -1]
        .groupby(["brand", "bertopic_topic_label"], observed=False)
        .size()
        .reset_index(name="n_reviews")
    )
    distribution["pct_reviews"] = distribution.groupby("brand", observed=False)["n_reviews"].transform(
        lambda s: 100 * s / s.sum()
    )
    return distribution.sort_values(["brand", "pct_reviews"], ascending=[True, False])


def aspect_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["brand", "aspect_top_category"], observed=False)
        .size()
        .reset_index(name="n_reviews")
    )
    counts["pct_reviews"] = counts.groupby("brand", observed=False)["n_reviews"].transform(
        lambda s: 100 * s / s.sum()
    )
    return counts.sort_values(["brand", "pct_reviews"], ascending=[True, False])


def emotion_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    emotion_columns = [column for column in df.columns if column.startswith("emotion_") and column != "emotion_top_category"]
    rows = []
    for brand, subset in df.groupby("brand", observed=False):
        total = len(subset)
        for column in emotion_columns:
            rows.append(
                {
                    "brand": brand,
                    "emotion": column.replace("emotion_", ""),
                    "n_reviews": int((subset[column] > 0).sum()),
                    "pct_reviews": 100 * float((subset[column] > 0).sum()) / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["brand", "pct_reviews"], ascending=[True, False])


def top_terms_by_brand(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    rows = []
    for brand, subset in df.groupby("brand", observed=False):
        counter = Counter()
        for text in subset["review_full_text"].astype(str):
            tokens = [token for token in normalized_analyzer(text) if len(token) > 3]
            counter.update(tokens)
        for rank, (term, count) in enumerate(counter.most_common(top_n), start=1):
            rows.append({"brand": brand, "rank": rank, "term": term, "count": count})
    return pd.DataFrame(rows)

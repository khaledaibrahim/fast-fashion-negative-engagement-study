from __future__ import annotations

import re

import pandas as pd

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)")


def _fix_mojibake(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.replace("\r", " ").replace("\n", " ")
    if any(token in cleaned for token in ["‚", "Ä", "Ã", "â"]):
        try:
            repaired = cleaned.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if repaired:
                cleaned = repaired
        except UnicodeError:
            pass
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _redact_pii(text: str) -> str:
    if not isinstance(text, str):
        return ""
    redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    redacted = re.sub(r"\s+", " ", redacted).strip()
    return redacted


def _standardize_brand(value: str) -> str:
    mapping = {
        "H&M": "H&M",
        "HM": "H&M",
        "ZARA": "ZARA",
        "Zara": "ZARA",
        "SHEIN": "SHEIN",
        "Shein": "SHEIN",
        "Urban Outfitters": "Urban Outfitters",
        "URBAN OUTFITTERS": "Urban Outfitters",
    }
    return mapping.get(str(value).strip(), str(value).strip())


def prepare_reviews(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    data = df.copy()
    data["brand"] = data["name"].map(_standardize_brand)
    data["review_text"] = data["reviewer_review"].fillna("").map(_fix_mojibake).map(_redact_pii)
    data["review_title"] = data["reviewer_title"].fillna("").map(_fix_mojibake).map(_redact_pii)
    data["review_full_text"] = (
        data["review_title"].fillna("").str.strip() + ". " + data["review_text"].fillna("").str.strip()
    ).str.strip(". ").str.strip()
    data["reviewer_rating"] = pd.to_numeric(data["reviewer_rating"], errors="coerce")
    data["review_date"] = pd.to_datetime(data["reviewer_published_date"], errors="coerce", utc=True)
    data["review_month"] = data["review_date"].dt.tz_convert(None).dt.to_period("M").astype(str)
    data["token_count"] = data["review_full_text"].str.split().str.len()

    min_star = config["analysis"]["min_star"]
    max_star = config["analysis"]["max_star"]
    min_tokens = config["analysis"]["min_tokens"]

    filtered = data.loc[
        data["reviewer_rating"].between(min_star, max_star, inclusive="both")
        & (data["token_count"] >= min_tokens)
        & data["review_full_text"].ne("")
    ].copy()

    filtered = filtered.drop_duplicates(subset=["brand", "reviewer_name", "review_full_text"])
    filtered = filtered.drop(columns=["email", "phone"], errors="ignore")
    filtered["brand"] = pd.Categorical(filtered["brand"], categories=config["analysis"]["brand_order"], ordered=True)
    return filtered.sort_values(["brand", "review_date", "url"], na_position="last").reset_index(drop=True)

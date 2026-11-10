"""Explicit, source-local behavior observations from sampled TwiBot-20 tweets."""

from __future__ import annotations

from collections import Counter
import re

import numpy as np


NAMES = (
    "normalized_repeat_fraction",
    "retweet_author_diversity",
    "url_domain_diversity",
    "top_url_domain_share",
    "hashtag_diversity",
    "mention_diversity",
    "mean_sampled_tweet_word_count",
    "sampled_tweet_token_diversity",
)
URL = re.compile(r"https?://[^\s]+", re.IGNORECASE)
DOMAIN = re.compile(r"^https?://(?:www\.)?([a-z0-9][a-z0-9.-]*)", re.IGNORECASE)
RETWEET = re.compile(r"^\s*RT\s+@([\w_]+):?", re.IGNORECASE)
HASHTAG = re.compile(r"(?<!\w)#([\w_]+)")
MENTION = re.compile(r"(?<!\w)@([\w_]+)")
TOKEN = re.compile(r"(?u)\b[a-zA-Z]{3,}\b")
WHITESPACE = re.compile(r"\s+")


def _fraction_unique(items: list[str]) -> float:
    return len(set(items)) / len(items) if items else float("nan")


def extract(tweets: list[str]) -> np.ndarray:
    """Return eight deterministic facts; missing denominators produce NaN."""
    texts = [str(item or "") for item in tweets]
    normalized = []
    retweet_authors: list[str] = []
    domains: list[str] = []
    hashtags: list[str] = []
    mentions: list[str] = []
    words: list[str] = []
    word_counts = []
    for text in texts:
        retweet = RETWEET.match(text)
        if retweet:
            retweet_authors.append(retweet.group(1).lower())
        for match in URL.finditer(text):
            candidate = match.group(0).rstrip(".,!?;:)]}")
            host = DOMAIN.match(candidate)
            if host:
                domain = host.group(1).lower().rstrip(".")
                if domain:
                    domains.append(domain)
        hashtags.extend(value.lower() for value in HASHTAG.findall(text))
        mentions.extend(value.lower() for value in MENTION.findall(text))
        current = [value.lower() for value in TOKEN.findall(text)]
        words.extend(current)
        word_counts.append(len(current))
        stripped = URL.sub("<url>", text.lower())
        stripped = MENTION.sub("@user", stripped)
        normalized.append(WHITESPACE.sub(" ", stripped).strip())
    n = len(texts)
    repeated = n - len(set(normalized)) if n else 0
    return np.asarray((
        repeated / n if n else float("nan"),
        _fraction_unique(retweet_authors),
        _fraction_unique(domains),
        max(Counter(domains).values()) / len(domains) if domains else float("nan"),
        _fraction_unique(hashtags),
        _fraction_unique(mentions),
        float(np.mean(word_counts)) if n else float("nan"),
        _fraction_unique(words),
    ), dtype=float)

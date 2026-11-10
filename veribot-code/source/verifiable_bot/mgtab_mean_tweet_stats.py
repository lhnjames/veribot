"""Fixed, source-computable summaries of MGTAB's published mean-tweet tensor."""

from __future__ import annotations

import numpy as np


STAT_NAMES = (
    "mean_tweet_tensor_signed_mean", "mean_tweet_tensor_std",
    "mean_tweet_tensor_mean_abs", "mean_tweet_tensor_rms",
    "mean_tweet_tensor_min", "mean_tweet_tensor_max",
    "mean_tweet_tensor_q10", "mean_tweet_tensor_q90",
    "mean_tweet_tensor_max_abs", "mean_tweet_tensor_top10_abs_mean",
    "mean_tweet_tensor_abs_q10", "mean_tweet_tensor_abs_q90",
    "mean_tweet_tensor_positive_fraction", "mean_tweet_tensor_zero_fraction",
    "mean_tweet_tensor_abs_entropy", "mean_tweet_tensor_median",
)


def summaries(published: np.ndarray) -> np.ndarray:
    """Return sixteen literal statistics from columns 20:788 of features.pt."""
    source = np.asarray(published, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 788 or not np.isfinite(source).all():
        raise ValueError("MGTAB published feature population differs")
    content = source[:, 20:]
    magnitude = np.abs(content)
    total = magnitude.sum(axis=1)
    weights = np.divide(magnitude, total[:, None],
                        out=np.zeros_like(magnitude), where=total[:, None] > 0)
    entropy = -(np.where(weights > 0, weights * np.log(np.maximum(weights, 1e-300)),
                         0.0)).sum(axis=1) / np.log(content.shape[1])
    columns = (
        content.mean(axis=1), content.std(axis=1), magnitude.mean(axis=1),
        np.sqrt(np.mean(content ** 2, axis=1)),
        content.min(axis=1), content.max(axis=1),
        np.quantile(content, 0.1, axis=1), np.quantile(content, 0.9, axis=1),
        magnitude.max(axis=1), np.partition(magnitude, -10, axis=1)[:, -10:].mean(axis=1),
        np.quantile(magnitude, 0.1, axis=1), np.quantile(magnitude, 0.9, axis=1),
        (content > 0).mean(axis=1), (content == 0).mean(axis=1),
        entropy, np.median(content, axis=1),
    )
    result = np.column_stack(columns)
    if result.shape != (len(source), len(STAT_NAMES)) or not np.isfinite(result).all():
        raise ValueError("MGTAB mean-tweet statistics are not finite")
    return result

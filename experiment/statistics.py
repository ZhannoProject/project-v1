"""Pair observations before aggregation; inference uses seeds or subjects."""

import math
import numpy as np


def mean_sd(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Sample SD requires at least two finite independent units")
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), n=int(x.size), sd_ddof=1)


def paired_tost(a, b, margin, unit, alpha=.05):
    """Two one-sided paired t tests, with a prespecified absolute margin."""
    from scipy.stats import t
    if unit not in ("seed", "subject"):
        raise ValueError("Inference units must be training seeds or subjects, never individual segments/blinks")
    if not math.isfinite(margin) or margin <= 0 or not 0 < alpha < .5:
        raise ValueError("TOST requires a prespecified positive margin and alpha in (0, .5)")
    if set(a) != set(b) or len(a) < 3:
        raise ValueError("TOST requires at least three identical paired independent unit IDs")
    ids = sorted(a)
    d = np.asarray([a[i] - b[i] for i in ids], dtype=float)
    summary = mean_sd(d)
    se = summary["sd"] / np.sqrt(len(d))
    if se == 0:
        p_lower = 0. if summary["mean"] > -margin else 1.
        p_upper = 0. if summary["mean"] < margin else 1.
        ci = [summary["mean"], summary["mean"]]
    else:
        p_lower = float(t.sf((summary["mean"] + margin) / se, len(d) - 1))
        p_upper = float(t.cdf((summary["mean"] - margin) / se, len(d) - 1))
        half = float(t.ppf(1 - alpha, len(d) - 1) * se)
        ci = [summary["mean"] - half, summary["mean"] + half]
    return {**summary, "difference": "a-b", "unit": unit, "unit_ids": ids,
            "margin": margin, "alpha": alpha, "p_lower": p_lower, "p_upper": p_upper,
            "p_value": max(p_lower, p_upper), "confidence_interval": ci,
            "confidence_level": 1 - 2 * alpha, "equivalent": max(p_lower, p_upper) < alpha}


def pair_segments(a, b, metric):
    """Same underlying inputs AND target positions, not matching list lengths."""
    def index(rows):
        ids = [r["segment_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate segment IDs")
        return {r["segment_id"]: r for r in rows}
    aa, bb = index(a), index(b)
    if set(aa) != set(bb) or not aa:
        raise ValueError("Cannot pair different test segments")
    av, bv = [], []
    for key in sorted(aa):
        for field in ("target_interval", "sample_rate_hz"):
            if aa[key][field] != bb[key][field]:
                raise ValueError(f"Paired segments have different {field}")
        av.append(aa[key][metric]); bv.append(bb[key][metric])
    if not np.isfinite(av).all() or not np.isfinite(bv).all():
        raise ValueError("Nonfinite paired observations")
    return float(np.mean(av)), float(np.mean(bv)), len(av)

"""Population statistics, following FlowJo's definitions where they exist."""

import numpy as np

from .spectral import robust_sd

# Statistics that need a channel; the rest describe the population itself.
CHANNEL_STATS = ("Mean", "Median", "Geometric Mean", "SD", "rSD", "CV", "rCV", "Percentile")
POPULATION_STATS = ("Count", "Freq. of Parent", "Freq. of Grandparent", "Freq. of Total")
ALL_STATS = POPULATION_STATS + CHANNEL_STATS


def channel_stat(values, stat, percentile=50.0):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    if stat == "Mean":
        return float(values.mean())
    if stat == "Median":
        return float(np.median(values))
    if stat == "Geometric Mean":
        # Undefined for values <= 0; those events are left out, as FlowJo does.
        pos = values[values > 0]
        return float(np.exp(np.log(pos).mean())) if pos.size else float("nan")
    if stat == "SD":
        return float(values.std(ddof=1)) if values.size > 1 else float("nan")
    if stat == "rSD":
        return float(robust_sd(values))
    if stat == "CV":
        mean = values.mean()
        return float(100.0 * values.std(ddof=1) / mean) if values.size > 1 and mean else float("nan")
    if stat == "rCV":
        med = np.median(values)
        return float(100.0 * robust_sd(values) / med) if med else float("nan")
    if stat == "Percentile":
        return float(np.percentile(values, percentile))
    raise ValueError(f"unknown statistic {stat!r}")


def frequency(count, reference_count):
    return 100.0 * count / reference_count if reference_count else float("nan")

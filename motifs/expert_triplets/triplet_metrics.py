"""Shared maths for expert "odd one out" triplets: scores, loss, intervals.

A triplet answer names the crop that does not belong with the other two. Three
things can be compared against it:

  motif_credit   a motif assignment (hard labels). 1 if it agrees with the
                 expert, 0 if it contradicts them, 1/3 if it has no opinion
                 (all three in one motif, or all three in different motifs).
                 1/3 is exactly the expected credit of a random answer, so
                 k = 1 and k = n_cells both score chance and only motifs that
                 split neighbourhoods the way the expert does score above it.

  embedding_odd  an embedding (distances). The model's odd one is the crop
                 left out of the closest pair.

  odd_one_out_nll  the same embedding as a probability model, which is the
                 LOSS used to fit expert-informed weights and to report how
                 confidently the embedding agrees:
                     P(odd = k) = exp(s_ij) / (exp(s_ij) + exp(s_ik) + exp(s_jk))
                 where (i, j) is the pair left when k is removed and
                 s = -d^2 / tau. Uniform guessing gives log(3) = 1.0986.

Intervals: `confidence_sequence` is valid at every sample size at once, so the
score can be watched while experts keep annotating without invalidating it
(predictable-plug-in empirical-Bernstein CS, Waudby-Smith & Ramdas 2023).
"""

from __future__ import annotations

import math

import numpy as np

CHANCE = 1.0 / 3.0
UNIFORM_NLL = math.log(3.0)


def motif_credit(motifs: tuple, odd: int) -> float:
    """Credit for one answer. `motifs` = motif of crops (0, 1, 2); `odd` = the
    expert's odd crop index."""
    a, b, c = motifs
    distinct = len({a, b, c})
    if distinct != 2:                          # all same / all different
        return CHANCE
    pair = [i for i in range(3) if i != odd]
    if motifs[pair[0]] == motifs[pair[1]]:     # expert's pair shares a motif
        return 1.0
    return 0.0


def pair_sq_dists(e: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """e: (n, 3, d) embeddings of each triplet's crops in crop order.
    Returns (n, 3) squared distances for the pair LEFT when crop k is removed:
    column k = d^2(the other two)."""
    if weights is not None:
        e = e * np.sqrt(np.asarray(weights, dtype=float))[None, None, :]
    d12 = ((e[:, 1] - e[:, 2]) ** 2).sum(-1)   # crop 0 removed
    d02 = ((e[:, 0] - e[:, 2]) ** 2).sum(-1)   # crop 1 removed
    d01 = ((e[:, 0] - e[:, 1]) ** 2).sum(-1)   # crop 2 removed
    return np.stack([d12, d02, d01], axis=1)


def embedding_odd(pair_d2: np.ndarray) -> np.ndarray:
    """Model's odd crop index: the one whose removal leaves the closest pair."""
    return np.argmin(pair_d2, axis=1)


def odd_one_out_log_probs(pair_d2: np.ndarray, tau: float) -> np.ndarray:
    """(n, 3) log P(odd = k) under the softmax model above."""
    s = -pair_d2 / tau
    s = s - s.max(axis=1, keepdims=True)
    return s - np.log(np.exp(s).sum(axis=1, keepdims=True))


def odd_one_out_nll(pair_d2: np.ndarray, odd: np.ndarray, tau: float) -> float:
    """Mean negative log-likelihood of the expert's answers (the loss)."""
    lp = odd_one_out_log_probs(pair_d2, tau)
    return float(-lp[np.arange(len(odd)), odd].mean())


def fit_tau(pair_d2: np.ndarray, odd: np.ndarray) -> float:
    """Temperature that minimises NLL, by a log-spaced grid search. Fit it on
    TRAIN answers only; scoring test answers with a tau fitted on them would
    leak."""
    scale = float(np.median(pair_d2)) or 1.0
    grid = scale * np.logspace(-3, 2, 121)
    losses = [odd_one_out_nll(pair_d2, odd, t) for t in grid]
    return float(grid[int(np.argmin(losses))])


def torch_odd_one_out_loss(e, odd, weights, tau: float):
    """Differentiable version of `odd_one_out_nll` for fitting `weights`.

    e: (n, 3, d) tensor, odd: (n,) long tensor, weights: (d,) tensor of
    non-negative per-dimension weights (e.g. softplus of a parameter).
    """
    import torch
    ew = e * torch.sqrt(weights)[None, None, :]
    d12 = ((ew[:, 1] - ew[:, 2]) ** 2).sum(-1)
    d02 = ((ew[:, 0] - ew[:, 2]) ** 2).sum(-1)
    d01 = ((ew[:, 0] - ew[:, 1]) ** 2).sum(-1)
    logits = -torch.stack([d12, d02, d01], dim=1) / tau
    return torch.nn.functional.cross_entropy(logits, odd)


def confidence_sequence(x: np.ndarray, alpha: float = 0.05,
                        c: float = 0.5) -> tuple[float, float]:
    """Anytime-valid (1 - alpha) interval for the mean of [0, 1] observations,
    taken in the order they were collected. Returns (lo, hi) at the last
    observation, intersected over all earlier ones."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    if x.size == 0:
        return 0.0, 1.0
    log_term = math.log(2.0 / alpha)
    mu_prev, var_prev = 0.5, 0.25
    s_x = s_sq = 0.0
    s_lam = s_lam_x = s_pen = 0.0
    lo, hi = 0.0, 1.0
    for t, xt in enumerate(x, start=1):
        lam = min(math.sqrt(2 * log_term / (var_prev * t * math.log(1 + t))), c)
        v = 4.0 * (xt - mu_prev) ** 2
        psi = (-math.log(1 - lam) - lam) / 4.0
        s_lam += lam
        s_lam_x += lam * xt
        s_pen += v * psi
        centre = s_lam_x / s_lam
        width = (log_term + s_pen) / s_lam
        lo, hi = max(lo, centre - width), min(hi, centre + width)
        s_x += xt
        mu_t = (0.5 + s_x) / (t + 1)
        s_sq += (xt - mu_t) ** 2
        mu_prev, var_prev = mu_t, (0.25 + s_sq) / (t + 1)
    return max(lo, 0.0), min(hi, 1.0)


def bootstrap_ci(x: np.ndarray, alpha: float = 0.05, n_boot: int = 4000,
                 seed: int = 0) -> tuple[float, float]:
    """Ordinary percentile bootstrap interval for the mean. Valid only for an
    analysis done once at a sample size fixed in advance (no peeking)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))

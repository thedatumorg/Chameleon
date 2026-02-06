import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def loading_scores(Candidate_Model_Set, args):
    scores_list = []
    for det in Candidate_Model_Set:

        path = f'{args.score_dir}/{det}/{args.filename.split(".")[0]}.npy'
        if os.path.exists(path):
            score = np.load(path)
        else:
            print('No score found, use random score instead')
            anomaly_score_pool = []
            for i in range(5):
                anomaly_score_pool.append(np.random.uniform(size=args.ts_len))
            score = np.mean(np.array(anomaly_score_pool), axis=0)

        if len(score) < args.ts_len:
            score = np.pad(score, (0, args.ts_len - len(score)), mode='constant')
        elif len(score) > args.ts_len:
            score = score[:args.ts_len]

        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        scores_list.append(score)
    det_scores = np.array(scores_list).T  # (score_len, num_score)
    return det_scores

def compute_alpha_from_uncertainty(per_det_hist, per_det_counts, args):
    # collect per-detector std and sample count
    stds = []
    ns = []
    for vals, c in zip(per_det_hist, per_det_counts):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size >= 2:
            stds.append(float(np.std(vals, ddof=1)))
            ns.append(int(vals.size))
        elif vals.size == 1:
            stds.append(0.25)  # conservative: one sample => uncertain
            ns.append(1)

    # default: trust prior if no evidence
    if len(stds) == 0:
        return float(getattr(args, "alpha_max", 0.9))

    unc = float(np.median(stds))      # robust uncertainty summary
    n_eff = float(np.median(ns))      # robust coverage summary

    # map uncertainty + coverage -> confidence in [0,1]
    tau = float(getattr(args, "alpha_tau", 0.15))          # uncertainty sensitivity
    target_n = float(getattr(args, "alpha_target_n", 12))  # enough trials threshold

    conf_unc = float(np.exp(-unc / max(tau, 1e-6)))        # low unc -> near 1
    conf_n = float(np.clip(n_eff / max(target_n, 1.0), 0.0, 1.0))
    conf = conf_unc * conf_n

    alpha_min = float(getattr(args, "alpha_min", 0.3))     # trust evidence
    alpha_max = float(getattr(args, "alpha_max", 0.9))     # trust prior
    alpha = alpha_max - (alpha_max - alpha_min) * conf
    return alpha


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)  # stability
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

def rank_to_weight(ranking, models, temp=1.0):
    """
    ranking: list[str] ordered best->worst
    models: selected_models list[str]
    returns weights aligned with models
    """
    pos = {m: i for i, m in enumerate(ranking)}  # smaller is better
    # if model not present, give it a low confidence rank at the end
    worst_rank = len(ranking) + 5
    ranks = np.array([pos.get(m, worst_rank) for m in models], dtype=float)

    # convert to scores: higher is better
    # using negative rank / temp then softmax
    scores = -ranks / max(temp, 1e-8)
    return softmax(scores)


def _to_1d_window(w):
    """
    Convert a window into 2D array shape (T, C) for unified scoring.
    Accepts common shapes: (T,), (T,1), (1,T), (T,C), (C,T).
    """
    w = np.asarray(w)
    if w.ndim == 1:
        w = w[:, None]  # (T,1)
    elif w.ndim == 2:
        # If it's (C,T) and C small, we guess transpose to (T,C)
        if w.shape[0] < w.shape[1] and w.shape[0] <= 32:
            # could be (C,T)
            # but if it's already (T,C), this doesn't hurt too much, so keep heuristic conservative
            pass
        # if it is (1,T), transpose
        if w.shape[0] == 1 and w.shape[1] > 1:
            w = w.T
    else:
        # fallback: flatten time dim
        w = w.reshape(w.shape[0], -1)
    return w  # (T,C)

def informative_score(window):
    """
    Information score: std(level) + 0.5 * std(diff)
    Works for uni/multi-variate windows.
    """
    w = _to_1d_window(window)            # (T,C)
    w = np.asarray(w, dtype=float)
    if w.shape[0] < 3:
        return 0.0

    # channel-wise
    level = np.nanstd(w, axis=0)
    diff  = np.nanstd(np.diff(w, axis=0), axis=0)

    # average across channels
    return float(np.nanmean(level) + 0.5 * np.nanmean(diff))

def select_informative_window_indices(windows, W_sample, rng=None, top_factor=3, spread=True):
    """
    Pick W_sample windows with highest informative_score.
    - top_factor: consider top (W_sample*top_factor) candidates by score, then optionally spread them.
    - spread: if True, choose evenly spaced indices within those top candidates to cover time.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    W_total = len(windows)
    if W_total == 0:
        return np.array([], dtype=int)

    W_sample = int(min(max(1, W_sample), W_total))

    scores = np.array([informative_score(w) for w in windows], dtype=float)
    # sort by score descending
    order = np.argsort(scores)[::-1]

    topN = min(W_total, max(W_sample, W_sample * int(top_factor)))
    top_idx = np.sort(order[:topN])  # sort by time index so we can spread

    if not spread or len(top_idx) == W_sample:
        return top_idx[:W_sample]

    # spread selection over time: pick evenly spaced positions within top_idx
    pos = np.linspace(0, len(top_idx) - 1, W_sample).round().astype(int)
    chosen = top_idx[pos]
    return chosen

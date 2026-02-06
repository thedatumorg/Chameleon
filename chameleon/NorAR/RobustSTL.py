import numpy as np 
import math

from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import toeplitz
from statsmodels.nonparametric.smoothers_lowess import lowess


# ----------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------
def bilateral_filter(j, t, y_j, y_t, delta1=1.0, delta2=1.0):
    idx1 = -1.0 * (abs(j - t) ** 2.0) / (2.0 * delta1 ** 2)
    idx2 = -1.0 * (abs(y_j - y_t) ** 2.0) / (2.0 * delta2 ** 2)
    return math.exp(idx1) * math.exp(idx2)


def get_neighbor_idx(total_len, target_idx, H=3):
    """
    Let i = target_idx.
    Then, return i-H, ..., i, ..., i+H, (i+H+1)
    """
    return [np.max([0, target_idx-H]), np.min([total_len, target_idx+H+1])]


def get_neighbor_range(total_len, target_idx, H=3):
    start_idx, end_idx = get_neighbor_idx(total_len, target_idx, H)
    return np.arange(start_idx, end_idx)


def get_season_idx(total_len, target_idx, T=10, K=2, H=5):
    num_season = np.min([K, int(target_idx/T)])
    if target_idx < T:
        key_idxs = target_idx + np.arange(0, num_season+1)*(-1*T)
    else:        
        key_idxs = target_idx + np.arange(1, num_season+1)*(-1*T)
    
    idxs = list(map(lambda idx: get_neighbor_range(total_len, idx, H), key_idxs))
    season_idxs = []
    for item in idxs:
        season_idxs += list(item)
    season_idxs = np.array(season_idxs)
    return season_idxs


def get_relative_trends(delta_trends):
    init_value = np.array([0])
    idxs = np.arange(len(delta_trends))
    relative_trends = np.array(list(map(lambda idx: np.sum(delta_trends[:idx]), idxs)))
    relative_trends = np.concatenate([init_value, relative_trends])
    return relative_trends


def get_toeplitz(shape, entry):
    h, w = shape
    num_entry = len(entry)
    assert np.ndim(entry) < 2
    if num_entry < 1:
        return np.zeros(shape)
    row = np.concatenate([entry[:1], np.zeros(h-1)])
    col = np.concatenate([np.array(entry), np.zeros(w-num_entry)])
    return toeplitz(row, col)


# ----------------------------------------------------------------------
# Denoising
# ----------------------------------------------------------------------
def denoise_step(sample, H=3, dn1=1., dn2=1.):
    sample = np.asarray(sample, dtype=float)

    def get_denoise_value(idx):
        start_idx, end_idx = get_neighbor_idx(len(sample), idx, H)
        idxs = np.arange(start_idx, end_idx)
        weight_sample = sample[idxs]

        weights = np.array(list(map(
            lambda j: bilateral_filter(j, idx, sample[j], sample[idx], dn1, dn2),
            idxs
        )), dtype=float)
        num = np.sum(weight_sample * weights)
        den = np.sum(weights)
        if den <= 1e-12 or not np.isfinite(den):
            return float(sample[idx])
        return float(num / den)

    idx_list = np.arange(len(sample))
    denoise_sample = np.array(list(map(get_denoise_value, idx_list)), dtype=float)
    return denoise_sample


def trend_extraction_LOESS(sample, season_len, reg1=10.0, reg2=0.5):
    sample = np.asarray(sample, dtype=float)
    n = sample.shape[0]

    if n <= 2:
        trend = sample.copy()
        return sample - trend, trend

    if season_len is None or season_len <= 0 or season_len >= n:
        base_win = max(3, int(0.1 * n))
    else:
        base_win = max(3, int(season_len))

    smooth_factor = 1.0 + math.log10(max(reg1, 1.0))
    win_len = int(base_win * smooth_factor)
    win_len = max(3, min(win_len, n))

    frac = win_len / float(n)
    frac = min(0.95, max(0.05, frac))

    x = np.arange(n, dtype=float)
    trend = lowess(sample, x, frac=frac, it=2, return_sorted=False)
    trend = np.asarray(trend, dtype=float)

    detrended = sample - trend
    return detrended, trend


def trend_extraction_l1(
    sample,
    season_len,
    reg1=10.0,
    reg2=0.5,
    max_iter=8,
    tol=1e-4,
    eps=1e-3):

    sample = np.asarray(sample, dtype=float)
    n = sample.shape[0]

    if n <= 2:
        trend = sample.copy()
        return sample - trend, trend

    main_diag = np.ones(n - 2, dtype=float)
    D2 = diags(
        diagonals=[ main_diag, -2.0 * main_diag, main_diag ],
        offsets=[0, 1, 2],
        shape=(n - 2, n),
        format="csr"
    )
    K = (D2.T @ D2).tocsr()
    I = eye(n, format="csr")

    A0 = I + reg1 * K
    trend = spsolve(A0, sample)

    for it in range(max_iter):
        r = sample - trend
        abs_r = np.abs(r)
        w = 1.0 / np.maximum(abs_r, eps)

        if not np.isfinite(w).all() or w.sum() == 0:
            break

        W = diags(w, offsets=0, shape=(n, n), format="csr")
        rhs = w * sample

        A = W + reg1 * K
        new_trend = spsolve(A, rhs)

        num = np.linalg.norm(new_trend - trend)
        den = np.linalg.norm(trend) + 1e-12
        rel_change = num / den

        trend = new_trend
        if rel_change < tol:
            break

    detrended = sample - trend
    return detrended, trend


# ----------------------------------------------------------------------
# Seasonality extraction: default (local bilateral)
# ----------------------------------------------------------------------
def seasonality_extraction(sample, season_len=10, K=2, H=5, ds1=50., ds2=1.):

    sample = np.asarray(sample, dtype=float)
    sample_len = len(sample)
    idx_list = np.arange(sample_len)

    def get_season_value(idx):
        idxs = get_season_idx(sample_len, idx, season_len, K, H)
        if idxs.size == 0:
            return sample[idx]

        weight_sample = sample[idxs]
        weights = np.array([
            bilateral_filter(j, idx, sample[j], sample[idx], ds1, ds2)
            for j in idxs
        ], dtype=float)

        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        denom = weights.sum()

        if not np.isfinite(denom) or denom <= 1e-12:
            return float(weight_sample.mean())

        num = np.sum(weight_sample * weights)
        return float(num / denom)

    seasons_tilda = np.array([get_season_value(i) for i in idx_list], dtype=float)
    return seasons_tilda


def trend_extraction_for_AD(
    sample,
    season_len,
    reg1=50.0,
    reg2=0.5,
    base_weights=None,
    max_iter=8,
    tol=1e-4,
    eps=1e-3,
):

    sample = np.asarray(sample, dtype=float)
    n = sample.shape[0]

    if n <= 2:
        trend = sample.copy()
        return sample - trend, trend

    if base_weights is None:
        base_weights = np.ones(n, dtype=float)
    else:
        base_weights = np.asarray(base_weights, dtype=float)
        base_weights = np.clip(base_weights, 0.0, 1.0)

    main_diag = np.ones(n - 2, dtype=float)
    D2 = diags(
        diagonals=[ main_diag, -2.0 * main_diag, main_diag ],
        offsets=[0, 1, 2],
        shape=(n - 2, n),
        format="csr"
    )
    K = (D2.T @ D2).tocsr()
    I = eye(n, format="csr")

    A0 = I + reg1 * K
    trend = spsolve(A0, sample)

    for it in range(max_iter):
        r = sample - trend
        abs_r = np.abs(r)

        w_l1 = 1.0 / np.maximum(abs_r, eps)
        w = base_weights * w_l1
        if not np.isfinite(w).all() or w.sum() == 0:
            break

        W = diags(w, offsets=0, shape=(n, n), format="csr")
        rhs = w * sample

        A = W + reg1 * K
        new_trend = spsolve(A, rhs)

        num = np.linalg.norm(new_trend - trend)
        den = np.linalg.norm(trend) + 1e-12
        rel_change = num / den

        trend = new_trend
        if rel_change < tol:
            break

    detrended = sample - trend
    return detrended, trend


def _weighted_median(values, weights):

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = (weights > 0) & np.isfinite(values) & np.isfinite(weights)
    if not np.any(mask):
        return float(np.nan)

    v = values[mask]
    w = weights[mask]

    idx = np.argsort(v)
    v_sorted = v[idx]
    w_sorted = w[idx]
    cumsum = np.cumsum(w_sorted)
    cutoff = 0.5 * np.sum(w_sorted)

    median_idx = np.searchsorted(cumsum, cutoff)
    median_idx = min(median_idx, len(v_sorted) - 1)
    return float(v_sorted[median_idx])


def seasonality_extraction_for_AD(
    detrended,
    season_len,
    weights=None,
    min_weight=0.1,
):

    detrended = np.asarray(detrended, dtype=float)
    n = len(detrended)
    if season_len is None or season_len <= 0 or season_len > n:
        # 不做季节建模，全部为 0
        seasons_tilda = np.zeros_like(detrended)
        return seasons_tilda

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = np.clip(weights, 0.0, 1.0)

    pattern = np.zeros(season_len, dtype=float)
    pattern_filled = np.zeros(season_len, dtype=bool)

    for k in range(season_len):
        idxs = np.arange(k, n, season_len)
        if idxs.size == 0:
            continue

        vals = detrended[idxs]
        wts = weights[idxs]

        mask = wts >= min_weight
        if not np.any(mask):
            continue

        val_sel = vals[mask]
        w_sel = wts[mask]

        med = _weighted_median(val_sel, w_sel)
        if np.isfinite(med):
            pattern[k] = med
            pattern_filled[k] = True

    if not np.any(pattern_filled):
        seasons_tilda = np.zeros_like(detrended)
        return seasons_tilda

    for k in range(season_len):
        if pattern_filled[k]:
            continue
        left = k - 1
        while left >= 0 and not pattern_filled[left]:
            left -= 1
        right = k + 1
        while right < season_len and not pattern_filled[right]:
            right += 1

        if left < 0 and right >= season_len:
            pattern[k] = 0.0
        elif left < 0:
            pattern[k] = pattern[right]
        elif right >= season_len:
            pattern[k] = pattern[left]
        else:
            alpha = (k - left) / float(right - left)
            pattern[k] = (1 - alpha) * pattern[left] + alpha * pattern[right]

    seasons_tilda = np.zeros_like(detrended)
    for t in range(n):
        seasons_tilda[t] = pattern[t % season_len]

    return seasons_tilda


def update_weights_for_AD(
    residual,
    prev_weights=None,
    c1=2.0,
    c2=4.0,
    alpha=0.7,
):

    r = np.asarray(residual, dtype=float)
    n = len(r)

    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-12
    sigma = 1.4826 * mad

    z = np.abs(r) / (sigma + 1e-12)

    raw_w = np.ones(n, dtype=float)
    # z <= c1 -> 1
    # z >= c2 -> 0
    # c1 < z < c2 -> 线性下降
    mask_mid = (z > c1) & (z < c2)
    raw_w[z >= c2] = 0.0
    raw_w[mask_mid] = (c2 - z[mask_mid]) / (c2 - c1)

    raw_w = np.clip(raw_w, 0.0, 1.0)

    if prev_weights is None:
        new_w = raw_w
    else:
        prev_weights = np.asarray(prev_weights, dtype=float)
        prev_weights = np.clip(prev_weights, 0.0, 1.0)
        new_w = alpha * prev_weights + (1.0 - alpha) * raw_w

    return new_w


# ----------------------------------------------------------------------
# Adjustment & convergence
# ----------------------------------------------------------------------
def adjustment(sample, relative_trends, seasons_tilda, season_len):
    sample = np.asarray(sample, dtype=float)
    seasons_tilda = np.asarray(seasons_tilda, dtype=float)
    relative_trends = np.asarray(relative_trends, dtype=float)

    if season_len is None or season_len <= 0:
        trends_hat = relative_trends.copy()
        seasons_hat = seasons_tilda.copy()
        remainders_hat = sample - trends_hat - seasons_hat
        return [trends_hat, seasons_hat, remainders_hat]

    num_season = int(len(sample) / season_len)
    if num_season <= 0:
        trend_init = float(np.mean(seasons_tilda))
    else:
        trend_init = float(np.mean(seasons_tilda[:season_len * num_season]))

    trends_hat = relative_trends + trend_init
    seasons_hat = seasons_tilda - trend_init
    remainders_hat = sample - trends_hat - seasons_hat
    return [trends_hat, seasons_hat, remainders_hat]


def check_converge_criteria(prev_remainders, remainders):
    prev_remainders = np.asarray(prev_remainders, dtype=float)
    remainders = np.asarray(remainders, dtype=float)
    diff = np.sqrt(np.mean(np.square(remainders-prev_remainders)))
    return diff < 1e-10


# ----------------------------------------------------------------------
# Core Robust STL for a single series
# ----------------------------------------------------------------------
def _RobustSTL(input, season_len, reg1=10.0, reg2=0.5, 
               K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.,
               max_iter=10,
               mode="default"):

    sample = np.asarray(input, dtype=float)
    trial = 1
    previous_remainders = None

    weights = None
    if mode == "ad_strict":
        weights = np.ones_like(sample, dtype=float)

    while True:
        print(f"[Iteration {trial}/{max_iter}] mode={mode}")

        # Step 1: Denoising
        denoise_sample = denoise_step(sample, H, dn1, dn2)

        # Step 2: Trend extraction
        if mode == "ad_strict":
            detrend_sample, trend_est = trend_extraction_for_AD(
                denoise_sample,
                season_len,
                reg1=reg1,
                reg2=reg2,
                base_weights=weights,
                max_iter=8,
                tol=1e-4,
                eps=1e-3,
            )
        else:
            detrend_sample, trend_est = trend_extraction_LOESS(
                denoise_sample, season_len, reg1, reg2
            )

        # Step 3: Seasonality extraction
        if mode == "ad_strict":
            seasons_tilda = seasonality_extraction_for_AD(
                detrend_sample,
                season_len,
                weights=weights,
                min_weight=0.1,
            )
        else:
            seasons_tilda = seasonality_extraction(
                detrend_sample, season_len, K, H, ds1, ds2
            )

        # Step 4: Adjustment
        trends_hat, seasons_hat, remainders_hat = adjustment(
            sample, trend_est, seasons_tilda, season_len
        )

        # Step 4.5: update weights (only in ad_strict)
        if mode == "ad_strict":
            weights = update_weights_for_AD(
                remainders_hat,
                prev_weights=weights,
                c1=2.0,
                c2=4.0,
                alpha=0.7,
            )

        # Step 5: Check convergence
        if previous_remainders is not None:
            converged = check_converge_criteria(previous_remainders, remainders_hat)
            if converged:
                return [input, trends_hat, seasons_hat, remainders_hat]

        if trial >= max_iter:
            return [input, trends_hat, seasons_hat, remainders_hat]

        previous_remainders = remainders_hat.copy()
        sample = trends_hat + seasons_hat + remainders_hat
        trial += 1


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def RobustSTL(input, season_len, reg1=10.0, reg2= 0.5,
              K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.,
              max_iter=10,
              mode="ad_strict"):

    arr = np.asarray(input)

    # 1D
    if arr.ndim < 2:
        return _RobustSTL(arr, season_len, reg1, reg2,
                          K, H, dn1, dn2, ds1, ds2,
                          max_iter=max_iter,
                          mode=mode)
    
    # 2D with shape [T, 1]
    if arr.ndim == 2 and arr.shape[1] == 1:
        return _RobustSTL(arr[:,0], season_len, reg1, reg2,
                          K, H, dn1, dn2, ds1, ds2,
                          max_iter=max_iter,
                          mode=mode)
    
    # 2D [N, T] or 3D [N, T, 1]
    if arr.ndim == 2 or arr.ndim == 3:
        if arr.ndim == 3 and arr.shape[2] > 1:
            print("[!] Valid input series shape: [# of Series, # of Time Steps] or [# of series, # of Time Steps, 1]")
            raise ValueError("Invalid input shape for RobustSTL.")
        elif arr.ndim == 3:
            arr = arr[:,:,0]

        num_series = arr.shape[0]
        input_list = [arr[i,:] for i in range(num_series)]
        
        from pathos.multiprocessing import ProcessingPool as Pool
        p = Pool(num_series)

        def run_RobustSTL_single(_input):
            return _RobustSTL(_input, season_len, reg1, reg2,
                              K, H, dn1, dn2, ds1, ds2,
                              max_iter=max_iter,
                              mode=mode)

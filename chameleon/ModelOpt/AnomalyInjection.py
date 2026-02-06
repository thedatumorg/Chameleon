#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import trange
from scipy.stats import bernoulli, norm
from sklearn.neighbors import NearestNeighbors
from scipy.signal import fftconvolve, find_peaks
from scipy import interpolate
from sklearn.metrics import average_precision_score
import contextlib
import io
from TSB_AD.model_wrapper import Semisupervise_AD_Pool, Unsupervise_AD_Pool, run_Semisupervise_AD, run_Unsupervise_AD
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict


# =========================
#   1) spikes        -> impulse/burst anomalies (localized, amplitude-calibrated)
#   2) wander        -> level shift / trend drift (step + drift, optionally persistent)
#   3) speedup       -> time warp / frequency change (fixed-length warp + resample back)
#   4) contextual    -> local shape corruption (affine distortion + optional smoothing)
# =========================

ANOMALY_PARAM_GRID = {
    "spikes": {
        "anomaly_type": ["spikes"],
        "random_parameters": [False],
        "max_anomaly_length": [4],
        "anomaly_size_type": ["mae"],
        "feature_id": [None],
        "correlation_scaling": [3],
        "scale": [2.0],
        "burst_frac": [1.0],
        "burst_k": [1, 3],
    },
    "wander": {
        "anomaly_type": ["wander"],
        "random_parameters": [False],
        "max_anomaly_length": [4],
        "anomaly_size_type": ["mae"],
        "feature_id": [None],
        "correlation_scaling": [3],
        "baseline": [-0.5, 0.5],
        "step_prob": [0.7],
        "persistent": [True],
    },
    "speedup": {
        "anomaly_type": ["speedup"],
        "random_parameters": [False],
        "max_anomaly_length": [4],
        "anomaly_size_type": ["mae"],
        "feature_id": [None],
        "correlation_scaling": [3],
        "speed": [0.7, 1.4],
    },
    "contextual": {
        "anomaly_type": ["contextual"],
        "random_parameters": [False],
        "max_anomaly_length": [4],
        "anomaly_size_type": ["mae"],
        "feature_id": [None],
        "correlation_scaling": [3],
        "scale": [1.0],
        "ma_window": [0, 4],
    },
}


def _constant_timseries(T: np.ndarray) -> bool:
    return np.all(T == T[0])


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    w = int(w)
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w), "same") / w


class InjectAnomalies:
    """
    A compact, higher-quality anomaly injector.
    Input convention: T is shaped (n_features, n_time).
    """

    def __init__(
        self,
        random_state: int = 0,
        verbose: bool = False,
        max_window_size: int = 128,
        min_window_size: int = 8,
    ):
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)
        self.verbose = bool(verbose)

        self.max_window_size = int(max_window_size)
        self.min_window_size = int(min_window_size)
        assert self.max_window_size > self.min_window_size, (
            "Maximum window size must be greater than the minimum window size."
        )

        # Option A only
        self._VALID_ANOMALY_TYPES = ["spikes", "wander", "speedup", "contextual"]

    def __str__(self):
        obj = {
            "random_state": self.random_state,
            "anomaly_types": self._VALID_ANOMALY_TYPES,
            "verbosity": self.verbose,
            "max_window_size": self.max_window_size,
            "min_window_size": self.min_window_size,
        }
        return f"InjectAnomaliesObject: {obj}"

    def compute_crosscorrelation(self, T: np.ndarray) -> np.ndarray:
        # T: (C, L)
        return np.corrcoef(T)

    def compute_window_size(self, ts_1d: np.ndarray) -> int:
        # robust heuristic: use autocorr peak distance when possible
        ts_1d = (ts_1d - ts_1d.mean()).squeeze()
        autocorr = fftconvolve(ts_1d, ts_1d, mode="same")
        self.peaks, _ = find_peaks(autocorr, distance=self.min_window_size)

        try:
            window_size = int(np.diff(self.peaks).mean())
            window_size = min(max(window_size, self.min_window_size), self.max_window_size)
        except Exception:
            window_size = self.max_window_size

        if self.verbose:
            print(f"Window size: {window_size}")
        return window_size

    # ---------- building blocks ----------

    def inject_contextual_delta_root(self, root_seg: np.ndarray, scale: float) -> np.ndarray:
        """
        Local shape corruption on root channel:
        delta = a*x + b - x  (affine distortion around the segment)
        """
        # a centered around 1, b around 0
        a = self.rng.normal(loc=1.0, scale=float(scale))
        b = self.rng.normal(loc=0.0, scale=float(scale))
        return a * root_seg + b - root_seg

    def compute_anomaly_properties(self, T: np.ndarray, max_anomaly_length: int) -> None:
        """
        Decide anomaly start/end.
        Modified: force anomaly segment to lie in the second half of the series.
        """
        _, n_time = T.shape

        # estimate cycle length on the selected feature
        self.estimated_window_size = self.compute_window_size(T[self.anomalous_feature, :])

        # sample anomaly length in units of estimated_window_size
        max_len = max(2, int(max_anomaly_length))
        self.anomaly_length = int(self.rng.integers(1, max_len + 1))

        # span in points
        span = int(self.anomaly_length * max(self.estimated_window_size, 1))
        span = max(1, min(span, n_time))  # safety

        # --- enforce: anomaly starts in second half ---
        half = n_time // 2

        # valid start range so that [start, start+span) stays inside [0, n_time)
        start_min = half
        start_max = max(start_min, n_time - span)  # inclusive upper bound candidate

        # default / fallback
        start = start_min

        # try peak-based anchor first (if available), but clamp to second-half feasible range
        n_peaks = len(getattr(self, "peaks", []))
        if n_peaks >= 1:
            # choose peak index preferentially from peaks that lie in second half
            peaks = np.asarray(self.peaks, dtype=int)
            peaks_2nd = peaks[peaks >= half]

            if peaks_2nd.size > 0:
                peak_idx = int(self.rng.choice(peaks_2nd))
            else:
                # no peaks in second half: fallback to any peak
                peak_idx = int(peaks[self.rng.integers(0, n_peaks)])

            start = int(peak_idx - self.estimated_window_size // 2)

        else:
            # random start in the feasible second-half range
            if start_max > start_min:
                start = int(self.rng.integers(start_min, start_max + 1))
            else:
                start = start_min

        # clamp start to feasible second-half range
        start = int(np.clip(start, start_min, start_max))

        self.anomaly_start = start
        self.anomaly_end = int(min(self.anomaly_start + span, n_time))

        if self.verbose:
            print(f"Anomaly start: {self.anomaly_start} end: {self.anomaly_end} (half={half}, span={span})")


    def _build_correlation_vec(self, T: np.ndarray, correlation_scaling: int) -> np.ndarray:
        """
        correlation_vec[i] = sign(r_i) * |r_i|^(1/correlation_scaling)
        Used to propagate injected delta from root channel to others.
        """
        if T.shape[0] <= 1:
            return np.ones(1, dtype=float)

        corr = self.compute_crosscorrelation(T)[self.anomalous_feature]
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        gamma = max(1, int(correlation_scaling))
        corr_vec = np.sign(corr) * np.power(np.abs(corr), 1.0 / gamma)
        corr_vec = np.nan_to_num(corr_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return corr_vec.astype(float)

    # ---------- main API ----------

    def inject_anomalies(
        self,
        T: np.ndarray,
        anomaly_type: str = "contextual",
        random_parameters: bool = False,
        max_anomaly_length: int = 4,
        anomaly_size_type: str = "mae",
        feature_id: int | None = None,
        correlation_scaling: int = 5,
        anomaly_propensity: float = 0.5,  # kept for API compatibility
        scale: float = 2.0,
        speed: float = 2.0,
        noise_std: float = 0.05,          # kept for API compatibility
        amplitude_scaling: float = 2.0,   # kept for API compatibility
        constant_type: str = "noisy_0",   # kept for API compatibility
        constant_quantile: float = 0.75, # kept for API compatibility
        baseline: float = 0.2,
        ma_window: int = 0,
        # new controls (safe defaults)
        burst_frac: float = 0.10,
        burst_k: int = 2,
        step_prob: float = 0.5,
        persistent: bool = True,
    ):
        """
        Inject anomalies into T (C, L). Returns:
          - timeseries_with_anomalies: (C, L)
          - anomaly_size: (L,)
          - anomaly_labels: (L,) binary
        """
        T = np.asarray(T, dtype=float)
        if T.ndim != 2:
            raise ValueError("T must be a 2D array with shape (n_features, n_time).")

        if anomaly_type not in self._VALID_ANOMALY_TYPES:
            raise ValueError(f"anomaly_type must be in {self._VALID_ANOMALY_TYPES} but {anomaly_type} was passed.")

        n_features, n_time = T.shape
        timeseries_with_anomalies = T.copy()

        # choose anomalous feature (root)
        if feature_id is None:
            features_with_signal = np.where(np.std(T, axis=1) > 0)[0]
            if len(features_with_signal) == 0:
                # degenerate: all constant
                self.anomalous_feature = 0
            else:
                self.anomalous_feature = int(self.rng.choice(features_with_signal))
        else:
            self.anomalous_feature = int(feature_id)

        if self.verbose:
            print(f"Feature {self.anomalous_feature} has an anomaly!")

        # correlation propagation vector (for multivariate coherence)
        correlation_vec = self._build_correlation_vec(T, correlation_scaling=correlation_scaling)
        if self.verbose:
            print(f"Correlation scaling vector: {correlation_vec}")

        # compute start/end and window size
        self.compute_anomaly_properties(T, max_anomaly_length=max_anomaly_length)
        seg_start, seg_end = int(self.anomaly_start), int(self.anomaly_end)
        seg_len = max(1, seg_end - seg_start)

        # ---------- inject by type ----------
        if anomaly_type == "spikes":
            # localized burst spikes on the root channel, then propagate delta to others
            root = self.anomalous_feature
            sigma = float(np.std(T[root, :]) + 1e-6)
            amp = float(scale) * sigma

            # choose burst length (in points)
            burst_frac = float(np.clip(burst_frac, 0.01, 1.0))
            burst_len = max(1, int(round(burst_frac * max(self.estimated_window_size, 1))))
            burst_len = min(burst_len, seg_len)

            # choose k spike positions within the segment
            burst_k = int(max(1, burst_k))
            positions = self.rng.choice(np.arange(seg_start, seg_end), size=min(burst_k, seg_len), replace=False)

            delta_root = np.zeros(n_time, dtype=float)
            for pos in positions:
                # add a short local spike "kernel" (triangular-ish)
                left = max(seg_start, pos - burst_len // 2)
                right = min(seg_end, pos + burst_len // 2 + 1)
                width = max(1, right - left)
                # symmetric weights
                w = 1.0 - np.abs(np.linspace(-1, 1, width))
                delta_root[left:right] += self.rng.normal(loc=0.0, scale=amp) * w

            # propagate to all channels using correlation weights
            delta = np.tile(delta_root, (n_features, 1)) * correlation_vec[:, None]
            timeseries_with_anomalies = timeseries_with_anomalies + delta

            spike_labels = np.zeros(n_time, dtype=int)
            # label the entire local support around each spike
            for pos in positions:
                left = max(seg_start, pos - burst_len // 2)
                right = min(seg_end, pos + burst_len // 2 + 1)
                spike_labels[left:right] = 1

        elif anomaly_type == "wander":
            # level shift / drift on root, then propagate
            root = self.anomalous_feature
            sigma = float(np.std(T[root, :]) + 1e-6)
            off = float(baseline) * sigma  # baseline interpreted relative to sigma

            do_step = (self.rng.random() < float(step_prob))

            delta_root = np.zeros(seg_len, dtype=float)
            if do_step:
                # step change inside segment
                delta_root[:] = off
            else:
                # linear drift across the segment
                delta_root[:] = np.linspace(0.0, off, seg_len)

            # propagate delta into all channels (coherent fault)
            delta = np.tile(delta_root, (n_features, 1)) * correlation_vec[:, None]
            timeseries_with_anomalies[:, seg_start:seg_end] += delta

            if bool(persistent):
                # regime change persists after the anomaly window
                timeseries_with_anomalies[:, seg_end:] += (off * correlation_vec[:, None])

        elif anomaly_type == "speedup":
            # time warp inside segment, then resample back to fixed length
            speed = float(speed)
            speed = max(0.1, speed)

            seg = timeseries_with_anomalies[:, seg_start:seg_end].copy()
            x_old = np.arange(seg_len)

            warped_len = max(2, int(round(seg_len / speed)))
            x_new = np.linspace(0, seg_len - 1, warped_len)

            seg_warped = np.zeros((n_features, warped_len), dtype=float)
            for i in range(n_features):
                interpf = interpolate.interp1d(x_old, seg[i], kind="linear", fill_value="extrapolate")
                seg_warped[i] = interpf(x_new)

            # resample back to seg_len
            x_back = np.linspace(0, warped_len - 1, seg_len)
            x_w = np.arange(warped_len)

            seg_back = np.zeros((n_features, seg_len), dtype=float)
            for i in range(n_features):
                interpf2 = interpolate.interp1d(x_w, seg_warped[i], kind="linear", fill_value="extrapolate")
                seg_back[i] = interpf2(x_back)

            timeseries_with_anomalies[:, seg_start:seg_end] = seg_back

        elif anomaly_type == "contextual":
            # local shape corruption: affine distortion on root + optional smoothing, then propagate
            root = self.anomalous_feature
            root_seg = timeseries_with_anomalies[root, seg_start:seg_end].copy()

            # distortion strength scaled by root sigma for stability
            sigma = float(np.std(T[root, :]) + 1e-6)
            # interpret `scale` as relative strength; convert to absolute perturbation scale
            s = float(scale)

            delta_root = self.inject_contextual_delta_root(root_seg, scale=s)  # already relative
            # calibrate to sigma to avoid exploding if series is tiny
            delta_root = np.clip(delta_root, -5.0 * sigma, 5.0 * sigma)

            # optional smoothing corruption (acts like blur / local averaging)
            mw = int(ma_window)
            if mw and mw > 1:
                sm = moving_average(root_seg, mw)
                delta_root += (sm - root_seg)

            delta = np.tile(delta_root, (n_features, 1)) * correlation_vec[:, None]
            timeseries_with_anomalies[:, seg_start:seg_end] += delta

        else:
            raise RuntimeError(f"Unhandled anomaly_type: {anomaly_type}")

        # ---------- anomaly size ----------
        if anomaly_size_type == "mae":
            anomaly_size = np.mean(np.abs(T - timeseries_with_anomalies), axis=0)
        elif anomaly_size_type == "mse":
            anomaly_size = np.mean((T - timeseries_with_anomalies) ** 2, axis=0)
        elif anomaly_size_type == "nearest":
            nearest = NearestNeighbors(n_neighbors=2, algorithm="ball_tree", metric="cityblock")
            nearest.fit(X=T.T)
            distances, _ = nearest.kneighbors(timeseries_with_anomalies.T)
            anomaly_size = distances.mean(axis=1) / max(1, len(T))
        else:
            raise ValueError("anomaly_size_type must be one of {'mae','mse','nearest'}")

        # ---------- labels (less aggressive dilation) ----------
        # Use a small dilation proportional to anomaly span (capped).
        span = max(1, seg_end - seg_start)
        dilate = int(round(0.10 * span))  # 10% of span
        dilate = max(0, min(dilate, self.min_window_size))

        anomaly_labels = np.zeros(timeseries_with_anomalies.shape[1], dtype=int)
        l0 = max(0, seg_start - dilate)
        l1 = min(timeseries_with_anomalies.shape[1], seg_end + dilate)
        anomaly_labels[l0:l1] = 1

        if anomaly_type == "spikes":
            # spikes use the more precise burst support labels
            anomaly_labels = spike_labels.astype(int)

        return timeseries_with_anomalies, anomaly_size, anomaly_labels

def recall_at_topk(score, label):
    T = len(score)
    k = max(1, int(np.ceil(label.mean() * T)))
    idx = np.argsort(score)[-k:]
    return label[idx].sum() / max(1, label.sum())

def score_separation(score, label):
    s1 = score[label == 1]
    s0 = score[label == 0]
    if len(s1) == 0 or len(s0) == 0:
        return 0.0
    return (np.mean(s1) - np.mean(s0)) / (np.std(s0) + 1e-6)

def combine_metrics_to_utility(m, weights):
    val = 0.0
    wsum = 0.0
    for k, w in weights.items():
        x = m.get(k, np.nan)
        if np.isfinite(x):
            val += float(w) * float(x)
            wsum += float(w)
    if wsum == 0:
        return np.nan
    return val / wsum

def compute_synth_metrics(score, label):
    ap = average_precision_score(label, score) if label.sum() > 0 else np.nan
    r_gt = recall_at_topk(score, label)
    # sep = score_separation(score, label)
    # return {"ap": ap, "recall_topk": r_gt, "sep": sep}
    return {"ap": ap, "recall_topk": r_gt}

def gen_synthetic_performance_list(data, label, Det_pool, verbose=False):
    score_list = []

    for det in Det_pool:

        print('Running:', det)

        Optimal_Det_HP = Optimal_Multi_algo_HP_dict[det] if data.shape[1] > 1 else Optimal_Uni_algo_HP_dict[det]
        for hp in Optimal_Det_HP.items():
            if "window_size" in hp:
                Optimal_Det_HP["window_size"] = 16

        data_train = data[:int(0.5*len(data)), :]
        
        with (contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()):
            if det in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(det, data_train, data, **Optimal_Det_HP)
            elif det in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(det, data, **Optimal_Det_HP)
            else:
                raise Exception(f"{det} is not defined")
        
        if isinstance(output, np.ndarray):
            score = output
        else:
            print('Warning: The output of the detector is not a numpy array. {}.'.format(output))
            score = np.random.rand(len(label))
        
        # Length reconcile
        if len(score) != len(label):
            if len(score) > len(label):
                score = score[:len(label)]
            else:
                # pad by edge
                pad = np.full(len(label) - len(score), score[-1] if len(score) > 0 else 0.0, dtype=np.float64)
                score = np.concatenate([score, pad], axis=0)

        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        score_list.append(score)

    perf_list = []
    metric_weights = {"ap": 0.5, "recall_topk": 0.5}
    for score in score_list:
        m = compute_synth_metrics(score, label)
        u = combine_metrics_to_utility(m, weights=metric_weights)
        # print('Metrics:', m, 'Utility:', u)
        perf_list.append(u)
    return perf_list

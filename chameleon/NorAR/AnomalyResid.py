#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import STL 

from .RobustSTL import RobustSTL

def split_like_original(arr: np.ndarray, window_size: int):
    """
    arr: (T, C)
    return: list of (window_size, C), following your original logic
    """
    T = arr.shape[0]
    if T < window_size:
        pad_len = window_size - T
        arr = np.pad(arr, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        return [arr]

    modulo = T % window_size
    rest = arr[modulo:]
    k = int(rest.shape[0] / window_size)
    chunks = list(np.split(rest, k, axis=0))
    if modulo != 0:
        first_window = arr[:window_size]
        chunks = [first_window] + chunks
    return chunks


def _safe_drop_const_cols(X, thr=1e-12):
    # X: (T, sw)
    std = np.nanstd(X, axis=0)
    keep = std > thr
    if not np.any(keep):
        return X[:, :0], keep  # no columns left
    return X[:, keep], keep

def _safe_pca_reconstruct_matrix(X, n_components, svd_solver="auto", jitter=0.0):
    """
    X: (T, d) sliding matrix with d<=sw after dropping constant cols
    returns: recon (T, d) reconstructed with up to n_components
    """
    # Clean numerical junk
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if jitter > 0.0:
        X = X + jitter * np.random.randn(*X.shape)

    # Effective rank cap for components
    # Upper bound by min(T, d) and also by # nonzero singular values if desired
    n_max = min(n_components, X.shape[0], X.shape[1])
    if n_max <= 0:
        return np.zeros_like(X)

    pca = PCA(n_components=n_max, svd_solver=svd_solver)
    Z = pca.fit_transform(X)           # (T, n_max)
    Z = np.nan_to_num(Z, nan=0.0)
    recon = pca.inverse_transform(Z)   # (T, d)
    recon = np.nan_to_num(recon, nan=0.0)
    return recon


class AnomalyResidualDecomposer(nn.Module):

    def __init__(
        self,
        sliding_window: int = 128,
        num_components: int = 3,
        mode: str = "pca_ad",
        season_len: int = 7,
        robust_fallback: bool = True,
        stl_kwargs: dict | None = None,
        zscore_normalize: bool = True,
    ):
        super().__init__()
        self.sliding_window = int(sliding_window)
        self.zscore_normalize = zscore_normalize

        if num_components == 0:
            self.num_components = int(0.2 * self.sliding_window)
        else:
            self.num_components = int(num_components)

        if self.sliding_window < 3:
            raise ValueError("sliding_window must be >= 3")

        self.mode = mode.lower()
        if self.mode not in ["pca", "pca_ad", "pca_ad_auto", "robust_stl", "stl", "stl_ad"]:
            raise ValueError(
                f"Unknown mode '{mode}', must be "
                f"'pca', 'pca_ad', 'pca_ad_auto', 'robust_stl', 'stl', or 'stl_ad'."
            )

        self.season_len = season_len
        if self.mode in ["robust_stl", "stl", "stl_ad"] and self.season_len is None:
            raise ValueError(
                "season_len must be provided when mode='robust_stl', 'stl', or 'stl_ad'."
            )

        self.robust_fallback = bool(robust_fallback)
        self.stl_kwargs = stl_kwargs or {}

    # ----------------------- shared helpers ----------------------- #
    @staticmethod
    def _moving_average(ts: np.ndarray, k: int) -> np.ndarray:
        # ensure odd window
        if k % 2 == 0:
            k = k + 1
        k = max(3, k)
        kernel = np.ones(k, dtype=np.float64) / float(k)
        return np.convolve(ts, kernel, mode="same")

    @staticmethod
    def _update_anomaly_weights(
        residual: np.ndarray,
        prev_weights: np.ndarray | None = None,
        c1: float = 2.0,
        c2: float = 4.0,
        alpha: float = 0.7,
    ) -> np.ndarray:

        r = np.asarray(residual, dtype=float)
        n = r.shape[0]

        med = np.median(r)
        mad = np.median(np.abs(r - med)) + 1e-12
        sigma = 1.4826 * mad

        z = np.abs(r) / (sigma + 1e-12)

        raw_w = np.ones(n, dtype=float)
        # z <= c1 -> 1
        # z >= c2 -> 0
        mask_mid = (z > c1) & (z < c2)
        raw_w[z >= c2] = 0.0
        raw_w[mask_mid] = (c2 - z[mask_mid]) / (c2 - c1)
        raw_w = np.clip(raw_w, 0.0, 1.0)

        if prev_weights is None:
            return raw_w

        prev_weights = np.asarray(prev_weights, dtype=float)
        prev_weights = np.clip(prev_weights, 0.0, 1.0)
        new_w = alpha * prev_weights + (1.0 - alpha) * raw_w
        return new_w

    # ----------------------- PCA backend -------------------------- #
    def _synthetic_pca_1d(self, ts: np.ndarray, sw: int, num_components: int) -> np.ndarray:
        """
        ts: (L,) -> returns reconstructed 'normal' signal (L,)
        Robust to zero-variance / degenerate windows to avoid total_var=0 warnings.
        """
        L = ts.shape[0]
        if sw > L:
            # pad then unpad
            pad_len = sw - L
            ts_padded = np.pad(ts, (0, pad_len), mode="edge")
            rec = self._synthetic_pca_1d(ts_padded, sw, num_components)
            return rec[:L]

        # Sliding matrix (T, sw)
        T = L - sw + 1
        sm = np.lib.stride_tricks.sliding_window_view(ts, sw)  # (T, sw)

        # Drop constant columns to avoid total_var==0
        sm_f, keep_mask = _safe_drop_const_cols(sm, thr=1e-12)

        if sm_f.shape[1] == 0:
            # Entire window is constant → nothing to reconstruct, return original
            rec_core = sm.mean(axis=1)  # identical across columns
        else:
            # Safe PCA reconstruction
            recon_f = _safe_pca_reconstruct_matrix(
                sm_f,
                n_components=min(num_components, sm_f.shape[1]),
                svd_solver="auto",
                jitter=0.0,
            )
            # Map back to full dimension if some columns were dropped
            recon_full = np.zeros_like(sm)
            recon_full[:, keep_mask] = recon_f
            # For dropped cols, copy the original (or mean) to keep alignment
            if (~keep_mask).any():
                recon_full[:, ~keep_mask] = sm[:, ~keep_mask]
            # Collapse sliding windows back to 1D (row mean)
            rec_core = recon_full.mean(axis=1)

        # Symmetric padding back to length L
        padding = (sw - 1) // 2
        left_pad = padding
        right_pad = L - (T + padding)
        if right_pad < 0:  # guard
            right_pad = padding
        rec = np.pad(rec_core, (left_pad, right_pad), mode="edge")

        # Final alignment
        if rec.shape[0] != L:
            if rec.shape[0] > L:
                rec = rec[:L]
            else:
                rec = np.pad(rec, (0, L - rec.shape[0]), mode="edge")
        return rec

    # ----------------------- Robust STL backend (custom) ---------- #
    def _robust_stl_1d(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        ts: (L,) → returns (normal, residual) both (L,)

        Uses your RobustSTL implementation:
            sample ≈ trend + season + remainder
        We take:
            normal   = trend + season
            residual = remainder
        """
        # RobustSTL returns: [input, trends_hat, seasons_hat, remainders_hat]
        inp, trend, season, remainder = RobustSTL(
            ts,
            season_len=self.season_len,
            **self.stl_kwargs,
        )
        normal = trend + season
        residual = remainder
        return normal, residual

    # ----------------------- statsmodels STL backend -------------- #
    def _stl_statsmodels_1d(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        ts: (L,) → returns (normal, residual) both (L,)

        Uses statsmodels.tsa.seasonal.STL:
            sample ≈ trend + seasonal + resid
        We take:
            normal   = trend + seasonal
            residual = resid
        """
        ts = np.asarray(ts, dtype=float)

        stl = STL(
            ts,
            period=self.season_len,
            robust=True,           # robust version of STL
            **self.stl_kwargs,     # extra STL parameters: seasonal, trend, low_pass, etc.
        )
        res = stl.fit()
        trend = res.trend
        seasonal = res.seasonal
        resid = res.resid

        normal = trend + seasonal
        residual = resid
        return normal, residual

    # ----------------------- STL-AD backend (anomaly-aware STL) --- #
    def _stl_ad_1d(
        self,
        ts: np.ndarray,
        max_outer: int = 3,
        c1: float = 2.0,
        c2: float = 4.0,
        alpha: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray]:

        ts = np.asarray(ts, dtype=float)
        L = ts.shape[0]

        stl = STL(
            ts,
            period=self.season_len,
            robust=True,
            **self.stl_kwargs,
        )
        res = stl.fit()
        trend = res.trend
        seasonal = res.seasonal
        normal = trend + seasonal

        weights = np.ones(L, dtype=float)

        for it in range(max_outer):
            residual = ts - normal
            weights = self._update_anomaly_weights(
                residual,
                prev_weights=weights,
                c1=c1,
                c2=c2,
                alpha=alpha,
            )

            x_clean = weights * ts + (1.0 - weights) * normal

            stl2 = STL(
                x_clean,
                period=self.season_len,
                robust=True,
                **self.stl_kwargs,
            )
            res2 = stl2.fit()
            trend2 = res2.trend
            seasonal2 = res2.seasonal
            normal_new = trend2 + seasonal2

            diff = np.linalg.norm(normal_new - normal) / (np.linalg.norm(normal) + 1e-12)
            normal = normal_new
            if diff < 1e-3:
                break

        residual_final = ts - normal
        return normal, residual_final


    # ----------------------- PCA-AD backend (anomaly-aware PCA) --- #
    def _pca_ad_1d(
        self,
        ts: np.ndarray,
        max_outer: int = 3,
        c1: float = 2.0,
        c2: float = 4.0,
        alpha: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray]:

        ts = np.asarray(ts, dtype=float)
        L = ts.shape[0]

        normal = self._synthetic_pca_1d(
            ts,
            sw=self.sliding_window,
            num_components=self.num_components,
        )

        weights = np.ones(L, dtype=float)

        for it in range(max_outer):
            residual = ts - normal

            weights = self._update_anomaly_weights(
                residual,
                prev_weights=weights,
                c1=c1,
                c2=c2,
                alpha=alpha,
            )

            x_clean = weights * ts + (1.0 - weights) * normal

            normal_new = self._synthetic_pca_1d(
                x_clean,
                sw=self.sliding_window,
                num_components=self.num_components,
            )

            diff = np.linalg.norm(normal_new - normal) / (np.linalg.norm(normal) + 1e-12)
            normal = normal_new
            if diff < 1e-3:
                break

        residual_final = ts - normal
        return normal, residual_final


    def _auto_num_components(
        self,
        sm_f: np.ndarray,
        energy_thresh: float = 0.95,
    ) -> int:
        """
        Given filtered sliding-window matrix sm_f (T, D), choose number of
        principal components based on explained variance threshold.

        - energy_thresh: cumulative variance ratio target (e.g., 0.9 or 0.95)
        - self.num_components is treated as an upper bound if > 0
        """
        T, D = sm_f.shape
        if D == 0:
            return 0

        # Center columns
        X = sm_f - sm_f.mean(axis=0, keepdims=True)

        # SVD: X ≈ U S V^T
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fall back: just use 1 component if SVD fails
            return 1

        eigvals = S ** 2
        total = eigvals.sum()
        if total <= 0:
            # Degenerate case: fall back to 1 or upper bound
            k = 1
        else:
            cum = np.cumsum(eigvals) / total
            # smallest k such that cumulative variance >= energy_thresh
            k = int(np.searchsorted(cum, energy_thresh) + 1)

        # Apply upper bound from self.num_components if it is positive
        # if self.num_components > 0:
        #     k = min(k, self.num_components)

        # Clamp to [1, D]
        k = max(1, min(k, D))
        print(f"Auto-selected PCA components: {k} (target energy: {energy_thresh})")
        return k


    def _synthetic_pca_auto_1d(
        self,
        ts: np.ndarray,
        sw: int,
        energy_thresh: float = 0.95,
    ) -> np.ndarray:
        """
        ts: (L,) -> reconstructed 'normal' signal (L,)
        PCA with automatically chosen number of components based on explained variance.
        """
        L = ts.shape[0]
        if sw > L:
            # pad then unpad
            pad_len = sw - L
            ts_padded = np.pad(ts, (0, pad_len), mode="edge")
            rec = self._synthetic_pca_auto_1d(ts_padded, sw, energy_thresh)
            return rec[:L]

        # Sliding matrix (T, sw)
        T = L - sw + 1
        sm = np.lib.stride_tricks.sliding_window_view(ts, sw)  # (T, sw)

        # Drop constant columns to avoid total_var==0
        sm_f, keep_mask = _safe_drop_const_cols(sm, thr=1e-12)

        if sm_f.shape[1] == 0:
            # Entire window is constant → nothing to reconstruct, return original mean
            rec_core = sm.mean(axis=1)
        else:
            # Decide rank automatically
            n_auto = self._auto_num_components(sm_f, energy_thresh=energy_thresh)

            # Safety: cap by effective dimension
            n_eff = min(n_auto, sm_f.shape[1])

            # Safe PCA reconstruction
            recon_f = _safe_pca_reconstruct_matrix(
                sm_f,
                n_components=n_eff,
                svd_solver="auto",
                jitter=0.0,
            )

            # Map back to full dimension if some columns were dropped
            recon_full = np.zeros_like(sm)
            recon_full[:, keep_mask] = recon_f
            if (~keep_mask).any():
                recon_full[:, ~keep_mask] = sm[:, ~keep_mask]

            # Collapse sliding windows back to 1D (row mean)
            rec_core = recon_full.mean(axis=1)

        # Symmetric padding back to length L
        padding = (sw - 1) // 2
        left_pad = padding
        right_pad = L - (T + padding)
        if right_pad < 0:  # guard
            right_pad = padding
        rec = np.pad(rec_core, (left_pad, right_pad), mode="edge")

        # Final alignment
        if rec.shape[0] != L:
            if rec.shape[0] > L:
                rec = rec[:L]
            else:
                rec = np.pad(rec, (0, L - rec.shape[0]), mode="edge")
        return rec


    def _pca_ad_auto_1d(
        self,
        ts: np.ndarray,
        max_outer: int = 3,
        c1: float = 2.0,
        c2: float = 4.0,
        alpha: float = 0.7,
        energy_thresh: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Anomaly-aware PCA decomposition with automatic rank selection.

        Same iterative scheme as _pca_ad_1d, but each PCA step chooses the
        number of components to reach a target explained variance.
        """
        ts = np.asarray(ts, dtype=float)
        L = ts.shape[0]

        # Initial: PCA with auto-selected components
        normal = self._synthetic_pca_auto_1d(
            ts,
            sw=self.sliding_window,
            energy_thresh=energy_thresh,
        )

        # Initialize weights: all ones
        weights = np.ones(L, dtype=float)

        for it in range(max_outer):
            residual = ts - normal

            # Update [0,1] anomaly weights via MAD-based robust z-score
            weights = self._update_anomaly_weights(
                residual,
                prev_weights=weights,
                c1=c1,
                c2=c2,
                alpha=alpha,
            )

            # Construct cleaned series: anomalies pulled towards normal
            x_clean = weights * ts + (1.0 - weights) * normal

            # Re-run PCA with auto rank on cleaned series
            normal_new = self._synthetic_pca_auto_1d(
                x_clean,
                sw=self.sliding_window,
                energy_thresh=energy_thresh,
            )

            # Convergence check (relative change)
            diff = np.linalg.norm(normal_new - normal) / (np.linalg.norm(normal) + 1e-12)
            normal = normal_new
            if diff < 1e-3:
                break

        residual_final = ts - normal
        return normal, residual_final


    # ----------------------- main forward ------------------------- #
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, L) -> returns (x_normal, x_resid) both (B, C, L)
        """
        if x.dim() != 3:
            raise ValueError("x must be (B, C, L)")

        B, C, L = x.shape
        device = x.device
        dtype = x.dtype

        normals = np.zeros((B, C, L), dtype=np.float64)
        residuals = np.zeros((B, C, L), dtype=np.float64)

        sw = int(self.sliding_window)
        x_cpu = x.detach().cpu().to(torch.float64).numpy()

        for b in range(B):
            for c in range(C):
                ts = x_cpu[b, c]  # (L,)
                try:
                    if self.mode == "pca":
                        normal_i = self._synthetic_pca_1d(ts, sw, self.num_components)
                        residual_i = ts - normal_i
                    
                    elif self.mode == "pca_ad":
                        # Anomaly-aware PCA: down-weight large residuals when estimating normal
                        normal_i, residual_i = self._pca_ad_1d(ts)

                    elif self.mode == "pca_ad_auto":
                        # Anomaly-aware PCA with automatic rank selection
                        normal_i, residual_i = self._pca_ad_auto_1d(ts)

                    elif self.mode == "robust_stl":
                        normal_i, residual_i = self._robust_stl_1d(ts)

                    elif self.mode == "stl":
                        normal_i, residual_i = self._stl_statsmodels_1d(ts)

                    elif self.mode == "stl_ad":
                        normal_i, residual_i = self._stl_ad_1d(ts)

                    else:
                        raise ValueError(f"Unknown mode '{self.mode}'")

                    # NaN check → trigger fallback
                    if np.isnan(normal_i).any() or np.isnan(residual_i).any():
                        raise ValueError("Decomposition resulted in NaN values.")

                except Exception as e:
                    if not self.robust_fallback:
                        raise

                    # robust fallback: moving average with a sane odd window
                    print(
                        f"⚠️ Decomposition failed for sample {b}, channel {c}, "
                        f"mode={self.mode}; falling back to moving average. Error: {e}"
                    )
                    k = max(3, min(9, (sw // 2) * 2 + 1))
                    normal_i = self._moving_average(ts, k=k)
                    residual_i = ts - normal_i

                # alignment safeguards
                if normal_i.shape[0] != ts.shape[0]:
                    m = min(normal_i.shape[0], ts.shape[0])
                    normals[b, c, :m] = normal_i[:m]
                    residuals[b, c, :m] = ts[:m] - normal_i[:m]
                else:
                    normals[b, c, :] = normal_i
                    residuals[b, c, :] = residual_i

        x_normal = torch.from_numpy(normals).to(device=device, dtype=dtype)
        x_resid  = torch.from_numpy(residuals).to(device=device, dtype=dtype)
        
        # mean centering to ensure zero-mean
        # x_normal = x_normal - x_normal.mean(dim=2, keepdim=True)
        # x_resid = x_resid - x_resid.mean(dim=2, keepdim=True)

        # z-socre normalization
        if self.zscore_normalize:
            normal_mean = x_normal.mean(dim=2, keepdim=True)
            normal_std = x_normal.std(dim=2, keepdim=True) + 1e-12
            x_normal = (x_normal - normal_mean) / normal_std  

            resid_mean = x_resid.mean(dim=2, keepdim=True)
            resid_std = x_resid.std(dim=2, keepdim=True) + 1e-12
            x_resid = (x_resid - resid_mean) / resid_std    

        return x_normal, x_resid

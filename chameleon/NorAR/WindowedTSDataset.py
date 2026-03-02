#!/usr/bin/env python3
import os
import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.decomposition import PCA
from pycatch22 import catch22_all
from sklearn.metrics import average_precision_score

import sys
sys.path.append("/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon")

from chameleon.NorAR.AnomalyResid import AnomalyResidualDecomposer

# ------------------------------------------------------------------------
# config: candidate models
# ------------------------------------------------------------------------
CANDIDATE_MODEL_SET = [
    'Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'POLY', 'MatrixProfile', 'KShapeAD', 'SAND',
    'Series2Graph', 'SR', 'Sub_PCA', 'Sub_HBOS', 'Sub_OCSVM',
    'Sub_MCD', 'Sub_KNN', 'KMeansAD_U', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD',
    'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut',
    'TimesNet', 'FITS', 'OFA', 'Lag_Llama', 'Chronos', 'TimesFM',
    'MOMENT_ZS', 'MOMENT_FT'
]
NUM_MODELS = len(CANDIDATE_MODEL_SET)

CANDIDATE_MODEL_SET_M = ['IForest', 'LOF', 'PCA', 'HBOS', 'OCSVM', 'MCD', 'KNN', 'KMeansAD', 'COPOD', 'CBLOF', 'EIF', 'RobustPCA', 'AutoEncoder', 
                    'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 'TimesNet', 'FITS', 'OFA']
NUM_MODELS_M = len(CANDIDATE_MODEL_SET_M)

def num_windows_for_length(T: int, window_size: int) -> int:
    if T < window_size:
        return 1
    modulo = T % window_size
    k = (T - modulo) / window_size
    assert math.ceil(k) == k
    k = int(k)
    if modulo != 0:
        k = k + 1
    return k


def split_like_original(arr: np.ndarray, window_size: int):
    """
    arr: (T, C)
    return: list of (window_size, C)
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


class WindowedTSDataset(Dataset):
    def __init__(self,
                 domain: str,
                 file_list_csv: str,
                 dataset_dir: str,
                 metric_dir: str,
                 metric_name: str,
                 window_size: int):
        self.dataset_dir = dataset_dir
        self.window_size = window_size
        self.metric_name = metric_name

        # read file list
        if domain == 'ID':
            df = pd.read_csv(file_list_csv)
            self.files = df['file_name'].values.tolist()
        else:
            df = pd.read_csv(file_list_csv)
            if domain in df["domain_name"].unique():
                self.files = df[df["domain_name"] != domain]['file_name'].values.tolist()
            else:
                print(f'No {domain}')
                exit()

        # pre-load all candidate metric tables into dict[file] -> vector
        self.file2label = self._load_all_labels(metric_dir, self.files, metric_name)

        # build global index
        self.index_map = []  # list of (file_name, window_idx)
        self.file2nwin = {}
        for fname in self.files:
            fpath = os.path.join(dataset_dir, fname)
            ts_df = pd.read_csv(fpath).dropna()
            data = ts_df.iloc[:, 0:-1].values.astype(np.float32)  # (T, C)
            T = data.shape[0]
            nwin = num_windows_for_length(T, window_size)
            self.file2nwin[fname] = nwin
            for w in range(nwin):
                self.index_map.append((fname, w))

    def _load_all_labels(self, metric_dir, file_list, metric_name):
        """
        returns: dict[file_name] -> np.array(M,)
        """
        file2label = {f: [] for f in file_list}
        for det in CANDIDATE_MODEL_SET:
            det_path = os.path.join(metric_dir, det + '.csv')
            det_df = pd.read_csv(det_path)
            for f in file_list:
                v = det_df[det_df['file'] == f][metric_name].values[0]
                file2label[f].append(float(v))
        for f in file2label:
            file2label[f] = np.array(file2label[f], dtype=np.float32)
        return file2label

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fname, widx = self.index_map[idx]
        fpath = os.path.join(self.dataset_dir, fname)
        df = pd.read_csv(fpath).dropna()
        data = df.iloc[:, 0:-1].values.astype(np.float32)  # (T, C)

        # per-series standardization
        data = StandardScaler().fit_transform(data)   # (T, C)

        # window both normal & residual series
        windows_data = split_like_original(data, self.window_size)  # list of (L, C)
        win_n = windows_data[widx]  # (L, C)

        # (C, L) for Conv1d / model
        x_n_t = torch.from_numpy(win_n).permute(1, 0).contiguous()

        # label
        y = torch.from_numpy(self.file2label[fname])  # (M,)

        # return a pair for the model: (normal, resid)
        return x_n_t, y


class WindowedTSDatasetPrecomputed(Dataset):

    def __init__(self,
                 domain: str,
                 file_list_csv: str,
                 metric_dir: str,
                 metric_name: str,
                 precomputed_dir: str):
        self.metric_name = metric_name
        self.precomputed_dir = precomputed_dir

        df = pd.read_csv(file_list_csv)
        if domain == 'ID':
            self.files = df['file_name'].values.tolist()
        else:
            if domain in df["domain_name"].unique():
                self.files = df[df["domain_name"] != domain]['file_name'].values.tolist()
            else:
                print(f'No {domain}')
                raise SystemExit(1)

        # labels
        self.file2label = self._load_all_labels(metric_dir, self.files, metric_name)

        self.windows_n = {}  # fname -> np.ndarray (B, L, C)
        self.windows_r = {}
        self.index_map = []  # list of (fname, widx)

        for fname in self.files:
            npz_path = os.path.join(self.precomputed_dir, f"{fname}.npz")
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"Precomputed file not found: {npz_path} "
                                        f"(Please run precompute_resid_windows.py first)")
            data = np.load(npz_path)
            win_n = data["windows_n"]  # (B, L, C)
            win_r = data["windows_r"]  # (B, L, C)

            assert win_n.shape == win_r.shape
            B = win_n.shape[0]

            self.windows_n[fname] = win_n
            self.windows_r[fname] = win_r

            for widx in range(B):
                self.index_map.append((fname, widx))

        print(f"[Dataset] total windows = {len(self.index_map)}")

    def _load_all_labels(self, metric_dir, file_list, metric_name):
        file2label = {f: [] for f in file_list}
        for det in CANDIDATE_MODEL_SET:
            det_path = os.path.join(metric_dir, det + '.csv')
            det_df = pd.read_csv(det_path)
            for f in file_list:
                v = det_df[det_df['file'] == f][metric_name].values[0]
                file2label[f].append(float(v))
        for f in file2label:
            file2label[f] = np.array(file2label[f], dtype=np.float32)
        return file2label

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fname, widx = self.index_map[idx]
        win_n = self.windows_n[fname][widx]  # (L, C)
        win_r = self.windows_r[fname][widx]  # (L, C)

        x_n_t = torch.from_numpy(win_n).permute(1, 0).contiguous()
        x_r_t = torch.from_numpy(win_r).permute(1, 0).contiguous()

        y = torch.from_numpy(self.file2label[fname])  # (M,)

        return (x_n_t, x_r_t), y

def split_like_original_1d(arr: np.ndarray, window_size: int):
    """
    1D version of your split_like_original, for score vectors and label vectors.
    arr: (T,)
    return: list of (L,)
    """
    T = arr.shape[0]
    if T < window_size:
        pad_len = window_size - T
        arr = np.pad(arr, (0, pad_len), mode='constant', constant_values=0)
        return [arr]

    modulo = T % window_size
    rest = arr[modulo:]
    k = int(rest.shape[0] / window_size)
    chunks = list(np.split(rest, k, axis=0))
    if modulo != 0:
        first_window = arr[:window_size]
        chunks = [first_window] + chunks
    return chunks


def window_metric_ap_or_sep(score_w: np.ndarray, label_w: np.ndarray) -> float:
    """
    score_w: (L,) float
    label_w: (L,) int {0,1}
    Returns a robust metric per window.

    - If window has both pos and neg: AP (average precision)
    - Else: separation-like proxy (reward low scores on all-neg, high scores on all-pos)
    """
    y = label_w.astype(np.int32)
    s = score_w.astype(np.float32)

    n_pos = int(y.sum())
    if 0 < n_pos < len(y):
        # standard AUC-PR (AP)
        return float(average_precision_score(y, s))

    # degenerate: all 0 or all 1
    if n_pos == 0:
        # all normal window: lower scores are better
        # return negative mean score (so "higher is better" in training if you like)
        return float(-np.mean(s))
    else:
        # all anomaly window: higher scores are better
        return float(np.mean(s))


class WindowedTSDatasetPrecomputed_FG(Dataset):

    def __init__(self,
                 domain: str,
                 file_list_csv: str,
                 metric_dir: str,
                 metric_name: str,
                 precomputed_dir: str,
                 score_dir: str = '/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/TSB-AutoAD/resource/score/uni/',
                 window_metric: str = "ap",         # "ap" (AP with fallback sep) or "sep" only
                 return_file_metric: bool = True,  # if True: also return original file metric
                 cache_scores: bool = True):       # optionally cache per-file scores
        self.metric_name = metric_name
        self.precomputed_dir = precomputed_dir
        self.score_dir = score_dir
        self.window_metric = window_metric
        self.return_file_metric = return_file_metric
        self.cache_scores = cache_scores

        df = pd.read_csv(file_list_csv)
        if domain == 'ID':
            self.files = df['file_name'].values.tolist()
        else:
            if domain in df["domain_name"].unique():
                self.files = df[df["domain_name"] != domain]['file_name'].values.tolist()
            else:
                raise SystemExit(f'No {domain}')

        # file-level labels (M,) (optional to return)
        self.file2label = self._load_all_labels(metric_dir, self.files, metric_name)

        self.windows_n = {}   # fname -> (B,L,C)
        self.windows_r = {}
        self.win_lab_pt = {}  # fname -> (B,L) int
        self.start_idxs = {}  # fname -> (B,) int, optional
        self.index_map = []   # list of (fname, widx)

        # optional score cache:
        # score_cache[fname][det] = np.ndarray(T,)
        self.score_cache = {}  # fname -> dict(det->score)

        for fname in self.files:
            npz_path = os.path.join(self.precomputed_dir, f"{fname}.npz")
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"Precomputed file not found: {npz_path}")

            data = np.load(npz_path, allow_pickle=True)
            win_n = data["windows_n"]
            win_r = data["windows_r"]
            assert win_n.shape == win_r.shape

            if "window_labels_point" not in data:
                raise ValueError(f"{npz_path} missing window_labels_point. Re-run precompute.")

            win_lab_pt = data["window_labels_point"].astype(np.int32)  # (B,L)

            self.windows_n[fname] = win_n
            self.windows_r[fname] = win_r
            self.win_lab_pt[fname] = win_lab_pt

            if "start_idxs" in data:
                self.start_idxs[fname] = data["start_idxs"].astype(np.int64)

            B = win_n.shape[0]
            if win_lab_pt.shape[0] != B:
                raise RuntimeError(f"Label/window mismatch for {fname}: {win_lab_pt.shape} vs {win_n.shape}")

            for widx in range(B):
                self.index_map.append((fname, widx))

            if self.cache_scores:
                self.score_cache[fname] = {det: self._load_score(det, fname) for det in CANDIDATE_MODEL_SET}

        print(f"[Dataset] total windows = {len(self.index_map)}")

    def _load_score(self, det: str, fname: str) -> np.ndarray:
        score_name = fname.split(".")[0]
        score_path = os.path.join(self.score_dir, det, f"{score_name}.npy")

        if os.path.exists(score_path):
            return np.load(score_path).astype(np.float32)

        # fallback: random score
        # print(f"[WARN] score missing for {det}/{fname}, using random score")

        # infer T from stored window metadata
        # safest is to use original length if saved, else B*L
        if fname in self.start_idxs:
            T = int(self.start_idxs[fname][-1] + self.win_lab_pt[fname].shape[1])
        else:
            B, L = self.win_lab_pt[fname].shape
            T = B * L

        return np.random.randn(T).astype(np.float32)

    def _get_score(self, det: str, fname: str) -> np.ndarray:
        if self.cache_scores:
            return self.score_cache[fname][det]
        return self._load_score(det, fname)

    def _load_all_labels(self, metric_dir, file_list, metric_name):
        file2label = {f: [] for f in file_list}
        for det in CANDIDATE_MODEL_SET:
            det_path = os.path.join(metric_dir, det + '.csv')
            det_df = pd.read_csv(det_path)
            for f in file_list:
                v = det_df[det_df['file'] == f][metric_name].values[0]
                file2label[f].append(float(v))
        for f in file2label:
            file2label[f] = np.array(file2label[f], dtype=np.float32)
        return file2label

    def __len__(self):
        return len(self.index_map)

    def _compute_window_metric_vector(self, fname: str, widx: int) -> np.ndarray:
        """
        Returns y_win_metric: (M,) float, per-detector window-level metric.
        """
        label_w = self.win_lab_pt[fname][widx]  # (L,)
        L = label_w.shape[0]

        y_win = np.zeros(len(CANDIDATE_MODEL_SET), dtype=np.float32)

        # Option A: use start_idxs to slice scores
        if fname in self.start_idxs:
            t0 = int(self.start_idxs[fname][widx])
            t1 = t0 + L
            for i, det in enumerate(CANDIDATE_MODEL_SET):
                s_full = self._get_score(det, fname)  # (T,)
                score_w = s_full[t0:t1]
                if score_w.shape[0] != L:
                    # safety (shouldn't happen unless T < window_size and you padded)
                    score_w = np.pad(score_w, (0, L - score_w.shape[0]), mode='constant', constant_values=0)

                if self.window_metric == "sep":
                    y_win[i] = self._sep_only(score_w, label_w)
                else:
                    y_win[i] = window_metric_ap_or_sep(score_w, label_w)

            return y_win

        # Option B: re-window scores using the same split logic (no start_idxs needed)
        for i, det in enumerate(CANDIDATE_MODEL_SET):
            s_full = self._get_score(det, fname)  # (T,)
            s_chunks = split_like_original_1d(s_full, window_size=L)
            score_w = s_chunks[widx].astype(np.float32)
            if self.window_metric == "sep":
                y_win[i] = self._sep_only(score_w, label_w)
            else:
                y_win[i] = window_metric_ap_or_sep(score_w, label_w)

        return y_win

    @staticmethod
    def _sep_only(score_w: np.ndarray, label_w: np.ndarray) -> float:
        y = label_w.astype(np.int32)
        s = score_w.astype(np.float32)
        s1 = s[y == 1]
        s0 = s[y == 0]
        if len(s1) == 0:
            return float(-np.mean(s))  # penalize false alarms
        if len(s0) == 0:
            return float(np.mean(s))   # reward high on all-anom
        return float((s1.mean() - s0.mean()) / (s0.std() + 1e-6))

    def __getitem__(self, idx):
        fname, widx = self.index_map[idx]
        win_n = self.windows_n[fname][widx]  # (L,C)
        win_r = self.windows_r[fname][widx]  # (L,C)

        x_n_t = torch.from_numpy(win_n).permute(1, 0).contiguous()
        x_r_t = torch.from_numpy(win_r).permute(1, 0).contiguous()

        # NEW: per-window detector metric label (M,)
        y_win_metric = torch.from_numpy(self._compute_window_metric_vector(fname, widx))

        if self.return_file_metric:
            y_file = torch.from_numpy(self.file2label[fname])  # (M,)
            return (x_n_t, x_r_t), (y_win_metric, y_file)

        return (x_n_t, x_r_t), y_win_metric


class WindowedTSDataset_M(Dataset):
    def __init__(self,
                 domain: str,
                 file_list_csv: str,
                 dataset_dir: str,
                 metric_dir: str,
                 metric_name: str,
                 window_size: int,
                 target_channels: int = 1,
                 pca_random_state: int = 42,
                 proj_method: list = ["pca"] # ["pca", "var_topk", "kurtosis_topk", "entropy_topk", "l1_topk"]
                ):
        self.dataset_dir = dataset_dir
        self.window_size = window_size
        self.metric_name = metric_name
        self.target_channels = target_channels
        self.pca_random_state = pca_random_state
        self.proj_method = proj_method

        # read file list
        if domain == 'ID':
            df = pd.read_csv(file_list_csv)
            self.files = df['file_name'].values.tolist()
        else:
            df = pd.read_csv(file_list_csv)
            if domain in df["domain_name"].unique():
                self.files = df[df["domain_name"] != domain]['file_name'].values.tolist()
            else:
                print(f'No {domain}')
                exit()

        # pre-load all candidate metric tables into dict[file] -> vector
        self.file2label = self._load_all_labels(metric_dir, self.files, metric_name)


        # build global index
        self.index_map = []  # list of (file_name, window_idx)
        self.file2nwin = {}
        for fname in self.files:
            fpath = os.path.join(dataset_dir, fname)
            ts_df = pd.read_csv(fpath).dropna()
            data = ts_df.iloc[:, 0:-1].values.astype(np.float32)  # (T, C)
            T = data.shape[0]
            nwin = num_windows_for_length(T, window_size)
            self.file2nwin[fname] = nwin
            for w in range(nwin):
                self.index_map.append((fname, w))

    def _load_all_labels(self, metric_dir, file_list, metric_name):
        """
        returns: dict[file_name] -> np.array(M,)
        """
        file2label = {f: [] for f in file_list}
        for det in CANDIDATE_MODEL_SET_M:
            det_path = os.path.join(metric_dir, det + '.csv')
            det_df = pd.read_csv(det_path)
            for f in file_list:
                v = det_df[det_df['file'] == f][metric_name].values[0]
                file2label[f].append(float(v))
        for f in file2label:
            file2label[f] = np.array(file2label[f], dtype=np.float32)
        return file2label

    def __len__(self):
        return len(self.index_map)

    def _project_single_window(self, win: np.ndarray) -> np.ndarray:
        eps_var = 1e-12
        L_local, C_local = win.shape
        C_target = self.target_channels

        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)

        if C_target is None:
            return win.astype(np.float32)

        win_out_list = []

        def _select_topk_channels(score: np.ndarray) -> np.ndarray:
            if C_local >= C_target:
                topk_idx = np.argsort(score)[-C_target:]
                topk_idx = np.sort(topk_idx) 
                out = win[:, topk_idx]
            else:
                pad_width = ((0, 0), (0, C_target - C_local))
                out = np.pad(win, pad_width, mode="constant", constant_values=0.0)
            return out

        for method in self.proj_method:
            m = method.lower()
            if m == "pca":
                if C_local > C_target:
                    total_var = float(np.var(win, axis=0).sum())
                    if (L_local >= 2) and (total_var > eps_var):
                        n_comp = min(C_target, C_local, L_local)
                        pca = PCA(n_components=n_comp, random_state=self.pca_random_state)
                        win_reduced = pca.fit_transform(win)  # (L_local, n_comp)
                        win_reduced = np.nan_to_num(
                            win_reduced, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        if n_comp < C_target:
                            pad_width = ((0, 0), (0, C_target - n_comp))
                            win_out = np.pad(
                                win_reduced, pad_width,
                                mode="constant", constant_values=0.0
                            )
                        else:
                            win_out = win_reduced
                    else:
                        variances = np.var(win, axis=0)  # (C_local,)
                        win_out = _select_topk_channels(variances)
                else:
                    pad_width = ((0, 0), (0, C_target - C_local))
                    win_out = np.pad(
                        win, pad_width,
                        mode="constant", constant_values=0.0
                    )

            elif m == "var_topk":
                variances = np.var(win, axis=0)  # (C_local,)
                win_out = _select_topk_channels(variances)

            elif m == "kurtosis_topk":
                x_centered = win - win.mean(axis=0, keepdims=True)
                m2 = np.mean(x_centered ** 2, axis=0)  # (C_local,)
                m4 = np.mean(x_centered ** 4, axis=0)  # (C_local,)
                kurt = m4 / (m2 ** 2 + eps_var)
                win_out = _select_topk_channels(kurt)

            elif m == "entropy_topk":
                nbins = 16
                x_min = win.min(axis=0)
                x_max = win.max(axis=0)
                span = x_max - x_min
                x_max = x_max + (span == 0) * 1e-6

                ent = np.zeros(C_local, dtype=np.float64)
                for c in range(C_local):
                    hist, _ = np.histogram(
                        win[:, c],
                        bins=nbins,
                        range=(x_min[c], x_max[c]),
                        density=False
                    )
                    p = hist.astype(np.float64)
                    p = p / (p.sum() + eps_var)
                    mask = p > 0
                    ent[c] = -(p[mask] * np.log(p[mask] + eps_var)).sum()

                win_out = _select_topk_channels(ent)

            elif m == "l1_topk":
                l1 = np.mean(np.abs(win), axis=0)  # (C_local,)
                win_out = _select_topk_channels(l1)

            else:
                raise ValueError(f"Unknown proj_method: {method}")

            win_out = np.nan_to_num(win_out, nan=0.0, posinf=0.0, neginf=0.0)
            win_out_list.append(win_out)

        # (num_methods, L, C_target) -> (L, num_methods * C_target)
        if len(win_out_list) == 1:
            win_cat = win_out_list[0]
        else:
            win_stack = np.stack(win_out_list, axis=0)
            win_cat = win_stack.reshape(L_local, -1)

        win_cat = np.nan_to_num(win_cat, nan=0.0, posinf=0.0, neginf=0.0)
        return win_cat.astype(np.float32)

    def __getitem__(self, idx):
        fname, widx = self.index_map[idx]
        fpath = os.path.join(self.dataset_dir, fname)
        df = pd.read_csv(fpath).dropna()
        data = df.iloc[:, 0:-1].values.astype(np.float32)  # (T, C)

        # per-series standardization
        data = StandardScaler().fit_transform(data)   # (T, C)

        # window both normal & residual series
        windows_data = split_like_original(data, self.window_size)  # list of (L, C)
        win_n = windows_data[widx]  # (L, C)

        win_n = self._project_single_window(win_n)  # (L, T)

        # (C, L) for Conv1d / model
        x_n_t = torch.from_numpy(win_n).permute(1, 0).contiguous()

        # label
        y = torch.from_numpy(self.file2label[fname])  # (M,)

        # return a pair for the model: (normal, resid)
        return x_n_t, y


class WindowedTSDatasetPrecomputed_M(Dataset):
    def __init__(self,
                 domain: str,
                 file_list_csv: str,
                 metric_dir: str,
                 metric_name: str,
                 precomputed_dir: str,
                 target_channels: int = 1,
                 pca_random_state: int = 42,
                 proj_method: list = ["pca"] # ["pca", "var_topk", "kurtosis_topk", "entropy_topk", "l1_topk"]
                 ):
        self.metric_name = metric_name
        self.precomputed_dir = precomputed_dir
        self.target_channels = target_channels
        self.pca_random_state = pca_random_state
        self.proj_method = proj_method

        df = pd.read_csv(file_list_csv)
        if domain == 'ID':
            self.files = df['file_name'].values.tolist()
        else:
            if domain in df["domain_name"].unique():
                self.files = df[df["domain_name"] != domain]['file_name'].values.tolist()
            else:
                print(f'No {domain}')
                raise SystemExit(1)

        # labels
        self.file2label = self._load_all_labels(metric_dir, self.files, metric_name)

        self.windows_n = {}  # fname -> np.ndarray (B, L, C_f)
        self.windows_r = {}
        self.index_map = []  # list of (fname, widx)
        self.file_channels = {}  # fname -> C_f

        for fname in self.files:
            npz_path = os.path.join(self.precomputed_dir, f"{fname}.npz")
            if not os.path.exists(npz_path):
                raise FileNotFoundError(
                    f"Precomputed file not found: {npz_path} "
                    f"(Please run precompute_resid_windows.py first)"
                )
            data = np.load(npz_path)
            win_n = data["windows_n"]  # (B, L, C_f)
            win_r = data["windows_r"]  # (B, L, C_f)

            assert win_n.shape == win_r.shape
            B, L, C_f = win_n.shape

            self.windows_n[fname] = win_n
            self.windows_r[fname] = win_r
            self.file_channels[fname] = C_f

            for widx in range(B):
                self.index_map.append((fname, widx))

        print(f"[Dataset] total windows = {len(self.index_map)}")

        if self.target_channels is None:
            all_C = set(self.file_channels.values())
            if len(all_C) != 1:
                raise ValueError(
                    f"[Dataset] target_channels=None, but files have different C: {all_C}. "
                    f"Please set target_channels to unify channel dim."
                )

    def _load_all_labels(self, metric_dir, file_list, metric_name):
        file2label = {f: [] for f in file_list}
        for det in CANDIDATE_MODEL_SET_M:
            det_path = os.path.join(metric_dir, det + '.csv')
            det_df = pd.read_csv(det_path)
            for f in file_list:
                v = det_df[det_df['file'] == f][metric_name].values[0]
                file2label[f].append(float(v))
        for f in file2label:
            file2label[f] = np.array(file2label[f], dtype=np.float32)
        return file2label

    def __len__(self):
        return len(self.index_map)

    def _project_single_window(self, win: np.ndarray) -> np.ndarray:
        eps_var = 1e-12
        L_local, C_local = win.shape
        C_target = self.target_channels

        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)

        if C_target is None:
            return win.astype(np.float32)

        win_out_list = []

        def _select_topk_channels(score: np.ndarray) -> np.ndarray:
            if C_local >= C_target:
                topk_idx = np.argsort(score)[-C_target:]
                topk_idx = np.sort(topk_idx)
                out = win[:, topk_idx]
            else:
                pad_width = ((0, 0), (0, C_target - C_local))
                out = np.pad(win, pad_width, mode="constant", constant_values=0.0)
            return out

        for method in self.proj_method:
            m = method.lower()
            if m == "pca":
                if C_local > C_target:
                    total_var = float(np.var(win, axis=0).sum())
                    if (L_local >= 2) and (total_var > eps_var):
                        n_comp = min(C_target, C_local, L_local)
                        pca = PCA(n_components=n_comp, random_state=self.pca_random_state)
                        win_reduced = pca.fit_transform(win)  # (L_local, n_comp)
                        win_reduced = np.nan_to_num(
                            win_reduced, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        if n_comp < C_target:
                            pad_width = ((0, 0), (0, C_target - n_comp))
                            win_out = np.pad(
                                win_reduced, pad_width,
                                mode="constant", constant_values=0.0
                            )
                        else:
                            win_out = win_reduced
                    else:
                        variances = np.var(win, axis=0)  # (C_local,)
                        win_out = _select_topk_channels(variances)
                else:
                    pad_width = ((0, 0), (0, C_target - C_local))
                    win_out = np.pad(
                        win, pad_width,
                        mode="constant", constant_values=0.0
                    )

            elif m == "var_topk":
                variances = np.var(win, axis=0)  # (C_local,)
                win_out = _select_topk_channels(variances)

            elif m == "kurtosis_topk":
                x_centered = win - win.mean(axis=0, keepdims=True)
                m2 = np.mean(x_centered ** 2, axis=0)  # (C_local,)
                m4 = np.mean(x_centered ** 4, axis=0)  # (C_local,)
                kurt = m4 / (m2 ** 2 + eps_var)
                win_out = _select_topk_channels(kurt)

            elif m == "entropy_topk":
                nbins = 16
                x_min = win.min(axis=0)
                x_max = win.max(axis=0)
                span = x_max - x_min
                x_max = x_max + (span == 0) * 1e-6

                ent = np.zeros(C_local, dtype=np.float64)
                for c in range(C_local):
                    hist, _ = np.histogram(
                        win[:, c],
                        bins=nbins,
                        range=(x_min[c], x_max[c]),
                        density=False
                    )
                    p = hist.astype(np.float64)
                    p = p / (p.sum() + eps_var)
                    mask = p > 0
                    ent[c] = -(p[mask] * np.log(p[mask] + eps_var)).sum()

                win_out = _select_topk_channels(ent)

            elif m == "l1_topk":
                l1 = np.mean(np.abs(win), axis=0)  # (C_local,)
                win_out = _select_topk_channels(l1)

            else:
                raise ValueError(f"Unknown proj_method: {method}")

            win_out = np.nan_to_num(win_out, nan=0.0, posinf=0.0, neginf=0.0)
            win_out_list.append(win_out)

        # (num_methods, L, C_target) -> (L, num_methods * C_target)
        if len(win_out_list) == 1:
            win_cat = win_out_list[0]
        else:
            win_stack = np.stack(win_out_list, axis=0)
            win_cat = win_stack.reshape(L_local, -1)

        win_cat = np.nan_to_num(win_cat, nan=0.0, posinf=0.0, neginf=0.0)
        return win_cat.astype(np.float32)


    def _extract_catch22_summary(self, win: np.ndarray) -> np.ndarray:
        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
        L_local, C_local = win.shape

        feats = []
        for c in range(C_local):
            ts = win[:, c]
            ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
            res = catch22_all(ts)
            vals = np.asarray(res["values"], dtype=np.float64)  # (22,)
            feats.append(vals)
        feats = np.stack(feats, axis=0)  # (C_local, 22)

        summary_features = np.vstack([
            np.nanmin(feats, axis=0),
            np.nanpercentile(feats, 25, axis=0),
            np.nanmean(feats, axis=0),
            np.nanpercentile(feats, 75, axis=0),
            np.nanmax(feats, axis=0),
        ])  # (5, 22)

        summary_features = np.nan_to_num(summary_features,
                                         nan=0.0, posinf=0.0, neginf=0.0)

        base_vec = summary_features.reshape(-1).astype(np.float32)  # (110,)
        return base_vec.reshape(-1, 1)  # (110, 1)

    def __getitem__(self, idx):
        fname, widx = self.index_map[idx]
        win_n = self.windows_n[fname][widx]  # (L, C_f)
        win_r = self.windows_r[fname][widx]  # (L, C_f)

        win_n = self._project_single_window(win_n)  # (L, T)
        win_r = self._project_single_window(win_r)  # (L, T)

        # win_n = self._extract_catch22_summary(win_n)  # (110, 1)
        # win_r = self._extract_catch22_summary(win_r)  # (110, 1)

        win_n = np.nan_to_num(win_n, nan=0.0, posinf=0.0, neginf=0.0)
        win_r = np.nan_to_num(win_r, nan=0.0, posinf=0.0, neginf=0.0)

        x_n_t = torch.from_numpy(win_n).permute(1, 0).contiguous()
        x_r_t = torch.from_numpy(win_r).permute(1, 0).contiguous()

        y = torch.from_numpy(self.file2label[fname])  # (M,)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        return (x_n_t, x_r_t), y


class WindowedTSDatasetPrecomputed_M_FG(Dataset):
    def __init__(self,
                 domain: str,
                 file_list_csv: str,
                 metric_dir: str,
                 metric_name: str,
                 precomputed_dir: str,
                 score_dir: str = '/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/TSB-AutoAD/resource/score/multi/',
                 # --- multivariate projection options (same spirit as WindowedTSDatasetPrecomputed_M) ---
                 target_channels: int = 1,
                 pca_random_state: int = 42,
                 proj_method: list = ["pca"],  # ["pca", "var_topk", "kurtosis_topk", "entropy_topk", "l1_topk"]
                 # --- FG options ---
                 window_metric: str = "ap",      # "ap" or "sep" (you can extend)
                 return_file_metric: bool = True,
                 cache_scores: bool = True):
        self.metric_name = metric_name
        self.precomputed_dir = precomputed_dir
        self.score_dir = score_dir

        self.target_channels = target_channels
        self.pca_random_state = pca_random_state
        self.proj_method = proj_method

        self.window_metric = window_metric.lower()
        self.return_file_metric = return_file_metric
        self.cache_scores = cache_scores

        df = pd.read_csv(file_list_csv)
        if domain == 'ID':
            self.files = df['file_name'].values.tolist()
        else:
            if domain in df["domain_name"].unique():
                self.files = df[df["domain_name"] != domain]['file_name'].values.tolist()
            else:
                raise SystemExit(f'No {domain}')

        # file-level labels (Mdet,)
        self.file2label = self._load_all_labels(metric_dir, self.files, metric_name)

        self.windows_n = {}    # fname -> (B, L, C_f)
        self.windows_r = {}
        self.win_lab_pt = {}   # fname -> (B, L)
        self.start_idxs = {}   # fname -> (B,) optional
        self.file_channels = {}  # fname -> C_f
        self.index_map = []    # (fname, widx)

        # score cache: score_cache[fname][det] = (T,)
        self.score_cache = {}

        for fname in self.files:
            npz_path = os.path.join(self.precomputed_dir, f"{fname}.npz")
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"Precomputed file not found: {npz_path}")

            data = np.load(npz_path, allow_pickle=True)
            win_n = data["windows_n"]  # (B, L, C_f)
            win_r = data["windows_r"]  # (B, L, C_f)
            assert win_n.shape == win_r.shape
            B, L, C_f = win_n.shape

            if "window_labels_point" not in data:
                raise ValueError(f"{npz_path} missing window_labels_point. Re-run precompute.")
            win_lab_pt = data["window_labels_point"].astype(np.int32)  # (B, L)

            if win_lab_pt.shape[0] != B or win_lab_pt.shape[1] != L:
                raise RuntimeError(f"Label/window mismatch for {fname}: {win_lab_pt.shape} vs {win_n.shape}")

            self.windows_n[fname] = win_n
            self.windows_r[fname] = win_r
            self.win_lab_pt[fname] = win_lab_pt
            self.file_channels[fname] = C_f

            if "start_idxs" in data:
                self.start_idxs[fname] = data["start_idxs"].astype(np.int64)

            for widx in range(B):
                self.index_map.append((fname, widx))

            if self.cache_scores:
                self.score_cache[fname] = {det: self._load_score(det, fname) for det in CANDIDATE_MODEL_SET_M}

        print(f"[Dataset-M-FG] total windows = {len(self.index_map)}")

        # same check as WindowedTSDatasetPrecomputed_M
        if self.target_channels is None:
            all_C = set(self.file_channels.values())
            if len(all_C) != 1:
                raise ValueError(
                    f"[Dataset-M-FG] target_channels=None, but files have different C: {all_C}. "
                    f"Please set target_channels to unify channel dim."
                )

    # ---------- score loading ----------
    def _load_score(self, det: str, fname: str) -> np.ndarray:
        score_name = fname.split(".")[0]
        score_path = os.path.join(self.score_dir, det, f"{score_name}.npy")

        if os.path.exists(score_path):
            s = np.load(score_path).astype(np.float32)
            if s.ndim == 2 and s.shape[1] == 1:
                s = s[:, 0]
            # sanitize here
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            return s

        # fallback random (deterministic recommended)
        if fname in self.start_idxs:
            L = self.win_lab_pt[fname].shape[1]
            T = int(self.start_idxs[fname][-1] + L)
        else:
            B, L = self.win_lab_pt[fname].shape
            T = B * L

        seed = abs(hash(f"{fname}-{det}")) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(T).astype(np.float32)

    def _get_score(self, det: str, fname: str) -> np.ndarray:
        if self.cache_scores:
            return self.score_cache[fname][det]
        return self._load_score(det, fname)

    # ---------- file-level metric labels ----------
    def _load_all_labels(self, metric_dir, file_list, metric_name):
        file2label = {f: [] for f in file_list}
        for det in CANDIDATE_MODEL_SET_M:
            det_path = os.path.join(metric_dir, det + '.csv')
            det_df = pd.read_csv(det_path)
            for f in file_list:
                v = det_df[det_df['file'] == f][metric_name].values[0]
                file2label[f].append(float(v))
        for f in file2label:
            file2label[f] = np.array(file2label[f], dtype=np.float32)
        return file2label

    def __len__(self):
        return len(self.index_map)

    # ---------- multivariate projection (copied from WindowedTSDatasetPrecomputed_M) ----------
    def _project_single_window(self, win: np.ndarray) -> np.ndarray:
        eps_var = 1e-12
        L_local, C_local = win.shape
        C_target = self.target_channels

        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)

        if C_target is None:
            return win.astype(np.float32)

        win_out_list = []

        def _select_topk_channels(score: np.ndarray) -> np.ndarray:
            if C_local >= C_target:
                topk_idx = np.argsort(score)[-C_target:]
                topk_idx = np.sort(topk_idx)
                out = win[:, topk_idx]
            else:
                pad_width = ((0, 0), (0, C_target - C_local))
                out = np.pad(win, pad_width, mode="constant", constant_values=0.0)
            return out

        for method in self.proj_method:
            m = method.lower()
            if m == "pca":
                if C_local > C_target:
                    total_var = float(np.var(win, axis=0).sum())
                    if (L_local >= 2) and (total_var > eps_var):
                        n_comp = min(C_target, C_local, L_local)
                        pca = PCA(n_components=n_comp, random_state=self.pca_random_state)
                        win_reduced = pca.fit_transform(win)  # (L_local, n_comp)
                        win_reduced = np.nan_to_num(win_reduced, nan=0.0, posinf=0.0, neginf=0.0)
                        if n_comp < C_target:
                            pad_width = ((0, 0), (0, C_target - n_comp))
                            win_out = np.pad(win_reduced, pad_width, mode="constant", constant_values=0.0)
                        else:
                            win_out = win_reduced
                    else:
                        variances = np.var(win, axis=0)
                        win_out = _select_topk_channels(variances)
                else:
                    pad_width = ((0, 0), (0, C_target - C_local))
                    win_out = np.pad(win, pad_width, mode="constant", constant_values=0.0)

            elif m == "var_topk":
                variances = np.var(win, axis=0)
                win_out = _select_topk_channels(variances)

            elif m == "kurtosis_topk":
                x_centered = win - win.mean(axis=0, keepdims=True)
                m2 = np.mean(x_centered ** 2, axis=0)
                m4 = np.mean(x_centered ** 4, axis=0)
                kurt = m4 / (m2 ** 2 + eps_var)
                win_out = _select_topk_channels(kurt)

            elif m == "entropy_topk":
                nbins = 16
                x_min = win.min(axis=0)
                x_max = win.max(axis=0)
                span = x_max - x_min
                x_max = x_max + (span == 0) * 1e-6

                ent = np.zeros(C_local, dtype=np.float64)
                for c in range(C_local):
                    hist, _ = np.histogram(win[:, c], bins=nbins, range=(x_min[c], x_max[c]), density=False)
                    p = hist.astype(np.float64)
                    p = p / (p.sum() + eps_var)
                    mask = p > 0
                    ent[c] = -(p[mask] * np.log(p[mask] + eps_var)).sum()
                win_out = _select_topk_channels(ent)

            elif m == "l1_topk":
                l1 = np.mean(np.abs(win), axis=0)
                win_out = _select_topk_channels(l1)

            else:
                raise ValueError(f"Unknown proj_method: {method}")

            win_out = np.nan_to_num(win_out, nan=0.0, posinf=0.0, neginf=0.0)
            win_out_list.append(win_out)

        # (num_methods, L, C_target) -> (L, num_methods*C_target)
        if len(win_out_list) == 1:
            win_cat = win_out_list[0]
        else:
            win_stack = np.stack(win_out_list, axis=0)
            win_cat = win_stack.reshape(L_local, -1)

        win_cat = np.nan_to_num(win_cat, nan=0.0, posinf=0.0, neginf=0.0)
        return win_cat.astype(np.float32)

    # ---------- window metric ----------
    @staticmethod
    def _sep_only(score_w: np.ndarray, label_w: np.ndarray) -> float:
        y = label_w.astype(np.int32)
        s = score_w.astype(np.float32)
        s1 = s[y == 1]
        s0 = s[y == 0]
        if len(s1) == 0:
            return float(-np.mean(s))
        if len(s0) == 0:
            return float(np.mean(s))
        return float((s1.mean() - s0.mean()) / (s0.std() + 1e-6))

    def _compute_window_metric_vector(self, fname: str, widx: int) -> np.ndarray:
        label_w = self.win_lab_pt[fname][widx]  # (L,)
        L = label_w.shape[0]
        y_win = np.zeros(len(CANDIDATE_MODEL_SET_M), dtype=np.float32)

        if fname in self.start_idxs:
            t0 = int(self.start_idxs[fname][widx])
            t1 = t0 + L
            for i, det in enumerate(CANDIDATE_MODEL_SET_M):
                s_full = self._get_score(det, fname)  # (T,)
                score_w = s_full[t0:t1]
                if score_w.shape[0] != L:
                    score_w = np.pad(score_w, (0, L - score_w.shape[0]), mode="constant", constant_values=0)

                if self.window_metric == "sep":
                    y_win[i] = self._sep_only(score_w, label_w)
                else:
                    y_win[i] = window_metric_ap_or_sep(score_w, label_w)
            return y_win

        # fallback: re-window score
        for i, det in enumerate(CANDIDATE_MODEL_SET_M):
            s_full = self._get_score(det, fname)
            s_chunks = split_like_original_1d(s_full, window_size=L)
            score_w = s_chunks[widx].astype(np.float32)
            if self.window_metric == "sep":
                y_win[i] = self._sep_only(score_w, label_w)
            else:
                y_win[i] = window_metric_ap_or_sep(score_w, label_w)
        return y_win

    def __getitem__(self, idx):
        fname, widx = self.index_map[idx]
        win_n = self.windows_n[fname][widx]  # (L, C_f)
        win_r = self.windows_r[fname][widx]  # (L, C_f)

        # multivariate -> fixed channel dim (like WindowedTSDatasetPrecomputed_M)
        win_n = self._project_single_window(win_n)  # (L, C_proj)
        win_r = self._project_single_window(win_r)  # (L, C_proj)

        win_n = np.nan_to_num(win_n, nan=0.0, posinf=0.0, neginf=0.0)
        win_r = np.nan_to_num(win_r, nan=0.0, posinf=0.0, neginf=0.0)

        x_n_t = torch.from_numpy(win_n).permute(1, 0).contiguous()
        x_r_t = torch.from_numpy(win_r).permute(1, 0).contiguous()

        y_win_metric = torch.from_numpy(self._compute_window_metric_vector(fname, widx))
        y_win_metric = torch.nan_to_num(y_win_metric, nan=0.0, posinf=0.0, neginf=0.0)

        if self.return_file_metric:
            y_file = torch.from_numpy(self.file2label[fname])
            y_file = torch.nan_to_num(y_file, nan=0.0, posinf=0.0, neginf=0.0)
            return (x_n_t, x_r_t), (y_win_metric, y_file)

        return (x_n_t, x_r_t), y_win_metric
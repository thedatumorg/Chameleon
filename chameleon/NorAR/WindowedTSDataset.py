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

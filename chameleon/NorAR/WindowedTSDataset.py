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
from sklearn.cluster import KMeans
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

# ------------------------------------------------------------------------
# util: same split logic as your original code
# but we turn it into a function that returns #windows for a length
# ------------------------------------------------------------------------
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



# ------------------------------------------------------------------------
# dataset: decompose the full series first, then window normal & resid
# ------------------------------------------------------------------------
class WindowedTSDatasetItem(Dataset):
    def __init__(self,
                 domain: str,
                 file_list_csv: str,
                 dataset_dir: str,
                 metric_dir: str,
                 metric_name: str,
                 window_size: int,
                 dec_sliding_window: int,
                 dec_num_components: int):
        """
        file_list_csv: csv that has column 'file_name' (and maybe 'domain_name')
        dataset_dir: where raw series csvs are
        metric_dir: where per-detector metric CSVs are stored (one csv per detector)
        metric_name: e.g. 'VUS-PR'
        """
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

        # decomposer applied on full series (CPU)
        self.decomposer = AnomalyResidualDecomposer(
            sliding_window=dec_sliding_window,
            num_components=dec_num_components,
            robust_fallback=True
        )

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

        # --- NEW: decompose FULL series into normal/resid first ---
        # to tensor: (1, C, T)
        with torch.no_grad():
            ts = torch.from_numpy(data.astype(np.float32)).permute(1, 0).unsqueeze(0)  # (1, C, T)
            x_n, x_r = self.decomposer(ts)  # (1, C, T), (1, C, T)

        # back to numpy (T, C)
        x_n = x_n.squeeze(0).permute(1, 0).cpu().numpy().astype(np.float32)
        x_r = x_r.squeeze(0).permute(1, 0).cpu().numpy().astype(np.float32)

        # window both normal & residual series
        windows_n = split_like_original(x_n, self.window_size)  # list of (L, C)
        windows_r = split_like_original(x_r, self.window_size)  # list of (L, C)

        win_n = windows_n[widx]  # (L, C)
        win_r = windows_r[widx]  # (L, C)

        # (C, L) for Conv1d / model
        x_n_t = torch.from_numpy(win_n).permute(1, 0).contiguous()
        x_r_t = torch.from_numpy(win_r).permute(1, 0).contiguous()

        # label
        y = torch.from_numpy(self.file2label[fname])  # (M,)

        # return a pair for the model: (normal, resid)
        return (x_n_t, x_r_t), y


class WindowedTSDatasetInit(Dataset):
    def __init__(self,
                 domain: str,
                 file_list_csv: str,
                 dataset_dir: str,
                 metric_dir: str,
                 metric_name: str,
                 window_size: int,
                 dec_sliding_window: int,
                 dec_num_components: int):
        self.dataset_dir = dataset_dir
        self.window_size = window_size
        self.metric_name = metric_name

        df = pd.read_csv(file_list_csv)
        if domain == 'ID':
            self.files = df['file_name'].values.tolist()
        else:
            if domain in df["domain_name"].unique():
                self.files = df[df["domain_name"] != domain]['file_name'].values.tolist()
            else:
                print(f'No {domain}')
                raise SystemExit(1)

        self.file2label = self._load_all_labels(metric_dir, self.files, metric_name)

        # 只初始化一次 decomposer
        self.decomposer = AnomalyResidualDecomposer(
            sliding_window=dec_sliding_window,
            num_components=dec_num_components,
            robust_fallback=True
        )

        # 预先把每个 file 的 (normal/resid) windows 都算好
        self.windows_n = {}   # fname -> list of (L,C)
        self.windows_r = {}   # fname -> list of (L,C)
        self.index_map = []   # list of (fname, widx)

        for fname in self.files:
            fpath = os.path.join(dataset_dir, fname)
            ts_df = pd.read_csv(fpath).dropna()
            data = ts_df.iloc[:, 0:-1].values.astype(np.float32)  # (T, C)

            # per-series 标准化
            data_std = StandardScaler().fit_transform(data)  # (T, C)

            # 一次性在整条序列上做 decomposition
            with torch.no_grad():
                ts = torch.from_numpy(data_std.astype(np.float32)).permute(1, 0).unsqueeze(0)  # (1,C,T)
                x_n, x_r = self.decomposer(ts)  # (1,C,T),(1,C,T)

            x_n = x_n.squeeze(0).permute(1, 0).cpu().numpy().astype(np.float32)  # (T,C)
            x_r = x_r.squeeze(0).permute(1, 0).cpu().numpy().astype(np.float32)  # (T,C)

            # split 成 windows
            win_list_n = split_like_original(x_n, self.window_size)  # list of (L,C)
            win_list_r = split_like_original(x_r, self.window_size)

            self.windows_n[fname] = win_list_n
            self.windows_r[fname] = win_list_r

            for widx in range(len(win_list_n)):
                self.index_map.append((fname, widx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fname, widx = self.index_map[idx]

        win_n = self.windows_n[fname][widx]  # (L,C)
        win_r = self.windows_r[fname][widx]  # (L,C)

        x_n_t = torch.from_numpy(win_n).permute(1, 0).contiguous()  # (C,L)
        x_r_t = torch.from_numpy(win_r).permute(1, 0).contiguous()  # (C,L)

        y = torch.from_numpy(self.file2label[fname])                # (M,)

        return (x_n_t, x_r_t), y
        
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

        # 预加载所有 npz 到内存（也可以改成 lazy-load + cache）
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

        # (C, L) for Conv1d / Conformer
        x_n_t = torch.from_numpy(win_n).permute(1, 0).contiguous()
        x_r_t = torch.from_numpy(win_r).permute(1, 0).contiguous()

        y = torch.from_numpy(self.file2label[fname])  # (M,)

        return (x_n_t, x_r_t), y


class WindowedTSDatasetPrecomputed_M_Padding(Dataset):
    """
    使用预处理好的 NorAR windows (.npz):
      每个文件一个 npz，里面有 windows_n, windows_r，shape: (B, L, C)

    可选：把通道数 pad 到固定的 target_channels。
    """
    def __init__(self,
                 domain: str,
                 file_list_csv: str,
                 metric_dir: str,
                 metric_name: str,
                 precomputed_dir: str,
                 target_channels: int = 256,
                 allow_truncate: bool = True):
        """
        Args
        ----
        target_channels : int | None
            - None: 不做 padding，保持原始 C（需要所有样本 C 一致）
            - int: 把所有样本的通道数 pad/截断到该值

        allow_truncate : bool
            - 如果为 False 且遇到 C > target_channels，则报错
            - 如果为 True 且遇到 C > target_channels，则直接截断多余通道
        """
        self.metric_name = metric_name
        self.precomputed_dir = precomputed_dir
        self.target_channels = target_channels
        self.allow_truncate = allow_truncate

        df = pd.read_csv(file_list_csv)
        if domain == 'ID':
            self.files = df['file_name'].values.tolist()
        else:
            if domain in df["domain_name"].unique():
                # 保持你原来的逻辑（!= domain）；如果想“只用当前 domain”，这里可以改成 ==
                self.files = df[df["domain_name"] != domain]['file_name'].values.tolist()
            else:
                print(f'No {domain}')
                raise SystemExit(1)

        # labels
        self.file2label = self._load_all_labels(metric_dir, self.files, metric_name)

        # 预加载所有 npz 到内存
        self.windows_n = {}  # fname -> np.ndarray (B, L, C)
        self.windows_r = {}
        self.index_map = []  # list of (fname, widx)

        for fname in self.files:
            npz_path = os.path.join(self.precomputed_dir, f"{fname}.npz")
            if not os.path.exists(npz_path):
                raise FileNotFoundError(
                    f"Precomputed file not found: {npz_path} "
                    f"(Please run precompute_resid_windows.py first)"
                )
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

    def _pad_channels(self, win: np.ndarray) -> np.ndarray:
        """
        win: (L, C) -> (L, C_padded)

        - 如果 target_channels is None: 返回原始 win
        - 否则 pad / truncate 到 target_channels
        """
        if self.target_channels is None:
            return win

        L, C = win.shape
        T = self.target_channels

        if C == T:
            return win

        if C > T:
            if not self.allow_truncate:
                raise ValueError(
                    f"Window has {C} channels, which exceeds target_channels={T}. "
                    f"Set allow_truncate=True if you want to cut off extra channels."
                )
            # 截断多余通道: (L, C) -> (L, T)
            return win[:, :T]

        # C < T: 在通道维度右侧 pad 0
        pad_width = ((0, 0), (0, T - C))  # (L-dim, C-dim)
        win_padded = np.pad(win, pad_width, mode="constant", constant_values=0.0)
        return win_padded  # (L, T)

    def __getitem__(self, idx):
        fname, widx = self.index_map[idx]
        win_n = self.windows_n[fname][widx]  # (L, C)
        win_r = self.windows_r[fname][widx]  # (L, C)

        # 通道 padding
        win_n = self._pad_channels(win_n)  # (L, C') where C' = target_channels or C
        win_r = self._pad_channels(win_r)  # (L, C')

        # (C', L) for Conv1d / TCN
        x_n_t = torch.from_numpy(win_n).permute(1, 0).contiguous()
        x_r_t = torch.from_numpy(win_r).permute(1, 0).contiguous()

        y = torch.from_numpy(self.file2label[fname])  # (M,)

        return (x_n_t, x_r_t), y


class WindowedTSDatasetPrecomputedDomain(Dataset):
    """
    使用预处理好的 NorAR windows (.npz):
      每个文件一个 npz，里面有 windows_n, windows_r，shape: (B, L, C)
    同时返回 domain label, 从 fname.split('_')[4] 提取.
    """
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
                # 训练时排除 held-out domain
                self.files = df[df["domain_name"] != domain]['file_name'].values.tolist()
            else:
                print(f'No {domain}')
                raise SystemExit(1)

        # ---------- domain vocab from fname.split('_')[4] ----------
        self.domain2id = {}
        self.file2domid = {}
        for fname in self.files:
            parts = fname.split('_')
            if len(parts) < 5:
                raise ValueError(f"Unexpected file name format (cannot get domain): {fname}")
            dom_str = parts[4]
            if dom_str not in self.domain2id:
                self.domain2id[dom_str] = len(self.domain2id)
            self.file2domid[fname] = self.domain2id[dom_str]
        self.num_domains = len(self.domain2id)
        print(f"[Dataset] num_domains = {self.num_domains}, mapping = {self.domain2id}")

        # ---------- labels ----------
        self.file2label = self._load_all_labels(metric_dir, self.files, metric_name)

        # ---------- pre-load npz ----------
        self.windows_n = {}  # fname -> np.ndarray (B, L, C)
        self.windows_r = {}
        self.index_map = []  # list of (fname, widx)

        for fname in self.files:
            npz_path = os.path.join(self.precomputed_dir, f"{fname}.npz")
            if not os.path.exists(npz_path):
                raise FileNotFoundError(
                    f"Precomputed file not found: {npz_path} "
                    f"(Please run precompute_resid_windows.py first)"
                )
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

        # (C, L) for Conv1d / ModernTCN
        x_n_t = torch.from_numpy(win_n).permute(1, 0).contiguous()
        x_r_t = torch.from_numpy(win_r).permute(1, 0).contiguous()

        y = torch.from_numpy(self.file2label[fname])  # (M,)

        dom_id = self.file2domid[fname]               # int
        dom_t  = torch.tensor(dom_id, dtype=torch.long)

        return (x_n_t, x_r_t), y, dom_t


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

        # 先清理一次 NaN / Inf，避免后续统计炸掉
        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)

        # 如果不需要统一通道数，或者本身刚好等于目标通道数且只用一个方法，就直接返回
        if C_target is None:
            return win.astype(np.float32)

        win_out_list = []

        # ------------------ 小工具：Top-K 通道选择 + pad ------------------
        def _select_topk_channels(score: np.ndarray) -> np.ndarray:
            """
            score: (C_local,)  每个通道的分数（越大越重要）
            返回: (L_local, C_target)
            """
            if C_local >= C_target:
                topk_idx = np.argsort(score)[-C_target:]
                topk_idx = np.sort(topk_idx)  # 保持通道顺序稳定
                out = win[:, topk_idx]
            else:
                # 通道不够，全部保留并在右侧 pad 0
                pad_width = ((0, 0), (0, C_target - C_local))
                out = np.pad(win, pad_width, mode="constant", constant_values=0.0)
            return out

        # ------------------ 主循环：多种方法拼接 ------------------
        for method in self.proj_method:
            m = method.lower()
            # 1) PCA-based 降维
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
                        # PCA 不稳定（样本太少或方差太小），退化为 var_topk
                        variances = np.var(win, axis=0)  # (C_local,)
                        win_out = _select_topk_channels(variances)
                else:
                    # 通道数 <= 目标通道数：不做真正的 PCA，只做 pad
                    pad_width = ((0, 0), (0, C_target - C_local))
                    win_out = np.pad(
                        win, pad_width,
                        mode="constant", constant_values=0.0
                    )

            # 2) Top-K variance
            elif m == "var_topk":
                variances = np.var(win, axis=0)  # (C_local,)
                win_out = _select_topk_channels(variances)

            # 3) Kurtosis Top-K（异常敏感，heavy-tail 通道得分高）
            elif m == "kurtosis_topk":
                # 手动算四阶矩和二阶矩来近似峰度
                x_centered = win - win.mean(axis=0, keepdims=True)
                m2 = np.mean(x_centered ** 2, axis=0)  # (C_local,)
                m4 = np.mean(x_centered ** 4, axis=0)  # (C_local,)
                kurt = m4 / (m2 ** 2 + eps_var)        # 不减 3，排序结果一样
                win_out = _select_topk_channels(kurt)

            # 4) Entropy Top-K（信息量大的通道得分高）
            elif m == "entropy_topk":
                # 用统一的 bin 边界估计每个通道的直方图熵
                nbins = 16
                # 全局最小值 / 最大值，用于统一 bins
                x_min = win.min(axis=0)
                x_max = win.max(axis=0)
                # 避免 max==min 导致单点区间
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
                    # 只对非零概率求 log
                    mask = p > 0
                    ent[c] = -(p[mask] * np.log(p[mask] + eps_var)).sum()

                win_out = _select_topk_channels(ent)

            # 5) L1-energy Top-K（平均绝对值越大，能量越高）
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
                # 保持你原来的逻辑（!= domain）；如果想“只用当前 domain”，这里可以改成 ==
                self.files = df[df["domain_name"] != domain]['file_name'].values.tolist()
            else:
                print(f'No {domain}')
                raise SystemExit(1)

        # labels
        self.file2label = self._load_all_labels(metric_dir, self.files, metric_name)

        # 预加载所有 npz 到内存
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

        # 若不指定 target_channels，则要求所有 C_f 一致（否则 DataLoader collate 不工作）
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

        # 先清理一次 NaN / Inf，避免后续统计炸掉
        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)

        # 如果不需要统一通道数，或者本身刚好等于目标通道数且只用一个方法，就直接返回
        if C_target is None:
            return win.astype(np.float32)

        win_out_list = []

        # ------------------ 小工具：Top-K 通道选择 + pad ------------------
        def _select_topk_channels(score: np.ndarray) -> np.ndarray:
            """
            score: (C_local,)  每个通道的分数（越大越重要）
            返回: (L_local, C_target)
            """
            if C_local >= C_target:
                topk_idx = np.argsort(score)[-C_target:]
                topk_idx = np.sort(topk_idx)  # 保持通道顺序稳定
                out = win[:, topk_idx]
            else:
                # 通道不够，全部保留并在右侧 pad 0
                pad_width = ((0, 0), (0, C_target - C_local))
                out = np.pad(win, pad_width, mode="constant", constant_values=0.0)
            return out

        # ------------------ 主循环：多种方法拼接 ------------------
        for method in self.proj_method:
            m = method.lower()
            # 1) PCA-based 降维
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
                        # PCA 不稳定（样本太少或方差太小），退化为 var_topk
                        variances = np.var(win, axis=0)  # (C_local,)
                        win_out = _select_topk_channels(variances)
                else:
                    # 通道数 <= 目标通道数：不做真正的 PCA，只做 pad
                    pad_width = ((0, 0), (0, C_target - C_local))
                    win_out = np.pad(
                        win, pad_width,
                        mode="constant", constant_values=0.0
                    )

            # 2) Top-K variance
            elif m == "var_topk":
                variances = np.var(win, axis=0)  # (C_local,)
                win_out = _select_topk_channels(variances)

            # 3) Kurtosis Top-K（异常敏感，heavy-tail 通道得分高）
            elif m == "kurtosis_topk":
                # 手动算四阶矩和二阶矩来近似峰度
                x_centered = win - win.mean(axis=0, keepdims=True)
                m2 = np.mean(x_centered ** 2, axis=0)  # (C_local,)
                m4 = np.mean(x_centered ** 4, axis=0)  # (C_local,)
                kurt = m4 / (m2 ** 2 + eps_var)        # 不减 3，排序结果一样
                win_out = _select_topk_channels(kurt)

            # 4) Entropy Top-K（信息量大的通道得分高）
            elif m == "entropy_topk":
                # 用统一的 bin 边界估计每个通道的直方图熵
                nbins = 16
                # 全局最小值 / 最大值，用于统一 bins
                x_min = win.min(axis=0)
                x_max = win.max(axis=0)
                # 避免 max==min 导致单点区间
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
                    # 只对非零概率求 log
                    mask = p > 0
                    ent[c] = -(p[mask] * np.log(p[mask] + eps_var)).sum()

                win_out = _select_topk_channels(ent)

            # 5) L1-energy Top-K（平均绝对值越大，能量越高）
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

        # 1) 对每个通道计算 22 个 catch22 特征 -> (C_local, 22)
        feats = []
        for c in range(C_local):
            # catch22 对常数序列有时会返回 NaN，先做一次清洗
            ts = win[:, c]
            ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
            res = catch22_all(ts)
            vals = np.asarray(res["values"], dtype=np.float64)  # (22,)
            feats.append(vals)
        feats = np.stack(feats, axis=0)  # (C_local, 22)

        # 2) 在通道维上做 summary: min / 25% / mean / 75% / max -> (5, 22)
        summary_features = np.vstack([
            np.nanmin(feats, axis=0),
            np.nanpercentile(feats, 25, axis=0),
            np.nanmean(feats, axis=0),
            np.nanpercentile(feats, 75, axis=0),
            np.nanmax(feats, axis=0),
        ])  # (5, 22)

        summary_features = np.nan_to_num(summary_features,
                                         nan=0.0, posinf=0.0, neginf=0.0)

        # 3) 展平为一个 feature 向量（5*22 = 110 维）
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




class WindowedTSDatasetPrecomputedClusterDomain(Dataset):
    """
    使用预处理好的 NorAR windows (.npz):
      每个文件一个 npz，里面有 windows_n, windows_r，shape: (B, L, C)

    不再使用文件名中的 domain，而是：
      对所有 normal windows 做聚类（KMeans），得到 cluster id 作为“domain label”.

    Args
    ----
    domain : str
        'ID' 使用所有 domain；否则排除该 domain 名对应的文件（和之前逻辑一致，只是 domain 不再用作标签）。
    file_list_csv : str
    metric_dir : str
    metric_name : str
    precomputed_dir : str
    num_clusters : int
        聚类的簇数，将作为 adversarial 头中的 num_domains.
    """
    def __init__(self,
                 domain: str,
                 file_list_csv: str,
                 metric_dir: str,
                 metric_name: str,
                 precomputed_dir: str,
                 num_clusters: int = 8):

        self.metric_name = metric_name
        self.precomputed_dir = precomputed_dir
        self.num_clusters = int(num_clusters)

        df = pd.read_csv(file_list_csv)
        if domain == 'ID':
            self.files = df['file_name'].values.tolist()
        else:
            if domain in df["domain_name"].unique():
                # 训练时排除 held-out domain（和原来一致）
                self.files = df[df["domain_name"] != domain]['file_name'].values.tolist()
            else:
                print(f'No {domain}')
                raise SystemExit(1)

        # ---------- labels ----------
        self.file2label = self._load_all_labels(metric_dir, self.files, metric_name)

        # ---------- pre-load npz + build index_map + features for clustering ----------
        self.windows_n = {}  # fname -> np.ndarray (B, L, C)
        self.windows_r = {}
        self.index_map = []  # list of (fname, widx)
        feat_list = []       # 每个 window 的特征向量 (flatten)

        for fname in self.files:
            npz_path = os.path.join(self.precomputed_dir, f"{fname}.npz")
            if not os.path.exists(npz_path):
                raise FileNotFoundError(
                    f"Precomputed file not found: {npz_path} "
                    f"(Please run precompute_resid_windows.py first)"
                )
            data = np.load(npz_path)
            win_n = data["windows_n"]  # (B, L, C)
            win_r = data["windows_r"]  # (B, L, C)
            assert win_n.shape == win_r.shape
            B, L, C = win_n.shape

            self.windows_n[fname] = win_n
            self.windows_r[fname] = win_r

            # index_map & features 保持同一顺序，方便之后对齐 cluster label
            for widx in range(B):
                self.index_map.append((fname, widx))
                # 简单做法：直接 flatten normal window 作为聚类特征
                feat_list.append(win_n[widx].reshape(-1))  # (L*C,)

        print(f"[Dataset] total windows = {len(self.index_map)}")

        feats = np.stack(feat_list, axis=0)  # (N_windows, L*C)

        # ---------- 聚类，得到“domain” = cluster id ----------
        print(f"[Clustering] KMeans on {feats.shape[0]} windows, dim={feats.shape[1]}, "
              f"num_clusters={self.num_clusters}")
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init="auto")
        cluster_labels = kmeans.fit_predict(feats).astype(np.int64)  # (N_windows,)

        self.cluster_labels = cluster_labels
        self.num_domains = self.num_clusters  # 提供给模型使用
        print(f"[Clustering] done. num_domains (clusters) = {self.num_domains}")

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

        # (C, L) for Conv1d / ModernTCN
        x_n_t = torch.from_numpy(win_n).permute(1, 0).contiguous()
        x_r_t = torch.from_numpy(win_r).permute(1, 0).contiguous()

        y = torch.from_numpy(self.file2label[fname])  # (M,)

        # cluster_id 作为“domain label”
        dom_id = int(self.cluster_labels[idx])
        dom_t  = torch.tensor(dom_id, dtype=torch.long)

        return (x_n_t, x_r_t), y, dom_t

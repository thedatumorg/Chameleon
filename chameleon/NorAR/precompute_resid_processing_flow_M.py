#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append("/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon")
from chameleon.NorAR.AnomalyResid import AnomalyResidualDecomposer


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
    return chunks  # list of (L, C)


def split_labels_like_original(labels: np.ndarray, window_size: int):
    """
    labels: (T,)
    return:
      - window_labels_point: (B, L) int32, point-wise GT labels per window (pad with 0 if needed)
      - window_labels_any:   (B,) int32, 1 if any anomaly in window else 0
      - window_labels_frac:  (B,) float32, fraction of anomaly points in window
      - start_idxs:          (B,) int64, start index in original series for each window
                             (start indices follow the same 'modulo' logic as split_like_original)
    """
    T = labels.shape[0]

    if T < window_size:
        pad_len = window_size - T
        lab = np.pad(labels, (0, pad_len), mode='constant', constant_values=0)
        window_labels_point = lab[None, :].astype(np.int32)  # (1, L)
        window_labels_any = (window_labels_point.sum(axis=1) > 0).astype(np.int32)
        window_labels_frac = (window_labels_point.mean(axis=1)).astype(np.float32)
        start_idxs = np.array([0], dtype=np.int64)
        return window_labels_point, window_labels_any, window_labels_frac, start_idxs

    modulo = T % window_size
    starts = []

    # windows list follows: [0] if modulo != 0, then [modulo, modulo+L, ...]
    if modulo != 0:
        starts.append(0)
    for s in range(modulo, T, window_size):
        if s + window_size <= T:  # only full windows (rest is exact multiple)
            starts.append(s)

    window_labels_point = np.stack([labels[s:s + window_size] for s in starts], axis=0).astype(np.int32)  # (B, L)
    window_labels_any = (window_labels_point.sum(axis=1) > 0).astype(np.int32)                             # (B,)
    window_labels_frac = (window_labels_point.mean(axis=1)).astype(np.float32)                             # (B,)
    start_idxs = np.array(starts, dtype=np.int64)                                                          # (B,)

    return window_labels_point, window_labels_any, window_labels_frac, start_idxs


def preprocess_one_file(
    fname: str,
    dataset_dir: str,
    window_size: int,
    decomposer: AnomalyResidualDecomposer,
    out_dir: Path,
):
    fpath = os.path.join(dataset_dir, fname)
    if not os.path.exists(fpath):
        print(f"[WARN] file not found: {fpath}, skip.")
        return

    ts_df = pd.read_csv(fpath).dropna()
    data = ts_df.iloc[:, 0:-1].values.astype(np.float32)   # (T, C)
    labels = ts_df.iloc[:, -1].values.astype(np.int32)     # (T,)

    # split windows (same as training)
    win_list = split_like_original(data, window_size=window_size)
    if len(win_list) == 0:
        print(f"[WARN] no windows produced for {fname}, skip.")
        return

    windows = np.stack(win_list, axis=0).astype(np.float32)  # (B, L, C)
    B, L, C = windows.shape

    # split labels with matching logic
    window_labels_point, window_labels_any, window_labels_frac, start_idxs = split_labels_like_original(
        labels, window_size=window_size
    )

    # sanity check alignment
    if window_labels_point.shape[0] != B or window_labels_point.shape[1] != L:
        raise RuntimeError(
            f"[ERR] window/label mismatch for {fname}: "
            f"windows={windows.shape}, window_labels_point={window_labels_point.shape}"
        )

    with torch.no_grad():
        ts = torch.from_numpy(windows).permute(0, 2, 1)  # (B, C, L)
        ts = ts.to(dtype=torch.float32)
        x_n, x_r = decomposer(ts)

    windows_n = x_n.permute(0, 2, 1).cpu().numpy().astype(np.float32)  # (B, L, C)
    windows_r = x_r.permute(0, 2, 1).cpu().numpy().astype(np.float32)  # (B, L, C)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fname}.npz"

    np.savez_compressed(
        out_path,
        windows_n=windows_n,
        windows_r=windows_r,
        window_labels_point=window_labels_point,  # (B, L) int32
        window_labels_any=window_labels_any,      # (B,) int32
        window_labels_frac=window_labels_frac,    # (B,) float32
        start_idxs=start_idxs,                    # (B,) int64
        orig_T=np.array([data.shape[0]], dtype=np.int64),
    )

    print(
        f"[OK] {fname}: T={data.shape[0]}, C={data.shape[1]}, "
        f"n_win={B}, L={L} -> {out_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Precompute NorAR decomposition windows to .npz (with window labels)")
    parser.add_argument('--domain', type=str, default='ID',
                        help='ID, WebService, Medical, Facility, Synthetic, HumanActivity, Sensor, Environment, Finance, Traffic')
    parser.add_argument('--file_list', type=str,
                        default='/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/file_list/TSB-AD-M-Label.csv')
    parser.add_argument('--dataset_dir', type=str,
                        default='/data/liuqinghua/code/ts/public_repo/TSB-AD/Datasets/TSB-AD-Datasets/TSB-AD-M/')
    parser.add_argument('--out_dir', type=str,
                        default='/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/precomputed_resid_processing_flow_stl_M/')

    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--dec_sliding_window', type=int, default=128)
    parser.add_argument('--dec_num_components', type=int, default=3)

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    df = pd.read_csv(args.file_list)
    if args.domain == 'ID':
        files = df['file_name'].values.tolist()
    else:
        if args.domain in df["domain_name"].unique():
            files = df[df["domain_name"] != args.domain]['file_name'].values.tolist()
        else:
            print(f'No domain {args.domain}')
            return

    print(f"Total files to preprocess: {len(files)}")

    decomposer = AnomalyResidualDecomposer(
        mode='stl_ad',
        robust_fallback=True
    )

    for fname in files:
        preprocess_one_file(
            fname=fname,
            dataset_dir=args.dataset_dir,
            window_size=args.window_size,
            decomposer=decomposer,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()

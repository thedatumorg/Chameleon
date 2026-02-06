#!/usr/bin/env python3
import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.append("/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon")
from chameleon.NorAR.AnomalyResid import AnomalyResidualDecomposer


# 和训练脚本里一致的 window 切分函数
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
    data = ts_df.iloc[:, 0:-1].values.astype(np.float32)  # (T, C)

    win_list = split_like_original(data, window_size=window_size)
    if len(win_list) == 0:
        print(f"[WARN] no windows produced for {fname}, skip.")
        return

    windows = np.stack(win_list, axis=0).astype(np.float32)  # (B, L, C)
    B, L, C = windows.shape

    with torch.no_grad():
        ts = torch.from_numpy(windows).permute(0, 2, 1)  # (B, C, L)
        ts = ts.to(dtype=torch.float32)
        x_n, x_r = decomposer(ts)

    windows_n = x_n.permute(0, 2, 1).cpu().numpy().astype(np.float32)
    windows_r = x_r.permute(0, 2, 1).cpu().numpy().astype(np.float32)

    

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fname}.npz"

    np.savez_compressed(
        out_path,
        windows_n=windows_n,
        windows_r=windows_r,
    )
    print(
        f"[OK] {fname}: T={data.shape[0]}, C={data.shape[1]}, "
        f"n_win={windows_n.shape[0]}, L={windows_n.shape[1]} -> {out_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Precompute NorAR decomposition windows to .npz")
    parser.add_argument('--domain', type=str, default='ID',
                        help='ID, WebService, Medical, Facility, Synthetic, HumanActivity, Sensor, Environment, Finance, Traffic')
    parser.add_argument('--file_list', type=str,
                        default='/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/file_list/TSB-AD-U-Label.csv')
    parser.add_argument('--dataset_dir', type=str,
                        default='/data/liuqinghua/code/ts/public_repo/TSB-AD/Datasets/TSB-AD-Datasets/TSB-AD-U/')
    parser.add_argument('--out_dir', type=str,
                        default='/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/precomputed_resid_processing_flow_stl_ad/')

    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--dec_sliding_window', type=int, default=128)
    parser.add_argument('--dec_num_components', type=int, default=3)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # 读 file_list & 按 domain 过滤，逻辑和训练脚本一致
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
        sliding_window=args.dec_sliding_window,
        num_components=args.dec_num_components,
        mode='stl_ad',
        season_len=7,
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

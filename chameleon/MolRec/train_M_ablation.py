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
from torch.utils.data import random_split

import sys
sys.path.append("/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon")
from chameleon.NorAR.WindowedTSDataset import WindowedTSDatasetPrecomputed_M as WindowedTSDatasetPrecomputed
from chameleon.MolRec.ChameleonRec import ChameleonRec

from chameleon.NorAR.WindowedTSDataset import WindowedTSDataset_M as WindowedTSDataset
from chameleon.MolRec.ChameleonRec_None import ChameleonRec_None
from chameleon.MolRec.ChameleonRec_Conv import ChameleonRec_Conv
from chameleon.MolRec.ChameleonRec_Transformer import ChameleonRec_Transformer


# ------------------------------------------------------------------------
# config: candidate models
# ------------------------------------------------------------------------
CANDIDATE_MODEL_SET = ['IForest', 'LOF', 'PCA', 'HBOS', 'OCSVM', 'MCD', 'KNN', 'KMeansAD', 'COPOD', 'CBLOF', 'EIF', 'RobustPCA', 'AutoEncoder', 
                    'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 'TimesNet', 'FITS', 'OFA']
NUM_MODELS = len(CANDIDATE_MODEL_SET)

import random
def set_seed(seed: int):
    print(f"[seed] Using seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------------
def pairwise_logistic_ranking_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.0):
    """
    Ranking loss: encourages ordering of models according to target scores.
    pred, target: (B, M)
    """
    diff_t = target.unsqueeze(-1) - target.unsqueeze(-2)    # (B,M,M)
    mask = (diff_t > margin).float()
    diff_p = pred.unsqueeze(-1) - pred.unsqueeze(-2)
    loss_mat = -F.logsigmoid(diff_p) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss_mat.sum() / denom


def regression_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    Regression loss: per-model score regression (MSE).
    pred, target: (B, M)
    """
    return F.mse_loss(pred, target)


def classification_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    Classification loss: treat target scores as utilities and convert to labels via argmax.
    - pred: (B, M) logits.
    - target: (B, M) scores; we take argmax along dim=-1 as the class index.
    """
    # target may be float, we only need argmax
    labels = torch.argmax(target, dim=-1)  # (B,)
    return F.cross_entropy(pred, labels)


def compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str, margin: float = 0.0):
    """
    Dispatch between ranking / regression / classification.
    """
    loss_type = loss_type.lower()
    if loss_type == "ranking":
        return pairwise_logistic_ranking_loss(pred, target, margin=margin)
    elif loss_type == "regression":
        target = target.to(pred.dtype)
        return regression_loss(pred, target)
    elif loss_type == "classification":
        return classification_loss(pred, target)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Expected 'ranking', 'regression', or 'classification'.")


# ------------------------------------------------------------------------
# training  (UPDATED to use (x_n, x_r, y) from dataset and dual-input model)
# ------------------------------------------------------------------------
def train_one_file(args):
    device = torch.device(args.device)
    set_seed(42)

    ds = WindowedTSDatasetPrecomputed(
        domain=args.domain,
        file_list_csv=args.file_list,
        metric_dir=args.metric_path,
        metric_name=args.metric,
        precomputed_dir=args.precomputed_dir,
    )

    # ds = WindowedTSDataset(
    #     domain=args.domain,
    #     file_list_csv=args.file_list,
    #     dataset_dir=args.dataset_dir,
    #     metric_dir=args.metric_path,
    #     metric_name=args.metric,
    #     window_size=args.window_size,
    # )

    # tiny split for val
    val_size = max(1, int(0.1 * len(ds)))
    train_size = len(ds) - val_size

    split_g = torch.Generator()
    split_g.manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size], generator=split_g)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    # get input channels from first sample
    (x_n0, x_r0), _ = ds[0]
    in_channels, L = x_n0.shape  # (C, L)

    d_model   = getattr(args, "d_model", 256)
    dropout   = getattr(args, "dropout", 0.1)

    mtc_patch_size   = getattr(args, "mtc_patch_size", 8)
    mtc_patch_stride = getattr(args, "mtc_patch_stride", 8)
    mtc_down_ratio   = getattr(args, "mtc_downsample_ratio", 2)
    mtc_num_blocks   = getattr(args, "mtc_num_blocks", (2, 2))
    mtc_large_sizes  = getattr(args, "mtc_large_sizes", (15, 15))
    mtc_small_sizes  = getattr(args, "mtc_small_sizes", (3, 3))
    mtc_dims         = getattr(args, "mtc_dims", (128, 256))
    mtc_dw_dims      = getattr(args, "mtc_dw_dims", (128, 256))

    backbone = getattr(args, "backbone_type", "TCN").lower()
    if backbone == "tcn":
        model = ChameleonRec(
            in_channels=in_channels, num_models=NUM_MODELS,
            d_model=d_model, dropout=dropout,
            patch_size=mtc_patch_size, patch_stride=mtc_patch_stride, downsample_ratio=mtc_down_ratio,
            num_blocks=mtc_num_blocks, large_size=mtc_large_sizes, small_size=mtc_small_sizes,
            dims=mtc_dims, dw_dims=mtc_dw_dims,
        ).to(device)
    elif backbone == "convnet":
        model = ChameleonRec_Conv(
            in_channels=in_channels, num_models=NUM_MODELS, window_size=args.window_size, d_model=d_model
        ).to(device)
    elif backbone == "transformer":
        model = ChameleonRec_Transformer(
        in_channels=in_channels, num_models=NUM_MODELS, d_model=d_model
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=3, verbose=True, min_lr=1e-6,
    )

    es_patience = getattr(args, 'patience', 5)
    es_counter = 0
    best_val = float("inf")

    loss_type = getattr(args, "loss_type", "ranking").lower()
    print(f"Using loss_type = {loss_type}")

    for epoch in range(args.epochs):
        start_time = time.time()

        # train
        model.train()
        total_loss = 0.0
        for (x_n, x_r), y in train_loader:
            x_n = x_n.to(device)
            x_r = x_r.to(device)
            y   = y.to(device)

            pred = model(x_n, x_r)
            loss = compute_loss(pred, y, loss_type, margin=0.02)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * x_n.size(0)
        total_loss /= len(train_loader.dataset)

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_n, x_r), y in val_loader:
                x_n = x_n.to(device)
                x_r = x_r.to(device)
                y   = y.to(device)
                pred = model(x_n, x_r)
                loss = compute_loss(pred, y, loss_type, margin=0.02)
                val_loss += loss.item() * x_n.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        epoch_time = time.time() - start_time
        current_lr = optim.param_groups[0]["lr"]

        print(f"[{epoch+1:03d}/{args.epochs}] "
              f"train={total_loss:.4f} val={val_loss:.4f} "
              f"lr={current_lr:.6f} time={epoch_time:.2f}s")

        # checkpoint + early stop
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            es_counter = 0
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"{args.domain}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "candidate_models": CANDIDATE_MODEL_SET,
                "window_size": args.window_size,
                "d_model": d_model,
                "mtc_patch_size": mtc_patch_size,
                "mtc_patch_stride": mtc_patch_stride,
                "mtc_downsample_ratio": mtc_down_ratio,
                "mtc_num_blocks": mtc_num_blocks,
                "mtc_large_sizes": mtc_large_sizes,
                "mtc_small_sizes": mtc_small_sizes,
                "mtc_dims": mtc_dims,
                "mtc_dw_dims": mtc_dw_dims,
                "dropout": dropout,
            }, save_path)
            print(f"✅ Saved to {save_path} (best_val={best_val:.4f})")
        else:
            es_counter += 1
            print(f"no improvement, early-stop counter = {es_counter}/{es_patience}")
        if es_counter >= es_patience:
            print("⏹ Early stopping triggered.")
            break

    print("done.")

# ------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end TS model recommender (ModernTCN backbone)")

    parser.add_argument('--domain', type=str, default='ID',
                        help='ID, WebService, Medical, Facility, Synthetic, HumanActivity, Sensor, Environment, Finance, Traffic')
    parser.add_argument('--file_list', type=str,
                        default='/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/file_list/TSB-AD-M-Label.csv')
    parser.add_argument('--dataset_dir', type=str,
                        default='/data/liuqinghua/code/ts/public_repo/TSB-AD/Datasets/TSB-AD-Datasets/TSB-AD-M/')
    parser.add_argument('--metric_path', type=str,
                        default='/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/Candidate/TSB-AD-M/')
                        
    parser.add_argument('--metric', type=str, default='VUS-PR')
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--save_dir', type=str,
                        default='/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/weights/TSB-AD-M/ChameleonRec_Ablation/')
    parser.add_argument('--precomputed_dir', type=str,
                        default='/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/precomputed_resid_processing_flow_stl_ad_M/')


    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)

    parser.add_argument('--mtc_patch_size', type=int, default=8)
    parser.add_argument('--mtc_patch_stride', type=int, default=8)
    parser.add_argument('--mtc_downsample_ratio', type=int, default=2)

    def _tuple_arg(s):
        if isinstance(s, tuple):
            return s
        if isinstance(s, list):
            return tuple(s)
        parts = [p for p in str(s).split(',') if p != '']
        return tuple(map(int, parts))

    parser.add_argument('--mtc_num_blocks', type=_tuple_arg, default=(2, 2))
    parser.add_argument('--mtc_large_sizes', type=_tuple_arg, default=(15, 15))
    parser.add_argument('--mtc_small_sizes', type=_tuple_arg, default=(3, 3))
    parser.add_argument('--mtc_dims',        type=_tuple_arg, default=(128, 256))
    parser.add_argument('--mtc_dw_dims',     type=_tuple_arg, default=(128, 256))

    parser.add_argument(
        '--loss_type',
        type=str,
        default='regression',
        choices=['ranking', 'regression', 'classification'],
        help='Which loss to use: ranking / regression / classification'
    )

    parser.add_argument(
        '--backbone_type',
        type=str,
        default='TCN',
        choices=['TCN', 'ConvNet', 'Transformer'],
    )

    args = parser.parse_args()
    train_one_file(args)

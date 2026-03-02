import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

import sys
sys.path.append("/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon")
from chameleon.NorAR.AnomalyResid import AnomalyResidualDecomposer, split_like_original
from chameleon.MolRec.ChameleonRec import ChameleonRec

# Ablation
from chameleon.MolRec.ChameleonRec_None import ChameleonRec_None
from chameleon.MolRec.ChameleonRec_Conv import ChameleonRec_Conv
from chameleon.MolRec.ChameleonRec_Transformer import ChameleonRec_Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decomposer = AnomalyResidualDecomposer(
    mode='stl_ad',
    robust_fallback=True,
    zscore_normalize=True
).to(device)

def _project_single_window(win, C_target, proj_method, pca_random_state=42):
    eps_var = 1e-12
    L_local, C_local = win.shape

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

    for method in proj_method:
        m = method.lower()
        if m == "pca":
            if C_local > C_target:
                total_var = float(np.var(win, axis=0).sum())
                if (L_local >= 2) and (total_var > eps_var):
                    n_comp = min(C_target, C_local, L_local)
                    pca = PCA(n_components=n_comp, random_state=pca_random_state)
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


def run_ChameleonRec(variant, data, Candidate_Model_Set, args, ranking=False):

    # 1) choose checkpoint by domain
    ckpt_dir = os.path.join(args.pretrained_weights, "ChameleonRec")
    if variant == "ID":
        model_path = os.path.join(ckpt_dir, "ID.pt")
    else:
        domain = args.filename.split("_")[4]
        dom_path = os.path.join(ckpt_dir, f"{domain}.pt")
        model_path = dom_path if os.path.exists(dom_path) else os.path.join(ckpt_dir, "ID.pt")

    ckpt_file = Path(model_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_file}")

    # 2) load checkpoint + metadata
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_file, map_location=device)

    saved_cands = ckpt.get("candidate_models", Candidate_Model_Set)
    cand_list   = saved_cands if saved_cands is not None else Candidate_Model_Set
    num_models  = len(cand_list)

    window_size = int(ckpt.get("window_size", getattr(args, "window_size", 1024)))
    d_model   = int(ckpt.get("d_model", getattr(args, "d_model", 256)))
    dropout   = float(ckpt.get("dropout", getattr(args, "dropout", 0.1)))
    mtc_patch_size   = int(ckpt.get("mtc_patch_size",   getattr(args, "mtc_patch_size", 8)))
    mtc_patch_stride = int(ckpt.get("mtc_patch_stride", getattr(args, "mtc_patch_stride", 8)))
    mtc_down_ratio   = int(ckpt.get("mtc_downsample_ratio", getattr(args, "mtc_downsample_ratio", 2)))
    mtc_num_blocks   = ckpt.get("mtc_num_blocks",       getattr(args, "mtc_num_blocks", (2, 2)))
    mtc_large_sizes  = ckpt.get("mtc_large_sizes",      getattr(args, "mtc_large_sizes", (15, 15)))
    mtc_small_sizes  = ckpt.get("mtc_small_sizes",      getattr(args, "mtc_small_sizes", (3, 3)))
    mtc_dims         = ckpt.get("mtc_dims",             getattr(args, "mtc_dims", (128, 256)))
    mtc_dw_dims      = ckpt.get("mtc_dw_dims",          getattr(args, "mtc_dw_dims", (128, 256)))

    data = data.astype(np.float32)  # (T, C)
    data = StandardScaler().fit_transform(data)
    win_list = split_like_original(data, window_size=window_size)
    if len(win_list) == 0:
        raise RuntimeError("No windows produced from input series.")
    windows = np.stack(win_list, axis=0)  # (B, L, C)
    B, L, C = windows.shape
    with torch.no_grad():
        ts = torch.from_numpy(windows).permute(0, 2, 1)   # (B, C, L)
        ts = ts.to(dtype=torch.float32)
        x_n, x_r = decomposer(ts)
    windows_n = x_n.permute(0, 2, 1).cpu().numpy().astype(np.float32)
    windows_r = x_r.permute(0, 2, 1).cpu().numpy().astype(np.float32)

    windows_n_orig = windows_n.copy()

    if C > 1:
        # MTS PCA-based channel alignment (per window, normal/residual processed separately)
        C_current = C
        C_target = 1
        proj_method = ["pca"]   # ["pca", "var_topk", "kurtosis_topk", "entropy_topk", "l1_topk"]
        C_Out = C_target * len(proj_method)

        B, L, C_current = windows_n.shape
        assert windows_r.shape == (B, L, C_current)

        windows_n_new = np.zeros((B, L, C_Out), dtype=np.float32)
        windows_r_new = np.zeros((B, L, C_Out), dtype=np.float32)

        for b in range(B):
            win_n = windows_n[b]  # (L, C_current)
            win_r = windows_r[b]  # (L, C_current)
            windows_n_new[b] = _project_single_window(win_n, C_target, proj_method)  # (L, C_target)
            windows_r_new[b] = _project_single_window(win_r, C_target, proj_method)  # (L, C_target)
        windows_n = windows_n_new
        windows_r = windows_r_new

        B, L, C_padded = windows_n.shape
        C = C_padded  # update C to target value

    # 3) build ChameleonRec
    in_channels = C
    model = ChameleonRec(
        in_channels=in_channels, num_models=num_models,
        d_model=d_model, dropout=dropout,
        patch_size=mtc_patch_size, patch_stride=mtc_patch_stride, downsample_ratio=mtc_down_ratio,
        num_blocks=mtc_num_blocks, large_size=mtc_large_sizes, small_size=mtc_small_sizes,
        dims=mtc_dims, dw_dims=mtc_dw_dims,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # 4) per-window prediction → aggregate ranks
    all_ranks = []
    with torch.no_grad():
        for wn, wr in zip(windows_n, windows_r):   # each (L, C)
            x_n = torch.from_numpy(wn).permute(1, 0).unsqueeze(0).to(device)  # (1, C, L)
            x_r = torch.from_numpy(wr).permute(1, 0).unsqueeze(0).to(device)  # (1, C, L)
            pred = model(x_n, x_r).squeeze(0).detach().cpu().numpy()  # (M,)
            r = np.argsort(pred)                 # ascending index list
            r = np.argsort(r)                    # convert to rank; highest pred -> M-1
            all_ranks.append(r)
    ranks = np.stack(all_ranks, axis=0)          # (W, M)
    agg_ranks = ranks.sum(axis=0)                # (M,)
    idx_sorted = np.argsort(agg_ranks)[::-1][:1]  # descending → top_k
    cand_list = saved_cands if saved_cands is not None else Candidate_Model_Set
    selected_model = [cand_list[i] for i in idx_sorted]
    print("selected_model:", selected_model[0])

    if ranking:
        return agg_ranks, (windows_n, windows_n_orig)
    
    # 5) load the corresponding detector score
    score_name = args.filename.split(".")[0]
    score_path = f"{args.score_dir}/{selected_model[0]}/{score_name}.npy"
    score = np.load(score_path)

    return score, True

def run_ChameleonRec_Sep(variant, data, Candidate_Model_Set, args, ranking=False):

    # 1) choose checkpoint by domain
    ckpt_dir = os.path.join(args.pretrained_weights, "ChameleonRec_Sep")
    if variant == "ID":
        model_path = os.path.join(ckpt_dir, "ID.pt")
    else:
        domain = args.filename.split("_")[4]
        dom_path = os.path.join(ckpt_dir, f"{domain}.pt")
        model_path = dom_path if os.path.exists(dom_path) else os.path.join(ckpt_dir, "ID.pt")

    ckpt_file = Path(model_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_file}")

    # 2) load checkpoint + metadata
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_file, map_location=device)

    saved_cands = ckpt.get("candidate_models", Candidate_Model_Set)
    cand_list   = saved_cands if saved_cands is not None else Candidate_Model_Set
    num_models  = len(cand_list)

    window_size = int(ckpt.get("window_size", getattr(args, "window_size", 1024)))
    d_model   = int(ckpt.get("d_model", getattr(args, "d_model", 256)))
    dropout   = float(ckpt.get("dropout", getattr(args, "dropout", 0.1)))
    mtc_patch_size   = int(ckpt.get("mtc_patch_size",   getattr(args, "mtc_patch_size", 8)))
    mtc_patch_stride = int(ckpt.get("mtc_patch_stride", getattr(args, "mtc_patch_stride", 8)))
    mtc_down_ratio   = int(ckpt.get("mtc_downsample_ratio", getattr(args, "mtc_downsample_ratio", 2)))
    mtc_num_blocks   = ckpt.get("mtc_num_blocks",       getattr(args, "mtc_num_blocks", (2, 2)))
    mtc_large_sizes  = ckpt.get("mtc_large_sizes",      getattr(args, "mtc_large_sizes", (15, 15)))
    mtc_small_sizes  = ckpt.get("mtc_small_sizes",      getattr(args, "mtc_small_sizes", (3, 3)))
    mtc_dims         = ckpt.get("mtc_dims",             getattr(args, "mtc_dims", (128, 256)))
    mtc_dw_dims      = ckpt.get("mtc_dw_dims",          getattr(args, "mtc_dw_dims", (128, 256)))

    data = data.astype(np.float32)  # (T, C)
    data = StandardScaler().fit_transform(data)
    win_list = split_like_original(data, window_size=window_size)
    if len(win_list) == 0:
        raise RuntimeError("No windows produced from input series.")
    windows = np.stack(win_list, axis=0)  # (B, L, C)
    B, L, C = windows.shape
    with torch.no_grad():
        ts = torch.from_numpy(windows).permute(0, 2, 1)   # (B, C, L)
        ts = ts.to(dtype=torch.float32)
        x_n, x_r = decomposer(ts)
    windows_n = x_n.permute(0, 2, 1).cpu().numpy().astype(np.float32)
    windows_r = x_r.permute(0, 2, 1).cpu().numpy().astype(np.float32)

    windows_n_orig = windows_n.copy()

    if C > 1:
        # MTS PCA-based channel alignment (per window, normal/residual processed separately)
        C_current = C
        C_target = 1
        proj_method = ["pca"]   # ["pca", "var_topk", "kurtosis_topk", "entropy_topk", "l1_topk"]
        C_Out = C_target * len(proj_method)

        B, L, C_current = windows_n.shape
        assert windows_r.shape == (B, L, C_current)

        windows_n_new = np.zeros((B, L, C_Out), dtype=np.float32)
        windows_r_new = np.zeros((B, L, C_Out), dtype=np.float32)

        for b in range(B):
            win_n = windows_n[b]  # (L, C_current)
            win_r = windows_r[b]  # (L, C_current)
            windows_n_new[b] = _project_single_window(win_n, C_target, proj_method)  # (L, C_target)
            windows_r_new[b] = _project_single_window(win_r, C_target, proj_method)  # (L, C_target)
        windows_n = windows_n_new
        windows_r = windows_r_new

        B, L, C_padded = windows_n.shape
        C = C_padded  # update C to target value

    # 3) build ChameleonRec
    in_channels = C
    model = ChameleonRec(
        in_channels=in_channels, num_models=num_models,
        d_model=d_model, dropout=dropout,
        patch_size=mtc_patch_size, patch_stride=mtc_patch_stride, downsample_ratio=mtc_down_ratio,
        num_blocks=mtc_num_blocks, large_size=mtc_large_sizes, small_size=mtc_small_sizes,
        dims=mtc_dims, dw_dims=mtc_dw_dims,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # 4) per-window prediction → aggregate ranks
    all_ranks = []
    with torch.no_grad():
        for wn, wr in zip(windows_n, windows_r):   # each (L, C)
            x_n = torch.from_numpy(wn).permute(1, 0).unsqueeze(0).to(device)  # (1, C, L)
            x_r = torch.from_numpy(wr).permute(1, 0).unsqueeze(0).to(device)  # (1, C, L)
            pred = model(x_n, x_r).squeeze(0).detach().cpu().numpy()  # (M,)
            r = np.argsort(pred)                 # ascending index list
            r = np.argsort(r)                    # convert to rank; highest pred -> M-1
            all_ranks.append(r)
    ranks = np.stack(all_ranks, axis=0)          # (W, M)
    agg_ranks = ranks.sum(axis=0)                # (M,)
    idx_sorted = np.argsort(agg_ranks)[::-1][:1]  # descending → top_k
    cand_list = saved_cands if saved_cands is not None else Candidate_Model_Set
    selected_model = [cand_list[i] for i in idx_sorted]
    print("selected_model:", selected_model[0])

    if ranking:
        return agg_ranks, (windows_n, windows_n_orig)
    
    # 5) load the corresponding detector score
    score_name = args.filename.split(".")[0]
    score_path = f"{args.score_dir}/{selected_model[0]}/{score_name}.npy"
    score = np.load(score_path)

    return score, True

'''
Ablation study
'''

def run_ChameleonRec_Ablation(variant, data, Candidate_Model_Set, args, ranking=False):

    # 1) choose checkpoint by domain
    ckpt_dir = os.path.join(args.pretrained_weights, "ChameleonRec_Ablation")
    if variant == "ID":
        model_path = os.path.join(ckpt_dir, "ID.pt")
    else:
        domain = args.filename.split("_")[4]
        dom_path = os.path.join(ckpt_dir, f"{domain}.pt")
        model_path = dom_path if os.path.exists(dom_path) else os.path.join(ckpt_dir, "ID.pt")

    ckpt_file = Path(model_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_file}")

    # 2) load checkpoint + metadata
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_file, map_location=device)

    saved_cands = ckpt.get("candidate_models", Candidate_Model_Set)
    cand_list   = saved_cands if saved_cands is not None else Candidate_Model_Set
    num_models  = len(cand_list)

    window_size = int(ckpt.get("window_size", getattr(args, "window_size", 1024)))
    d_model   = int(ckpt.get("d_model", getattr(args, "d_model", 256)))
    dropout   = float(ckpt.get("dropout", getattr(args, "dropout", 0.1)))
    mtc_patch_size   = int(ckpt.get("mtc_patch_size",   getattr(args, "mtc_patch_size", 8)))
    mtc_patch_stride = int(ckpt.get("mtc_patch_stride", getattr(args, "mtc_patch_stride", 8)))
    mtc_down_ratio   = int(ckpt.get("mtc_downsample_ratio", getattr(args, "mtc_downsample_ratio", 2)))
    mtc_num_blocks   = ckpt.get("mtc_num_blocks",       getattr(args, "mtc_num_blocks", (2, 2)))
    mtc_large_sizes  = ckpt.get("mtc_large_sizes",      getattr(args, "mtc_large_sizes", (15, 15)))
    mtc_small_sizes  = ckpt.get("mtc_small_sizes",      getattr(args, "mtc_small_sizes", (3, 3)))
    mtc_dims         = ckpt.get("mtc_dims",             getattr(args, "mtc_dims", (128, 256)))
    mtc_dw_dims      = ckpt.get("mtc_dw_dims",          getattr(args, "mtc_dw_dims", (128, 256)))

    data = data.astype(np.float32)  # (T, C)
    data = StandardScaler().fit_transform(data)
    win_list = split_like_original(data, window_size=window_size)
    if len(win_list) == 0:
        raise RuntimeError("No windows produced from input series.")
    windows = np.stack(win_list, axis=0)  # (B, L, C)
    B, L, C = windows.shape
    with torch.no_grad():
        ts = torch.from_numpy(windows).permute(0, 2, 1)   # (B, C, L)
        ts = ts.to(dtype=torch.float32)
        x_n, x_r = decomposer(ts)
    windows_n = x_n.permute(0, 2, 1).cpu().numpy().astype(np.float32)
    windows_r = x_r.permute(0, 2, 1).cpu().numpy().astype(np.float32)

    windows_n_orig = windows_n.copy()

    if C > 1:
        # MTS PCA-based channel alignment (per window, normal/residual processed separately)
        C_current = C
        C_target = 1
        proj_method = ["pca"]   # ["pca", "var_topk", "kurtosis_topk", "entropy_topk", "l1_topk"]
        C_Out = C_target * len(proj_method)

        B, L, C_current = windows_n.shape
        assert windows_r.shape == (B, L, C_current)

        windows_n_new = np.zeros((B, L, C_Out), dtype=np.float32)
        windows_r_new = np.zeros((B, L, C_Out), dtype=np.float32)

        for b in range(B):
            win_n = windows_n[b]  # (L, C_current)
            win_r = windows_r[b]  # (L, C_current)
            windows_n_new[b] = _project_single_window(win_n, C_target, proj_method)  # (L, C_target)
            windows_r_new[b] = _project_single_window(win_r, C_target, proj_method)  # (L, C_target)
        windows_n = windows_n_new
        windows_r = windows_r_new

        B, L, C_padded = windows_n.shape
        C = C_padded  # update C to target value

    # 3) build ChameleonRec
    in_channels = C
    # model = ChameleonRec(
    #     in_channels=in_channels, num_models=num_models,
    #     d_model=d_model, dropout=dropout,
    #     patch_size=mtc_patch_size, patch_stride=mtc_patch_stride, downsample_ratio=mtc_down_ratio,
    #     num_blocks=mtc_num_blocks, large_size=mtc_large_sizes, small_size=mtc_small_sizes,
    #     dims=mtc_dims, dw_dims=mtc_dw_dims,
    # ).to(device)

    model = ChameleonRec_Transformer(
        in_channels=in_channels, num_models=num_models,
        d_model=d_model
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # 4) per-window prediction → aggregate ranks
    all_ranks = []
    with torch.no_grad():
        for wn, wr in zip(windows_n, windows_r):   # each (L, C)
            x_n = torch.from_numpy(wn).permute(1, 0).unsqueeze(0).to(device)  # (1, C, L)
            x_r = torch.from_numpy(wr).permute(1, 0).unsqueeze(0).to(device)  # (1, C, L)
            pred = model(x_n, x_r).squeeze(0).detach().cpu().numpy()  # (M,)
            r = np.argsort(pred)                 # ascending index list
            r = np.argsort(r)                    # convert to rank; highest pred -> M-1
            all_ranks.append(r)
    ranks = np.stack(all_ranks, axis=0)          # (W, M)
    agg_ranks = ranks.sum(axis=0)                # (M,)
    idx_sorted = np.argsort(agg_ranks)[::-1][:1]  # descending → top_k
    cand_list = saved_cands if saved_cands is not None else Candidate_Model_Set
    selected_model = [cand_list[i] for i in idx_sorted]
    print("selected_model:", selected_model[0])

    if ranking:
        return agg_ranks, (windows_n, windows_n_orig)
    
    # 5) load the corresponding detector score
    score_name = args.filename.split(".")[0]
    score_path = f"{args.score_dir}/{selected_model[0]}/{score_name}.npy"
    score = np.load(score_path)

    return score, True

def run_ChameleonRec_Ablation_decomp(variant, data, Candidate_Model_Set, args, ranking=False):

    # 1) choose checkpoint by domain
    ckpt_dir = os.path.join(args.pretrained_weights, "ChameleonRec_Ablation_decomp")
    if variant == "ID":
        model_path = os.path.join(ckpt_dir, "ID.pt")
    else:
        domain = args.filename.split("_")[4]
        dom_path = os.path.join(ckpt_dir, f"{domain}.pt")
        model_path = dom_path if os.path.exists(dom_path) else os.path.join(ckpt_dir, "ID.pt")

    ckpt_file = Path(model_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_file}")

    # 2) load checkpoint + metadata
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_file, map_location=device)

    saved_cands = ckpt.get("candidate_models", Candidate_Model_Set)
    cand_list   = saved_cands if saved_cands is not None else Candidate_Model_Set
    num_models  = len(cand_list)

    window_size = int(ckpt.get("window_size", getattr(args, "window_size", 1024)))
    d_model   = int(ckpt.get("d_model", getattr(args, "d_model", 256)))
    dropout   = float(ckpt.get("dropout", getattr(args, "dropout", 0.1)))
    mtc_patch_size   = int(ckpt.get("mtc_patch_size",   getattr(args, "mtc_patch_size", 8)))
    mtc_patch_stride = int(ckpt.get("mtc_patch_stride", getattr(args, "mtc_patch_stride", 8)))
    mtc_down_ratio   = int(ckpt.get("mtc_downsample_ratio", getattr(args, "mtc_downsample_ratio", 2)))
    mtc_num_blocks   = ckpt.get("mtc_num_blocks",       getattr(args, "mtc_num_blocks", (2, 2)))
    mtc_large_sizes  = ckpt.get("mtc_large_sizes",      getattr(args, "mtc_large_sizes", (15, 15)))
    mtc_small_sizes  = ckpt.get("mtc_small_sizes",      getattr(args, "mtc_small_sizes", (3, 3)))
    mtc_dims         = ckpt.get("mtc_dims",             getattr(args, "mtc_dims", (128, 256)))
    mtc_dw_dims      = ckpt.get("mtc_dw_dims",          getattr(args, "mtc_dw_dims", (128, 256)))

    data = data.astype(np.float32)  # (T, C)
    data = StandardScaler().fit_transform(data)
    win_list = split_like_original(data, window_size=window_size)
    if len(win_list) == 0:
        raise RuntimeError("No windows produced from input series.")
    windows_n = np.stack(win_list, axis=0)  # (B, L, C)
    B, L, C = windows_n.shape

    windows_n_orig = windows_n.copy()

    if C > 1:
        # MTS PCA-based channel alignment (per window, normal/residual processed separately)
        C_current = C
        C_target = 1
        proj_method = ["pca"]   # ["pca", "var_topk", "kurtosis_topk", "entropy_topk", "l1_topk"]
        C_Out = C_target * len(proj_method)

        B, L, C_current = windows_n.shape
        assert windows_n.shape == (B, L, C_current)

        windows_n_new = np.zeros((B, L, C_Out), dtype=np.float32)

        for b in range(B):
            win_n = windows_n[b]  # (L, C_current)
            windows_n_new[b] = _project_single_window(win_n, C_target, proj_method)  # (L, C_target)
        windows_n = windows_n_new

        B, L, C_padded = windows_n.shape
        C = C_padded  # update C to target value

    # 3) build ChameleonRec
    in_channels = C
    model = ChameleonRec_None(
        in_channels=in_channels, num_models=num_models,
        d_model=d_model, dropout=dropout,
        patch_size=mtc_patch_size, patch_stride=mtc_patch_stride, downsample_ratio=mtc_down_ratio,
        num_blocks=mtc_num_blocks, large_size=mtc_large_sizes, small_size=mtc_small_sizes,
        dims=mtc_dims, dw_dims=mtc_dw_dims,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # 4) per-window prediction → aggregate ranks
    all_ranks = []
    with torch.no_grad():
        for wn in windows_n:   # each (L, C)
            x_n = torch.from_numpy(wn).permute(1, 0).unsqueeze(0).to(device)  # (1, C, L)
            pred = model(x_n).squeeze(0).detach().cpu().numpy()  # (M,)
            r = np.argsort(pred)                 # ascending index list
            r = np.argsort(r)                    # convert to rank; highest pred -> M-1
            all_ranks.append(r)
    ranks = np.stack(all_ranks, axis=0)          # (W, M)
    agg_ranks = ranks.sum(axis=0)                # (M,)
    idx_sorted = np.argsort(agg_ranks)[::-1][:1]  # descending → top_k
    cand_list = saved_cands if saved_cands is not None else Candidate_Model_Set
    selected_model = [cand_list[i] for i in idx_sorted]
    print("selected_model:", selected_model[0])

    if ranking:
        return agg_ranks, (windows_n, windows_n_orig)
    
    # 5) load the corresponding detector score
    score_name = args.filename.split(".")[0]
    score_path = f"{args.score_dir}/{selected_model[0]}/{score_name}.npy"
    score = np.load(score_path)

    return score, True


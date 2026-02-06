import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.model_selection import ParameterGrid
import ast

import sys
sys.path.append("/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon")

from chameleon.NorAR.AnomalyResid import AnomalyResidualDecomposer, split_like_original
from chameleon.MolRec.utils import *
from chameleon.ModelOpt.AnomalyInjection import ANOMALY_PARAM_GRID, InjectAnomalies, gen_synthetic_performance_list
from chameleon.ModelOpt.utils import *


def _minmax(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (max(hi - lo, eps))

def _even_sample_indices(n_items: int, k: int) -> np.ndarray:
    if k >= n_items:
        return np.arange(n_items, dtype=int)
    return np.unique(np.linspace(0, n_items - 1, num=k, dtype=int))

def run_Avg_Ens(variant, data, Candidate_Model_Set, args, precomputed=True):
    from chameleon.ModelOpt.OE import Avg_Ens
    flag = True
    if precomputed:
        det_scores = loading_scores(Candidate_Model_Set, args)
    score = Avg_Ens(det_scores)
    return score, flag

def run_ChameleonOpt_precomputed(variant, data, Candidate_Model_Set, args):

    ranking_list = []

    agg_ranks, (windows_n, windows_n_orig) = run_ChameleonRec(variant=variant, data=data, Candidate_Model_Set=Candidate_Model_Set, args=args, ranking=True)

    num_models  = len(Candidate_Model_Set)
    idx_sorted = np.argsort(agg_ranks)[::-1][:]  # descending → top_k
    selected_model = [Candidate_Model_Set[i] for i in idx_sorted]

    ranking_list.append(selected_model)
    print("ranking orig:", ranking_list[0])

    try:

        # 7) partial re-ranking via synthetic injection
        rng = np.random.default_rng(getattr(args, "seed", 0))

        # --- TOP-K DETECTOR SAMPLING (NOT RANDOM) ---
        # choose top-K detectors by original aggregated ranks
        # topK_default = max(1, int(round(0.20 * num_models)))
        topK_default = 5
        topK = int(getattr(args, "topK", topK_default))
        topK = max(1, min(topK, num_models))

        # indices of models sorted by prior rank (descending: best first)
        prior_order   = np.argsort(agg_ranks)[::-1]
        sampled_indices = prior_order[:topK]              # global indices of sampled detectors
        sampled_Det_pool = [Candidate_Model_Set[i] for i in sampled_indices]

        # choose windows
        W_total = windows_n.shape[0]
        W_sample = min(
            max(getattr(args, "min_windows", 4), int(round(0.1 * W_total))),
            min(getattr(args, "max_windows", 8), W_total)
        )

        # --- informative windows sampling ---
        use_informative = getattr(args, "informative_windows", True)
        spread = getattr(args, "informative_spread", True)
        top_factor = getattr(args, "informative_top_factor", 3)

        # choose which window bank to score
        window_bank = windows_n if data.shape[1] == 1 else windows_n_orig

        if use_informative:
            sampled_w_idx = select_informative_window_indices(
                windows=window_bank,
                W_sample=W_sample,
                rng=rng,
                top_factor=top_factor,
                spread=spread
            )
        else:
            sampled_w_idx = _even_sample_indices(W_total, W_sample)

        # sample normal windows for injection
        sampled_windows = [window_bank[i] for i in sampled_w_idx]

        # pick anomaly types and iterate ParameterGrid for each
        types_available = list(ANOMALY_PARAM_GRID.keys())
        max_types = getattr(args, "max_anomaly_types", 4)
        if len(types_available) <= max_types:
            anomaly_type_pool = types_available
        else:
            anomaly_type_pool = list(rng.choice(np.array(types_available, dtype=object), size=max_types, replace=False))

        # anomaly_type_pool = ['speedup']

        anomaly_obj = InjectAnomalies(
            random_state=int(rng.integers(0, 10_000)),
            verbose=False,
            max_window_size=128,
            min_window_size=8
        )

        # accumulate performance per sampled detector
        per_det_scores = np.zeros(len(sampled_Det_pool), dtype=float)
        per_det_counts = np.zeros(len(sampled_Det_pool), dtype=int)
        per_det_hist = [[] for _ in range(len(sampled_Det_pool))]

        grid_sizes = {a: len(list(ParameterGrid(ANOMALY_PARAM_GRID[a]))) for a in anomaly_type_pool}
        n_calls = len(sampled_windows) * sum(grid_sizes.values())
        print("W_sample =", len(sampled_windows))
        print("Grid sizes:", grid_sizes)
        print("Total gen_synthetic_performance_list calls =", n_calls)
        print("Total detector runs =", n_calls * len(sampled_Det_pool))

        for w in sampled_windows:
            w_std = float(np.maximum(np.std(w), 1e-2))

            for a_type in anomaly_type_pool:
                grid_dict = ANOMALY_PARAM_GRID.get(a_type, None)

                if grid_dict is None:
                    continue

                for params in ParameterGrid(grid_dict):
                    p = dict(params)  # copy
                    # scale amplitude-like knobs by window std (if present)
                    if "scale" in p and isinstance(p["scale"], (int, float)):
                        p["scale"] = float(p["scale"]) * w_std
                    # attach series and type
                    p["T"] = w.T
                    p["anomaly_type"] = a_type

                    try:
                        T_a, anomaly_sizes, anomaly_labels = anomaly_obj.inject_anomalies(**p)
                    except Exception as e:
                        print(f"[inject_anomalies] skip due to error: {e}")
                        continue

                    if T_a is None or anomaly_labels is None:
                        continue
                    if len(anomaly_labels) == 0:
                        continue

                    syn_data = T_a.T
                    lbl = np.asarray(anomaly_labels).astype(int)
                    perf_list = gen_synthetic_performance_list(syn_data, lbl, sampled_Det_pool)

                    perf_arr = np.asarray(perf_list, dtype=float)
                    valid = np.isfinite(perf_arr)
                    per_det_scores[valid] += perf_arr[valid]
                    per_det_counts[valid] += 1
                    for j in np.where(valid)[0]:
                        per_det_hist[j].append(float(perf_arr[j]))


        # ---------- RANKING UPDATE WITH PARTIAL EVIDENCE ----------
        # Fallback to original best if nothing evaluated
        if per_det_counts.sum() == 0 or not np.any(per_det_counts):
            print('Fallback to original best if nothing evaluated ⚠️')
            # best_idx = int(np.argmax(agg_ranks))
            # selected_model = cand_list[best_idx]
            idx_sorted = np.argsort(agg_ranks)[::-1][:]
        else:
            # K-length: average performance within the sampled pool (local indexing)
            avg_perf = np.divide(per_det_scores, np.maximum(per_det_counts, 1), dtype=float)  # shape (K,)

            # normalize sampled detectors' perf to [0,1] (local)
            finite_mask = np.isfinite(avg_perf)
            if not finite_mask.any():
                perf01 = np.zeros_like(avg_perf)
            else:
                lo = float(np.nanmin(avg_perf[finite_mask]))
                hi = float(np.nanmax(avg_perf[finite_mask]))
                denom = (hi - lo) if (hi > lo) else 1.0
                perf01 = np.clip((np.nan_to_num(avg_perf, nan=lo) - lo) / denom, 0.0, 1.0)

            base01 = _minmax(agg_ranks)  # global length (num_models,)

            alpha = compute_alpha_from_uncertainty(per_det_hist, per_det_counts, args)
            blended = base01.copy()

            # Blend in logit space for more contrast
            def _logit(x, eps=1e-6):
                x = np.clip(x, eps, 1.0 - eps)
                return np.log(x / (1.0 - x))
            base_logit = _logit(base01)
            blended_logit = base_logit.copy()
            for j, det_idx in enumerate(sampled_indices):
                blended_logit[det_idx] = alpha * base_logit[det_idx] + (1.0 - alpha) * _logit(perf01[j])
            blended = 1.0 / (1.0 + np.exp(-blended_logit))

            # best_idx = int(np.argmax(blended))
            # selected_model = cand_list[best_idx]
            idx_sorted = np.argsort(blended)[::-1][:]

        # print(f"[re-ranking] picked: {selected_model}")

    except:
        print('Exception occurred during re-ranking, fallback to original ranking.')
        idx_sorted = np.argsort(agg_ranks)[::-1][:]
    
    # Select top models for ensembling
    selected_models = [Candidate_Model_Set[i] for i in idx_sorted]
    ranking_list.append(selected_models)
    print("ranking opt:", ranking_list[1])
    return ranking_list, True


def run_ChameleonEns_U_ID(variant, data, Candidate_Model_Set, args):

    df = pd.read_csv(f"{args.save_dir}/ChameleonOpt_precomputed_ID.csv")

    s = df[df['file'] == args.filename]['RankingOrig'].to_list()[0]
    selected_models = ast.literal_eval(s)[:int(variant)]
    print("Selected models for ensembling:", selected_models)

    scores_list = []
    for selected_model in selected_models:
        score_name = args.filename.split('.')[0]
        if os.path.exists(f'{args.score_dir}/{selected_model}/{score_name}.npy'):
            score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            if len(score) == len(data):
                scores_list.append(score)            
    det_scores = np.array(scores_list).T

    if len(scores_list) == 1:
        return det_scores.ravel(), True
    scaler = StandardScaler()
    scaler.fit(det_scores)
    standardized_det_scores = scaler.transform(det_scores)
    avg_ens_scores = np.mean(standardized_det_scores, axis=1)
    return avg_ens_scores, True

def run_ChameleonEns_U_OOD(variant, data, Candidate_Model_Set, args):

    df = pd.read_csv(f"{args.save_dir}/ChameleonOpt_precomputed_OOD.csv")

    s = df[df['file'] == args.filename]['RankingOrig'].to_list()[0]
    selected_models = ast.literal_eval(s)[:int(variant)]
    print("Selected models for ensembling:", selected_models)

    scores_list = []
    for selected_model in selected_models:
        score_name = args.filename.split('.')[0]
        if os.path.exists(f'{args.score_dir}/{selected_model}/{score_name}.npy'):
            score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            if len(score) == len(data):
                scores_list.append(score)            
    det_scores = np.array(scores_list).T

    if len(scores_list) == 1:
        return det_scores.ravel(), True
    scaler = StandardScaler()
    scaler.fit(det_scores)
    standardized_det_scores = scaler.transform(det_scores)
    avg_ens_scores = np.mean(standardized_det_scores, axis=1)
    return avg_ens_scores, True

def run_ChameleonEns_M_ID(variant, data, Candidate_Model_Set, args):

    df = pd.read_csv(f"{args.save_dir}/ChameleonOpt_precomputed_ID.csv")

    s = df[df['file'] == args.filename]['RankingOrig'].to_list()[0]
    selected_models = ast.literal_eval(s)[:int(variant)]
    print("Selected models for ensembling:", selected_models)

    scores_list = []
    for selected_model in selected_models:
        score_name = args.filename.split('.')[0]
        if os.path.exists(f'{args.score_dir}/{selected_model}/{score_name}.npy'):
            score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            if len(score) == len(data):
                scores_list.append(score)            
    det_scores = np.array(scores_list).T

    if len(scores_list) == 1:
        return det_scores.ravel(), True
    scaler = StandardScaler()
    scaler.fit(det_scores)
    standardized_det_scores = scaler.transform(det_scores)
    avg_ens_scores = np.mean(standardized_det_scores, axis=1)
    return avg_ens_scores, True

def run_ChameleonEns_M_OOD(variant, data, Candidate_Model_Set, args):

    df = pd.read_csv(f"{args.save_dir}/ChameleonOpt_precomputed_OOD.csv")

    s = df[df['file'] == args.filename]['RankingOrig'].to_list()[0]
    selected_models = ast.literal_eval(s)[:int(variant)]
    print("Selected models for ensembling:", selected_models)

    scores_list = []
    for selected_model in selected_models:
        score_name = args.filename.split('.')[0]
        if os.path.exists(f'{args.score_dir}/{selected_model}/{score_name}.npy'):
            score = np.load(f'{args.score_dir}/{selected_model}/{score_name}.npy')
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            if len(score) == len(data):
                scores_list.append(score)            
    det_scores = np.array(scores_list).T

    if len(scores_list) == 1:
        return det_scores.ravel(), True
    scaler = StandardScaler()
    scaler.fit(det_scores)
    standardized_det_scores = scaler.transform(det_scores)
    avg_ens_scores = np.mean(standardized_det_scores, axis=1)
    return avg_ens_scores, True

def run_ChameleonOpt_U_ID(variant, data, Candidate_Model_Set, args, alpha=0.7, temp=1.0, w_clip=(0.05, 0.60)):

    df = pd.read_csv(f"{args.save_dir}/ChameleonOpt_precomputed_ID.csv")

    s_orig = df[df["file"] == args.filename]["RankingOrig"].to_list()[0]
    s_opt  = df[df["file"] == args.filename]["RankingOpt"].to_list()[0]

    ranking_orig = ast.literal_eval(s_orig)
    ranking_opt  = ast.literal_eval(s_opt)

    # ---- pick top-k from ranking_opt (as you already do)
    k = int(variant)
    selected_models = ranking_opt[:k]
    print("Selected models for ensembling:", selected_models)

    # ---- load scores (keep only valid models)
    score_name = args.filename.split(".")[0]
    scores_list = []
    valid_models = []

    for m in selected_models:
        p = f"{args.score_dir}/{m}/{score_name}.npy"
        if os.path.exists(p):
            score = np.load(p)
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
            if len(score) == len(data):
                scores_list.append(score)
                valid_models.append(m)

    if len(scores_list) == 0:
        return None, False

    det_scores = np.array(scores_list).T  # [T, M]

    if det_scores.shape[1] == 1:
        return det_scores.ravel(), True

    # ---- standardize per time point across models
    scaler = StandardScaler()
    standardized = scaler.fit_transform(det_scores)
    standardized[standardized < 0] = 0  # keep your non-neg truncation

    # ---- compute weights from both rankings (aligned with valid_models)
    w_opt  = rank_to_weight(ranking_opt,  valid_models, temp=temp)
    w_orig = rank_to_weight(ranking_orig, valid_models, temp=temp)

    # combine: alpha * opt + (1-alpha) * orig
    w = alpha * w_opt + (1.0 - alpha) * w_orig

    # ---- robustness: clip weights then renormalize
    if w_clip is not None:
        lo, hi = w_clip
        w = np.clip(w, lo, hi)
        w = w / (w.sum() + 1e-12)

    # ---- weighted ensemble
    ens = standardized @ w  # [T,]
    return ens, True

def run_ChameleonOpt_U_OOD(variant, data, Candidate_Model_Set, args, alpha=0.7, temp=1.0, w_clip=(0.05, 0.60)):

    df = pd.read_csv(f"{args.save_dir}/ChameleonOpt_precomputed_OOD.csv")

    s_orig = df[df["file"] == args.filename]["RankingOrig"].to_list()[0]
    s_opt  = df[df["file"] == args.filename]["RankingOpt"].to_list()[0]

    ranking_orig = ast.literal_eval(s_orig)
    ranking_opt  = ast.literal_eval(s_opt)

    # ---- pick top-k from ranking_opt (as you already do)
    k = int(variant)
    selected_models = ranking_opt[:k]
    print("Selected models for ensembling:", selected_models)

    # ---- load scores (keep only valid models)
    score_name = args.filename.split(".")[0]
    scores_list = []
    valid_models = []

    for m in selected_models:
        p = f"{args.score_dir}/{m}/{score_name}.npy"
        if os.path.exists(p):
            score = np.load(p)
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
            if len(score) == len(data):
                scores_list.append(score)
                valid_models.append(m)

    if len(scores_list) == 0:
        return None, False

    det_scores = np.array(scores_list).T  # [T, M]

    if det_scores.shape[1] == 1:
        return det_scores.ravel(), True

    # ---- standardize per time point across models
    scaler = StandardScaler()
    standardized = scaler.fit_transform(det_scores)
    standardized[standardized < 0] = 0  # keep your non-neg truncation

    # ---- compute weights from both rankings (aligned with valid_models)
    w_opt  = rank_to_weight(ranking_opt,  valid_models, temp=temp)
    w_orig = rank_to_weight(ranking_orig, valid_models, temp=temp)

    # combine: alpha * opt + (1-alpha) * orig
    w = alpha * w_opt + (1.0 - alpha) * w_orig

    # ---- robustness: clip weights then renormalize
    if w_clip is not None:
        lo, hi = w_clip
        w = np.clip(w, lo, hi)
        w = w / (w.sum() + 1e-12)

    # ---- weighted ensemble
    ens = standardized @ w  # [T,]
    return ens, True

def run_ChameleonOpt_M_ID(variant, data, Candidate_Model_Set, args, alpha=0.7, temp=1.0, w_clip=(0.05, 0.60)):

    df = pd.read_csv(f"{args.save_dir}/ChameleonOpt_precomputed_ID.csv")

    s_orig = df[df["file"] == args.filename]["RankingOrig"].to_list()[0]
    s_opt  = df[df["file"] == args.filename]["RankingOpt"].to_list()[0]

    ranking_orig = ast.literal_eval(s_orig)
    ranking_opt  = ast.literal_eval(s_opt)

    # ---- pick top-k from ranking_opt (as you already do)
    k = int(variant)
    selected_models = ranking_opt[:k]
    print("Selected models for ensembling:", selected_models)

    # ---- load scores (keep only valid models)
    score_name = args.filename.split(".")[0]
    scores_list = []
    valid_models = []

    for m in selected_models:
        p = f"{args.score_dir}/{m}/{score_name}.npy"
        if os.path.exists(p):
            score = np.load(p)
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
            if len(score) == len(data):
                scores_list.append(score)
                valid_models.append(m)

    if len(scores_list) == 0:
        return None, False

    det_scores = np.array(scores_list).T  # [T, M]

    if det_scores.shape[1] == 1:
        return det_scores.ravel(), True

    # ---- standardize per time point across models
    scaler = StandardScaler()
    standardized = scaler.fit_transform(det_scores)
    standardized[standardized < 0] = 0  # keep your non-neg truncation

    # ---- compute weights from both rankings (aligned with valid_models)
    w_opt  = rank_to_weight(ranking_opt,  valid_models, temp=temp)
    w_orig = rank_to_weight(ranking_orig, valid_models, temp=temp)

    # combine: alpha * opt + (1-alpha) * orig
    w = alpha * w_opt + (1.0 - alpha) * w_orig

    # ---- robustness: clip weights then renormalize
    if w_clip is not None:
        lo, hi = w_clip
        w = np.clip(w, lo, hi)
        w = w / (w.sum() + 1e-12)

    # ---- weighted ensemble
    ens = standardized @ w  # [T,]
    return ens, True

def run_ChameleonOpt_M_OOD(variant, data, Candidate_Model_Set, args, alpha=0.7, temp=1.0, w_clip=(0.05, 0.60)):

    df = pd.read_csv(f"{args.save_dir}/ChameleonOpt_precomputed_OOD.csv")

    s_orig = df[df["file"] == args.filename]["RankingOrig"].to_list()[0]
    s_opt  = df[df["file"] == args.filename]["RankingOpt"].to_list()[0]

    ranking_orig = ast.literal_eval(s_orig)
    ranking_opt  = ast.literal_eval(s_opt)

    # ---- pick top-k from ranking_opt (as you already do)
    k = int(variant)
    selected_models = ranking_opt[:k]
    print("Selected models for ensembling:", selected_models)

    # ---- load scores (keep only valid models)
    score_name = args.filename.split(".")[0]
    scores_list = []
    valid_models = []

    for m in selected_models:
        p = f"{args.score_dir}/{m}/{score_name}.npy"
        if os.path.exists(p):
            score = np.load(p)
            score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
            if len(score) == len(data):
                scores_list.append(score)
                valid_models.append(m)

    if len(scores_list) == 0:
        return None, False

    det_scores = np.array(scores_list).T  # [T, M]

    if det_scores.shape[1] == 1:
        return det_scores.ravel(), True

    # ---- standardize per time point across models
    scaler = StandardScaler()
    standardized = scaler.fit_transform(det_scores)
    standardized[standardized < 0] = 0  # keep your non-neg truncation

    # ---- compute weights from both rankings (aligned with valid_models)
    w_opt  = rank_to_weight(ranking_opt,  valid_models, temp=temp)
    w_orig = rank_to_weight(ranking_orig, valid_models, temp=temp)

    # combine: alpha * opt + (1-alpha) * orig
    w = alpha * w_opt + (1.0 - alpha) * w_orig

    # ---- robustness: clip weights then renormalize
    if w_clip is not None:
        lo, hi = w_clip
        w = np.clip(w, lo, hi)
        w = w / (w.sum() + 1e-12)

    # ---- weighted ensemble
    ens = standardized @ w  # [T,]
    return ens, True


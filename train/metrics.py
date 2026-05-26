"""
train/metrics.py

Comprehensive retrieval metrics for ImageWeave.
All metrics support both directions (Image→Text and Text→Image).

Metrics included:
  RECALL:   R@1, R@5, R@10, R@20, R@50
  RANKING:  MRR, MAP, Median Rank, Mean Rank
  NDCG:     nDCG@5, nDCG@10
  EMBEDDING: Alignment, Uniformity, Mean Cosine Sim (pos/neg)
  ACCURACY: Top-1, Top-5 (== R@1, R@5 — explicit for clarity)
  SPREAD:   Std Rank, % in top-5%, % in top-10%
  TEMPERATURE: Effective temperature (if available)
"""

import torch
import torch.nn.functional as F
import math


# ============================================================
# CORE RANK COMPUTATION
# ============================================================

def _compute_ranks(similarity_matrix: torch.Tensor) -> torch.Tensor:
    """
    Given NxN similarity matrix, compute the rank of the correct
    match (diagonal) for each query.
    Returns: 1-indexed rank tensor of shape [N].
    """
    N = similarity_matrix.shape[0]
    ranks = []
    for i in range(N):
        ranking = torch.argsort(similarity_matrix[i], descending=True)
        rank    = (ranking == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    return torch.tensor(ranks, dtype=torch.float32)


# ============================================================
# NDCG (Normalized Discounted Cumulative Gain)
# For single relevant item: nDCG@k = 1/log2(rank+1) if rank<=k, else 0
# Ideal DCG@k = 1.0 always (since best case = rank 1)
# ============================================================

def _ndcg_at_k(ranks: torch.Tensor, k: int) -> float:
    dcg   = (ranks <= k).float() * (1.0 / torch.log2(ranks + 1.0))
    ideal = 1.0   # ideal DCG is always 1.0 (rank=1 → 1/log2(2)=1)
    return (dcg / ideal).mean().item()


# ============================================================
# MAP (Mean Average Precision)
# For single relevant item: AP = 1/rank, same as MRR
# Kept separate for explicitness in paper tables.
# ============================================================

def _map(ranks: torch.Tensor) -> float:
    return (1.0 / ranks).mean().item()


# ============================================================
# ALIGNMENT & UNIFORMITY (Wang et al. 2020)
# Alignment: how close are positive pairs in embedding space?
# Uniformity: how spread are embeddings on the hypersphere?
# Both are useful contrastive learning diagnostics.
# ============================================================

def _alignment_uniformity(
    image_embeddings: torch.Tensor,
    text_embeddings:  torch.Tensor
) -> dict:
    """
    image_embeddings, text_embeddings: L2-normalized [N, D]
    """
    # Alignment: mean L2 distance between positive pairs (lower is better)
    alignment = (image_embeddings - text_embeddings).norm(dim=1).pow(2).mean().item()

    # Uniformity: log of mean pairwise Gaussian kernel (lower magnitude = more uniform)
    sq_pdist_img = torch.pdist(image_embeddings, p=2).pow(2)
    sq_pdist_txt = torch.pdist(text_embeddings,  p=2).pow(2)
    uniformity_img = sq_pdist_img.mul(-2).exp().mean().log().item()
    uniformity_txt = sq_pdist_txt.mul(-2).exp().mean().log().item()

    return {
        "alignment":     alignment,
        "uniformity_img": uniformity_img,
        "uniformity_txt": uniformity_txt,
    }


# ============================================================
# COSINE SIMILARITY STATS
# ============================================================

def _cosine_stats(similarity_matrix: torch.Tensor) -> dict:
    """
    Compute statistics over the similarity matrix.
    Diagonal = positive pairs, off-diagonal = negative pairs.
    """
    N = similarity_matrix.shape[0]

    # Positive pair similarities (diagonal)
    pos_sim = similarity_matrix.diagonal()

    # Negative pair similarities (off-diagonal)
    mask     = ~torch.eye(N, dtype=torch.bool)
    neg_sim  = similarity_matrix[mask]

    return {
        "pos_cosine_mean": pos_sim.mean().item(),
        "pos_cosine_std":  pos_sim.std().item(),
        "neg_cosine_mean": neg_sim.mean().item(),
        "neg_cosine_std":  neg_sim.std().item(),
        "pos_neg_gap":     (pos_sim.mean() - neg_sim.mean()).item(),
    }


# ============================================================
# MAIN METRICS FUNCTION
# ============================================================

def retrieval_metrics(
    similarity_matrix:  torch.Tensor,
    image_embeddings:   torch.Tensor = None,
    text_embeddings:    torch.Tensor = None,
    direction:          str          = "i2t",   # "i2t" or "t2i"
    prefix:             str          = "",       # e.g. "val_" or "train_"
) -> dict:
    """
    Compute ALL retrieval metrics from a similarity matrix.

    Args:
        similarity_matrix : [N, N] float tensor (rows=queries, cols=gallery)
        image_embeddings  : optional [N, D] L2-normalized, for alignment/uniformity
        text_embeddings   : optional [N, D] L2-normalized, for alignment/uniformity
        direction         : "i2t" (image query → text gallery)
                          or "t2i" (text query → image gallery)
        prefix            : string prefix for all metric keys

    Returns:
        dict of all metrics with string keys
    """
    N     = similarity_matrix.shape[0]
    ranks = _compute_ranks(similarity_matrix)

    p = prefix   # shorthand

    metrics = {
        # ------------------------------------------------
        # RECALL @ K
        # ------------------------------------------------
        f"{p}R@1":   (ranks <= 1).float().mean().item(),
        f"{p}R@5":   (ranks <= 5).float().mean().item(),
        f"{p}R@10":  (ranks <= 10).float().mean().item(),
        f"{p}R@20":  (ranks <= 20).float().mean().item(),
        f"{p}R@50":  (ranks <= 50).float().mean().item(),

        # ------------------------------------------------
        # RANKING QUALITY
        # ------------------------------------------------
        f"{p}MRR":          (1.0 / ranks).mean().item(),
        f"{p}MAP":          _map(ranks),                        # same as MRR for single-relevant
        f"{p}median_rank":  float(ranks.median().item()),
        f"{p}mean_rank":    float(ranks.mean().item()),
        f"{p}std_rank":     float(ranks.std().item()),

        # ------------------------------------------------
        # NDCG
        # ------------------------------------------------
        f"{p}nDCG@5":   _ndcg_at_k(ranks, 5),
        f"{p}nDCG@10":  _ndcg_at_k(ranks, 10),

        # ------------------------------------------------
        # ACCURACY (explicit aliases — same as R@1, R@5)
        # ------------------------------------------------
        f"{p}top1_acc":  (ranks <= 1).float().mean().item(),
        f"{p}top5_acc":  (ranks <= 5).float().mean().item(),

        # ------------------------------------------------
        # PRECISION & F1 SCORE
        # ------------------------------------------------
        f"{p}Precision@1":  (ranks <= 1).float().mean().item(),
        f"{p}Precision@5":  (ranks <= 5).float().mean().item() / 5.0,
        f"{p}Precision@10": (ranks <= 10).float().mean().item() / 10.0,
        f"{p}F1@1":          (ranks <= 1).float().mean().item(),
        f"{p}F1@5":          2.0 * (ranks <= 5).float().mean().item() / 6.0,
        f"{p}F1@10":         2.0 * (ranks <= 10).float().mean().item() / 11.0,

        # ------------------------------------------------
        # SPREAD / COVERAGE
        # ------------------------------------------------
        f"{p}pct_top5pct":  (ranks <= max(1, int(N * 0.05))).float().mean().item(),
        f"{p}pct_top10pct": (ranks <= max(1, int(N * 0.10))).float().mean().item(),

        # ------------------------------------------------
        # META
        # ------------------------------------------------
        f"{p}direction":    direction,
        f"{p}n_queries":    N,
    }

    # ------------------------------------------------
    # COSINE SIMILARITY STATS (from matrix)
    # ------------------------------------------------
    cosine_stats = _cosine_stats(similarity_matrix.float())
    for k, v in cosine_stats.items():
        metrics[f"{p}{k}"] = v

    # ------------------------------------------------
    # ALIGNMENT & UNIFORMITY (from embeddings)
    # ------------------------------------------------
    if image_embeddings is not None and text_embeddings is not None:
        au = _alignment_uniformity(image_embeddings.float(), text_embeddings.float())
        for k, v in au.items():
            metrics[f"{p}{k}"] = v

    return metrics


# ============================================================
# BIDIRECTIONAL METRICS (both i2t and t2i in one call)
# ============================================================

def bidirectional_metrics(
    image_embeddings: torch.Tensor,
    text_embeddings:  torch.Tensor,
) -> dict:
    """
    Compute metrics in BOTH retrieval directions.

    Args:
        image_embeddings: [N, D] L2-normalized
        text_embeddings:  [N, D] L2-normalized

    Returns:
        dict with i2t_ and t2i_ prefixed keys, plus averaged keys.
    """
    # Similarity matrix: rows = image queries, cols = text gallery
    sim = image_embeddings @ text_embeddings.T

    i2t = retrieval_metrics(
        sim,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        direction="i2t",
        prefix="i2t_"
    )

    t2i = retrieval_metrics(
        sim.T,
        image_embeddings=text_embeddings,
        text_embeddings=image_embeddings,
        direction="t2i",
        prefix="t2i_"
    )

    # Averaged metrics (standard practice in retrieval papers)
    avg = {
        "avg_R@1":        (i2t["i2t_R@1"]  + t2i["t2i_R@1"])  / 2,
        "avg_R@5":        (i2t["i2t_R@5"]  + t2i["t2i_R@5"])  / 2,
        "avg_R@10":       (i2t["i2t_R@10"] + t2i["t2i_R@10"]) / 2,
        "avg_MRR":        (i2t["i2t_MRR"]  + t2i["t2i_MRR"])  / 2,
        "avg_nDCG@10":    (i2t["i2t_nDCG@10"] + t2i["t2i_nDCG@10"]) / 2,
        "avg_Precision@5": (i2t["i2t_Precision@5"] + t2i["t2i_Precision@5"]) / 2,
        "avg_F1@5":        (i2t["i2t_F1@5"] + t2i["t2i_F1@5"]) / 2,
        "rSum":           i2t["i2t_R@1"] + i2t["i2t_R@5"] + i2t["i2t_R@10"]
                        + t2i["t2i_R@1"] + t2i["t2i_R@5"] + t2i["t2i_R@10"],
    }

    return {**i2t, **t2i, **avg}


# ============================================================
# PRETTY PRINTER FOR TRAINER / EVALUATE.PY
# ============================================================

# ============================================================
# SIMPLE ALIAS — used for last-batch TRAIN metrics (fast, i2t only)
# ============================================================

def simple_retrieval_metrics(similarity_matrix: torch.Tensor) -> dict:
    """Lightweight single-direction i2t metrics for train-batch reporting."""
    ranks = _compute_ranks(similarity_matrix.float())
    return {
        "R@1":  (ranks <= 1).float().mean().item(),
        "R@5":  (ranks <= 5).float().mean().item(),
        "R@10": (ranks <= 10).float().mean().item(),
        "MRR":  (1.0 / ranks).mean().item(),
    }


def print_metrics(metrics: dict, header: str = "METRICS"):
    """Formatted print of all computed metrics."""
    width = 62
    print("\n" + "=" * width)
    print(f"  {header}")
    print("=" * width)

    # Group by prefix
    groups = {}
    for k, v in metrics.items():
        if "_" in k:
            prefix = k.split("_")[0]
        else:
            prefix = "general"
        groups.setdefault(prefix, {})[k] = v

    for group, kv in groups.items():
        print(f"\n  [{group.upper()}]")
        for k, v in kv.items():
            if isinstance(v, float):
                if "rank" in k.lower() or "n_queries" in k:
                    print(f"    {k:<30s}: {v:.1f}")
                elif "R@" in k or "acc" in k or "pct" in k or "nDCG" in k:
                    print(f"    {k:<30s}: {v*100:.2f}%")
                elif k.endswith("direction") or k.endswith("n_queries"):
                    pass
                else:
                    print(f"    {k:<30s}: {v:.4f}")
            elif isinstance(v, int):
                print(f"    {k:<30s}: {v}")
            elif isinstance(v, str):
                print(f"    {k:<30s}: {v}")

    print("=" * width + "\n")
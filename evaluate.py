"""
evaluate.py — Canonical locked evaluation script for ImageWeave.

This script is the SINGLE source of truth for all reported metrics.
Never modify the evaluation logic. Only checkpoint paths change.

Computes ALL metrics in BOTH retrieval directions:
  - Image → Text  (i2t)
  - Text  → Image (t2i)
  - Averaged (avg_*)
  - rSum (sum of all 6 recalls — standard in retrieval papers)

Usage:
    python evaluate.py --checkpoint checkpoints/baseline/best_model.pth
    python evaluate.py --checkpoint checkpoints/no_qformer/best_model.pth --output results/no_qformer.json
"""

import os
import sys
import json
import argparse
import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser(description="ImageWeave canonical evaluation")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output",     type=str, default=None,
                    help="Optional JSON file to save results")
parser.add_argument("--val_split",  type=str,
                    default="dataset/processed/metadata/val_split.json")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device",     type=str, default="cuda")
args = parser.parse_args()

# ============================================================
# IMPORTS
# ============================================================

from dataset.story_dataset import StoryTripletDataset
from model.imageweave      import ImageWeave
from train.metrics         import bidirectional_metrics, print_metrics

# ============================================================
# HEADER
# ============================================================

WIDTH = 66
print("\n" + "=" * WIDTH)
print("  IMAGEWEAVE — CANONICAL EVALUATION")
print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * WIDTH)
print(f"  Checkpoint : {args.checkpoint}")
print(f"  Val split  : {args.val_split}")
print(f"  Device     : {args.device}")
print(f"  Batch size : {args.batch_size}")
print("=" * WIDTH + "\n")

# ============================================================
# VALIDATION DATASET
# ============================================================

val_dataset = StoryTripletDataset(args.val_split, is_train=False)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

print(f"[evaluate] Val samples : {len(val_dataset)}")

# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint   = torch.load(args.checkpoint, map_location=args.device)
saved_config = checkpoint.get("config", None)

model = ImageWeave(config=saved_config).to(args.device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

epoch        = checkpoint.get("epoch",        "unknown")
best_val_mrr = checkpoint.get("best_val_mrr", checkpoint.get("best_r1", "unknown"))
ablation     = checkpoint.get("ablation_name", "unknown")

print(f"[evaluate] Ablation    : {ablation}")
print(f"[evaluate] Epoch       : {epoch}")
print(f"[evaluate] Saved MRR   : {best_val_mrr}\n")

# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

all_image_embeddings = []
all_text_embeddings  = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Extracting embeddings"):
        query_pixel_values  = batch["query_pixel_values"].to(args.device, non_blocking=True)
        memory_pixel_values = batch["memory_pixel_values"].to(args.device, non_blocking=True)
        texts               = batch["text"]

        with torch.amp.autocast("cuda"):
            image_emb, text_emb = model(query_pixel_values, memory_pixel_values, texts)

        # L2 normalize immediately
        all_image_embeddings.append(F.normalize(image_emb, dim=-1).cpu())
        all_text_embeddings.append(F.normalize(text_emb,  dim=-1).cpu())

image_embeddings = torch.cat(all_image_embeddings, dim=0)
text_embeddings  = torch.cat(all_text_embeddings,  dim=0)

assert image_embeddings.shape[0] == text_embeddings.shape[0] == len(val_dataset), \
    f"Shape mismatch: {image_embeddings.shape[0]} vs {len(val_dataset)}"

print(f"\n[evaluate] Image emb shape : {image_embeddings.shape}")
print(f"[evaluate] Text emb shape  : {text_embeddings.shape}")

# ============================================================
# COMPUTE ALL METRICS (BIDIRECTIONAL)
# ============================================================

metrics = bidirectional_metrics(image_embeddings, text_embeddings)

# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * WIDTH)
print(f"  RESULTS — {ablation.upper()} | Epoch {epoch} | N={len(val_dataset)}")
print("=" * WIDTH)

# ---- IMAGE → TEXT ----
print("\n  IMAGE → TEXT  (i2t)")
print("  " + "-" * 50)
print(f"  R@1          : {metrics['i2t_R@1']*100:>7.2f}%")
print(f"  R@5          : {metrics['i2t_R@5']*100:>7.2f}%")
print(f"  R@10         : {metrics['i2t_R@10']*100:>7.2f}%")
print(f"  R@20         : {metrics['i2t_R@20']*100:>7.2f}%")
print(f"  R@50         : {metrics['i2t_R@50']*100:>7.2f}%")
print(f"  Precision@1  : {metrics['i2t_Precision@1']*100:>7.2f}%")
print(f"  Precision@5  : {metrics['i2t_Precision@5']*100:>7.2f}%")
print(f"  Precision@10 : {metrics['i2t_Precision@10']*100:>7.2f}%")
print(f"  F1@1         : {metrics['i2t_F1@1']*100:>7.2f}%")
print(f"  F1@5         : {metrics['i2t_F1@5']*100:>7.2f}%")
print(f"  F1@10        : {metrics['i2t_F1@10']*100:>7.2f}%")
print(f"  MRR          : {metrics['i2t_MRR']:>8.4f}")
print(f"  MAP          : {metrics['i2t_MAP']:>8.4f}")
print(f"  nDCG@5       : {metrics['i2t_nDCG@5']:>8.4f}")
print(f"  nDCG@10      : {metrics['i2t_nDCG@10']:>8.4f}")
print(f"  Median Rank  : {metrics['i2t_median_rank']:>7.1f}")
print(f"  Mean Rank    : {metrics['i2t_mean_rank']:>7.1f}")
print(f"  Std Rank     : {metrics['i2t_std_rank']:>7.1f}")
print(f"  Top-1 Acc    : {metrics['i2t_top1_acc']*100:>7.2f}%")
print(f"  Top-5 Acc    : {metrics['i2t_top5_acc']*100:>7.2f}%")
print(f"  % in top-5%  : {metrics['i2t_pct_top5pct']*100:>7.2f}%")
print(f"  % in top-10% : {metrics['i2t_pct_top10pct']*100:>7.2f}%")
print(f"  Pos cos sim  : {metrics['i2t_pos_cosine_mean']:>8.4f} ± {metrics['i2t_pos_cosine_std']:.4f}")
print(f"  Neg cos sim  : {metrics['i2t_neg_cosine_mean']:>8.4f} ± {metrics['i2t_neg_cosine_std']:.4f}")
print(f"  Pos/Neg gap  : {metrics['i2t_pos_neg_gap']:>8.4f}")

# ---- TEXT → IMAGE ----
print("\n  TEXT → IMAGE  (t2i)")
print("  " + "-" * 50)
print(f"  R@1          : {metrics['t2i_R@1']*100:>7.2f}%")
print(f"  R@5          : {metrics['t2i_R@5']*100:>7.2f}%")
print(f"  R@10         : {metrics['t2i_R@10']*100:>7.2f}%")
print(f"  R@20         : {metrics['t2i_R@20']*100:>7.2f}%")
print(f"  R@50         : {metrics['t2i_R@50']*100:>7.2f}%")
print(f"  Precision@1  : {metrics['t2i_Precision@1']*100:>7.2f}%")
print(f"  Precision@5  : {metrics['t2i_Precision@5']*100:>7.2f}%")
print(f"  Precision@10 : {metrics['t2i_Precision@10']*100:>7.2f}%")
print(f"  F1@1         : {metrics['t2i_F1@1']*100:>7.2f}%")
print(f"  F1@5         : {metrics['t2i_F1@5']*100:>7.2f}%")
print(f"  F1@10        : {metrics['t2i_F1@10']*100:>7.2f}%")
print(f"  MRR          : {metrics['t2i_MRR']:>8.4f}")
print(f"  MAP          : {metrics['t2i_MAP']:>8.4f}")
print(f"  nDCG@5       : {metrics['t2i_nDCG@5']:>8.4f}")
print(f"  nDCG@10      : {metrics['t2i_nDCG@10']:>8.4f}")
print(f"  Median Rank  : {metrics['t2i_median_rank']:>7.1f}")
print(f"  Mean Rank    : {metrics['t2i_mean_rank']:>7.1f}")
print(f"  Std Rank     : {metrics['t2i_std_rank']:>7.1f}")

# ---- EMBEDDING QUALITY ----
print("\n  EMBEDDING QUALITY")
print("  " + "-" * 50)
if "i2t_alignment" in metrics:
    print(f"  Alignment    : {metrics['i2t_alignment']:>8.4f}  (lower=better)")
    print(f"  Uniformity ↑ : img={metrics['i2t_uniformity_img']:.4f}  txt={metrics['i2t_uniformity_txt']:.4f}  (more negative=better)")

# ---- AVERAGED / COMBINED ----
print("\n  AVERAGED (i2t + t2i) / 2")
print("  " + "-" * 50)
print(f"  avg R@1      : {metrics['avg_R@1']*100:>7.2f}%")
print(f"  avg R@5      : {metrics['avg_R@5']*100:>7.2f}%")
print(f"  avg R@10     : {metrics['avg_R@10']*100:>7.2f}%")
print(f"  avg MRR      : {metrics['avg_MRR']:>8.4f}")
print(f"  avg nDCG@10  : {metrics['avg_nDCG@10']:>8.4f}")
print(f"  rSum         : {metrics['rSum']:>8.4f}  (sum of 6 recalls — paper standard)")
print("=" * WIDTH + "\n")

# ============================================================
# SAVE JSON
# ============================================================

output_data = {
    "checkpoint":    args.checkpoint,
    "ablation_name": ablation,
    "epoch":         epoch,
    "val_samples":   len(val_dataset),
    "val_split":     args.val_split,
    "timestamp":     datetime.datetime.now().isoformat(),
    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
}

if args.output:
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"[evaluate] Results saved → {args.output}")
else:
    # Default: save next to checkpoint
    default_out = os.path.join(
        os.path.dirname(args.checkpoint),
        "eval_results.json"
    )
    with open(default_out, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"[evaluate] Results saved → {default_out}")

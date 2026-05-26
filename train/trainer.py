import os
import sys
import json
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

from transformers import get_cosine_schedule_with_warmup

from configs.train_config import CONFIG

from dataset.story_dataset import StoryTripletDataset

from model.imageweave import ImageWeave

from train.losses   import ContrastiveLoss
from train.metrics  import bidirectional_metrics, simple_retrieval_metrics

# ====================================================
# ABLATION NAME — controls checkpoint & log directories
# ====================================================

ABLATION_NAME = CONFIG["ablation_name"]

# ====================================================
# DIRECTORIES
# ====================================================

os.makedirs("logs", exist_ok=True)
os.makedirs(f"checkpoints/{ABLATION_NAME}", exist_ok=True)

# ====================================================
# TEE LOGGER
# Writes to:
#   1. terminal (live output)
#   2. logs/{ablation_name}.log  (ablation-specific log)
#   3. train.log                 (legacy path, kept for viewer compat)
# ====================================================

LOG_PATH       = f"logs/{ABLATION_NAME}.log"
LEGACY_LOG_PATH = "train.log"

import datetime

class Logger:
    """Tees stdout/stderr to multiple log files simultaneously."""
    def __init__(self, *filepaths):
        self.terminal = sys.__stdout__
        self.logs = [open(fp, "a", buffering=1) for fp in filepaths]

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        for log in self.logs:
            log.write(message)
            log.flush()

    def flush(self):
        self.terminal.flush()
        for log in self.logs:
            log.flush()

import datetime

_logger = Logger(LOG_PATH, LEGACY_LOG_PATH)
sys.stdout = _logger
sys.stderr = _logger

# ====================================================
# SESSION HEADER
# ====================================================

START_TIME = datetime.datetime.now()
print("\n" + "#" * 66)
print(f"  IMAGEWEAVE TRAINING SESSION")
print(f"  Ablation   : {ABLATION_NAME}")
print(f"  Started    : {START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Log files  : {LOG_PATH}  |  {LEGACY_LOG_PATH}")
print("#" * 66 + "\n")

# ====================================================
# REPRODUCIBILITY
# ====================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ====================================================
# GPU OPTIMIZATION
# ====================================================

cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ====================================================
# DEVICE
# ====================================================

device = CONFIG["device"]
print(f"\nUsing device: {device}")
print(f"Ablation name: {ABLATION_NAME}\n")

# ====================================================
# DATASETS
# ====================================================

train_dataset = StoryTripletDataset(
    "dataset/processed/metadata/train_split.json",
    is_train=True
)

val_dataset = StoryTripletDataset(
    "dataset/processed/metadata/val_split.json"
)

print(f"Train samples : {len(train_dataset)}")
print(f"Val samples   : {len(val_dataset)}\n")

# ====================================================
# DATALOADERS
# ====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

# ====================================================
# MODEL
# ====================================================

model = ImageWeave(config=CONFIG).to(device)

# ====================================================
# PARAM REPORT
# ====================================================

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 40)
print(f"Total Params    : {total_params:,}")
print(f"Trainable Params: {trainable_params:,}")
print("=" * 40 + "\n")

# ====================================================
# LOSS
# ====================================================

criterion = ContrastiveLoss(
    use_queue=CONFIG["use_queue"],
    queue_size=CONFIG["queue_size"],
    label_smoothing=0.1
).to(device)

# ====================================================
# OPTIMIZER
# ====================================================

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"]
)

# ====================================================
# SCHEDULER
# ====================================================

total_steps = (
    len(train_loader)
    * CONFIG["epochs"]
) // CONFIG["gradient_accumulation"]

warmup_steps = int(0.1 * total_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ====================================================
# AMP
# ====================================================

scaler = torch.amp.GradScaler("cuda")

# ====================================================
# LOGGING
# ====================================================

training_log_path = f"logs/training_log_{ABLATION_NAME}.json"
training_history  = []

# ====================================================
# CHECKPOINT PATH
# ====================================================

checkpoint_path = os.path.join(
    CONFIG["checkpoint_dir"],
    ABLATION_NAME,
    "best_model.pth"
)

# ====================================================
# VALIDATION
# ====================================================

@torch.no_grad()
def validate(model, loader, device):

    model.eval()

    all_image_embeddings = []
    all_text_embeddings  = []

    for batch in loader:

        query_pixel_values  = batch["query_pixel_values"].to(device, non_blocking=True)
        memory_pixel_values = batch["memory_pixel_values"].to(device, non_blocking=True)
        texts               = batch["text"]

        with torch.amp.autocast("cuda"):
            image_embeddings, text_embeddings = model(
                query_pixel_values,
                memory_pixel_values,
                texts
            )

        # L2 normalize before storing
        all_image_embeddings.append(F.normalize(image_embeddings, dim=-1).cpu())
        all_text_embeddings.append(F.normalize(text_embeddings,   dim=-1).cpu())

    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    text_embeddings  = torch.cat(all_text_embeddings,  dim=0)

    metrics = bidirectional_metrics(image_embeddings, text_embeddings)
    return metrics, image_embeddings, text_embeddings

# ====================================================
# TRAINING LOOP
# ====================================================

best_val_mrr         = 0.0    # track MRR (more stable than R@1)
epochs_without_improvement = 0

for epoch in range(CONFIG["epochs"]):

    model.train()

    epoch_loss = 0
    optimizer.zero_grad(set_to_none=True)

    # ================================================
    # TRAIN LOOP
    # ================================================

    for step, batch in enumerate(train_loader):

        query_pixel_values  = batch["query_pixel_values"].to(device, non_blocking=True)
        memory_pixel_values = batch["memory_pixel_values"].to(device, non_blocking=True)
        texts               = batch["text"]

        with torch.amp.autocast("cuda"):
            image_embeddings, text_embeddings = model(
                query_pixel_values,
                memory_pixel_values,
                texts
            )
            loss = criterion(image_embeddings, text_embeddings)
            loss = loss / CONFIG["gradient_accumulation"]

        scaler.scale(loss).backward()

        if (step + 1) % CONFIG["gradient_accumulation"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                CONFIG["max_grad_norm"]
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        epoch_loss += loss.item() * CONFIG["gradient_accumulation"]

        # Track last-batch accuracy (indicative only, not printed per step)
        with torch.no_grad():
            sim     = torch.matmul(image_embeddings.detach(), text_embeddings.detach().T)
            preds   = sim.argmax(dim=1)
            targets = torch.arange(sim.size(0), device=sim.device)
            accuracy = (preds == targets).float().mean().item()

    # ================================================
    # TRAIN EPOCH METRICS (last batch only, indicative)
    # ================================================

    avg_loss = epoch_loss / len(train_loader)
    lr_now   = scheduler.get_last_lr()[0]

    with torch.no_grad():
        last_sim = torch.matmul(
            image_embeddings.detach().cpu(),
            text_embeddings.detach().cpu().T
        )
        train_metrics = simple_retrieval_metrics(last_sim)

    # ================================================
    # VALIDATION
    # ================================================

    val_metrics, val_img_emb, val_txt_emb = validate(model, val_loader, device)

    # ------------------------------------------------
    # ONE CLEAN LINE PER EPOCH
    # ------------------------------------------------

    improved_marker = "  [BEST]" if val_metrics["avg_MRR"] > best_val_mrr else f"  [-{epochs_without_improvement + (0 if val_metrics['avg_MRR'] > best_val_mrr else 1)}/{CONFIG['early_stopping_patience']}]"

    print(
        f"[Ep {epoch+1:02d}/{CONFIG['epochs']}] "
        f"loss={avg_loss:.4f}  lr={lr_now:.2e}  "
        f"| i2t R@1={val_metrics['i2t_R@1']*100:5.2f}% "
        f"R@10={val_metrics['i2t_R@10']*100:5.2f}% "
        f"MRR={val_metrics['i2t_MRR']:.4f}  "
        f"| t2i R@1={val_metrics['t2i_R@1']*100:5.2f}% "
        f"| rSum={val_metrics['rSum']:.3f}"
        f"{improved_marker}"
    )

    # ------------------------------------------------
    # DETAILED BLOCK EVERY EPOCH
    # ------------------------------------------------

    if True:
        W = 66
        print("\n" + "=" * W)
        print(f"  EPOCH {epoch+1:02d} — DETAILED METRICS  [{ABLATION_NAME}]")
        print("=" * W)
        print(f"  Train  | loss={avg_loss:.4f}  batch_acc={accuracy:.3f}  LR={lr_now:.2e}")
        print(f"  i2t    | R@1={val_metrics['i2t_R@1']*100:.2f}%  R@5={val_metrics['i2t_R@5']*100:.2f}%  "
              f"R@10={val_metrics['i2t_R@10']*100:.2f}%  R@20={val_metrics['i2t_R@20']*100:.2f}%")
        print(f"         | P@1={val_metrics['i2t_Precision@1']*100:.2f}%  P@5={val_metrics['i2t_Precision@5']*100:.2f}%  "
              f"P@10={val_metrics['i2t_Precision@10']*100:.2f}%")
        print(f"         | F1@1={val_metrics['i2t_F1@1']*100:.2f}%  F1@5={val_metrics['i2t_F1@5']*100:.2f}%  "
              f"F1@10={val_metrics['i2t_F1@10']*100:.2f}%")
        print(f"         | MRR={val_metrics['i2t_MRR']:.4f}  MAP={val_metrics['i2t_MAP']:.4f}  "
              f"nDCG@10={val_metrics['i2t_nDCG@10']:.4f}")
        print(f"         | Median Rank={val_metrics['i2t_median_rank']:.0f}  "
              f"Mean Rank={val_metrics['i2t_mean_rank']:.1f}")
        print(f"  t2i    | R@1={val_metrics['t2i_R@1']*100:.2f}%  R@5={val_metrics['t2i_R@5']*100:.2f}%  "
              f"R@10={val_metrics['t2i_R@10']*100:.2f}%  MRR={val_metrics['t2i_MRR']:.4f}")
        print(f"         | P@1={val_metrics['t2i_Precision@1']*100:.2f}%  P@5={val_metrics['t2i_Precision@5']*100:.2f}%  "
              f"P@10={val_metrics['t2i_Precision@10']*100:.2f}%")
        print(f"         | F1@1={val_metrics['t2i_F1@1']*100:.2f}%  F1@5={val_metrics['t2i_F1@5']*100:.2f}%  "
              f"F1@10={val_metrics['t2i_F1@10']*100:.2f}%")
        print(f"  Avg    | R@1={val_metrics['avg_R@1']*100:.2f}%  R@5={val_metrics['avg_R@5']*100:.2f}%  "
              f"R@10={val_metrics['avg_R@10']*100:.2f}%  MRR={val_metrics['avg_MRR']:.4f}  "
              f"rSum={val_metrics['rSum']:.3f}")
        print(f"  Embed  | pos_cos={val_metrics['i2t_pos_cosine_mean']:.4f}  "
              f"neg_cos={val_metrics['i2t_neg_cosine_mean']:.4f}  "
              f"gap={val_metrics['i2t_pos_neg_gap']:.4f}")
        if 'i2t_alignment' in val_metrics:
            print(f"         | align={val_metrics['i2t_alignment']:.4f}  "
                  f"unif={val_metrics['i2t_uniformity_img']:.4f}")
        gap = train_metrics['R@1'] - val_metrics['i2t_R@1']
        warn = "  ⚠ OVERFIT" if gap > 0.5 else ""
        print(f"  Gap    | train_R@1={train_metrics['R@1']:.4f}  val_R@1={val_metrics['i2t_R@1']:.4f}  "
              f"delta={gap:.4f}{warn}")
        print(f"  Best   | MRR={best_val_mrr:.4f}  patience={epochs_without_improvement}/{CONFIG['early_stopping_patience']}")
        print("=" * W + "\n")

    # ================================================
    # SAVE TRAINING LOG
    # ================================================

    training_history.append({
        "epoch":            epoch + 1,
        "ablation_name":    ABLATION_NAME,
        "timestamp":        datetime.datetime.now().isoformat(),
        # ---- Train (last batch, indicative) ----
        "train_loss":       avg_loss,
        "train_accuracy":   accuracy,
        "train_r1":         train_metrics["R@1"],
        "train_r5":         train_metrics["R@5"],
        "train_mrr":        train_metrics["MRR"],
        # ---- Val: Image → Text ----
        "val_i2t_r1":       val_metrics["i2t_R@1"],
        "val_i2t_r5":       val_metrics["i2t_R@5"],
        "val_i2t_r10":      val_metrics["i2t_R@10"],
        "val_i2t_r20":      val_metrics["i2t_R@20"],
        "val_i2t_r50":      val_metrics["i2t_R@50"],
        "val_i2t_precision1":  val_metrics["i2t_Precision@1"],
        "val_i2t_precision5":  val_metrics["i2t_Precision@5"],
        "val_i2t_precision10": val_metrics["i2t_Precision@10"],
        "val_i2t_f1_1":     val_metrics["i2t_F1@1"],
        "val_i2t_f1_5":     val_metrics["i2t_F1@5"],
        "val_i2t_f1_10":    val_metrics["i2t_F1@10"],
        "val_i2t_mrr":      val_metrics["i2t_MRR"],
        "val_i2t_map":      val_metrics["i2t_MAP"],
        "val_i2t_ndcg10":   val_metrics["i2t_nDCG@10"],
        "val_i2t_ndcg5":    val_metrics["i2t_nDCG@5"],
        "val_i2t_top1_acc": val_metrics["i2t_top1_acc"],
        "val_i2t_top5_acc": val_metrics["i2t_top5_acc"],
        "val_i2t_median_rank": val_metrics["i2t_median_rank"],
        "val_i2t_mean_rank":   val_metrics["i2t_mean_rank"],
        "val_i2t_pos_neg_gap": val_metrics["i2t_pos_neg_gap"],
        # ---- Val: Text → Image ----
        "val_t2i_r1":       val_metrics["t2i_R@1"],
        "val_t2i_r5":       val_metrics["t2i_R@5"],
        "val_t2i_r10":      val_metrics["t2i_R@10"],
        "val_t2i_r20":      val_metrics["t2i_R@20"],
        "val_t2i_precision1":  val_metrics["t2i_Precision@1"],
        "val_t2i_precision5":  val_metrics["t2i_Precision@5"],
        "val_t2i_precision10": val_metrics["t2i_Precision@10"],
        "val_t2i_f1_1":     val_metrics["t2i_F1@1"],
        "val_t2i_f1_5":     val_metrics["t2i_F1@5"],
        "val_t2i_f1_10":    val_metrics["t2i_F1@10"],
        "val_t2i_mrr":      val_metrics["t2i_MRR"],
        "val_t2i_map":      val_metrics["t2i_MAP"],
        "val_t2i_ndcg10":   val_metrics["t2i_nDCG@10"],
        # ---- Val: Averaged ----
        "val_avg_r1":       val_metrics["avg_R@1"],
        "val_avg_r5":       val_metrics["avg_R@5"],
        "val_avg_r10":      val_metrics["avg_R@10"],
        "val_avg_precision5": val_metrics["avg_Precision@5"],
        "val_avg_f1_5":      val_metrics["avg_F1@5"],
        "val_avg_mrr":      val_metrics["avg_MRR"],
        "val_avg_ndcg10":   val_metrics["avg_nDCG@10"],
        "val_rsum":         val_metrics["rSum"],
        # ---- Embedding quality ----
        "val_pos_cosine":   val_metrics.get("i2t_pos_cosine_mean", None),
        "val_neg_cosine":   val_metrics.get("i2t_neg_cosine_mean", None),
        "val_pos_neg_gap":  val_metrics.get("i2t_pos_neg_gap",     None),
        "val_alignment":    val_metrics.get("i2t_alignment",        None),
        "val_uniformity":   val_metrics.get("i2t_uniformity_img",   None),
    })

    with open(training_log_path, "w") as f:
        json.dump(training_history, f, indent=4)

    # ================================================
    # CHECKPOINTING — track by avg_MRR (both directions)
    # ================================================

    if val_metrics["avg_MRR"] > best_val_mrr:

        best_val_mrr               = val_metrics["avg_MRR"]
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch":               epoch,
                "ablation_name":       ABLATION_NAME,
                "config":              CONFIG,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mrr":        best_val_mrr,
                "best_r1":             val_metrics["i2t_R@1"],
                "best_rsum":           val_metrics["rSum"],
                "val_metrics":         {k: v for k, v in val_metrics.items()
                                        if isinstance(v, (int, float))},
            },
            checkpoint_path
        )

        print(f"  → Saved best checkpoint [avg_MRR={best_val_mrr:.4f}  i2t R@1={val_metrics['i2t_R@1']*100:.2f}%  rSum={val_metrics['rSum']:.3f}]")

    else:

        epochs_without_improvement += 1

        if epochs_without_improvement >= CONFIG["early_stopping_patience"]:
            print(f"\nEARLY STOPPING at epoch {epoch+1} — no improvement for {CONFIG['early_stopping_patience']} epochs.")
            break

END_TIME = datetime.datetime.now()
print("\n" + "#" * 66)
print(f"  Training Complete — {ABLATION_NAME}")
print(f"  Best avg MRR  : {best_val_mrr:.4f}")
print(f"  Checkpoint    : {checkpoint_path}")
print(f"  Started       : {START_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Finished      : {END_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Duration      : {str(END_TIME - START_TIME).split('.')[0]}")
print("#" * 66)
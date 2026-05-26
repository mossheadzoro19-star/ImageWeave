"""
scripts/summarize_ablations.py

Reads all result JSON files in results/ and prints the ablation table.

Usage:
    python scripts/summarize_ablations.py
"""

import os
import json
import glob

# Expected ablation order for the paper table
ABLATION_ORDER = [
    "baseline",
    "no_qformer",
    "no_cross_attention",
    "no_memory",
    "mean_pooling",
    "clip_only",
]

DISPLAY_NAMES = {
    "baseline":            "Full ImageWeave (Ours)",
    "no_qformer":          "w/o QFormer",
    "no_cross_attention":  "w/o Cross-Image Attention",
    "no_memory":           "w/o Memory Fusion",
    "mean_pooling":        "w/o Attention Pooling (Mean Pool)",
    "clip_only":           "CLIP Baseline (No Sequential Modules)",
}

results = {}
for path in glob.glob("results/*.json"):
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path) as f:
        results[name] = json.load(f)

if not results:
    print("No results found in results/. Run evaluate.py for each checkpoint first.")
    exit(0)

def g(r, *keys):
    """Get first available key from result dict."""
    for k in keys:
        if k in r:
            return r[k]
    return float("nan")

# ====================================================
# TABLE HEADER
# ====================================================

print()
print("=" * 110)
print("  IMAGEWEAVE ABLATION STUDY — RESULTS")
print("  Evaluation: full val split | Both Retrieval Directions")
print("=" * 110)
print(f"{'Ablation':<40}  {'i2t R@1':>8}  {'i2t R@10':>9}  {'t2i R@1':>8}  {'t2i R@10':>9}  {'avg MRR':>8}  {'rSum':>8}  {'Epoch':>6}")
print("-" * 110)

ordered = [k for k in ABLATION_ORDER if k in results]
extras  = [k for k in results if k not in ABLATION_ORDER]

for name in ordered + extras:
    r = results[name]
    display = DISPLAY_NAMES.get(name, name)
    print(
        f"{display:<40}  "
        f"{g(r,'i2t_R@1','R@1')*100:>7.2f}%  "
        f"{g(r,'i2t_R@10','R@10')*100:>8.2f}%  "
        f"{g(r,'t2i_R@1')*100:>7.2f}%  "
        f"{g(r,'t2i_R@10')*100:>8.2f}%  "
        f"{g(r,'avg_MRR','MRR'):>8.4f}  "
        f"{g(r,'rSum'):>8.4f}  "
        f"{str(r.get('epoch', '?')):>6}"
    )

print("=" * 110)
print()

# ====================================================
# LATEX TABLE
# ====================================================

print("% LaTeX table (copy into paper)")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\begin{tabular}{lccccccc}")
print(r"\hline")
print(r"\textbf{Method} & \textbf{i2t R@1} & \textbf{i2t R@5} & \textbf{i2t R@10} & \textbf{t2i R@1} & \textbf{t2i R@5} & \textbf{avg MRR} & \textbf{rSum} \\")
print(r"\hline")
for name in ordered + extras:
    r = results[name]
    display = DISPLAY_NAMES.get(name, name)
    vals = (
        f"{g(r,'i2t_R@1','R@1')*100:.2f}",
        f"{g(r,'i2t_R@5','R@5')*100:.2f}",
        f"{g(r,'i2t_R@10','R@10')*100:.2f}",
        f"{g(r,'t2i_R@1')*100:.2f}",
        f"{g(r,'t2i_R@5')*100:.2f}",
        f"{g(r,'avg_MRR','MRR'):.4f}",
        f"{g(r,'rSum'):.4f}",
    )
    if name == "baseline":
        bvals = " & ".join(f"\\textbf{{{v}}}" for v in vals)
        print(f"\\textbf{{{display}}} & {bvals} \\\\")
    else:
        print(f"{display} & {' & '.join(vals)} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Ablation study on full validation split. i2t=image-to-text, t2i=text-to-image. Best in \textbf{bold}.}")
print(r"\label{tab:ablation}")
print(r"\end{table}")


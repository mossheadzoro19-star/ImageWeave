#!/usr/bin/env bash
# ============================================================
# run_train.sh — Launch ImageWeave training in a tmux session
#
# USAGE
#   bash run_train.sh                    # run 'baseline' config (default)
#   bash run_train.sh no_qformer         # run named ablation
#   tmux attach -t imageweave-baseline   # re-attach
#   Ctrl-b d                             # detach (training keeps running)
#   Ctrl-c                               # stop training
# ============================================================

set -euo pipefail

ABLATION="${1:-baseline}"
GPU="${2:-0}"
SESSION="imageweave-${ABLATION}"
WORKDIR="/home/drive2/user1_workspace/imageweave_project"
ENV_PYTHON="${WORKDIR}/../envs/imageweave/bin/python"

# Fall back to system python if venv not found
if [ ! -f "$ENV_PYTHON" ]; then
    ENV_PYTHON="$(which python3)"
    echo "[run_train] venv not found, using: $ENV_PYTHON"
fi

echo ""
echo "============================================="
echo "  ImageWeave — Ablation: $ABLATION"
echo "============================================="
echo ""

# -----------------------------------------------------------
# If the session already exists, just re-attach
# -----------------------------------------------------------
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[run_train] Session '$SESSION' already running."
    echo "[run_train] Attaching..."
    # tmux attach-session -t "$SESSION"
    exit 0
fi

# -----------------------------------------------------------
# Patch ablation_name in train_config.py before launch.
# This lets us run different ablations without editing files.
# -----------------------------------------------------------
sed -i "s/\"ablation_name\": \"[^\"]*\"/\"ablation_name\": \"${ABLATION}\"/" \
    "${WORKDIR}/configs/train_config.py"

# Also set the ablation flags for known ablation names
case "$ABLATION" in
    baseline)
        sed -i 's/"use_qformer": .*/"use_qformer": True,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_cross_attention": .*/"use_cross_attention": True,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_memory": .*/"use_memory": True,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_attention_pooling": .*/"use_attention_pooling": True,/' "${WORKDIR}/configs/train_config.py"
        ;;
    no_qformer)
        sed -i 's/"use_qformer": .*/"use_qformer": False,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_cross_attention": .*/"use_cross_attention": True,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_memory": .*/"use_memory": True,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_attention_pooling": .*/"use_attention_pooling": True,/' "${WORKDIR}/configs/train_config.py"
        ;;
    no_cross_attention)
        sed -i 's/"use_qformer": .*/"use_qformer": True,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_cross_attention": .*/"use_cross_attention": False,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_memory": .*/"use_memory": True,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_attention_pooling": .*/"use_attention_pooling": True,/' "${WORKDIR}/configs/train_config.py"
        ;;
    no_memory)
        sed -i 's/"use_qformer": .*/"use_qformer": True,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_cross_attention": .*/"use_cross_attention": True,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_memory": .*/"use_memory": False,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_attention_pooling": .*/"use_attention_pooling": True,/' "${WORKDIR}/configs/train_config.py"
        ;;
    mean_pooling)
        sed -i 's/"use_qformer": .*/"use_qformer": True,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_cross_attention": .*/"use_cross_attention": True,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_memory": .*/"use_memory": True,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_attention_pooling": .*/"use_attention_pooling": False,/' "${WORKDIR}/configs/train_config.py"
        ;;
    clip_only)
        sed -i 's/"use_qformer": .*/"use_qformer": False,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_cross_attention": .*/"use_cross_attention": False,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_memory": .*/"use_memory": False,/' "${WORKDIR}/configs/train_config.py"
        sed -i 's/"use_attention_pooling": .*/"use_attention_pooling": False,/' "${WORKDIR}/configs/train_config.py"
        ;;
    *)
        echo "[run_train] WARNING: Unknown ablation '$ABLATION'. Using current config flags."
        ;;
esac

echo "[run_train] Config set for: $ABLATION"

# -----------------------------------------------------------
# Create a new detached tmux session
# -----------------------------------------------------------
tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux send-keys -t "$SESSION" "cd $WORKDIR" Enter
tmux send-keys -t "$SESSION" "echo '========================================'" Enter
tmux send-keys -t "$SESSION" "echo '  ImageWeave Training: $ABLATION'" Enter
tmux send-keys -t "$SESSION" "echo '  Log: logs/${ABLATION}.log'" Enter
tmux send-keys -t "$SESSION" "echo '========================================'" Enter

tmux send-keys -t "$SESSION" "CUDA_VISIBLE_DEVICES=$GPU $ENV_PYTHON -m train.trainer" Enter

echo ""
echo "============================================="
echo "  Training launched: $ABLATION"
echo "============================================="
echo "  Attach   :  tmux attach -t $SESSION"
echo "  Detach   :  Ctrl-b then d"
echo "  Log      :  tail -f $WORKDIR/logs/${ABLATION}.log"
echo "  Checkpoint: $WORKDIR/checkpoints/${ABLATION}/best_model.pth"
echo ""

# # tmux attach-session -t "$SESSION"

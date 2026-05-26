# ============================================================
# train_config.py
#
# Single source of truth for all training hyperparameters.
# Ablation experiments change only the flags below.
# Never hard-code these values in model or trainer code.
# ============================================================

CONFIG = {

    # =================================================
    # TRAINING
    # =================================================

    "batch_size": 16,

    "gradient_accumulation": 4,        # effective batch size = 64

    "epochs": 60,                      # hard cap; early stopping fires first

    "learning_rate": 1e-5,             # reduced from 2e-5 — better generalization

    "weight_decay": 5e-2,              # increased to 5e-2 for final overfitting mitigation

    "max_grad_norm": 1.0,

    # =================================================
    # EARLY STOPPING
    # =================================================

    "early_stopping_patience": 8,      # reduced from 15; model peaks ~epoch 30-35

    # =================================================
    # DEVICE
    # =================================================

    "device": "cuda",

    # =================================================
    # CHECKPOINTS & LOGGING
    # =================================================

    "checkpoint_dir": "checkpoints",

    # Ablation name controls where checkpoints and logs are saved.
    # Full model (no ablation): "baseline"
    # Ablation runs: "no_qformer", "no_cross_attention", etc.
    "ablation_name": "clip_only",

    # =================================================
    # ABLATION FLAGS
    # All True = full ImageWeave model.
    # Set one to False to run that ablation.
    # =================================================

    # QFormer: temporally-aware query token extractor.
    # If False: CLIP patch tokens are mean-pooled directly.
    "use_qformer": False,

    # Cross-Image Attention: attention across T×Q query tokens.
    # If False: each frame is processed independently.
    "use_cross_attention": False,

    # Memory fusion: add encoded memory frames to query tokens.
    # If False: memory_embedding is zeroed out (query-only baseline).
    "use_memory": False,

    # Attention Pooling (StoryPooling CLS-token).
    # If False: use StoryHead (global mean pooling) instead.
    "use_attention_pooling": False,

    # =================================================
    # CONTRASTIVE LOSS FLAGS
    # =================================================

    # MoCo-style memory queue.
    # DISABLED: without momentum encoder, causes representation drift
    # and degrades validation performance (confirmed experimentally).
    "use_queue": False,

    "queue_size": 8192,

    # =================================================
    # CLIP BACKBONE
    # =================================================

    # Freeze entire CLIP backbone (both vision and text).
    # ENABLED: frozen CLIP gives better generalization (confirmed).
    # Only the final block is unfrozen when this is True,
    # based on the unfreezing logic in ImageWeave.__init__.
    "freeze_clip": True,

    # =================================================
    # INFERENCE CONFIG (not used in training)
    # =================================================

    "top_k_retrieval": 5,
    "reasoning_model": "gemini-1.5-pro",
    "max_generation_tokens": 80
}

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# BIDIRECTIONAL INFONCE CONTRASTIVE LOSS
# =====================================================
#
# Key design decisions:
#
# 1. SYMMETRIC bidirectional loss (image→text AND text→image).
#
# 2. LEARNABLE temperature with clamping.
#
# 3. LABEL SMOOTHING = 0.1 for mild regularization.
#
# 4. use_queue=False by default.
#    MoCo-style queue REQUIRES a momentum encoder to prevent
#    representation drift. Without it, queue negatives become
#    stale and degrade validation performance (confirmed
#    experimentally: archived no-queue run achieved R@1=12.16%
#    vs queue run R@1=7.92%). Queue is kept as optional for
#    ablation purposes only.
#
# =====================================================

class ContrastiveLoss(nn.Module):

    def __init__(
        self,
        init_temperature=0.07,
        label_smoothing=0.1,
        feature_dim=512,
        queue_size=8192,
        use_queue=False,          # disabled by default — see note above
    ):

        super().__init__()

        self.use_queue      = use_queue
        self.label_smoothing = label_smoothing

        # Learnable log-temperature (clamped in forward)
        self.log_temperature = nn.Parameter(
            torch.tensor(init_temperature).log()
        )

        # ------------------------------------------------
        # Queue buffers (only used if use_queue=True)
        # ------------------------------------------------

        if self.use_queue:
            self.queue_size = queue_size

            self.register_buffer(
                "image_queue",
                F.normalize(torch.randn(feature_dim, queue_size), dim=0)
            )

            self.register_buffer(
                "text_queue",
                F.normalize(torch.randn(feature_dim, queue_size), dim=0)
            )

            self.register_buffer(
                "queue_ptr",
                torch.zeros(1, dtype=torch.long)
            )

    # =====================================================
    # QUEUE UPDATE (only called when use_queue=True)
    # =====================================================

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        batch_size = image_feat.shape[0]
        ptr        = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:
            self.image_queue[:, ptr:ptr + batch_size] = image_feat.T
            self.text_queue[:, ptr:ptr + batch_size]  = text_feat.T
            ptr = (ptr + batch_size) % self.queue_size
        else:
            remainder = self.queue_size - ptr
            self.image_queue[:, ptr:self.queue_size] = image_feat[:remainder].T
            self.text_queue[:, ptr:self.queue_size]  = text_feat[:remainder].T
            overflow = batch_size - remainder
            self.image_queue[:, :overflow] = image_feat[remainder:].T
            self.text_queue[:, :overflow]  = text_feat[remainder:].T
            ptr = overflow

        self.queue_ptr[0] = ptr

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        image_embeddings,
        text_embeddings
    ):

        # ------------------------------------------------
        # L2 NORMALIZE
        # ------------------------------------------------

        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings  = F.normalize(text_embeddings,  dim=-1)

        # ------------------------------------------------
        # LEARNABLE TEMPERATURE (clamped for stability)
        # ------------------------------------------------

        temperature = self.log_temperature.exp().clamp(min=0.01, max=0.5)

        # ------------------------------------------------
        # IN-BATCH SIMILARITY MATRIX   [B, B]
        # ------------------------------------------------

        sim_inbatch = image_embeddings @ text_embeddings.T

        if self.use_queue:

            # --------------------------------------------
            # QUEUE NEGATIVES   [B, K]
            # --------------------------------------------

            sim_queue_i2t = image_embeddings @ self.text_queue.clone().detach()
            sim_queue_t2i = text_embeddings  @ self.image_queue.clone().detach()

            logits_i2t = torch.cat([sim_inbatch, sim_queue_i2t], dim=1) / temperature
            logits_t2i = torch.cat([sim_inbatch.T, sim_queue_t2i], dim=1) / temperature

        else:

            # --------------------------------------------
            # IN-BATCH ONLY (cleaner, no drift)
            # --------------------------------------------

            logits_i2t = sim_inbatch   / temperature
            logits_t2i = sim_inbatch.T / temperature

        # ------------------------------------------------
        # LABELS (diagonal = correct match)
        # ------------------------------------------------

        B      = sim_inbatch.shape[0]
        labels = torch.arange(B, device=image_embeddings.device)

        # ------------------------------------------------
        # SYMMETRIC LOSS  (image→text  +  text→image)
        # ------------------------------------------------

        loss_i2t = F.cross_entropy(
            logits_i2t,
            labels,
            label_smoothing=self.label_smoothing
        )

        loss_t2i = F.cross_entropy(
            logits_t2i,
            labels,
            label_smoothing=self.label_smoothing
        )

        loss = (loss_i2t + loss_t2i) / 2.0

        # ------------------------------------------------
        # UPDATE QUEUE (only if enabled)
        # ------------------------------------------------

        if self.use_queue:
            self._dequeue_and_enqueue(image_embeddings, text_embeddings)

        return loss
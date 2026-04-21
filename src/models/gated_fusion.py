import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    """
    Robust mid-level gated fusion (final production version).
    Handles any batch/seq shape safely.
    Equation: h_fused = σ(W_g [h_text; h_vis]) ⊙ h_text + (1-σ) ⊙ h_vis
    """
    def __init__(self, hidden_dim: int = 768, vis_vocab_size: int = 256):
        super().__init__()
        self.proj_vis = nn.Linear(vis_vocab_size, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, h_text: torch.Tensor, soft_probs: torch.Tensor, confidence: torch.Tensor):
        """
        h_text:       (batch, seq_len, hidden_dim)
        soft_probs:   (batch, seq_len, vocab_size)
        confidence:   (batch, seq_len, 1)
        """
        batch_size, seq_len = h_text.shape[0], h_text.shape[1]

        # Flatten batch × seq for Linear layers (safest pattern)
        h_text_flat = h_text.view(batch_size * seq_len, -1)
        soft_probs_flat = soft_probs.view(batch_size * seq_len, -1)
        confidence_flat = confidence.view(batch_size * seq_len, -1)

        # Project vision and multiply by confidence
        h_vis_flat = self.proj_vis(soft_probs_flat) * confidence_flat

        # Gate computation
        combined = torch.cat([h_text_flat, h_vis_flat], dim=-1)
        gate = torch.sigmoid(self.gate(combined))

        # Gated combination
        h_fused_flat = gate * h_text_flat + (1 - gate) * h_vis_flat

        # Reshape back to original (batch, seq_len, hidden_dim)
        h_fused = h_fused_flat.view(batch_size, seq_len, -1)

        return h_fused
import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Mid-level gated fusion layer.
    Mathematically weighs textual context (h_text) against Vision Module soft probabilities.
    Equation: h_fused = σ(W_g [h_text; h_vis]) ⊙ h_text + (1-σ) ⊙ h_vis
    """

    def __init__(self, hidden_dim: int = 768, vis_vocab_size: int = 384):
        super().__init__()
        self.proj_vis = nn.Linear(vis_vocab_size, hidden_dim)  # Project vision soft_probs
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)  # Gate network
        self.hidden_dim = hidden_dim

    def forward(self, h_text: torch.Tensor, soft_probs: torch.Tensor, confidence: torch.Tensor):
        """
        h_text:       (batch, seq_len, hidden_dim)     ← ByT5 encoder output
        soft_probs:   (batch, seq_len, vis_vocab_size) ← Vision Module character probabilities
        confidence:   (batch, seq_len, 1)              ← Vision confidence scores
        """
        # Project and weight vision features by confidence
        h_vis = self.proj_vis(soft_probs) * confidence.unsqueeze(-1)

        # Concatenate and compute gate
        combined = torch.cat([h_text, h_vis], dim=-1)
        gate = torch.sigmoid(self.gate(combined))  # σ(W_g [h_text; h_vis])

        # Gated combination
        h_fused = gate * h_text + (1 - gate) * h_vis
        return h_fused
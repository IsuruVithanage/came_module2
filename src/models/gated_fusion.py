import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int = 512, vis_vocab_size: int = 256):
        super().__init__()
        self.proj_vis = nn.Linear(vis_vocab_size, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, h_text: torch.Tensor, soft_probs: torch.Tensor, confidence: torch.Tensor):
        batch_size, seq_len = h_text.shape[0], h_text.shape[1]

        # Flatten for Linear layers
        h_text_flat = h_text.view(batch_size * seq_len, -1)
        soft_probs_flat = soft_probs.view(batch_size * seq_len, -1)
        confidence_flat = confidence.view(batch_size * seq_len, -1)

        h_vis_flat = self.proj_vis(soft_probs_flat) * confidence_flat

        combined = torch.cat([h_text_flat, h_vis_flat], dim=-1)
        gate = torch.sigmoid(self.gate(combined))

        h_fused_flat = gate * h_text_flat + (1 - gate) * h_vis_flat

        # Reshape back
        h_fused = h_fused_flat.view(batch_size, seq_len, -1)
        return h_fused
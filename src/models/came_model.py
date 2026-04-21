import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, ByT5Tokenizer
from .gated_fusion import GatedFusion


class CAMEModel(nn.Module):
    """
    Full Context-Aware Multimodal Epigrapher (CAME) model.
    - ByT5 encoder-decoder (byte-level, perfect for Brahmi Unicode)
    - Gated Fusion with Vision soft_probs + confidence
    - 3 multitask heads: restoration, syllable validity, confidence calibration
    """

    def __init__(self, model_name: str = "google/byt5-small", hidden_dim: int = 768):
        super().__init__()
        self.byt5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = ByT5Tokenizer.from_pretrained(model_name)

        self.fusion = GatedFusion(hidden_dim=hidden_dim, vis_vocab_size=self.tokenizer.vocab_size)

        # Multitask heads (applied to decoder last_hidden_state)
        self.restoration_head = nn.Linear(hidden_dim, self.tokenizer.vocab_size)
        self.syllable_head = nn.Linear(hidden_dim, 2)  # 0=invalid, 1=valid
        self.confidence_head = nn.Linear(hidden_dim, 1)  # regression 0–1

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                soft_probs: torch.Tensor = None,  # From Vision Module
                confidence: torch.Tensor = None,  # From Vision Module
                labels: torch.Tensor = None):  # Teacher forcing during training

        # 1. Encoder (textual context from noisy Brahmi with [MASK])
        encoder_outputs = self.byt5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        h_text = encoder_outputs.last_hidden_state

        # 2. Gated Fusion (only when Vision input is provided)
        if soft_probs is not None and confidence is not None:
            h_fused = self.fusion(h_text, soft_probs, confidence)
        else:
            h_fused = h_text

        # 3. Decoder (uses fused representation as encoder_hidden_states)
        decoder_outputs = self.byt5.decoder(
            input_ids=labels if self.training and labels is not None else None,
            encoder_hidden_states=h_fused,
            encoder_attention_mask=attention_mask
        )
        hidden = decoder_outputs.last_hidden_state

        # 4. Multitask heads
        restoration_logits = self.restoration_head(hidden)
        syllable_logits = self.syllable_head(hidden)
        confidence_score = torch.sigmoid(self.confidence_head(hidden))

        return {
            "restoration_logits": restoration_logits,
            "syllable_logits": syllable_logits,
            "confidence_score": confidence_score,
            "decoder_hidden": hidden  # for beam search later
        }

    def generate(self, *args, **kwargs):
        """Forward compatibility with Hugging Face generate API."""
        return self.byt5.generate(*args, **kwargs)
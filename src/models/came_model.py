import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, ByT5Tokenizer
from .gated_fusion import GatedFusion


class CAMEModel(nn.Module):
    def __init__(self, model_name: str = "google/byt5-small"):
        super().__init__()
        self.byt5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = ByT5Tokenizer.from_pretrained(model_name)

        # === DYNAMIC HIDDEN DIM FROM BYT5 ===
        self.hidden_dim = self.byt5.config.d_model  # 512 for byt5-small

        self.fusion = GatedFusion(
            hidden_dim=self.hidden_dim,
            vis_vocab_size=self.tokenizer.vocab_size
        )

        # Multitask heads
        self.restoration_head = nn.Linear(self.hidden_dim, self.tokenizer.vocab_size)
        self.syllable_head = nn.Linear(self.hidden_dim, 2)
        self.confidence_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, input_ids, attention_mask=None, soft_probs=None, confidence=None, labels=None):
        encoder_outputs = self.byt5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_text = encoder_outputs.last_hidden_state

        if soft_probs is not None and confidence is not None:
            h_fused = self.fusion(h_text, soft_probs, confidence)
        else:
            h_fused = h_text

        decoder_outputs = self.byt5.decoder(
            input_ids=labels if self.training and labels is not None else None,
            encoder_hidden_states=h_fused,
            encoder_attention_mask=attention_mask
        )
        hidden = decoder_outputs.last_hidden_state

        return {
            "restoration_logits": self.restoration_head(hidden),
            "syllable_logits": self.syllable_head(hidden),
            "confidence_score": torch.sigmoid(self.confidence_head(hidden)),
        }

    def generate(self, *args, **kwargs):
        return self.byt5.generate(*args, **kwargs)
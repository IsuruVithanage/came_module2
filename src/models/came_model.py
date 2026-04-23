import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, ByT5Tokenizer
from .gated_fusion import GatedFusion
from ..utils.build_syllable_vocab import get_safe_syllable_tokens

# 1. Define the base single characters explicitly
BASE_BRAHMI_CHARS = [
    "_",  # mask token
    "𑀓", "𑀔", "𑀕", "𑀖", "𑀗", "𑀘", "𑀙", "𑀚", "𑀛", "𑀜",
    "𑀝", "𑀞", "𑀟", "𑀠", "𑀡", "𑀢", "𑀣", "𑀤", "𑀥", "𑀦",
    "𑀧", "𑀨", "𑀩", "𑀪", "𑀫", "𑀬", "𑀭", "𑀮", "𑀯", "𑀰",
    "𑀱", "𑀲", "𑀳", "𑀴", "𑀵", "𑀶", "𑀷",
    "𑀅", "𑀆", "𑀇", "𑀈", "𑀉", "𑀊", "𑀋", "𑀌", "𑀍", "𑀎", "𑀏", "𑀐", "𑀑", "𑀒",
    "𑀸", "𑀹", "𑀺", "𑀻", "𑀼", "𑀽", "𑀾", "𑀿", "𑁀", "𑁁", "𑁂", "𑁃", "𑁄", "𑁅", "\U00011046"
]

# 2. Combine base chars with dynamic composite syllables
BRAHMI_TOKENS = BASE_BRAHMI_CHARS + get_safe_syllable_tokens(min_count=5)

class CAMEModel(nn.Module):
    def __init__(self, model_name: str = "google/byt5-small"):
        super().__init__()
        self.byt5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = ByT5Tokenizer.from_pretrained(model_name)

        # Force tokenizer to recognize Brahmi as single tokens
        num_added = self.tokenizer.add_tokens(BRAHMI_TOKENS)
        self.byt5.resize_token_embeddings(len(self.tokenizer))

        self.hidden_dim = self.byt5.config.d_model
        self.vocab_size = len(self.tokenizer)

        self.fusion = GatedFusion(
            hidden_dim=self.hidden_dim,
            vis_vocab_size=self.vocab_size
        )

        self.restoration_head = nn.Linear(self.hidden_dim, self.vocab_size)
        self.syllable_head = nn.Linear(self.hidden_dim, 2)
        self.confidence_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, input_ids, attention_mask=None, soft_probs=None, confidence=None, labels=None):
        encoder_outputs = self.byt5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_text = encoder_outputs.last_hidden_state

        if soft_probs is not None and confidence is not None:
            h_fused = self.fusion(h_text, soft_probs, confidence)
        else:
            h_fused = h_text

        # === FIXED: Provide decoder_input_ids for inference ===
            # === FIXED: Allow full-sequence decoding if labels are provided (even in eval mode) ===
        if labels is not None:
            decoder_input_ids = self.byt5._shift_right(labels)
        else:
            # During standard generation without labels, use the pad token
            batch_size = input_ids.shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.byt5.config.pad_token_id,
                dtype=torch.long,
                device=input_ids.device
            )

        decoder_outputs = self.byt5.decoder(
            input_ids=decoder_input_ids,
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
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.came_model import CAMEModel
from src.utils.brahmi_converter import brahmi_to_iast
from src.utils.validity import is_valid_akshara


class BeamSearchRestorer:
    def __init__(self, checkpoint_path: str = "checkpoints/came_latest.pt", beam_size: int = 20):
        self.model = CAMEModel()
        if Path(checkpoint_path).exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            print("✅ Loaded trained checkpoint")
        else:
            print("⚠️ No checkpoint – using random weights (demo mode)")
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.beam_size = beam_size
        self.lambda_valid = 1.5
        self.mu_conf = 0.3

    def restore(self, noisy_brahmi: str, soft_probs=None, confidence=None):
        # 1. Tokenize FIRST to get the true mathematical sequence length
        inputs = self.tokenizer(noisy_brahmi, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        seq_len = input_ids.shape[1]

        # 2. Generate dummy Vision data matching the exact seq_len AND true vocab size
        if soft_probs is None:
            # === CHANGED HERE: Use self.model.vocab_size instead of self.tokenizer.vocab_size ===
            soft_probs = torch.ones((1, seq_len, self.model.vocab_size)) / self.model.vocab_size
        if confidence is None:
            confidence = torch.ones((1, seq_len, 1)) * 0.8

        beams = [(input_ids.clone(), 0.0)]

        # Look for the "_" mask token
        mask_token_id = self.tokenizer.encode("_", add_special_tokens=False)[0]
        mask_positions = [i for i, t in enumerate(input_ids[0]) if t == mask_token_id]

        for mask_pos in mask_positions:
            new_beams = []
            for seq, score in beams:
                with torch.no_grad():
                    # === CRITICAL FIX: Pass seq as labels ===
                    outputs = self.model(
                        input_ids=seq,
                        attention_mask=attention_mask,
                        soft_probs=soft_probs,
                        confidence=confidence,
                        labels=seq
                    )

                logits = outputs["restoration_logits"][0, mask_pos]
                probs = torch.softmax(logits, dim=-1)
                topk_probs, topk_ids = torch.topk(probs, k=self.beam_size)

                for k in range(self.beam_size):
                    new_seq = seq.clone()
                    new_seq[0, mask_pos] = topk_ids[k]

                    candidate = self.tokenizer.decode(new_seq[0], skip_special_tokens=True)
                    new_score = score + torch.log(topk_probs[k]).item()

                    if is_valid_akshara(candidate):
                        new_score += self.lambda_valid

                    new_score += self.mu_conf * outputs["confidence_score"][0, mask_pos].item()
                    new_beams.append((new_seq, new_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.beam_size]

        results = []
        for seq, score in beams:
            brahmi = self.tokenizer.decode(seq[0], skip_special_tokens=True)
            brahmi = brahmi.replace("_", "").strip()
            iast = brahmi_to_iast(brahmi)
            conf = round(torch.sigmoid(torch.tensor(score)).item(), 4)
            results.append((brahmi, iast, conf))

        return sorted(results, key=lambda x: x[2], reverse=True)
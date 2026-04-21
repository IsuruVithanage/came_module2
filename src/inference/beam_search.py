import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.came_model import CAMEModel
from src.utils.brahmi_converter import brahmi_to_iast
from src.utils.validity import is_valid_akshara


class BeamSearchRestorer:
    def __init__(self, checkpoint_path: str = "checkpoints/came_latest.pt", beam_size: int = 5):
        self.model = CAMEModel()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.beam_size = beam_size
        self.lambda_valid = 0.4
        self.mu_conf = 0.2

    def restore(self, noisy_brahmi: str, soft_probs: torch.Tensor = None, confidence: torch.Tensor = None):
        """
        Position-wise constrained beam search.
        Returns list of (iast_text, confidence_score) sorted by final score.
        """
        if soft_probs is None:
            soft_probs = torch.ones((1, len(noisy_brahmi), self.tokenizer.vocab_size)) / self.tokenizer.vocab_size
        if confidence is None:
            confidence = torch.ones((1, len(noisy_brahmi), 1)) * 0.8

        # Tokenize noisy input (with <mask>)
        inputs = self.tokenizer(noisy_brahmi, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Start with empty beam: [(sequence, score)]
        beams = [(input_ids.clone(), 0.0)]

        mask_positions = [i for i, token in enumerate(input_ids[0]) if
                          token == self.tokenizer.convert_tokens_to_ids("<mask>")]

        for pos_idx, mask_pos in enumerate(mask_positions):
            new_beams = []
            for seq, score in beams:
                # Run decoder step for this position only
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=seq,
                        attention_mask=attention_mask,
                        soft_probs=soft_probs,
                        confidence=confidence,
                        labels=None
                    )

                # Get logits for the current mask position
                logits = outputs["restoration_logits"][0, mask_pos]
                probs = torch.softmax(logits, dim=-1)

                # Top-k candidates
                topk_probs, topk_ids = torch.topk(probs, k=self.beam_size)

                for k in range(self.beam_size):
                    new_seq = seq.clone()
                    new_seq[0, mask_pos] = topk_ids[k]

                    new_score = score + torch.log(topk_probs[k]).item()

                    # Apply syllable validity
                    candidate_str = self.tokenizer.decode(new_seq[0], skip_special_tokens=True)
                    if is_valid_akshara(candidate_str):
                        new_score += self.lambda_valid

                    # Add confidence head bonus
                    new_score += self.mu_conf * outputs["confidence_score"][0, mask_pos].item()

                    new_beams.append((new_seq, new_score))

            # Keep top beam_size
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.beam_size]

        # Convert best beams to IAST + final confidence
        results = []
        for seq, score in beams:
            restored_brahmi = self.tokenizer.decode(seq[0], skip_special_tokens=True)
            iast = brahmi_to_iast(restored_brahmi)
            final_conf = torch.sigmoid(torch.tensor(score)).item()  # normalize to 0-1
            results.append((iast, round(final_conf, 4)))

        return sorted(results, key=lambda x: x[1], reverse=True)


# Quick test when run directly
if __name__ == "__main__":
    restorer = BeamSearchRestorer()
    example_noisy = "𑀛𑁄𑀢𑀺𑀰<MASK>𑀦𑀢𑁂𑀭𑀰𑀅<MASK>𑀺𑀯𑀰𑀺𑀓𑀩𑀢𑀰𑀼𑀫𑀦𑀤𑀢𑀢𑁂𑀭𑀰𑀮𑁂𑀦𑁂𑀰𑀕𑀰"
    results = restorer.restore(example_noisy)
    print("🔥 Restoration results:")
    for iast, conf in results:
        print(f"   {iast}   confidence: {conf}")
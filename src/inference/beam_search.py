import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.came_model import CAMEModel
from src.utils.brahmi_converter import brahmi_to_iast
from src.utils.validity import is_valid_akshara


class BeamSearchRestorer:
    def __init__(self, checkpoint_path: str = "checkpoints/came_latest.pt", beam_size: int = 12):
        self.model = CAMEModel()
        if Path(checkpoint_path).exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            print("✅ Loaded trained checkpoint")
        else:
            print("⚠️ No checkpoint – using random weights (demo mode)")
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.beam_size = beam_size
        self.lambda_valid = 1.5  # ← Stronger validity enforcement
        self.mu_conf = 0.3

    def restore(self, noisy_brahmi: str, soft_probs=None, confidence=None):
        if soft_probs is None:
            soft_probs = torch.ones((1, len(noisy_brahmi), self.tokenizer.vocab_size)) / self.tokenizer.vocab_size
        if confidence is None:
            confidence = torch.ones((1, len(noisy_brahmi), 1)) * 0.8

        inputs = self.tokenizer(noisy_brahmi, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        beams = [(input_ids.clone(), 0.0)]
        mask_positions = [i for i, t in enumerate(input_ids[0]) if t == self.tokenizer.convert_tokens_to_ids("<mask>")]

        for mask_pos in mask_positions:
            new_beams = []
            for seq, score in beams:
                with torch.no_grad():
                    outputs = self.model(input_ids=seq, attention_mask=attention_mask,
                                         soft_probs=soft_probs, confidence=confidence)

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

        # Final post-processing to clean output
        results = []
        for seq, score in beams:
            brahmi = self.tokenizer.decode(seq[0], skip_special_tokens=True)
            brahmi = brahmi.replace("<mask>", "").replace("<MASK>", "").strip()
            iast = brahmi_to_iast(brahmi)
            conf = round(torch.sigmoid(torch.tensor(score)).item(), 4)
            results.append((iast, conf))

        return sorted(results, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    restorer = BeamSearchRestorer()
    test = "𑀕<MASK>𑀢𑀺𑀰𑀼𑀫𑀧𑀼𑀢𑀰𑀼𑀫<MASK>𑀳"
    print(restorer.restore(test))
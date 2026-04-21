import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.beam_search import BeamSearchRestorer
from src.training.trainer import CAMETrainer   # just to check training status

print("🚀 CAME Module 2 - Context-Aware Multimodal Epigrapher")
print("=" * 70)

# Quick status check
checkpoint = Path("checkpoints/came_latest.pt")
if checkpoint.exists():
    print(f"✅ Model ready → {checkpoint}")
else:
    print("⚠️  No checkpoint yet. Train first or use a pre-trained one.")

# Example restoration (replace with real Vision output later)
restorer = BeamSearchRestorer(checkpoint_path=str(checkpoint) if checkpoint.exists() else None)

noisy_example = "𑀛𑁄𑀢𑀺𑀰<MASK>𑀦𑀢𑁂𑀭𑀰𑀅<MASK>𑀺𑀯𑀰𑀺𑀓𑀩𑀢𑀰𑀼𑀫𑀦𑀤𑀢𑀢𑁂𑀭𑀰𑀮𑁂𑀦𑁂𑀰𑀕𑀰"

print(f"\n🔥 Testing restoration on noisy input:")
print(f"   Noisy Brahmi : {noisy_example}")

results = restorer.restore(noisy_example)

print("\n📊 Ranked Results:")
for i, (iast, conf) in enumerate(results[:5]):
    print(f"   {i+1:2d}. {iast:<40}  confidence: {conf:.4f}")

print("\n🎉 CAME Module 2 is ready!")
print("   Next: Integrate with Component 1 (Vision Module)")
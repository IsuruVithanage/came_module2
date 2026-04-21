import yaml
from pathlib import Path

print("🚀 CAME Module 2 - Context-Aware Multimodal Epigrapher")
print("=" * 60)

config_path = Path("configs/model_config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

print(f"✅ Loaded config for model: {config['model']['name']}")
print(f"   Hidden dim: {config['model']['hidden_dim']}")
print(f"   Training epochs: {config['training']['epochs']}")

print("\n📁 Project structure ready!")
print("Next step: Reply with **'Step 2'** and I will give you the exact code for:")
print("   - Data preparation (Step 2)")
print("   - Brahmi converter & validity checker (Step 3)")
import streamlit as st
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.inference.beam_search import BeamSearchRestorer

st.set_page_config(page_title="CAME Epigrapher", page_icon="🪨", layout="wide")
st.title("🪨 CAME Module 2 – Semantic Restoration of Brahmi Inscriptions")
st.markdown("**Advanced Multimodal Epigrapher** | Gated Fusion + Constrained Beam Search")


# Load model (graceful fallback)
@st.cache_resource
def load_restorer():
    checkpoint = Path("checkpoints/came_latest.pt")
    if checkpoint.exists():
        st.success("✅ Loaded trained checkpoint")
        return BeamSearchRestorer(checkpoint_path=str(checkpoint))
    else:
        st.warning(
            "⚠️ No checkpoint found. Using model with random weights (demo only). Train longer for better results.")
        return BeamSearchRestorer(checkpoint_path=None)


restorer = load_restorer()

# Sidebar
st.sidebar.header("Demo Controls")
example = st.sidebar.selectbox(
    "Choose example inscription",
    [
        "𑀛𑁄𑀢𑀺𑀰_𑀦𑀢𑁂𑀭𑀰𑀅_𑀺𑀯𑀰𑀺𑀓𑀩𑀢𑀰𑀼𑀫𑀦𑀤𑀢𑀢𑁂𑀭𑀰𑀮𑁂𑀦𑁂𑀰𑀕𑀰",
        "𑀫𑀳𑀭𑀸𑀚𑀧𑀼_𑀳",
        "𑀢𑀺𑀰𑀰𑀫𑀡𑀺𑀬𑀮𑁂𑀦𑁂𑀰𑀕𑀰",
        "Custom"
    ]
)

if example == "Custom":
    noisy_input = st.text_area("Paste noisy Brahmi text (use <MASK> for missing characters)",
                               value="𑀛𑀛𑁄𑀢𑀺𑀰_𑀦𑀢𑁂𑀭𑀰𑀅_𑀺𑀯𑀰𑀺𑀓𑀩𑀢𑀰𑀼𑀫𑀦𑀤𑀢𑀢𑁂𑀭𑀰𑀮𑁂𑀦𑁂𑀰𑀕𑀰",
                               height=100)
else:
    noisy_input = st.text_area("Noisy Brahmi text", value=example, height=100)

if st.button("🚀 Restore Missing Characters", type="primary"):
    with st.spinner("Running gated fusion + position-wise beam search..."):
        results = restorer.restore(noisy_input)

    st.subheader("📊 Ranked Restoration Results")

    # UNPACK ALL 3 VARIABLES (brahmi, iast, conf)
    for rank, (brahmi, iast, conf) in enumerate(results[:5], 1):
        st.metric(
            label=f"Rank {rank} (Confidence: {conf:.4f})",
            value=brahmi,
            delta=iast,
            delta_color="off"  # Keeps the IAST text gray instead of green/red
        )
    st.divider()
    st.caption("Powered by CAME: Gated Fusion • ByT5 • Syllable Validity • Constrained Beam Search")

st.info("💡 Tip: After longer training, accuracy improves dramatically. Use the evaluation script to measure progress.")

st.caption("Component 2 – Final-Year Project | Built with ❤️ for supervisor demo")
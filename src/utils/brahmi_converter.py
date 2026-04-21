import json
from pathlib import Path

# ==================== COMPLETE BRAHMI ↔ IAST MAPPING ====================
# Based on Unicode Standard Brahmi (U+11000–U+1107F) + your inscriptions
BRAHMI_TO_IAST = {
    # Independent vowels
    '\U00011005': 'a', '\U00011006': 'ā', '\U00011007': 'i', '\U00011008': 'ī',
    '\U00011009': 'u', '\U0001100A': 'ū', '\U0001100B': 'ṛ', '\U0001100C': 'ṝ',
    '\U0001100D': 'ḷ', '\U0001100E': 'ḹ', '\U0001100F': 'e', '\U00011010': 'ai',
    '\U00011011': 'o', '\U00011012': 'au',
    # Consonants
    '\U00011013': 'k', '\U00011014': 'kh', '\U00011015': 'g', '\U00011016': 'gh',
    '\U00011017': 'ṅ', '\U00011018': 'c', '\U00011019': 'ch', '\U0001101A': 'j',
    '\U0001101B': 'jh', '\U0001101C': 'ñ', '\U0001101D': 'ṭ', '\U0001101E': 'ṭh',
    '\U0001101F': 'ḍ', '\U00011020': 'ḍh', '\U00011021': 'ṇ', '\U00011022': 't',
    '\U00011023': 'th', '\U00011024': 'd', '\U00011025': 'dh', '\U00011026': 'n',
    '\U00011027': 'p', '\U00011028': 'ph', '\U00011029': 'b', '\U0001102A': 'bh',
    '\U0001102B': 'm', '\U0001102C': 'y', '\U0001102D': 'r', '\U0001102E': 'l',
    '\U0001102F': 'v', '\U00011030': 'ś', '\U00011031': 'ṣ', '\U00011032': 's',
    '\U00011033': 'h', '\U00011034': 'ḷ', '\U00011035': 'ḻ', '\U00011036': 'ṟ',
    '\U00011037': 'ṉ',
    # Vowel signs (matras)
    '\U00011038': '', '\U00011039': 'ā', '\U0001103A': 'i', '\U0001103B': 'ī',
    '\U0001103C': 'u', '\U0001103D': 'ū', '\U0001103E': 'ṛ', '\U0001103F': 'ṝ',
    '\U00011040': 'ḷ', '\U00011041': 'e', '\U00011042': 'ai', '\U00011043': 'o',
    '\U00011044': 'au', '\U00011045': 'ṃ', '\U00011046': '',  # virama
    # Additional signs from your inscriptions
    '\U00011047': 'ḥ', '\U00011048': 'ṃ', '\U00011049': 'ṁ',
}

# Reverse mapping (IAST → Brahmi) - built automatically
IAST_TO_BRAHMI = {v: k for k, v in BRAHMI_TO_IAST.items() if v}  # skip empty virama

def brahmi_to_iast(brahmi_text: str) -> str:
    """Convert full Brahmi Unicode string to readable IAST."""
    result = []
    for char in brahmi_text:
        result.append(BRAHMI_TO_IAST.get(char, char))  # unknown chars unchanged
    return ''.join(result).strip()

def iast_to_brahmi(iast_text: str) -> str:
    """Convert IAST back to Brahmi Unicode (for data prep if needed)."""
    result = []
    i = 0
    while i < len(iast_text):
        # Greedy longest match for digraphs (kh, gh, etc.)
        for length in range(3, 0, -1):
            chunk = iast_text[i:i+length]
            if chunk in IAST_TO_BRAHMI:
                result.append(IAST_TO_BRAHMI[chunk])
                i += length
                break
        else:
            # Single char fallback
            result.append(IAST_TO_BRAHMI.get(iast_text[i], iast_text[i]))
            i += 1
    return ''.join(result)

# Quick test when run directly
if __name__ == "__main__":
    test = "𑀛𑁄𑀢𑀺𑀰𑀦𑀢𑁂𑀭𑀰"
    print("Brahmi → IAST:", brahmi_to_iast(test))
    print("IAST → Brahmi:", iast_to_brahmi("jhotiśenaterasa"))
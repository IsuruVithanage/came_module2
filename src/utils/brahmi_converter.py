import json
from pathlib import Path

# ==================== COMPLETE BRAHMI ↔ IAST MAPPING ====================
BRAHMI_TO_IAST_MAP = {
    '\U00011000': 'm̐', '\U00011001': 'ṃ', '\U00011002': 'ḥ',
    '\U00011005': 'a', '\U00011006': 'ā', '\U00011007': 'i', '\U00011008': 'ī',
    '\U00011009': 'u', '\U0001100A': 'ū', '\U0001100B': 'ṛ', '\U0001100C': 'ṝ',
    '\U0001100D': 'ḷ', '\U0001100E': 'ḹ', '\U0001100F': 'e', '\U00011010': 'ai',
    '\U00011011': 'o', '\U00011012': 'au',
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
    '\U00011038': 'ā', '\U00011039': 'i', '\U0001103A': 'ī', '\U0001103B': 'u',
    '\U0001103C': 'ū', '\U0001103D': 'ṛ', '\U0001103E': 'ṝ', '\U0001103F': 'ḷ',
    '\U00011040': 'ḹ', '\U00011041': 'e', '\U00011042': 'ai', '\U00011043': 'o',
    '\U00011044': 'au'
}
VIRAMA = '\U00011046'
CONSONANT_RANGE = set(chr(i) for i in range(0x11013, 0x11038))

def brahmi_to_iast(brahmi_text: str) -> str:
    """State-aware transliteration handling the inherent 'a', matras, and viramas."""
    result = ""
    for i, char in enumerate(brahmi_text):
        if char in CONSONANT_RANGE:
            # Add consonant base + inherent 'a'
            result += BRAHMI_TO_IAST_MAP.get(char, char) + 'a'
        elif char in BRAHMI_TO_IAST_MAP and char not in CONSONANT_RANGE:
            # If it's a dependent vowel, replace the inherent 'a' from the previous consonant
            if i > 0 and brahmi_text[i-1] in CONSONANT_RANGE and result.endswith('a'):
                result = result[:-1]
            result += BRAHMI_TO_IAST_MAP[char]
        elif char == VIRAMA:
            # Suppress the inherent 'a'
            if i > 0 and brahmi_text[i-1] in CONSONANT_RANGE and result.endswith('a'):
                result = result[:-1]
        else:
            result += char
    return result.strip()
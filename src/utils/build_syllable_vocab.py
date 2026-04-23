import os
from pathlib import Path
from collections import Counter

VOWEL_SIGNS = set('𑀸𑀹𑀺𑀻𑀼𑀽𑀾𑀿𑁀𑁁𑁂𑁃𑁄𑁅')
CONSONANTS = set('𑀓𑀔𑀕𑀖𑀗𑀘𑀙𑀚𑀛𑀜𑀝𑀞𑀟𑀠𑀡𑀢𑀣𑀤𑀥𑀦𑀧𑀨𑀩𑀪𑀫𑀬𑀭𑀮𑀯𑀰𑀱𑀲𑀳𑀴𑀵𑀶𑀷')
IND_VOWELS = set('𑀅𑀆𑀇𑀈𑀉𑀊𑀋𑀌𑀍𑀎𑀏𑀐𑀑𑀒')
VIRAMA = '\U00011046'


def extract_aksharas(text):
    """Split a Brahmi string into akshara units (C, C+VS, C+V+C)."""
    units, i, chars = [], 0, list(text)
    while i < len(chars):
        ch = chars[i]
        if ch in CONSONANTS:
            unit = ch
            if i + 1 < len(chars) and chars[i + 1] in VOWEL_SIGNS:
                unit += chars[i + 1]
                i += 1
            elif (i + 1 < len(chars) and chars[i + 1] == VIRAMA
                  and i + 2 < len(chars) and chars[i + 2] in CONSONANTS):
                unit += chars[i + 1] + chars[i + 2]
                i += 2
            units.append(unit)
        elif ch in (IND_VOWELS | VOWEL_SIGNS):
            units.append(ch)
        else:
            # Catch spaces or unknown characters safely
            units.append(ch)
        i += 1
    return units


def get_safe_syllable_tokens(min_count=5):
    """Return multi-char aksharas that appear >= min_count times in training data."""
    # Ensure it works dynamically regardless of where the script is run from
    project_root = Path(__file__).parent.parent.parent
    train_path = project_root / "data" / "processed" / "train.txt"

    if not train_path.exists():
        print("⚠️ Warning: train.txt not found. Returning empty syllable list. Run prepare_data.py first.")
        return []

    txt = train_path.read_text(encoding="utf-8")
    counts = Counter()
    for line in txt.strip().splitlines():
        for unit in extract_aksharas(line):
            if len(unit) > 1:  # only composite syllables
                counts[unit] += 1

    safe_tokens = [token for token, count in counts.items() if count >= min_count]
    print(f"📚 Built dynamic Akshara vocabulary: Found {len(safe_tokens)} composite syllables.")
    return safe_tokens
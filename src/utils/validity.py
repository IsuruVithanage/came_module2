# Complete Brahmi Unicode Sets (U+11000 to U+1107F)
CONSONANTS = set('𑀓𑀔𑀕𑀖𑀗𑀘𑀙𑀚𑀛𑀜𑀝𑀞𑀟𑀠𑀡𑀢𑀣𑀤𑀥𑀦𑀧𑀨𑀩𑀪𑀫𑀬𑀭𑀮𑀯𑀰𑀱𑀲𑀳𑀴𑀵𑀶𑀷')
VOWEL_SIGNS = set('𑀸𑀹𑀺𑀻𑀼𑀽𑀾𑀿𑁀𑁁𑁂𑁃𑁄𑁅')
INDEPENDENT_VOWELS = set('𑀅𑀆𑀇𑀈𑀉𑀊𑀋𑀌𑀍𑀎𑀏𑀐𑀑𑀒')
MODIFIERS = set('𑀀𑀁𑀂')
VIRAMA = '\U00011046'

def is_valid_akshara(seq: str) -> bool:
    """Rule-based Brahmi akshara validator for Constrained Beam Search."""
    if not seq:
        return False

    if seq[0] in VOWEL_SIGNS or seq[0] == VIRAMA or seq[0] in MODIFIERS:
        return False

    for i in range(len(seq)):
        char = seq[i]
        if i > 0:
            prev_char = seq[i - 1]
            if char in CONSONANTS and prev_char in CONSONANTS:
                return False
            if char in VOWEL_SIGNS and prev_char not in CONSONANTS:
                return False
            if char == VIRAMA and prev_char not in CONSONANTS:
                return False
            if char in MODIFIERS:
                if prev_char not in CONSONANTS and prev_char not in VOWEL_SIGNS and prev_char not in INDEPENDENT_VOWELS:
                    return False
    return True
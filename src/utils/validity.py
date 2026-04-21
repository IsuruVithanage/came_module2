CONSONANTS = set('𑀓𑀔𑀕𑀖𑀗𑀘𑀙𑀚𑀛𑀜𑀝𑀞𑀟𑀠𑀡𑀢𑀣𑀤𑀥𑀦𑀧𑀨𑀩𑀪𑀫𑀬𑀭𑀮𑀯𑀰𑀱𑀲𑀳𑀴𑀵𑀶𑀷')
VOWEL_SIGNS = set('𑁀𑁁𑁂𑁃𑁄𑁅𑁆𑁇𑁈𑁉𑁊𑁋𑁌𑁍𑁎𑁏𑁐𑁑𑁒𑁓𑁔𑁕𑁖𑁗𑁘𑁙')
VIRAMA = '\U00011046'
INDEPENDENT_VOWELS = set('𑀅𑀆𑀇𑀈𑀉𑀊𑀋𑀌𑀍𑀎𑀏𑀐𑀑𑀒')

def is_valid_akshara(seq: str) -> bool:
    """
    Rule-based Brahmi akshara validator (directly from Grammer (1).pdf + Glossary.pdf).
    Used by Syllable Validity Head + constrained beam search.
    """
    if not seq:
        return False

    # Rule 1: No two consonants without virama between them (Grammar Rule xviii, lxxvii)
    for i in range(len(seq) - 1):
        if (seq[i] in CONSONANTS and
            seq[i + 1] in CONSONANTS and
            seq[i + 1] != VIRAMA):
            return False

    # Rule 2: Dependent vowel signs must follow a consonant
    for i in range(1, len(seq)):
        if seq[i] in VOWEL_SIGNS and seq[i - 1] not in CONSONANTS:
            return False

    # Rule 3: Virama can only appear after a consonant
    for i in range(1, len(seq)):
        if seq[i] == VIRAMA and seq[i - 1] not in CONSONANTS:
            return False

    # Rule 4: Cannot start with a dependent vowel sign
    if seq[0] in VOWEL_SIGNS:
        return False

    return True
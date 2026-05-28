"""
──────────────────────────────────────────────────────────────────────────────
 Preprocessor — Text normalization for the Aria RAG pipeline

 Pipeline:  raw text → lowercase → expand contractions → synonym expansion
            → strip punctuation → tokenize → remove stopwords → stem (Porter)
──────────────────────────────────────────────────────────────────────────────
"""

import re
from typing import Dict, List, Set


# ── Contraction Map ───────────────────────────────────────────────────────────

CONTRACTIONS: Dict[str, str] = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "he's": "he is", "she's": "she is", "it's": "it is", "that's": "that is",
    "what's": "what is", "where's": "where is", "who's": "who is",
    "how's": "how is", "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have", "i'll": "i will",
    "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "we'll": "we will", "they'll": "they will", "i'd": "i would",
    "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "we'd": "we would", "they'd": "they would", "let's": "let us",
    "ain't": "is not", "gonna": "going to", "wanna": "want to",
    "gotta": "got to", "kinda": "kind of", "sorta": "sort of",
}

# ── Stopwords ─────────────────────────────────────────────────────────────────

STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "up",
    "that", "this", "these", "those", "am", "its",
}

# ── Synonym / Alias Expansion ─────────────────────────────────────────────────

SYNONYMS: Dict[str, str] = {
    "send back": "return", "give back": "return", "bring back": "return",
    "ship back": "return", "sending back": "return",
    "money back": "refund", "get my money": "refund", "cash back": "refund",
    "my money": "refund",
    "swap": "exchange", "replace": "exchange", "switch": "exchange",
    "broken": "damaged", "busted": "damaged", "cracked": "damaged",
    "smashed": "damaged", "defective": "damaged",
    "cost": "price", "costs": "price", "fee": "price", "charge": "price",
    "login": "log in", "sign in": "log in", "sign up": "register",
    "eta": "estimated delivery", "delivery estimate": "estimated delivery",
    "promo": "promo code", "coupon": "promo code", "voucher": "promo code",
    "discount code": "promo code",
    "track": "track order", "tracking": "track order",
    "where is": "track order", "locate": "track order",
    "cancel order": "cancel", "cancellation": "cancel",
    "hack": "unauthorized access", "hacked": "unauthorized access",
    "stolen": "unauthorized access",
    "sue": "legal action", "lawyer": "legal action", "court": "legal action",
}


def stem(word: str) -> str:
    """
    Simplified Porter Stemmer.
    Reduces words to root forms: 'shipping' → 'ship', 'returned' → 'return'
    """
    if len(word) < 3:
        return word

    suffix_rules = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("alli", "al"),
        ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
        ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
        ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
        ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
        ("biliti", "ble"),
    ]

    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("ly") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("ment") and len(word) > 6:
        word = word[:-4]
    elif word.endswith("ness") and len(word) > 5:
        word = word[:-4]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]

    for pattern, replacement in suffix_rules:
        if word.endswith(pattern):
            word = word[: -len(pattern)] + replacement
            break

    return word


def expand_contractions(text: str) -> str:
    """Expand contractions: don't → do not, can't → cannot"""
    for contraction, expansion in CONTRACTIONS.items():
        pattern = re.compile(re.escape(contraction), re.IGNORECASE)
        text = pattern.sub(expansion, text)
        curly = contraction.replace("'", "\u2019")
        text = re.sub(re.escape(curly), expansion, text, flags=re.IGNORECASE)
    return text


def expand_synonyms(text: str) -> str:
    """Expand synonyms: 'send back' → 'return', 'money back' → 'refund'"""
    sorted_syns = sorted(SYNONYMS.items(), key=lambda x: -len(x[0]))
    for phrase, canonical in sorted_syns:
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        text = pattern.sub(canonical, text)
    return text


def tokenize(text: str) -> List[str]:
    """Split text into tokens, keeping only alphanumeric and hyphens."""
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return [t for t in cleaned.split() if len(t) > 0]


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove common English stopwords, keeping tokens that carry intent."""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def process(text: str) -> dict:
    """
    Full preprocessing pipeline.

    Returns:
        {
            "original": str,
            "cleaned": str,
            "tokens": list[str],      # after stopword removal
            "stems": list[str],        # stemmed tokens
            "bigrams": list[str],      # adjacent stem pairs
            "raw_tokens": list[str],   # before stopword removal
        }
    """
    original = text
    cleaned = text.lower().strip()
    cleaned = expand_contractions(cleaned)
    cleaned = expand_synonyms(cleaned)
    raw_tokens = tokenize(cleaned)
    tokens = remove_stopwords(raw_tokens)
    stems = [stem(t) for t in tokens]
    bigrams = [f"{stems[i]}_{stems[i+1]}" for i in range(len(stems) - 1)]

    return {
        "original": original,
        "cleaned": " ".join(raw_tokens),
        "tokens": tokens,
        "stems": stems,
        "bigrams": bigrams,
        "raw_tokens": raw_tokens,
    }


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def fuzzy_match(a: str, b: str, threshold: float = 0.3) -> bool:
    """Returns True if normalized edit distance is below threshold."""
    dist = edit_distance(a.lower(), b.lower())
    max_len = max(len(a), len(b))
    return (dist / max_len <= threshold) if max_len > 0 else True

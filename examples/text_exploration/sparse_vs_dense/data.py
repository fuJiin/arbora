"""Data loaders for ARB-139: text8 corpus + SimLex-999 + Google analogy.

text8 is the canonical small-data benchmark for word2vec-class
comparisons (~17M tokens, 71k unique, ~100MB). It's already preprocessed
(lowercase, alphanumerics only, single-space-separated), so tokenization
is trivial — both word2vec and our T1 ingest the exact same token stream
via `text.split()`. That makes tokenization a controlled variable.

Cached datasets land under `data/cache/` (top-level, gitignored).
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
SIMLEX_URL = "https://fh295.github.io/SimLex-999.zip"
# Google analogy: bundled with gensim datasets but easier to fetch
# directly. Mikolov's original location is gone; this mirror is stable.
ANALOGY_URL = (
    "https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/"
    "questions-words.txt"
)


# ---------------------------------------------------------------------------
# Generic cache helper
# ---------------------------------------------------------------------------


def _cached(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def _download(url: str, dest: Path) -> None:
    """Download `url` to `dest` if not already present. Idempotent."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"[fetching] {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


# ---------------------------------------------------------------------------
# text8
# ---------------------------------------------------------------------------


def load_text8(*, max_tokens: int | None = None) -> list[str]:
    """Load text8 as a list of word tokens.

    text8 is one big lowercase string with words separated by spaces. We
    split on whitespace; no fancier tokenization needed (and using one
    here would be a confound — both architectures must see identical
    sequences).

    Caches the unzipped corpus locally. ~100MB on disk.
    """
    zip_path = _cached("text8.zip")
    extracted = _cached("text8")
    if not extracted.exists():
        _download(TEXT8_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("text8", path=CACHE_DIR)
    tokens = extracted.read_text(encoding="utf-8").split()
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------


def build_vocab(
    tokens: list[str],
    *,
    vocab_size: int,
    unk_token: str = "<unk>",
) -> tuple[dict[str, int], list[str]]:
    """Top-N vocabulary by frequency. Returns (token_to_id, id_to_token).

    Index 0 is reserved for `<unk>`. Real tokens get ids 1..vocab_size-1.
    `vocab_size` includes the unk slot.
    """
    from collections import Counter

    counts = Counter(tokens)
    most_common = [tok for tok, _ in counts.most_common(vocab_size - 1)]
    id_to_token = [unk_token, *most_common]
    token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
    return token_to_id, id_to_token


def encode_tokens(tokens: list[str], token_to_id: dict[str, int]) -> list[int]:
    """Map a token stream to int IDs. OOV tokens become 0 (`<unk>`)."""
    return [token_to_id.get(t, 0) for t in tokens]


# ---------------------------------------------------------------------------
# SimLex-999
# ---------------------------------------------------------------------------


def load_simlex(*, vocab: set[str] | None = None) -> list[tuple[str, str, float]]:
    """Load SimLex-999 word-pair similarity benchmark.

    Returns `(word_a, word_b, human_similarity_in_0_to_10)`. If `vocab`
    is provided, restricts to pairs where both words appear in vocab.
    """
    zip_path = _cached("SimLex-999.zip")
    txt_path = _cached("SimLex-999.txt")
    if not txt_path.exists():
        _download(SIMLEX_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("SimLex-999/SimLex-999.txt", path=CACHE_DIR)
        # Move out of the nested dir for easier access.
        nested = CACHE_DIR / "SimLex-999" / "SimLex-999.txt"
        if nested.exists():
            nested.rename(txt_path)

    pairs: list[tuple[str, str, float]] = []
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    # Format (tab-separated): word1, word2, POS, SimLex999, conc(w1),
    # conc(w2), concQ, Assoc(USF), SimAssoc333, SD(SimLex). First line
    # is a header.
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        a, b = parts[0].lower(), parts[1].lower()
        score = float(parts[3])
        if vocab is not None and (a not in vocab or b not in vocab):
            continue
        pairs.append((a, b, score))
    return pairs


# ---------------------------------------------------------------------------
# Google analogy
# ---------------------------------------------------------------------------


def load_analogy(
    *, vocab: set[str] | None = None
) -> list[tuple[str, tuple[str, str, str, str]]]:
    """Load the Google analogy benchmark.

    Returns `(category, (a, b, c, d))` tuples. Categories like
    `: capital-common-countries`, `: family`, `: gram1-adjective-to-adverb`
    etc. If `vocab` is provided, restricts to entries where all four
    words appear in vocab.
    """
    txt_path = _cached("questions-words.txt")
    _download(ANALOGY_URL, txt_path)

    out: list[tuple[str, tuple[str, str, str, str]]] = []
    current_category = "uncategorized"
    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(":"):
            current_category = line[1:].strip()
            continue
        parts = line.lower().split()
        if len(parts) != 4:
            continue
        a, b, c, d = parts
        if vocab is not None and not all(w in vocab for w in (a, b, c, d)):
            continue
        out.append((current_category, (a, b, c, d)))
    return out

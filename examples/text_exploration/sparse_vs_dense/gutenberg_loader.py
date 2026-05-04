"""Gutenberg corpus loader for ARB-139 continual learning experiments.

Downloads a curated list of public-domain novels from Project Gutenberg
and tokenizes them in text8-compatible format (lowercase, alphanumeric
only, whitespace-split). Cached to data/cache/gutenberg/.

Used as corpus B for cross-domain continual learning: Wikipedia (text8)
→ fiction (Gutenberg). Different vocabulary distribution, different
sentence structures, different statistics — a real domain shift.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "gutenberg"

# Curated list of Gutenberg book IDs spanning genres and eras. Size estimates
# from PG metadata. Total ~5M tokens after cleaning.
GUTENBERG_BOOK_IDS = [
    11,  # Alice's Adventures in Wonderland (Carroll)
    12,  # Through the Looking-Glass (Carroll)
    16,  # Peter Pan (Barrie)
    74,  # The Adventures of Tom Sawyer (Twain)
    76,  # Adventures of Huckleberry Finn (Twain)
    84,  # Frankenstein (Shelley)
    98,  # A Tale of Two Cities (Dickens)
    174,  # The Picture of Dorian Gray (Wilde)
    345,  # Dracula (Stoker)
    1342,  # Pride and Prejudice (Austen)
    1661,  # The Adventures of Sherlock Holmes (Doyle)
    2701,  # Moby Dick (Melville)
    2814,  # Dubliners (Joyce)
    4300,  # Ulysses (Joyce)
    5200,  # Metamorphosis (Kafka)
    1400,  # Great Expectations (Dickens)
    1184,  # The Count of Monte Cristo (Dumas)
    1497,  # Republic (Plato)
    2542,  # A Doll's House (Ibsen)
    100,  # Complete Works of Shakespeare
    158,  # Emma (Austen)
    161,  # Sense and Sensibility (Austen)
    768,  # Wuthering Heights (Bronte)
    1232,  # The Prince (Machiavelli)
    1260,  # Jane Eyre (Bronte)
    35,  # The Time Machine (Wells)
    36,  # The War of the Worlds (Wells)
    120,  # Treasure Island (Stevenson)
    144,  # The Strange Case of Dr. Jekyll and Mr. Hyde (Stevenson)
    219,  # Heart of Darkness (Conrad)
    1080,  # A Modest Proposal (Swift)
    829,  # Gulliver's Travels (Swift)
    23,  # Narrative of the Life of Frederick Douglass
    600,  # Notes from the Underground (Dostoevsky)
    2554,  # Crime and Punishment (Dostoevsky)
    28054,  # The Brothers Karamazov (Dostoevsky)
    2600,  # War and Peace (Tolstoy)
    1399,  # Anna Karenina (Tolstoy)
    140,  # The Jungle (Sinclair)
    779,  # Babbit (Sinclair Lewis)
]


def _gutenberg_url(book_id: int) -> str:
    """Try the standard /cache/epub URL pattern first."""
    return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"


def _alt_url(book_id: int) -> str:
    """Some older books are at /files/."""
    return f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"


def _strip_pg_header_footer(text: str) -> str:
    """Remove Project Gutenberg's standard header and footer boilerplate."""
    start_re = re.compile(r"\*\*\* START OF [^*]*\*\*\*", re.IGNORECASE)
    end_re = re.compile(r"\*\*\* END OF [^*]*\*\*\*", re.IGNORECASE)
    m_start = start_re.search(text)
    m_end = end_re.search(text)
    if m_start:
        text = text[m_start.end() :]
    if m_end:
        text = text[: m_end.start()]
    return text


def _tokenize_text8_style(text: str) -> list[str]:
    """Lowercase, replace non-alphanumeric with space, collapse whitespace.
    Mirrors text8's preprocessing.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def _download_book(book_id: int) -> str | None:
    """Download a single Gutenberg book, falling back across URL patterns.
    Returns raw text or None on failure.
    """
    for url in (_gutenberg_url(book_id), _alt_url(book_id)):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "arbora-research/1.0 (research@arbora.local)"},
            )
            with urllib.request.urlopen(req, timeout=30) as f:
                raw = f.read()
            # Try multiple encodings.
            for enc in ("utf-8", "latin-1"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"  fetch failed for book_id={book_id} via {url}: {e}")
            continue
    return None


def _cached_book_path(book_id: int) -> Path:
    return CACHE_DIR / f"pg{book_id}.txt"


def load_gutenberg_corpus(
    max_tokens: int | None = None, verbose: bool = True
) -> list[str]:
    """Download (or load from cache) Gutenberg books, return tokenized list.

    Args:
        max_tokens: Stop accumulating once we have this many tokens (or
            after exhausting the book list).
        verbose: Print per-book diagnostics.

    Returns:
        Token stream tokenized text8-style.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_tokens: list[str] = []
    for book_id in GUTENBERG_BOOK_IDS:
        cache_path = _cached_book_path(book_id)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            with cache_path.open(encoding="utf-8") as f:
                cleaned = f.read()
            tokens = cleaned.split()
            source = "cached"
        else:
            raw = _download_book(book_id)
            if raw is None:
                if verbose:
                    print(f"  skip book_id={book_id}: download failed")
                continue
            stripped = _strip_pg_header_footer(raw)
            tokens = _tokenize_text8_style(stripped)
            cleaned = " ".join(tokens)
            with cache_path.open("w", encoding="utf-8") as f:
                f.write(cleaned)
            source = "downloaded"
        if verbose:
            print(f"  book_id={book_id:>5d}: {len(tokens):>9,} tokens ({source})")
        all_tokens.extend(tokens)
        if max_tokens is not None and len(all_tokens) >= max_tokens:
            break

    if max_tokens is not None:
        all_tokens = all_tokens[:max_tokens]

    if verbose:
        print(
            f"\n  total: {len(all_tokens):,} tokens from "
            f"{len([p for p in CACHE_DIR.glob('pg*.txt') if p.stat().st_size > 0])} books"
        )
    return all_tokens


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--max-tokens", type=int, default=5_000_000)
    args = p.parse_args()
    tokens = load_gutenberg_corpus(max_tokens=args.max_tokens)
    print(f"\nFirst 30 tokens: {tokens[:30]}")
    print(f"Last 10 tokens: {tokens[-10:]}")

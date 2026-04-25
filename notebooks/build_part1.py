"""Generate notebooks/arb139_part1_word2vec_to_ssh.ipynb.

Run:  uv run python notebooks/build_part1.py
Output: notebooks/arb139_part1_word2vec_to_ssh.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# -- 1. Title ----------------------------------------------------------------
md(
    """
# ARB-139 Part 1 — From word2vec to SSH

This notebook walks through, with runnable code:

1. What word2vec actually computes (skip-gram + negative sampling + sigmoid gradient)
2. Why the sigmoid factor is doing **three jobs at once** — magnitude bounding, surprise modulation, and a sliding threshold
3. How **SSH** (Sparse Skip-gram Hebbian) translates the same idea to sparse binary
   representations + local Hebbian / anti-Hebbian updates
4. Why SSH plateaus and regresses without modulation — the missing "stabilizer" row
   in the translation table
5. The actual sample-efficiency curves measured on text8

By the end you should be able to step through both update rules by hand on a toy corpus
and see why the difference between word2vec and SSH is not "gradient vs Hebbian" but
"stabilized vs not."

Requires: `numpy`, `matplotlib`. The final cell loads the sweep CSVs from
`data/runs/arb139/` produced by the overnight sweep.
"""
)

# -- 2. Setup ----------------------------------------------------------------
md("## Setup")

code(
    """
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
"""
)

# -- 3. Toy corpus -----------------------------------------------------------
md(
    """
## A toy corpus

Strip away corpus-loading complexity. Six-word universe, ~30-token stream with
deliberate co-occurrence structure: `cat` & `sat` go together, `bird` & `flew`
go together, `dog` mixes with both.
"""
)

code(
    """
vocab = ['cat', 'dog', 'bird', 'sat', 'flew', 'mat']
W2I = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

sentences = [
    "cat sat mat",
    "dog sat mat",
    "dog cat sat",
    "bird flew mat",
    "dog flew bird",
    "cat sat",
    "bird flew",
    "dog bird flew",
    "cat sat mat",
    "bird flew mat",
]
tokens = ' '.join(sentences).split()
token_ids = [W2I[t] for t in tokens]

print(f"Vocab ({V} words):", vocab)
print(f"Stream ({len(tokens)} tokens):", tokens)
"""
)

# -- 4. Skip-gram sampling ---------------------------------------------------
md(
    """
## Step 1 — Skip-gram + negative sampling

This piece is **identical between word2vec and SSH** — it shapes the data into
`(center, context, ±1)` triples. For each token, look at its window neighbors
(positives) and draw random words from `unigram^0.75` as negatives.

Note: this has nothing to do with vectors yet. It's pure data shaping.
"""
)

code(
    """
def skipgram_pairs(token_ids, window=2, n_neg=2, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    V = max(token_ids) + 1
    counts = np.bincount(token_ids, minlength=V).astype(float)
    probs = counts ** 0.75
    probs /= probs.sum()
    for i, center in enumerate(token_ids):
        lo, hi = max(0, i - window), min(len(token_ids), i + window + 1)
        for j in range(lo, hi):
            if j == i:
                continue
            yield center, token_ids[j], 1
            for _ in range(n_neg):
                yield center, int(rng.choice(V, p=probs)), -1

print("First 12 pairs from the stream:")
for ct, (c, ctx, label) in enumerate(skipgram_pairs(token_ids, window=2, n_neg=2)):
    if ct >= 12:
        break
    tag = 'POS' if label > 0 else 'neg'
    print(f"  {tag}  ({vocab[c]:>5} -> {vocab[ctx]:>5})")
"""
)

# -- 5. word2vec representation ----------------------------------------------
md(
    """
## Step 2a — word2vec representation (CDR)

Each word gets two `D`-dimensional dense float vectors: one for "center" role,
one for "context." Compatibility between two words is the **dot product** of
their vectors.

We use `D=8` here so we can print the actual numbers and see them change.
"""
)

code(
    """
D = 8
v_center = np.random.normal(0, 0.1, (V, D))
v_context = np.random.normal(0, 0.1, (V, D))

def w2v_score(w, c):
    return float(v_center[w] @ v_context[c])

print("Initial v_center[cat]:")
print(np.array2string(v_center[W2I['cat']], precision=3, suppress_small=True))
print()
print(f"score(cat, sat)  = {w2v_score(W2I['cat'], W2I['sat']):+.4f}  "
      f"(should be near zero — vectors are random)")
print(f"score(cat, bird) = {w2v_score(W2I['cat'], W2I['bird']):+.4f}")
"""
)

# -- 6. word2vec update on one pair ------------------------------------------
md(
    r"""
## Step 3a — word2vec update (one positive pair)

The loss for a positive pair $(w, c)$ is:

$$L = -\log \sigma(v_w \cdot v_c)$$

The gradient w.r.t. $v_w$ is:

$$\frac{\partial L}{\partial v_w} = -\sigma(-\text{score}) \cdot v_c$$

So the SGD update is:

$$v_w \mathrel{+}= \eta \cdot \sigma(-\text{score}) \cdot v_c$$

Notice the structure: the update direction is `v_c` (push toward the context vector),
and the magnitude is scaled by `σ(-score)` — the **error signal**. Big when score is
low (model is wrong), tiny when score is high (model already correct).

Let's apply one positive update for `(cat, sat)`:
"""
)

code(
    """
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def w2v_update(v_center, v_context, w, c, label, lr=0.1):
    score = float(v_center[w] @ v_context[c])
    if label > 0:
        error = sigmoid(-score)
        d_w = lr * error * v_context[c]
        d_c = lr * error * v_center[w]
        v_center[w] += d_w
        v_context[c] += d_c
    else:
        error = sigmoid(score)
        d_w = -lr * error * v_context[c]
        d_c = -lr * error * v_center[w]
        v_center[w] += d_w
        v_context[c] += d_c
    return score, error

w, c = W2I['cat'], W2I['sat']
before = v_center[w].copy()
score, err = w2v_update(v_center, v_context, w, c, label=1)
after = v_center[w].copy()

print(f"Before update: v_center[cat] = {np.array2string(before, precision=3, suppress_small=True)}")
print(f"score = {score:+.3f}   error_signal sigma(-score) = {err:.3f}")
print(f"      (score near zero -> model uncertain -> full update)")
print(f"After update:  v_center[cat] = {np.array2string(after, precision=3, suppress_small=True)}")
print(f"Diff (all 8 dims touched):  {np.array2string(after - before, precision=3, suppress_small=True)}")
"""
)

# -- 7. Sigmoid as triple-duty stabilizer ------------------------------------
md(
    """
## The crucial insight — sigmoid is doing **three jobs at once**

The factor `σ(-score)` is doing more than just "smooth gradient." It's simultaneously
acting as:

1. **Magnitude bound** — as score grows, the update vanishes, so vectors can't
   blow up
2. **Surprise modulator** — already-correct pairs get tiny updates, wrong pairs
   get big ones
3. **Sliding threshold** — the steep slope at score≈0 means the system spends
   most of its learning where it's most uncertain

Plotting these three regimes:
"""
)

code(
    """
scores = np.linspace(-6, 6, 200)
err = sigmoid(-scores)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))

axes[0].plot(scores, err, 'b-', lw=2)
axes[0].set_xlabel('current score (v_w . v_c)')
axes[0].set_ylabel('error signal sigma(-score)')
axes[0].set_title('Job 1: Magnitude bounding\\n(update vanishes as score grows)')
axes[0].axhline(0, color='gray', lw=0.5)
axes[0].axvline(0, color='gray', lw=0.5)

axes[1].plot(scores, err, 'b-', lw=2)
axes[1].fill_between(scores, 0, err, where=scores > 2, color='green', alpha=0.3,
                     label='already correct -> small update')
axes[1].fill_between(scores, 0, err, where=scores < -2, color='red', alpha=0.3,
                     label='wrong -> big update')
axes[1].set_xlabel('current score')
axes[1].set_ylabel('error signal')
axes[1].set_title('Job 2: Surprise modulation\\n(strong update when wrong)')
axes[1].legend(fontsize=8)

axes[2].plot(scores, err, 'b-', lw=2, label='sigma(-score)')
slope = (sigmoid(-0.01) - sigmoid(0.01)) / 0.02
axes[2].plot(scores, 0.5 + slope * scores, 'r--', alpha=0.6, label='steepest slope at score=0')
axes[2].set_xlabel('current score')
axes[2].set_ylabel('error signal')
axes[2].set_title('Job 3: Sliding threshold\\n(steepest gradient near uncertainty)')
axes[2].set_ylim(-0.1, 1.1)
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()
"""
)

# -- 8. SBR representation ---------------------------------------------------
md(
    """
## Step 2b — SSH representation (SBR)

Now the SSH side. Instead of one dense float vector per word, each word has:
- A **real-valued accumulator** `A_w ∈ R^D` (never exposed externally)
- A **derived sparse binary code** `E_w = top_k(A_w) ∈ {0, 1}^D`

The accumulator is what learning updates; the binary code is what we read out.
We use `D=16, k=4` here so we can print readable values.

Compatibility is **bit overlap** — equivalently, the dot product on binary vectors —
which is what Jaccard similarity normalizes.
"""
)

code(
    """
D_ssh = 16
k = 4

A_center = np.random.normal(0, 0.1, (V, D_ssh))
A_context = np.random.normal(0, 0.1, (V, D_ssh))

def top_k_indices(row, k):
    return np.argpartition(-row, k)[:k]

def to_sdr(row, k, D):
    sdr = np.zeros(D, dtype=bool)
    sdr[top_k_indices(row, k)] = True
    return sdr

def jaccard(a, b):
    union = (a | b).sum()
    return float((a & b).sum()) / union if union > 0 else 0.0

cat_idx = W2I['cat']
print("Accumulator A_center[cat] (16 dims, real-valued):")
print(np.array2string(A_center[cat_idx], precision=3, suppress_small=True))
print()
print(f"top-{k} indices: {sorted(top_k_indices(A_center[cat_idx], k))}")
print(f"E_cat (binary SDR): {to_sdr(A_center[cat_idx], k, D_ssh).astype(int)}")
print()
print(f"jaccard(E_cat, E_sat)  = {jaccard(to_sdr(A_center[cat_idx], k, D_ssh), to_sdr(A_context[W2I['sat']], k, D_ssh)):.3f}")
print(f"jaccard(E_cat, E_bird) = {jaccard(to_sdr(A_center[cat_idx], k, D_ssh), to_sdr(A_context[W2I['bird']], k, D_ssh)):.3f}")
"""
)

# -- 9. SSH update ------------------------------------------------------------
md(
    r"""
## Step 3b — SSH update (one positive pair)

The Hebbian update fires at fixed rates — no error signal.

For positive `(w, c)`:
$$A_{\text{center}}[w, \text{bits}(E_c)] \mathrel{+}= \eta_{\text{pos}}$$
$$A_{\text{context}}[c, \text{bits}(E_w)] \mathrel{+}= \eta_{\text{pos}}$$

For each negative `(w, n)`:
$$A_{\text{center}}[w, \text{bits}(E_n)] \mathrel{-}= \eta_{\text{neg}}$$

Crucially, only the `k` bits in `E_c`'s active set are touched — the other `D - k`
bits of `A_center[w]` are exactly preserved.

This is the **within-row locality** word2vec doesn't have.
"""
)

code(
    """
def ssh_update(A_center, A_context, w, c, label, k, lr_pos=0.05, lr_neg=0.02):
    if label > 0:
        e_context = top_k_indices(A_context[c], k)
        e_center  = top_k_indices(A_center[w],  k)
        A_center[w,  e_context] += lr_pos
        A_context[c, e_center]  += lr_pos
    else:
        e_context = top_k_indices(A_context[c], k)
        A_center[w, e_context] -= lr_neg

w, c = W2I['cat'], W2I['sat']
before = A_center[w].copy()
ssh_update(A_center, A_context, w, c, label=1, k=k)
after = A_center[w].copy()

diff = after - before
touched = np.flatnonzero(np.abs(diff) > 1e-6)
print(f"Before A_center[cat]: {np.array2string(before, precision=3, suppress_small=True)}")
print(f"After  A_center[cat]: {np.array2string(after,  precision=3, suppress_small=True)}")
print(f"Diff:                 {np.array2string(diff,   precision=3, suppress_small=True)}")
print(f"Touched bits: {touched.tolist()} (only {len(touched)} of {D_ssh} -- the rest preserved exactly)")
"""
)

# -- 10. Locality side-by-side -----------------------------------------------
md(
    """
## Locality, side by side

The same single `(cat, sat)` update applied with `D=64` so the contrast is visible:

- word2vec: every one of the 64 dimensions of `v_center[cat]` shifts
- SSH: only `k` (here 16) of the 64 bits of `A_center[cat]` shift; the other 48
  are preserved exactly

This is the structural reason SSH should be better at continual learning — when a
new word arrives, its update is constrained to a small slice of accumulator space,
so unrelated words' representations are not perturbed.
"""
)

code(
    """
rng = np.random.default_rng(0)
D_demo = 64
k_demo = 16

# word2vec demo
v_c_d = rng.normal(0, 0.1, (V, D_demo))
v_x_d = rng.normal(0, 0.1, (V, D_demo))
w2v_before = v_c_d[w].copy()
score = float(v_c_d[w] @ v_x_d[c])
err = sigmoid(-score)
v_c_d[w] += 0.1 * err * v_x_d[c]
w2v_diff = v_c_d[w] - w2v_before

# SSH demo
A_c_d = rng.normal(0, 0.1, (V, D_demo))
A_x_d = rng.normal(0, 0.1, (V, D_demo))
ssh_before = A_c_d[w].copy()
e_ctx = top_k_indices(A_x_d[c], k_demo)
A_c_d[w, e_ctx] += 0.05
ssh_diff = A_c_d[w] - ssh_before

fig, axes = plt.subplots(2, 1, figsize=(14, 4.5))
axes[0].bar(range(D_demo), w2v_diff, color='steelblue', edgecolor='black', linewidth=0.3)
axes[0].set_title(f'word2vec: change in v_center[cat] after one update'
                  f' -- touches all {D_demo} dims')
axes[0].axhline(0, color='black', lw=0.5)
axes[0].set_ylabel('Δ value')

axes[1].bar(range(D_demo), ssh_diff, color='darkorange', edgecolor='black', linewidth=0.3)
axes[1].set_title(f'SSH: change in A_center[cat] after one update'
                  f' -- touches only {(np.abs(ssh_diff) > 1e-9).sum()} of {D_demo} bits')
axes[1].axhline(0, color='black', lw=0.5)
axes[1].set_xlabel('dimension index')
axes[1].set_ylabel('Δ value')

plt.tight_layout()
plt.show()
"""
)

# -- 11. The translation table ----------------------------------------------
md(
    """
## The translation table — and the empty row

| word2vec piece | CDR + SGD form | SBR + Hebbian form (SSH) |
|---|---|---|
| Sampling | skip-gram + unigram^0.75 negatives | **identical** |
| Representation | `v ∈ R^D` (dense) | `A ∈ R^D` accumulator + `E = top_k(A) ∈ {0,1}^D` |
| Compatibility | `v_w · v_c` (dot product) | `\\|E_w ∩ E_c\\|` (bit overlap / Jaccard) |
| Positive update | `v_w += η · σ(-score) · v_c` | `A_center[w, E_c bits] += η_pos` |
| Negative update | `v_w -= η · σ(score) · v_n` | `A_center[w, E_n bits] -= η_neg` |
| Locality | row-local (touches all D dims of touched rows) | row-local **and** within-row (only k of D) |
| **Stabilizer** | sigmoid (triple-duty) | **EMPTY** ← the gap |

The whole story of "SSH plateaus at ~5M tokens and regresses at 10M" is in that
empty row. We'll fill it in Part 2 (Oja decay, BCM threshold, surprise modulation).
"""
)

# -- 12. Sample efficiency curves -------------------------------------------
md(
    """
## What the empty stabilizer row looks like at scale

Loading the actual sweep CSVs from `data/runs/arb139/`. The plot shows SimLex-999
Spearman correlation as a function of training tokens, for all five baselines we ran:

- **word2vec** — dense, gradient. Slow start, dominates at 10M.
- **sparse_skipgram_hebbian (SSH)** — sparse, Hebbian. Wins at 500k. Non-monotonic.
- **brown_cluster** — sparse, hierarchical. Strong start, plateaus ~2M, regresses.
- **random_indexing** — sparse, no learning. Stuck near zero (saturation pathology).
- **t1_sparse** — sparse, cortical Hebbian. Goes anti-correlated past 1M.

If the CSVs aren't present, this cell just prints a message — running the overnight
sweep (`bash examples/text_exploration/sparse_vs_dense/sweep_arb139.sh`) generates them.
"""
)

code(
    """
import csv
from collections import defaultdict
from pathlib import Path

# Try a few likely paths so the notebook works whether opened from notebooks/ or repo root
candidates = [Path('../data/runs/arb139'), Path('data/runs/arb139'),
              Path(__file__).resolve().parent.parent / 'data' / 'runs' / 'arb139'
              if '__file__' in dir() else Path('data/runs/arb139')]
csv_dir = next((p for p in candidates if p.exists()), None)

if csv_dir is None:
    print("No CSVs found. Run the sweep first:")
    print("  bash examples/text_exploration/sparse_vs_dense/sweep_arb139.sh")
else:
    rows = []
    for p in csv_dir.glob('*.csv'):
        with open(p) as f:
            rows.extend(csv.DictReader(f))

    agg = defaultdict(list)
    for r in rows:
        try:
            agg[(r['model'], int(r['n_tokens']))].append(float(r['simlex_spearman']))
        except (ValueError, KeyError):
            continue

    models = ['word2vec', 'sparse_skipgram_hebbian', 'brown_cluster',
              'random_indexing', 't1_sparse']
    colors = {
        'word2vec': 'steelblue',
        'sparse_skipgram_hebbian': 'darkorange',
        'brown_cluster': 'forestgreen',
        'random_indexing': 'gray',
        't1_sparse': 'crimson',
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model in models:
        keys = sorted([k for k in agg if k[0] == model], key=lambda x: x[1])
        if not keys:
            continue
        xs = [k[1] for k in keys]
        ys = [np.mean(agg[k]) for k in keys]
        stds = [np.std(agg[k]) if len(agg[k]) > 1 else 0 for k in keys]
        ax.errorbar(xs, ys, yerr=stds, marker='o',
                    label=model, color=colors.get(model, 'black'),
                    lw=2, capsize=4)

    ax.set_xscale('log')
    ax.set_xlabel('Training tokens (text8)')
    ax.set_ylabel('SimLex-999 Spearman correlation')
    ax.set_title('Sample-efficiency curves — five baselines on text8')
    ax.axhline(0, color='black', lw=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()
"""
)

# -- 13. What we learned -----------------------------------------------------
md(
    """
## Takeaways

- **Word2vec's three pieces** — sampling, scoring, update — are independent.
  We can swap any one without touching the others.
- **The sigmoid factor is the secret** — it's not just smooth gradient flow,
  it's a triple-duty stabilizer: bound + modulate + sliding threshold.
- **SSH retains word2vec's sampling** but swaps the representation (dense → sparse
  binary) and the update rule (gradient → Hebbian). Same data, different machinery.
- **SSH's update is more local than word2vec's** — only `k` of `D` bits change per
  update vs. all `D` for word2vec. This is the structural advantage for continual
  learning.
- **SSH plateaus because the stabilizer row is empty.** Without something playing
  sigmoid's role, accumulator magnitudes drift and the top-k pattern stops
  reorganizing meaningfully past 5M tokens.

## Where Part 2 goes

Part 2 will cover:
1. **Oja's rule** — Hebbian + the right decay term provably converges to the first
   principal component of the input. PCA from local updates only.
2. **Sanger's rule (GHA)** — extension to multiple components. Streaming PCA.
3. **The unifying mathematical equation** for all Hebbian variants, term by term.
4. **What word2vec's sigmoid is doing in the language of those terms** — making
   precise the analogy we sketched above.
"""
)

# -- assemble ----------------------------------------------------------------
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
    },
}

out_path = Path(__file__).parent / "arb139_part1_word2vec_to_ssh.ipynb"
nbf.write(nb, out_path)
print(f"Wrote {out_path}  ({len(cells)} cells)")

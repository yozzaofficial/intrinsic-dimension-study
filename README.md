# Intrinsic Dimension of Qwen2.5-Coder Models — Measurement, Compressibility, Native Low-Dim Training, and Distillation

An empirical four-part study of the **intrinsic dimension** of internal
representations in Qwen2.5-Coder models on coding snippets:

1. **Measure** the intrinsic dimension across three model sizes (1.5B, 7B, 14B)
2. **Test whether the measured redundancy is exploitable** via post-hoc
   low-rank compression with brief recovery fine-tuning
3. **Verify whether training natively in low ambient dimension converges**
   at all, by training two small models from scratch head-to-head
4. **Test extreme distillation** — can a 10.5M-parameter low-dim student
   inherit useful knowledge from a 14B teacher (1400× compression)?

**Short answer:** the intrinsic dimension of coding representations is
strikingly low (~11-16) compared to the ambient dimension the models
use (1536-5120); the last FFN layer of Qwen 14B can be compressed
**233× smaller** with only **+1.3%** perplexity degradation after brief
recovery on a disjoint domain; a from-scratch transformer at hidden_dim=64
(close to the intrinsic dimension) converges to the same best validation
loss as a 4× larger 256d model on the same corpus; and distillation from
Qwen 14B into a 10.5M-parameter low-dim student produces good perplexity
(2.66) and clean KL convergence (-96%), but the student is **too small
to generate structurally valid code** — exposing a critical lesson: low
perplexity ≠ practical usefulness. All four parts together verify the
necessary conditions but reveal that student capacity, not intrinsic
dimension, is now the practical bottleneck.

---

## Part 1 — Measuring intrinsic dimension

### The problem

A modern LLM represents each token as a vector of thousands of numbers:
Qwen2.5-Coder-1.5B uses 1536 dimensions, 7B uses 3584, 14B uses 5120.
This is the **ambient dimension**: the number of scalars needed to store
the vector.

But how many of those scalars carry useful information, and how many are
redundancy? Real representations of real data (text, code) don't fill
ambient space uniformly — they lie on a **manifold** of much lower
dimension. That lower number is the **intrinsic dimension**.

If the intrinsic dimension of coding representations is, say, ~15, but a
model uses 5120 dimensions to store them, there is a ~340× waste in
representation — waste that translates directly into parameters, memory,
and compute.

This measurement had not been published specifically for Qwen2.5-Coder
on the coding domain. This study produces it.

### Methodology

**Models tested:** Qwen2.5-Coder 1.5B / 7B / 14B (Instruct versions,
original bfloat16 weights, not quantized).

**Dataset:** 2000 coding snippets generated from 25 base patterns
(algorithms, React/JSX, CSS, SQL, TypeScript, Rust, Bash, Go, testing,
design patterns) with multiple variable substitutions and unique comments
to ensure no two vectors are identical.

**Extraction:** for each snippet, the model produces a hidden state per
layer. We take the vector at the **last token position** of each layer —
the vector that summarizes the entire snippet up to that processing depth.

**Intrinsic dimension estimation** — three independent methods to triangulate:

- **TwoNN** (Facco et al. 2017): uses the ratio between distances to the
  first and second nearest neighbors. Robust, non-linear, no assumptions
  on manifold shape.
- **MLE** (Levina-Bickel): maximum likelihood estimator with k=5 neighbors.
  More aggressive local estimator, tends to underestimate.
- **PCA cumulative**: how many principal components are needed to capture
  95% (or 99%) of variance. Assumes linearity → generally over-estimates
  the true intrinsic dimension, but useful as an upper reference bound.

**Hardware:** Vast.ai VPS with A100 80GB. Total experiment cost: ~$3
of GPU time.

### Results — Part 1

#### Summary: average intrinsic dimension across all layers

| Model           | hidden_dim | TwoNN | MLE  | PCA-95 | Theoretical compression |
|-----------------|-----------:|------:|-----:|-------:|------------------------:|
| Qwen 1.5B-Coder | 1536       | 11.6  | 3.7  | 11.1   | **132×**                |
| Qwen 7B-Coder   | 3584       | 14.4  | 3.6  | 11.9   | **249×**                |
| Qwen 14B-Coder  | 5120       | 16.1  | 8.9  | 19.2   | **317×**                |

Theoretical compression = `hidden_dim / average TwoNN` — the pure
redundancy factor between ambient dimension and useful information
encoded.

#### Observation 1: sublinear growth with scale

The 14B model has **9.3× more parameters** and **3.3× larger ambient
dimension** than the 1.5B — but only **~1.4× larger intrinsic dimension**
(12→16). The useful information for representing a coding snippet **does
not scale with model size** the same way ambient dimension does.

#### Observation 2: minimum intrinsic dimension converges across scales

Looking at the last layer of each model (where representations are most
"concentrated" for next-token prediction):

| Model     | Last layer TwoNN | Minimum TwoNN across all layers |
|-----------|-----------------:|--------------------------------:|
| Qwen 1.5B | 8.8              | 7.1 (layer 25)                  |
| Qwen 7B   | 9.8              | 8.8 (layer 24)                  |
| Qwen 14B  | 12.5             | 12.5 (layer 47)                 |

All three models converge to representing snippets in **7-13 intrinsic
dimensions** in their final layers, regardless of their ambient
dimension (which ranges from 1536 to 5120).

#### Observation 3: highly structured layer-by-layer pattern

All three models follow the same profile:

- **Early layers**: higher dimension (12-27) — the model "opens up" the
  representation from raw tokens
- **Middle layers (~40-60% of depth)**: intrinsic dimension peak (14-19) —
  rich processing phase
- **Late layers (~20% final)**: minimum dimension (7-13) — compression
  toward the target representation for prediction

This profile is **the same shape** across models of different sizes,
suggesting an architectural regularity in how depth compresses
information.

#### Observation 4: theoretical compression grows with scale

Theoretical compression factor (ambient_dim / intrinsic_dim) doubles
from 1.5B to 14B (132× → 317×). The larger model is proportionally
**more redundant** — it uses more ambient dimensions to represent the
same amount of useful information.

---

## Part 2 — Is the measured redundancy actually exploitable?

Measuring intrinsic dimension is a **necessary but not sufficient**
condition for compression: it says "the information fits in a small
subspace", but not "we can build a smaller model that exploits this
without losing capability".

Part 2 tests whether the redundancy is actually exploitable in practice,
via low-rank SVD compression of the last FFN block of Qwen2.5-Coder-14B,
followed by a brief recovery fine-tune.

### Methodology

**Target of compression:** the FFN of the last transformer layer of
Qwen2.5-Coder-14B (three matrices: `gate_proj`, `up_proj`, `down_proj`,
totaling **212M parameters**, all with shape 5120×13824 or 13824×5120).

**Compression method:** for each matrix W, apply SVD and keep only the
top-k singular values, replacing the original matrix with two smaller
matrices `down: 5120→k` and `up: k→13824`. Total new params: `k × (5120+13824)`.

**Ranks tested:** 16, 32, 64. Rank 16 is close to the measured intrinsic
dimension of the last layer (~13), rank 32-64 provide progressive safety
margins.

**Two-phase evaluation:**
1. **Cold**: measure perplexity on 12 held-out test snippets immediately
   after compression, no training.
2. **Warm**: fine-tune ONLY the compressed matrices for 100 steps on a
   **completely disjoint** recovery corpus (500 snippets covering different
   domains: parsers, sockets, ML, cryptography, game dev, cloud infra…),
   then re-measure perplexity on the same test snippets.

**Critical methodological point:** the recovery corpus contains no
snippet (or variant) that appears in the test set. The script includes an
explicit contamination check that fails if any test snippet substring
appears in the recovery corpus. This distinguishes real recovery of
capability from mere memorization of test data — a distinction the first
run of this experiment did NOT enforce (see `svd_recovery_results.json` for
the invalid earlier results, kept for transparency).

### Results — Part 2

#### Compression + recovery on Qwen2.5-Coder-14B last FFN layer

Baseline perplexity (uncompressed model on test set): **1.7460**

| Rank | Compression | PPL cold | Δ cold  | PPL warm | Δ warm  |
|-----:|------------:|---------:|--------:|---------:|--------:|
| 16   | **233×**    | 1.8492   | +5.9%   | 1.7689   | **+1.3%** |
| 32   | 117×        | 1.8511   | +6.0%   | 1.7811   | +2.0%   |
| 64   | 58×         | 1.8454   | +5.7%   | 1.7979   | +3.0%   |

Recovery time: ~5 seconds per rank (100 fine-tune steps, only compressed
matrices are trainable, everything else frozen).

#### Observation 5: cold compression is surprisingly robust

Even at 233× compression (rank 16, close to the measured intrinsic
dimension of the layer), the perplexity increases only **+5.9%** without
any recovery. This alone is a strong signal: the SVD retains the
principal components, and those components already capture most of the
useful information.

The fact that Δcold is essentially identical across rank 16, 32, and 64
(+5.7% to +6.0%) suggests that once you keep more directions than the
intrinsic dimension, adding more doesn't help — additional singular
components carry noise, not useful information.

#### Observation 6: brief recovery essentially closes the gap

After only 100 fine-tune steps on a disjoint domain, rank 16 recovers to
**+1.3%** of baseline perplexity. The recovery is more effective for
lower ranks (rank 16 warm is better than rank 32 or 64 warm), which is
counter-intuitive but consistent: at low rank, the compressed subspace
aligns tightly with the intrinsic manifold and the fine-tune can adjust
the projections without competing with noisy dimensions.

#### Observation 7: 212M parameters → 909K parameters, ~1.3% quality cost

The last FFN layer goes from **212,336,640 parameters (~424MB in bf16)**
to **909,312 parameters (~1.8MB in bf16)** with a **1.3% perplexity
degradation** on unseen coding tasks. This is a real, reproducible
demonstration that the redundancy measured in Part 1 is not just
theoretical.

### Important caveats (do not overstate)

- **Only the last layer was compressed.** Whether all 48 layers can be
  compressed this aggressively simultaneously is an open question — errors
  from stacked compressions likely compound.
- **Perplexity ≠ generative quality.** The model may still produce good
  perplexity while degrading in tool-calling, instruction-following, or
  long-form generation. A stronger test would evaluate the compressed
  model on HumanEval or MBPP.
- **Small recovery corpus.** 500 snippets, 100 steps. Production
  compression would need much more data/steps to stabilize.
- **Only FFN, not attention.** Attention matrices (Q, K, V, O projections)
  were not tested. They may have different intrinsic dimensions.

### What this changes about the possibility

Part 1 said: **there is theoretical room** to compress by orders of
magnitude. Part 2 says: **at least for one layer, that room is real** —
100 steps of recovery bring quality within a few percent of baseline
after a 233× compression.

This does not immediately give you a 100× smaller LLM. But it makes it
much less speculative to say that a natively-designed low-dimensional
architecture, trained with the right geometry from scratch, could work.
Before Part 2, that was speculation. Now it has one empirical data
point in its favor.

---

## Part 3 — Does native low-dimensional training actually converge?

Parts 1 and 2 tested compression **after** training. But the strongest
claim would be that a model **trained from scratch** with an ambient
dimension close to the measured intrinsic dimension can converge without
the "safety margin" of extra dimensions that transformer training
traditionally assumes it needs.

Part 3 tests this precondition directly: train two small transformers
from scratch on the same coding corpus, one with a "standard" ambient
dimension and one with an ambient dimension close to the intrinsic
dimension measured in Part 1, and compare convergence behavior.

### Methodology

**Two models trained head-to-head on identical data and identical
training schedule:**

| Model | hidden_dim | n_layers | Params |
|-------|-----------:|---------:|-------:|
| **MODEL A (control)** | 256 | 8  | **84.0M** |
| **MODEL B (low-dim)** | 64  | 16 | **20.2M** |

Both use the same tokenizer (Qwen2.5-Coder-1.5B-Instruct BPE, vocab
151665), identical corpus (~5000 blocks of 256 tokens from ~20,000
coding snippets), identical batch size, learning rate schedule (cosine
with 200 warmup steps, peak lr=3e-4), same 10,000 training steps.
MODEL A has **4.2× more parameters** than MODEL B.

The two models differ intentionally in *both* width and depth to keep
parameter counts comparable at the two extremes: MODEL B compensates
its narrower ambient dimension with double the depth. This is the
architectural pattern you would expect if the measured intrinsic
dimension holds — depth still helps for the sequential reasoning, but
width can shrink to the intrinsic size.

### Results — Part 3

#### Final validation losses

| Model | Params | Best val loss (peak) | Final val loss (step 10000) |
|-------|-------:|---------------------:|----------------------------:|
| MODEL A (256d) | 84.0M | ~0.325 (at step ~1000) | **1.236** |
| MODEL B (64d)  | 20.2M | ~0.325 (at step ~3000) | **0.364** |

The final numbers alone are misleading — MODEL A's final val loss is
inflated by severe overfitting, not by an architectural failure.
Reading only the last row would suggest MODEL B is 3× better, which
overstates the case.

#### Observation 8: MODEL A overfits catastrophically, MODEL B does not

MODEL A shows textbook overfitting: train loss falls to ~0.015 while
val loss climbs continuously from ~0.325 (step ~1000, best point) to
1.236 (step 10000, worst point). With 84M parameters on ~5000 unique
blocks, the model has enough capacity to memorize the training set
verbatim.

MODEL B stays remarkably stable: train and val losses track each other
closely for the entire 10,000 steps (train ~0.30, val ~0.35 throughout
steps 2000-10000). No overfitting. No divergence. The lower ambient
dimension acts as an implicit regularizer.

#### Observation 9: both models reach the same best generalization point

Read at their respective *best* validation loss (not just final), both
models converge to essentially the same point (~0.325). MODEL B gets
there more slowly (step ~3000 vs step ~1000), which is expected because
each parameter update carries less "capacity" in a lower-dimensional
model — but it gets there.

**This is the key finding of Part 3: low ambient dimension is not a
barrier to reaching the same generalization quality**. It slows
convergence in per-step terms but does not prevent it.

#### Observation 10: generative samples confirm the pattern

Both models produce syntactically valid Python code after training.
MODEL A tends to reproduce training snippets more verbatim (a symptom of
memorization), while MODEL B produces more mixed/composed outputs
(healthier generalization behavior). Both correctly complete prompts
like `def binary_search`, `class Stack:`, `def is_prime(n):`.

### What Part 3 changes about the claim

Part 1 said: **the intrinsic dimension is low** (measurement).
Part 2 said: **that redundancy is exploitable via post-hoc compression**
(compression of last layer, 233×, ~1.3% degradation).
Part 3 now says: **a model trained natively in low ambient dimension
does converge**, reaching the same generalization quality as a
4× larger model on the same data.

The three parts together verify the necessary conditions:
1. The intrinsic dimension is low (empirical measurement).
2. Existing models are compressible into that low dimension (post-hoc).
3. Training natively in that low dimension is possible (from scratch).

This does **not** yet prove that a 20M-parameter model can match
Qwen 14B on real coding tasks — the corpus and vocabulary here are
tiny compared to a production model's training data. But it removes the
last speculative element: the optimizer *can* converge in low ambient
dimension. Whether the resulting model would generalize well at scale
is now purely an empirical scaling question, not a fundamental one.

### Caveats specific to Part 3

- **Tiny corpus** (~5000 blocks from ~20,000 template-based snippets).
  Real coding has vastly more diversity. A production-scale test would
  need millions of unique snippets from GitHub / StackExchange.
- **Character-level content, BPE tokenization mismatch potential.** The
  models use Qwen's BPE tokenizer but train on a corpus much smaller and
  more repetitive than what that vocabulary was designed for. Some
  vocabulary is essentially unused during training.
- **Same final loss ≠ same final capability.** Perplexity/cross-entropy
  is a proxy. A stronger test would generate longer code and evaluate
  functional correctness on HumanEval or MBPP.
- **Only one architecture ratio tested.** The width×depth trade-off (64d
  × 16 layers vs 256d × 8 layers) is one point on a 2D grid. A proper
  scaling study would sweep both axes.

---

## Part 4 — Can a 700× smaller student learn from a 14B teacher via distillation?

Part 3 showed that a small (20M-parameter) transformer can be **trained
from scratch** in low ambient dimension and reach the same generalization
as a 4× larger model on the same corpus. Part 4 asks a much more ambitious
question: can that same low-dim student inherit knowledge from a much
larger, pre-trained teacher via **logit distillation**? If so, the student
might reach far better quality than what training from scratch on a small
corpus can provide.

The distillation ratio here is **~1400×**: teacher is Qwen2.5-Coder-14B
(14.77B parameters), student is a 10.5M-parameter transformer at
hidden_dim=64, 16 layers. This is far more aggressive than any published
distillation study.

### Methodology

**Teacher:** Qwen2.5-Coder-14B-Instruct, frozen, bf16 on GPU. Produces
logits for every training batch.

**Student:** identical architecture to MODEL B of Part 3 (hidden_dim=64,
16 layers, 4 attention heads, d_ff=256, seq_len=256), tied embedding
weights, 10.55M parameters total.

**Critical detail (bug fix that had to be tracked down):** Qwen's
tokenizer reports `vocab_size=151665`, but the model's embedding is
padded to `152064` (a multiple of 128 for GPU efficiency). Using
`len(tokenizer)` as the student's vocab size causes a shape mismatch in
the KL divergence between student and teacher logits, crashing training
on the first step. The student must adopt `teacher.config.vocab_size`.
This is a common pitfall when distilling from Qwen models.

**Combined loss:**

```
L = α · CrossEntropy(student, targets) + (1-α) · T² · KL(student/T || teacher/T)
```

with α=0.5, temperature T=2.0 (standard values for distillation, softens
the teacher distribution to expose more of its "dark knowledge").

**Corpus:** ~3000 unique coding snippets from ~20 base patterns (same
generation approach as Part 3, with unique comments per variant to
guarantee no duplicates). Split 90/10 train/val. Total: 656 blocks of
256 tokens.

**Training:** 5000 steps, batch size 4, AdamW, cosine LR schedule with
200 warmup steps, peak lr=3e-4. Total time: ~16 minutes on A100.

### Results — Part 4

#### Loss curves

| Step | Total loss | CE | KL | val PPL |
|-----:|-----------:|---:|---:|--------:|
| 1    | 7198       | 39.7 | 14356 | ~3.5·10¹⁷ (random) |
| 500  | 1466       | 6.6 | 2925 | 693 |
| 1000 | 700        | 3.1 | 1397 | 16.4 |
| 2000 | 370        | 1.48 | 738 | 3.87 |
| 3000 | 275        | 1.09 | 548 | 2.97 |
| 5000 | 267        | 1.15 | 533 | **2.66** |

**KL divergence dropped 96%** during training (14356 → 533), and validation
perplexity fell from essentially random to **2.66**. Both metrics indicate
successful convergence: the student is actively tracking the teacher's
next-token distribution, and no overfitting occurs (train and val CE
stay aligned around 1.0 for the last 3000 steps).

#### Observation 11: the distillation signal transfers cleanly

By any statistical measure, the training worked. KL loss decreases
monotonically and val PPL reaches a value (2.66) typical of models
that have properly modeled the language. There is no gradient collapse,
no divergence, no training instability. **The 1400× compression ratio
is not, by itself, a barrier to the distillation signal reaching the
student.**

#### Observation 12: BUT generated code is not syntactically correct

After 5000 training steps, generation from the student produces text
that *resembles* Python code (correct indentation, valid keywords like
`def`, `class`, `if`, `return`, `for`, correct-looking structure) but
is not valid syntax:

```
def binary_search, import = 0, right = 0, len(data    if len(seq):
        if len(data    while left < right:
        if not in range(2
        if n == 1:
        if n == 1:
        return True
    return True
    return True
    return True
```

The model has learned the **statistical distribution** of coding tokens
(hence the low perplexity) but has **insufficient structural reasoning
capacity** to produce coherent programs. It knows *that* `def` is followed
by a name, that indented blocks follow `:`, that `return` appears near
ends of functions — but cannot maintain semantic state over dozens of
consecutive tokens.

The gap between "modeling token distributions well" and "generating
structurally coherent code" is significant, and Part 4 exposes it clearly.

#### Observation 13: perplexity is not the same as usefulness

This is the most important lesson of Part 4, and it applies to all
compression/distillation work in general. A model can achieve very good
perplexity while being **practically unusable**. The 10.5M-parameter
student here has better token-level statistics than one might expect
from a model 1400× smaller than the teacher, but it cannot reliably
produce working code.

The mistake to avoid is treating low perplexity as a green light for
"the compression / distillation worked". It means "the model learned
the statistical distribution", which is a necessary but very weak
condition for practical usefulness.

### What Part 4 does and does not prove

**Proves:**
- The distillation signal from a 1400× larger teacher does reach the
  student and produces measurable learning (KL -96%, val PPL 2.66)
- Optimization is stable, no gradient collapse at this compression ratio
- Small students can capture teacher token distributions even at extreme
  compression

**Does not prove:**
- That 10M parameters are enough to write coherent code (they are not,
  based on the generated samples)
- That scaling the corpus would automatically improve structural quality
  (likely but not demonstrated)
- That the resulting model is usable for a real coding-agent objective
  (it clearly is not, yet)

### Interpretation for the broader project

The four parts together give a clear layered picture:
- **Part 1**: the intrinsic dimension of coding representations is low (~11-16).
- **Part 2**: post-hoc SVD can compress one layer 233× with minimal degradation
  after brief recovery.
- **Part 3**: native training in low ambient dimension converges without breaking.
- **Part 4**: distillation from a huge teacher into a low-dim student
  reaches good statistical fit, but 10M parameters are too few for
  coherent generation on this domain.

The bottleneck moving forward is not the intrinsic dimension (which is
demonstrably low) but the **student's total parameter count and corpus
diversity**. A next-step experiment would be to test the same
distillation setup with a larger student (e.g., 50-100M parameters,
still 140-280× smaller than the teacher) on a real-world coding corpus
(e.g., a subset of The Stack), and evaluate not just perplexity but
also generative correctness on HumanEval or MBPP.

### Caveats specific to Part 4

- **Small student.** 10.5M parameters is at the very low end of what
  produces coherent generation for language models. Even Qwen2.5-0.5B
  (500M parameters) is 50× larger.
- **Small, homogeneous corpus.** 3000 template-based snippets are far
  fewer than what a from-scratch model needs for varied generation.
- **Only text-level distillation.** More advanced techniques (feature
  distillation on intermediate layers, patient distillation with
  progressive width expansion) were not tested.
- **No evaluation of generative correctness beyond visual inspection.**
  A rigorous test would run HumanEval on the compressed model and
  measure pass@k.

---

## Files in this repository

```
data/
  intrinsic_dim_Qwen_Qwen2.5-Coder-1.5B-Instruct.json   # Part 1 — 1.5B results (28 layers)
  intrinsic_dim_Qwen_Qwen2.5-Coder-7B-Instruct.json     # Part 1 — 7B results (28 layers)
  intrinsic_dim_Qwen_Qwen2.5-Coder-14B-Instruct.json    # Part 1 — 14B results (48 layers)
  svd_recovery_v2_results.json                          # Part 2 — valid results (train/test separated)
  svd_recovery_results.json                             # Part 2 — earlier invalid results (kept for transparency)
  low_dim_native_results.json                           # Part 3 — head-to-head training A vs B
  distill_14b_to_lowdim_results.json                    # Part 4 — 14B → 10.5M distillation

scripts/
  measure_intrinsic_dim.py           # Part 1 — reproduces the intrinsic dimension measurement
  svd_compress_and_recover_v2.py     # Part 2 — reproduces the compression + recovery test (VALID)
  svd_compress_and_recover.py        # Part 2 — earlier version with contaminated recovery corpus (invalid)
  train_low_dim_native.py            # Part 3 — trains MODEL A (256d) and MODEL B (64d) from scratch
  distill_14b_to_lowdim.py           # Part 4 — distills Qwen 14B into a 10.5M-param low-dim student
```

Each Part 1 JSON contains: model, snippet count, ambient dimension,
number of layers, and per-layer TwoNN, MLE, PCA-95, PCA-99 estimates.

The Part 2 JSON contains: baseline perplexities (on test and on recovery
domain), and for each rank: parameter counts, compression ratio, cold
and warm perplexities with delta percentages.

The Part 3 JSON contains: full training history for both models (train
loss, val loss, val perplexity, learning rate at every 100 steps),
configuration of each model (hidden_dim, n_layers, n_heads, d_ff,
total params), and generated samples from both models after training.

The Part 4 JSON contains: teacher/student configuration and parameter
counts (with compression ratio), full training history (total loss,
cross-entropy component, KL divergence component, val perplexity at
every 100 steps), and generated samples from the distilled student.

## Why two versions of Part 2

The initial `svd_compress_and_recover.py` used a recovery corpus generated
from the same base patterns as the test set. The resulting fine-tune
effectively memorized test-adjacent data, producing suspiciously large
"improvements" (Δwarm = -18% to -23%) that were not real capability
recovery. The v2 script uses a completely disjoint recovery corpus (25
different seed patterns covering unrelated coding domains) and includes
a contamination check that halts execution if any test snippet appears
in the recovery corpus. Both scripts and results are included for full
methodological transparency.

## Reproduction

Requires a GPU with ≥24GB VRAM for 1.5B/7B, ≥40GB for 14B (Part 1);
Part 2 (14B compression + recovery) needs ≥45GB to also hold the
`deepcopy` during compression.

```bash
pip3 install transformers accelerate scikit-learn numpy torch

# Part 1 — intrinsic dimension measurement
python3 scripts/measure_intrinsic_dim.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --n-snippets 2000
python3 scripts/measure_intrinsic_dim.py --model Qwen/Qwen2.5-Coder-7B-Instruct --n-snippets 2000
python3 scripts/measure_intrinsic_dim.py --model Qwen/Qwen2.5-Coder-14B-Instruct --n-snippets 2000

# Part 2 — compression + recovery on the 14B (valid version)
python3 scripts/svd_compress_and_recover_v2.py --ranks 16 32 64 --recover-steps 100

# Part 3 — head-to-head from-scratch training of MODEL A (256d) and B (64d)
python3 scripts/train_low_dim_native.py

# Part 4 — distillation of Qwen 14B into a 10.5M-param low-dim student
python3 scripts/distill_14b_to_lowdim.py
```

Approximate run times:
- Part 1 — 1.5B: 5-8 minutes
- Part 1 — 7B: 10-15 minutes
- Part 1 — 14B: 15-25 minutes
- Part 2 — 14B compression cycle: 15-20 minutes (dominated by `deepcopy` of the 14B model, ~2 min per rank)
- Part 3 — training MODEL A (256d, 10000 steps): ~29 minutes on A100
- Part 3 — training MODEL B (64d, 10000 steps): ~12 minutes on A100
- Part 4 — 14B → 10.5M distillation (5000 steps): ~16 minutes on A100

All results are saved as JSON in the current directory.

---

## Context and scope

This study grew out of an investigation into running large LLMs on
consumer hardware (16GB Macs). The original question was: "is the
model actually large, or just represented inefficiently?". The empirical
answer on Qwen2.5-Coder is: **it is represented very inefficiently for
its purpose** — there is a theoretical compression margin of over 100×
that no production model has yet exploited, and at least the trivial
end of that margin (last FFN layer) is empirically exploitable with a
few seconds of recovery training after a 233× compression.

**Domain tested is coding-specific.** On general natural language, the
numbers might differ — likely higher, since the manifold is more complex.
It would be interesting to repeat the measurement on Qwen 7B **base**
(non-Coder) to quantify how much fine-tuning on coding reduces intrinsic
dimension compared to the generalist starting point.

**MLE consistently underestimates TwoNN.** This is expected — MLE is a
more aggressive local estimator. The two numbers should be read together:
TwoNN is the robust/conservative estimate on which to base main
interpretations; MLE serves as a lower bound.

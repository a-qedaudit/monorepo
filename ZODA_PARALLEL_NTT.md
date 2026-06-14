# Parallelizing ZODA's NTT / coding math kernels

This branch makes ZODA encode/decode use the `commonware-parallel` `Strategy`
that is **already plumbed through the API** but was being ignored by the heaviest
math kernels. The result is a large speedup on the parallel path with **no change
to the sequential path**, and bit-for-bit identical output.

Branch is for review/experimentation only (no PR).

---

## 1. Motivation: a `Strategy` that wasn't reaching the hot loops

`coding::zoda` already threads a `&impl Strategy` through `encode`/`decode`, and
uses it for the embarrassingly parallel parts (row hashing, per-shard generation).
But the three most expensive steps ran **single-threaded** regardless of the
strategy:

- **encode step 2** – the Reed-Solomon extension: `data.as_polynomials(..).evaluate()`
  (a forward NTT over the encoded matrix). `evaluate()` ignored the strategy.
- **encode step 5** – the checksum `data.mul(&checking_matrix)` (a dense matrix
  multiply over the full data matrix). `Matrix::mul` ignored the strategy.
- **decode** – `EvaluationVector::recover()`, which runs ~3 full-matrix NTTs
  (`interpolate` + the forward/inverse pair inside `divide`). `decode` even took
  the strategy as `_strategy` and dropped it on the floor.

Measured on `origin/main` (commit `e68fede`) on a 128-core box, an 8 MiB block at
`chunks=100` barely benefited from concurrency, which is the tell-tale sign of a
serial bottleneck inside an otherwise parallel pipeline:

| op | conc=1 | conc=8 | scaling |
|----|-------:|-------:|--------:|
| encode | 569.8 ms | 551.2 ms | 1.03x |
| decode | 2.165 s  | 2.145 s  | 1.00x  |

The NTT dominates the serial time; the checksum multiply is secondary.

---

## 2. What changed

Two files, no public-API breakage (only additive `*_with` methods; the existing
no-arg methods are preserved and behave exactly as before).

### `math/src/ntt.rs`

- **`par_matrix_ntt::<FORWARD, F, S>(&mut Matrix<F>, &S)`** — a column-parallel NTT.
  Each column of the matrix is an independent NTT lane (the per-stage butterflies
  mix *rows*, never *columns*), so transforming the columns in groups is identical
  to the in-place whole-matrix NTT. It splits the columns into **one contiguous
  block per worker** (count taken from `Strategy::parallelism_hint()`), copies each
  block out as a small row-major sub-matrix, runs the existing `ntt` on it, and
  scatters it back.
  - **Sequential fast path:** when the hint is `1` (or there is `<= 1` block), it
    runs the original in-place `ntt` over the whole matrix — **no copies, no
    behavior change**.
  - **Why contiguous blocks (not one strided column per task):** a column-at-a-time
    gather touches one element per cache line and was ~1.6x *slower* on the
    sequential strategy in an earlier iteration. Contiguous blocks keep both the
    copy and the transform on contiguous memory, and keep the NTT inner loop wide
    enough to amortize the twiddle-factor multiply.
- **`Matrix::ntt_with`**, **`PolynomialVector::evaluate_with`**,
  **`EvaluationVector::interpolate_with`**, **`PolynomialVector::divide_with`**,
  **`EvaluationVector::recover_with`** — strategy-aware variants that route the
  multi-column NTTs through `par_matrix_ntt`. The single-column NTTs (e.g. the
  vanishing polynomial inside `divide`) stay serial; parallelizing one column is
  pointless.
- **`Matrix::mul_with`** — row-parallel matrix multiply (output rows are
  independent). Falls back to the in-place `Matrix::mul` when
  `parallelism_hint() <= 1`, so the sequential strategy pays nothing.
- The no-arg `evaluate`/`interpolate`/`divide`/`recover`/`mul` keep their original
  sequential implementations.

### `coding/src/zoda/mod.rs`

- `encode`: `…evaluate_with(strategy)` and `data.mul_with(&checking_matrix, strategy)`.
- `decode`: renamed `_strategy` → `strategy` and `…recover_with(strategy)`.

---

## 3. Why this is correct (not just plausible)

- **Algebraic equivalence.** Columns are independent lanes in this NTT, so a
  per-column-block transform equals the whole-matrix transform. The sequential
  path is literally the original `ntt` call.
- **Order preservation.** The scatter writes block `b` back to columns
  `[b*block_cols, …)`, relying on `Strategy::map_collect_vec(0..n, …)` returning
  results in index order. That ordering is part of the existing contract and is
  already relied on by ZODA's own row-hashing (`row_hashes` → BMT build).
- **Tests.** `commonware-math` (23) and `commonware-coding` (34, incl. the ZODA
  round-trip / minifuzz tests) pass.
- **Rayon ≡ Sequential, byte for byte.** A temporary equivalence test (removed
  before commit) encoded the same data under `Sequential` and `Rayon(t)` and
  asserted the **commitment and every strong shard are identical** (`Eq`), then
  ran a full `Rayon` weaken/check/decode round-trip. Verified across data sizes
  `{64 KiB, 256 KiB, 1 MiB}` x thread counts `{2, 8, 13}` (13 exercises uneven,
  non-power-of-two blocks).
- `cargo fmt` (nightly) and `clippy -D warnings` are clean. The new public methods
  inherit the `ntt` module's `stability_scope!(ALPHA)`.

---

## 4. Benchmarks

`cargo bench -p commonware-coding --bench phased_coding_scheme_times`, release,
128-core Linux, 8 MiB block, `chunks=100` (min=33 / total=100), Criterion median.
Baseline = `origin/main` `e68fede`; After = this branch. Same machine, same
toolchain (`git stash` between runs so the two share the dependency build).

### encode (`zoda_phased::encode/msg_len=8388608 chunks=100`)

| conc | baseline | after | speedup |
|-----:|---------:|------:|--------:|
| 1 (Sequential) | 569.8 ms | 559.2 ms | ~1.0x (no regression) |
| 8 (Rayon-8)    | 551.2 ms | **173.7 ms** | **3.2x** |

### decode (`…/conc=. shard_selection=best`)

| conc | baseline | after | speedup |
|-----:|---------:|------:|--------:|
| 1 (Sequential) | 2.165 s | 2.186 s | ~1.0x (no regression) |
| 8 (Rayon-8)    | 2.145 s | **0.794 s** | **2.7x** |

The consensus marshal (`consensus/src/marshal/coding`) threads a real `Strategy`
into encode/decode, so production (Rayon) gets the parallel-path win; the
`Sequential` path is unchanged.

---

## 5. Caveats and future work

- **Not 8x (Amdahl).** The remaining serial work — BMT construction, the
  Fiat-Shamir commitment/shuffle, the single-column vanishing-polynomial steps in
  `divide`, and the fill/collect in `decode` — bounds the speedup, and Rayon was
  capped at 8 threads here. Larger pools scale further because the block count
  follows `parallelism_hint()`.
- **Verify-side checksum encode still serial.** `CheckingData::reckon`
  (`coding/src/zoda/mod.rs`) re-encodes the checksum with a plain `evaluate()`; it
  has no `Strategy` in scope. Threading one in is a small, separate follow-up.
- **Biggest untouched lever: SIMD Goldilocks.** `math/src/fields/goldilocks.rs`
  is scalar `u64`/`u128`. An AVX2/AVX-512 (or NEON) field would speed up *both* the
  sequential and parallel paths and compounds with this change. Larger effort.
- **Memory.** The parallel path allocates one sub-matrix buffer per block
  (total ~= one extra copy of the matrix). The sequential path allocates nothing
  extra.

---

## 6. Reproduce

```bash
# correctness
cargo test -p commonware-math  --lib
cargo test -p commonware-coding --lib

# before/after (stash this branch's edits for the baseline)
cargo bench -p commonware-coding --bench phased_coding_scheme_times -- \
  'zoda_phased::encode/msg_len=8388608 chunks=100'
cargo bench -p commonware-coding --bench phased_coding_scheme_times -- \
  'zoda_phased::decode/msg_len=8388608 chunks=100 conc=. shard_selection=best'
```

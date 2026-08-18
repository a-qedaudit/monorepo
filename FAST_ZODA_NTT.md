# Fast ZODA NTT: thread-parallel x SIMD Goldilocks

This branch combines two orthogonal optimizations of the Goldilocks NTT, the
kernel under ZODA encode/decode (and BLS-threshold recovery):

1. **Thread parallelism** — the dense NTT's columns are independent lanes, so they
   are split into one contiguous block per worker and run across a
   `commonware-parallel` `Strategy`.
2. **SIMD field arithmetic** — each block runs a vectorized Goldilocks NTT (AVX2 on
   x86-64, NEON on aarch64), so the per-element butterfly arithmetic is done
   `WIDTH` lanes at a time.

They compose: **threads spread the column blocks across cores; SIMD vectorizes the
arithmetic within each block**, giving `cores x lanes` of speedup. The sequential
path (`Sequential` strategy) still gets the full SIMD win with zero threading
overhead, and all paths produce byte-identical output to the original scalar code.

This is the union of two earlier standalone branches (`zoda-parallel-ntt`,
`goldilocks-simd-ntt`); the only integration point is that `par_matrix_ntt` calls
the per-field `FieldNTT::ntt_dense` (SIMD) on each block instead of the generic
scalar NTT.

Branch on `a-qedaudit/monorepo` for review (this writeup is not meant to merge).

---

## 1. Design

### `FieldNTT::ntt_dense` (the seam)
A new trait method does an in-place dense NTT over a row-major `rows x cols`
slice. The default is the scalar butterfly (`ntt::ntt_dense_scalar`, a faithful
refactor of the previous generic `Matrix` NTT). Goldilocks overrides it with a
SIMD kernel. This single hook is where vectorization plugs in.

### `par_matrix_ntt` (the parallel layer)
Splits the matrix columns into one contiguous block per worker (count from
`Strategy::parallelism_hint()`), and for each block calls `F::ntt_dense`:
- `Sequential` hints one block, so it runs `ntt_dense` in place over the whole
  matrix, no copies, no thread overhead;
- `Rayon(n)` hints `n` blocks, copies each out as a small row-major sub-matrix
  (contiguous, cache-friendly), runs `ntt_dense` on it, and scatters back.

`evaluate`/`recover`/`divide` expose `*_with(strategy)` variants used by ZODA
encode/decode; the no-arg versions delegate to `Sequential`.

### SIMD Goldilocks (`fields::goldilocks::simd`)
A packed field whose `add`/`sub`/`mul`/`div2` mirror `F::add_inner`/`sub_inner`/
`reduce_128`/`div_2` operation-for-operation, so the vector result equals the
scalar field bit-for-bit. The butterfly does `WIDTH` contiguous columns per
instruction with a scalar tail for `cols % WIDTH`.
- **AVX2** (x86-64 with `std`, 4 lanes): selected per call via runtime
  `is_x86_feature_detected!`, with the scalar path taken otherwise.
- **NEON** (aarch64, 2 lanes): part of the aarch64 baseline, so it is always used,
  no detection needed (also works in `no_std`).
- **scalar**: everything else, including x86-64 **without AVX2 at runtime**, x86-64
  with the `std` feature off (no runtime detection available), and any other
  architecture. `F` is `#[repr(transparent)]` so `&[F]` can be viewed as `&[u64]`
  for SIMD loads.

The reduction is identical across all three backends, so the exhaustively tested
x86 path validates the NEON math; only intrinsics differ.

The per-field hook validates its shape on entry (in release too):
`dense_ntt_lg_rows` asserts `rows` is a power of two and `data.len() == rows*cols`
(via `checked_mul`). This matters because the SIMD kernels are `unsafe` and index
the buffer by raw pointer up to `rows*cols - 1`, so the length invariant is a
memory-safety precondition, not just a debug aid.

The checksum matrix multiply (`Matrix::mul_with`) is parallelized the same way:
output rows are independent, so they are split into one contiguous block per
worker (with an empty-matrix guard), each block computed into a single buffer.

## 2. Correctness

- **Packed field ops == scalar `F`, exhaustively** (edges + pseudo-random, all
  lanes), for `add`/`sub`/`mul`/`div2`. AVX2 here; NEON on Apple Silicon.
- **Dense NTT == scalar** (the dispatched SIMD path vs `ntt_dense_scalar`) for both
  directions, `rows = 2^0..2^12`, and every `cols % WIDTH` remainder. Runs on
  x86-64 (AVX2) and aarch64 (NEON). Plus `should_panic` tests for the shape asserts
  (short data, zero rows) and `mul_with == mul`.
- **Cross-machine determinism.** Because AVX2, NEON, and scalar are all bit-identical
  to `F`'s scalar arithmetic, a participant on a non-AVX2 / non-SIMD machine computes
  the *same* ZODA commitment and shards as one on an AVX2 box. There is no consensus
  hazard from mixed hardware.
- **Combined path == sequential, byte for byte.** ZODA encoded under `Sequential`
  (SIMD, no threads) and `Rayon(t)` (threads x SIMD) produce identical commitments
  and strong shards, and round-trip correctly, across sizes `{64 KiB, 256 KiB,
  1 MiB}` x threads `{2, 8, 13}` (13 = uneven blocks).
- `commonware-math` (29) and `commonware-coding` (34, incl. ZODA round-trips /
  minifuzz) pass. `clippy -D warnings` + nightly `fmt` clean on x86-64 **and**
  aarch64, in both default and `--no-default-features` (no_std) builds.

## 3. Benchmarks (128-core x86-64, AVX2, release, 8 MiB block, chunks=100)

`cargo bench -p commonware-coding --bench phased_coding_scheme_times`.
Baseline = `e68fede` (scalar, serial). `conc=1` = `Sequential`, `conc=8` =
`Rayon(8)`. Same machine / toolchain.

| op | path | baseline | this branch | speedup |
|----|------|---------:|------------:|--------:|
| encode | conc=1 (SIMD only)      | 569.8 ms | 286.1 ms | 2.0x |
| encode | conc=8 (threads x SIMD) | 551.2 ms | 127.9 ms | **4.3x** |
| decode | conc=1 (SIMD only)      | 2.165 s  | 643.1 ms | 3.4x |
| decode | conc=8 (threads x SIMD) | 2.145 s  | 569.8 ms | **3.8x** |

Forward NTT kernel alone (`cargo bench -p commonware-math --bench ntt`, 2^15 rows)
is ~4.0x on AVX2 for cols >= 16 (cols=192: 347 ms -> 86 ms).

How the two layers contribute (8 MiB, vs baseline):

| | encode c1 | encode c8 | decode c1 | decode c8 |
|--|--:|--:|--:|--:|
| threads only (`zoda-parallel-ntt`) | 1.0x | 3.2x | 1.0x | 2.7x |
| SIMD only (`goldilocks-simd-ntt`)  | 2.0x | -    | 3.4x | -    |
| **combined (this branch)**         | 2.0x | **4.3x** | 3.4x | **3.8x** |

Reading the table: encode threads and SIMD stack cleanly (3.2x x ~1.3x more from
SIMD). For decode, SIMD already captures most of the available gain (3.4x at
conc=1); threading adds less on top because, with the NTTs now ~4x faster, decode's
serial remainder (vanishing-polynomial build, the single-column `q` NTT, pointwise
divide) is a larger fraction (Amdahl). Both conc=8 numbers are the fastest overall.

## 4. Notes / future work

- **AVX-512 / IFMA** (8 lanes) would roughly double the x86 kernel again.
- Decode's serial remainder is now the next bottleneck for that path; vectorizing
  the single-column vanishing/`q` NTTs or the pointwise divide would help.
- Intrinsics are confined to `simd.rs`, `cfg(target_arch)` + runtime-gated, with
  documented `// SAFETY:` blocks and the scalar field as the equivalence oracle.

## 5. Reproduce

```bash
cargo test  -p commonware-math  --lib            # field + dense NTT equivalence
cargo test  -p commonware-coding --lib           # ZODA round-trips (threads x SIMD)
cargo bench -p commonware-math  --bench ntt
cargo bench -p commonware-coding --bench phased_coding_scheme_times -- \
  'zoda_phased::(encode|decode)/msg_len=8388608 chunks=100'
# aarch64 (NEON) on Apple Silicon:
cargo test  -p commonware-math --lib goldilocks::simd
```

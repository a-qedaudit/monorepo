# SIMD-vectorized Goldilocks NTT (AVX2 + NEON)

This branch hardware-accelerates the Goldilocks dense NTT, the kernel under ZODA
encode/decode and BLS-threshold recovery. It is a `math`-crate-only change: every
caller benefits transparently, with a scalar fallback and bit-identical output.

Branch on `a-qedaudit/monorepo` for review (this writeup is not meant to merge).

---

## 1. Motivation

`math`'s Goldilocks field (`fields/goldilocks`) and NTT (`ntt.rs`) are scalar
`u64`/`u128`. The NTT dominates ZODA encode (one forward NTT) and decode (~three
NTTs via `recover`), and also BLS-threshold recovery. The field's modulus
`P = 2^64 - 2^32 + 1` has a famously SIMD-friendly reduction, and the butterfly
inner loop already walks contiguous columns with a shared twiddle, the perfect
vectorization axis. No prior SIMD work existed for either the field or the NTT.

## 2. What changed

All in `math`:

- **`FieldNTT::ntt_dense`** (new trait method, `algebra.rs`): in-place dense NTT
  over a row-major `rows x cols` slice. The default is the existing scalar
  butterfly (`ntt::ntt_dense_scalar`), a faithful refactor of the previous generic
  `Matrix` NTT. `Matrix::ntt` and `PolynomialVector::divide` route through it.
  Single-column / partial-segment NTTs keep using the generic `ntt`.
- **`fields::goldilocks::F` is `#[repr(transparent)]`** so `&[F]` can be viewed as
  `&[u64]` for SIMD loads/stores.
- **`fields::goldilocks::simd`** (new): overrides `ntt_dense` for Goldilocks with a
  vectorized kernel. A packed field (`add`/`sub`/`mul`/`div2`) mirrors `F`'s
  `add_inner`/`sub_inner`/`reduce_128`/`div_2` operation-for-operation, so the
  vector result equals the scalar field bit-for-bit. The butterfly processes
  `WIDTH` contiguous columns per instruction and handles the `cols % WIDTH`
  remainder with the scalar field. Backends:
  - **AVX2** (x86-64): 4 `u64` lanes, runtime-detected (`is_x86_feature_detected`),
    scalar fallback otherwise. `64x64 -> 128` via `_mm256_mul_epu32` partials;
    unsigned compares via the sign-flip trick; selects via `blendv`.
  - **NEON** (aarch64): 2 `u64` lanes, part of the aarch64 baseline so no runtime
    detection. `64x64 -> 128` via `vmull_u32`; native unsigned compares
    (`vcltq_u64`/`vcgeq_u64`) and `vbslq_u64` select.
  - **scalar**: any other target, or x86 without AVX2, or `no_std` x86.

The reduction algorithm is identical across all three backends, so the
exhaustively tested x86 path validates the NEON path's math; only the intrinsics
differ.

## 3. Correctness

- **Field ops, exhaustive vs scalar.** For ~213 canonical samples (incl. edges:
  `0, 1, P-1, P-2, EPSILON, 2^32, 2^63, ...`) crossed pairwise over all lanes, the
  packed `add`/`sub`/`mul`/`div2` equal `F`'s scalar ops bit-for-bit. Run on AVX2
  here; the same tests run on aarch64 hosts (your MacBook) for NEON.
- **Dense NTT vs scalar.** `ntt_dense` (dispatched to AVX2 here) equals
  `ntt_dense_scalar` for both directions, `rows = 2^0..2^12`, and
  `cols in {1,2,3,4,5,7,8,9,16,17,33,64}` (every `cols % WIDTH` remainder).
- **Existing suites.** `commonware-math` (26) and `commonware-coding` (34, incl.
  ZODA round-trips / minifuzz) pass with the AVX2 NTT live.
- `clippy -D warnings` and nightly `fmt` clean on **both** x86-64 and aarch64;
  aarch64 builds and tests compile-check (`--target aarch64-unknown-linux-gnu`).

## 4. Benchmarks (128-core x86-64, AVX2, release)

### Forward NTT kernel (`math`, `2^15` rows)

`cargo bench -p commonware-math --bench ntt`; scalar = `e68fede`, AVX2 = this branch.

| cols | scalar | AVX2 | speedup |
|-----:|-------:|-----:|--------:|
| 1   | 3.74 ms  | 3.80 ms  | ~1.0x (single column; scalar tail) |
| 4   | 9.36 ms  | 3.31 ms  | 2.8x |
| 16  | 31.36 ms | 7.76 ms  | 4.0x |
| 64  | 117.72 ms| 29.56 ms | 4.0x |
| 192 | 347.21 ms| 86.22 ms | **4.0x** |

4 `u64` lanes give the expected ~4x once columns fill a vector. NEON (2 lanes)
should give ~2x.

### End-to-end ZODA, sequential (`coding`, 8 MiB block, chunks=100)

`cargo bench -p commonware-coding --bench phased_coding_scheme_times`, `conc=1`
(the coding crate is unmodified; SIMD flows in via `ntt_dense`).

| op | scalar (`e68fede`) | AVX2 (this branch) | speedup |
|----|-------------------:|-------------------:|--------:|
| encode | 569.8 ms | 288.7 ms | 1.97x |
| decode | 2.165 s  | 0.660 s  | **3.28x** |

Decode gains more because it is dominated by NTTs (`recover` runs ~three); encode
has more non-NTT work (Merkle commit, Fiat-Shamir, checksum).

## 5. Notes and future work

- **Composes with thread-parallelism.** This is orthogonal to a parallel NTT: SIMD
  speeds each lane's arithmetic (and the serial path), threading spreads columns
  across cores. Stacking both multiplies the gains.
- **AVX-512 / IFMA.** An 8-lane AVX-512 backend (or `madd52` IFMA) would roughly
  double the x86 win again; left as a follow-up.
- **Verify-side checksum encode** in `CheckingData::reckon` already benefits (it
  calls `evaluate`).
- **Safety.** Intrinsics are `unsafe` but confined to `simd.rs`, gated by
  `cfg(target_arch)` + runtime detection, with documented `// SAFETY:` blocks and
  the scalar fallback as the equivalence oracle. `F: repr(transparent)` makes the
  `&[F] -> &[u64]` reinterpret sound.

## 6. Reproduce

```bash
cargo test  -p commonware-math  --lib            # field + dense NTT equivalence
cargo test  -p commonware-coding --lib           # ZODA round-trips on AVX2 NTT
cargo bench -p commonware-math  --bench ntt      # forward NTT kernel
cargo bench -p commonware-coding --bench phased_coding_scheme_times -- \
  'zoda_phased::(encode|decode)/msg_len=8388608 chunks=100 conc=1'
# aarch64 (NEON) on an Apple Silicon Mac:
cargo test  -p commonware-math --lib goldilocks::simd
```

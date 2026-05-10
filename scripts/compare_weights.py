"""Compare two weights ``.npz`` files (per-key + global max abs diff).

Useful for checking that two training runs (e.g. single-host vs multi-host)
produce identical weights up to numerical noise.

Usage::

    python scripts/compare_weights.py weights_a.npz weights_b.npz
"""

import argparse
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_a", help="First weights .npz")
    parser.add_argument("weights_b", help="Second weights .npz")
    parser.add_argument(
        "--top", type=int, default=10, help="Show top-N per-key diffs"
    )
    parser.add_argument(
        "--atol-strict",
        type=float,
        default=4.69e-2,
        help="Strict atol",
    )
    parser.add_argument(
        "--atol-lenient", type=float, default=1e-1, help="Lenient backstop atol"
    )
    parser.add_argument(
        "--atol-catastrophic", type=float, default=5e-1, help="Catastrophic threshold"
    )
    args = parser.parse_args()

    a = np.load(args.weights_a)
    b = np.load(args.weights_b)

    keys_a = set(a.files)
    keys_b = set(b.files)
    if keys_a != keys_b:
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        print(f"[error] Key mismatch: {len(only_a)} only in A, {len(only_b)} only in B", file=sys.stderr)
        for k in sorted(only_a)[:5]:
            print(f"  only_a: {k}", file=sys.stderr)
        for k in sorted(only_b)[:5]:
            print(f"  only_b: {k}", file=sys.stderr)
        return 1

    diffs = []
    bit_exact_count = 0
    for k in sorted(keys_a):
        va = a[k].astype(np.float64)
        vb = b[k].astype(np.float64)
        if va.shape != vb.shape:
            print(f"[error] Shape mismatch for {k}: {va.shape} vs {vb.shape}", file=sys.stderr)
            return 1
        diff = np.max(np.abs(va - vb))
        diffs.append((k, float(diff), va.shape, va.dtype))
        if diff == 0.0:
            bit_exact_count += 1

    diffs.sort(key=lambda x: -x[1])

    print(f"=== Comparing {args.weights_a} vs {args.weights_b} ===")
    print(f"Total keys: {len(diffs)}")
    print(f"Bit-exact (diff == 0.0): {bit_exact_count}/{len(diffs)} ({100*bit_exact_count/len(diffs):.1f}%)")
    print(f"\nTop {args.top} largest diffs:")
    for k, d, shape, dtype in diffs[:args.top]:
        print(f"  {k:60s} diff={d:.6e} shape={shape} dtype={dtype}")

    print("\nBottom 5 smallest non-zero diffs:")
    nonzero = [d for d in diffs if d[1] > 0]
    for k, d, shape, dtype in nonzero[-5:]:
        print(f"  {k:60s} diff={d:.6e} shape={shape} dtype={dtype}")

    all_diffs = np.array([d[1] for d in diffs])
    global_max = float(all_diffs.max())
    global_mean = float(all_diffs.mean())
    global_p99 = float(np.percentile(all_diffs, 99))
    nonzero_diffs = all_diffs[all_diffs > 0]
    nz_min = float(nonzero_diffs.min()) if len(nonzero_diffs) > 0 else 0.0

    print("\n=== Summary ===")
    print(f"Global max abs diff: {global_max:.6e}")
    print(f"Global mean abs diff: {global_mean:.6e}")
    print(f"P99 abs diff: {global_p99:.6e}")
    print(f"Non-zero min: {nz_min:.6e}")
    print(f"Bit-exact ratio: {bit_exact_count}/{len(diffs)} = {100*bit_exact_count/len(diffs):.1f}%")

    print("\n=== Verdict ===")
    if global_max < args.atol_strict:
        print(f"PASS strict (< {args.atol_strict:.2e})")
        verdict = 0
    elif global_max < args.atol_lenient:
        print(f"PASS lenient ({args.atol_strict:.2e} < diff < {args.atol_lenient:.2e})")
        verdict = 0
    elif global_max < args.atol_catastrophic:
        print(f"WARN above lenient ({args.atol_lenient:.2e} < diff < {args.atol_catastrophic:.2e})")
        verdict = 1
    else:
        print(f"FAIL catastrophic (>= {args.atol_catastrophic:.2e})")
        verdict = 2

    return verdict


if __name__ == "__main__":
    sys.exit(main())

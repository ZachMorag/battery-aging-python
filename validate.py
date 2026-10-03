import json
import sys

# --- acceptance criteria: decided BEFORE running ---
MIN_OUTLIER_RECALL = 0.80
MAX_FALSE_POSITIVE_RATE = 0.05
MAX_EOL_ERROR_CYCLES = 15


def check(name, value, limit, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<30} "
          f"measured={value:<10} limit={limit}")
    return ok


def main():
    truth = json.load(open("truth.json", encoding="utf-8"))
    res = json.load(open("results.json", encoding="utf-8"))

    true_out = set(truth["outlier_cycles_true"])
    flagged = set(res["outlier_cycles_flagged"])
    n_points = truth["n_cycles"]

    hits = len(true_out & flagged)
    recall = hits / len(true_out) if true_out else 0.0
    fp_rate = len(flagged - true_out) / (n_points - len(true_out))

    eol_true = truth["eol_cycle_true"]
    eol_est = res["eol_cycle_estimated"]
    eol_err = abs(eol_est - eol_true) if (eol_true and eol_est) else None

    print("\nVALIDATION - analysis output vs known ground truth")
    print(f"  injected: {len(true_out)}  flagged: {len(flagged)}  correct: {hits}")
    print(f"  true EOL: {eol_true}  estimated: {eol_est}\n")

    results = [
        check("outlier recall", f"{recall:.2f}",
              f">= {MIN_OUTLIER_RECALL}", recall >= MIN_OUTLIER_RECALL),
        check("false positive rate", f"{fp_rate:.3f}",
              f"<= {MAX_FALSE_POSITIVE_RATE}", fp_rate <= MAX_FALSE_POSITIVE_RATE),
        check("EOL error (cycles)", eol_err,
              f"<= {MAX_EOL_ERROR_CYCLES}",
              eol_err is not None and eol_err <= MAX_EOL_ERROR_CYCLES),
    ]

    if all(results):
        print("\nRESULT: PASS - all acceptance criteria met.")
        sys.exit(0)
    print("\nRESULT: FAIL - at least one criterion not met.")
    sys.exit(1)


if __name__ == "__main__":
    main()
import json
import numpy as np
import pandas as pd

SEED = 42
N_CYCLES = 500
RATED_CAPACITY_AH = 2.00
FADE_PER_CYCLE = 0.00060
KNEE_CYCLE = 350
EXTRA_FADE_AFTER_KNEE = 0.00090
NOISE_SIGMA_AH = 0.012
N_OUTLIERS = 18
OUTLIER_SIZE_AH = 0.18
EOL_THRESHOLD_PCT = 80.0


def true_capacity(cycle):
    """The real capacity curve, before any measurement noise."""
    cap = RATED_CAPACITY_AH - FADE_PER_CYCLE * cycle
    after_knee = np.clip(cycle - KNEE_CYCLE, 0, None)
    return cap - EXTRA_FADE_AFTER_KNEE * after_knee


def main():
    rng = np.random.default_rng(SEED)

    cycle = np.arange(1, N_CYCLES + 1)
    cap_true = true_capacity(cycle)
    cap_meas = cap_true + rng.normal(0.0, NOISE_SIGMA_AH, size=N_CYCLES)

    outlier_idx = rng.choice(np.arange(10, N_CYCLES - 10),
                             size=N_OUTLIERS, replace=False)
    outlier_idx = np.sort(outlier_idx)
    signs = rng.choice([-1.0, 1.0], size=N_OUTLIERS)
    cap_meas[outlier_idx] += signs * OUTLIER_SIZE_AH

    temperature_c = 25.0 + rng.normal(0.0, 0.8, size=N_CYCLES)

    df = pd.DataFrame({
        "cycle": cycle,
        "capacity_ah": np.round(cap_meas, 5),
        "temperature_c": np.round(temperature_c, 2),
    })
    df.to_csv("data.csv", index=False)

    soh_true = 100.0 * cap_true / RATED_CAPACITY_AH
    below = np.where(soh_true < EOL_THRESHOLD_PCT)[0]
    eol_true = int(cycle[below[0]]) if below.size else None

    truth = {
        "rated_capacity_ah": RATED_CAPACITY_AH,
        "eol_threshold_pct": EOL_THRESHOLD_PCT,
        "eol_cycle_true": eol_true,
        "outlier_cycles_true": [int(c) for c in cycle[outlier_idx]],
        "noise_sigma_ah": NOISE_SIGMA_AH,
        "n_cycles": int(N_CYCLES),
    }
    with open("truth.json", "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)

    print(f"wrote data.csv ({len(df)} rows)")
    print(f"wrote truth.json (true EOL cycle = {eol_true})")


if __name__ == "__main__":
    main()
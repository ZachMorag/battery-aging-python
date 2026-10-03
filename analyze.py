import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- parameters: match these to your MATLAB script ---
RATED_CAPACITY_AH = 2.00
IQR_WINDOW = 31
IQR_K = 1.5
SAVGOL_WINDOW = 21
SAVGOL_POLYORDER = 2
EOL_THRESHOLD_PCT = 80.0


def flag_outliers_rolling_iqr(y, window, k):
    """Tukey IQR test applied inside a moving window."""
    q1 = y.rolling(window, center=True, min_periods=5).quantile(0.25)
    q3 = y.rolling(window, center=True, min_periods=5).quantile(0.75)
    iqr = q3 - q1
    return (y < q1 - k * iqr) | (y > q3 + k * iqr)


def main():
    df = pd.read_csv("data.csv")
    print(f"loaded data.csv: {len(df)} rows")

    df["is_outlier"] = flag_outliers_rolling_iqr(df["capacity_ah"],
                                                 IQR_WINDOW, IQR_K)
    n_out = int(df["is_outlier"].sum())
    print(f"step 1  IQR: flagged {n_out} outliers")
    clean = df.loc[~df["is_outlier"], ["cycle", "capacity_ah"]].copy()

    full = pd.DataFrame({"cycle": df["cycle"]})
    full = full.merge(clean, on="cycle", how="left")
    full["capacity_ah"] = full["capacity_ah"].interpolate(limit_direction="both")

    window = SAVGOL_WINDOW
    if window % 2 == 0:
        window -= 1
    full["capacity_smooth_ah"] = savgol_filter(
        full["capacity_ah"].to_numpy(),
        window_length=window,
        polyorder=SAVGOL_POLYORDER,
    )
    print(f"step 2  Savitzky-Golay: window={window}, polyorder={SAVGOL_POLYORDER}")

    full["soh_pct"] = 100.0 * full["capacity_smooth_ah"] / RATED_CAPACITY_AH

    below = full.index[full["soh_pct"] < EOL_THRESHOLD_PCT]
    eol_cycle = int(full.loc[below[0], "cycle"]) if len(below) else None
    soh_final = float(full["soh_pct"].iloc[-1])
    print(f"step 3  SOH final = {soh_final:.1f}%  |  EOL at cycle {eol_cycle}")

    out = full[["cycle", "capacity_ah", "capacity_smooth_ah", "soh_pct"]].copy()
    out["is_outlier"] = df["is_outlier"].to_numpy()
    out.to_csv("results.csv", index=False)

    results = {
        "rated_capacity_ah": RATED_CAPACITY_AH,
        "eol_threshold_pct": EOL_THRESHOLD_PCT,
        "eol_cycle_estimated": eol_cycle,
        "soh_final_pct": round(soh_final, 3),
        "outlier_cycles_flagged": [int(c) for c in df.loc[df["is_outlier"], "cycle"]],
        "n_outliers_flagged": n_out,
        "params": {
            "iqr_window": IQR_WINDOW, "iqr_k": IQR_K,
            "savgol_window": int(window), "savgol_polyorder": SAVGOL_POLYORDER,
        },
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df["cycle"], df["capacity_ah"], s=8, alpha=0.35,
               label="measured", color="#8899a6")
    ax.scatter(df.loc[df["is_outlier"], "cycle"],
               df.loc[df["is_outlier"], "capacity_ah"],
               s=34, marker="x", label="flagged outlier", color="#993a32")
    ax.plot(full["cycle"], full["capacity_smooth_ah"], lw=2.2,
            label="smoothed", color="#185c7a")

    eol_cap = RATED_CAPACITY_AH * EOL_THRESHOLD_PCT / 100.0
    ax.axhline(eol_cap, ls="--", lw=1.2, color="#8c5a12",
               label=f"EOL = {EOL_THRESHOLD_PCT:.0f}% SOH")
    if eol_cycle:
        ax.axvline(eol_cycle, ls=":", lw=1.2, color="#8c5a12")
        ax.annotate(f"EOL @ cycle {eol_cycle}", xy=(eol_cycle, eol_cap),
                    xytext=(8, 14), textcoords="offset points", fontsize=9)

    ax.set_xlabel("cycle")
    ax.set_ylabel("capacity (Ah)")
    ax.set_title("Battery capacity fade - outliers removed, Savitzky-Golay smoothed")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("soh_curve.png", dpi=150)
    print("wrote results.csv, results.json, soh_curve.png")
if __name__ == "__main__":
    main()
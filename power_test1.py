"""Power of primary test 1 with the real L4 uncertainty series (2015-2024) and simulated errors.

No forecast error is used: z is simulated as |N(0,1)| plus beta * (u - mean u), so the result
shows which effect sizes the design can detect, before the link is tested.
"""
import sys
import numpy as np
import pandas as pd
import step6_link_analysis as S

unc = pd.read_csv("results/step1c/station_monthly_l4_uncertainty.csv", parse_dates=["date"])
u = unc[unc["date"].dt.year.between(2015, 2024)].copy()
rng = np.random.default_rng(11)
rows = []
n_sim, n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 100, 199
for eff in [0.0, 0.05, 0.10, 0.20, 0.30]:          # change in z per 0.1 K
    rej = 0
    for _ in range(n_sim):
        d = u[["station", "date", "unc_mean_k"]].copy()
        d["z"] = np.abs(rng.normal(0, 1, len(d))) + eff * 10 * (d["unc_mean_k"] - d["unc_mean_k"].mean())
        d["year"], d["month"] = d["date"].dt.year, d["date"].dt.month
        r = S.freedman_lane(d, "unc_mean_k", n_perm, rng)
        rej += r["p_perm"] < 0.05
    rows.append(dict(effect_z_per_0p1K=eff, power=rej / n_sim, n_sim=n_sim))
    print(rows[-1], flush=True)
x = u.copy()
x["year"], x["month"] = x["date"].dt.year, x["date"].dt.month
within = x["unc_mean_k"] - x.groupby(["station", "month"])["unc_mean_k"].transform("mean")
print("SD of u within station x calendar month (K):", round(float(within.std()), 4))
pd.DataFrame(rows).to_csv("results/power_test1.csv", index=False)

"""Exploratory check of forecast skill by monsoon phase (not in analysis plan v1.0; run 13 Oct 2026).

Question: SARIMA reached MSSS >= 0.10 against climatology at every station in the monsoon. Does this skill come from
forecasting month-to-month anomalies, or from a smaller mean bias than the training climatology?
For each phase (and each station within the monsoon), the mean squared error of SARIMA and climatology is split into
the squared mean bias and the remainder, and the skill score is recomputed with the mean bias of each method removed.
Deterministic; no random numbers.

  python step6f_exploratory_skill_bias.py --out results/step6/exploratory
"""
import argparse
import os

import numpy as np
import pandas as pd

PHASES = ["pre-monsoon", "monsoon", "post-monsoon", "winter"]


def summary(x):
    out = {"n": len(x)}
    for m in ("sarima", "climatology"):
        e = x["pred_" + m] - x["obs"]
        out[f"bias_{m}"] = e.mean()
        out[f"mse_{m}"] = (e ** 2).mean()
        out[f"mse_debiased_{m}"] = ((e - e.mean()) ** 2).mean()
    out["msss"] = 1 - out["mse_sarima"] / out["mse_climatology"]
    out["msss_debiased"] = 1 - out["mse_debiased_sarima"] / out["mse_debiased_climatology"]
    out["bias_share_of_mse_climatology"] = out["bias_climatology"] ** 2 / out["mse_climatology"]
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hindcast", default="results/step2/hindcast_station_month_lstm5seeds.csv")
    ap.add_argument("--out", default="results/step6/exploratory")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = pd.read_csv(a.hindcast)
    by_phase = d.groupby("phase").apply(summary).reindex(PHASES)
    by_phase.loc["all"] = summary(d)
    by_phase.to_csv(os.path.join(a.out, "skill_bias_by_phase.csv"))
    mon = d[d.phase == "monsoon"].groupby("station").apply(summary)
    mon.to_csv(os.path.join(a.out, "skill_bias_monsoon_by_station.csv"))
    print(by_phase.round(3).to_string())
    print(mon[["bias_climatology", "bias_sarima", "msss", "msss_debiased"]].round(2).to_string())


if __name__ == "__main__":
    main()

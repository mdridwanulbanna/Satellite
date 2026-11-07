"""Exploratory (not in the analysis plan): does test 1 depend on how daily analysis uncertainty is turned into
an uncertainty for the monthly mean SST?

The forecast target is the monthly mean of daily L4 SST. If the daily analysis errors e_d have standard
uncertainties u_d and correlation rho_|i-j| between days, the uncertainty of the monthly mean is

    u_month = (1/N) * sqrt( sum_i sum_j u_i u_j rho_|i-j| ).

It equals the mean of u_d only when the daily errors are fully correlated, and u_rms / sqrt(N) when they are
independent. The C3S product does not publish the time correlation of its errors, so this script brackets it:

  u_mean    mean of daily u (the plan's definition; fully correlated errors, constant u)
  u_rms     root mean square of daily u (fully correlated errors; family A of the plan)
  u_indep   u_rms / sqrt(N)                                  (independent daily errors)
  u_tau3, u_tau7, u_tau15: u_rms * sqrt(F(N, rho) / N^2), rho_k = exp(-k / tau), tau in days,
            F = sum_i sum_j rho_|i-j| (first-order autoregressive errors; u assumed constant within the month)
  u_log     log of u_mean (proportional changes)
  u_rank    rank of u_mean within each station and calendar month (0-1), robust to single large values

For each, test 1 is repeated (SARIMA, station x calendar-month fixed effects and trend), with beta per SD of u
left after the fixed effects (scale-free), the plan's Freedman-Lane permutation p (classical t), the same
permutation with the cluster-robust t, and the restricted wild cluster bootstrap p (Webb weights).

  python step6e_exploratory_u_definition.py --out results/step6/exploratory
"""
import argparse
import os

import numpy as np
import pandas as pd

import step6_link_analysis as L
from step6d_exploratory_test1_inference import setup, stats_of, permutation_tests, wcr

SEED = 20261013


def corr_factor(n, tau):
    k = np.arange(n)
    rho = np.exp(-np.abs(k[:, None] - k[None, :]) / tau)
    return rho.sum() / n ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hindcast", default="results/step2/hindcast_station_month_lstm5seeds.csv")
    ap.add_argument("--uncertainty", default="results/step1c/station_monthly_l4_uncertainty.csv")
    ap.add_argument("--n-perm", type=int, default=9999)
    ap.add_argument("--out", default="results/step6/exploratory")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    unc = pd.read_csv(a.uncertainty)
    n = unc["n_days"].astype(float)
    unc["u_indep"] = unc["unc_rms_k"] / np.sqrt(n)
    for tau in (3, 7, 15):
        f = n.map(lambda v: corr_factor(int(v), tau))
        unc[f"u_tau{tau}"] = unc["unc_rms_k"] * np.sqrt(f)
    unc["u_log"] = np.log(unc["unc_mean_k"])
    d = L.build_station_months(pd.read_csv(a.hindcast), unc, "sarima")
    extra = unc[["station", "date", "u_indep", "u_tau3", "u_tau7", "u_tau15", "u_log"]].copy()
    extra["date"] = pd.to_datetime(extra["date"])
    d = d.merge(extra, on=["station", "date"], how="left")
    d["u_rank"] = d.groupby(["station", "month"])["unc_mean_k"].rank(pct=True)

    defs = [("u_mean", "unc_mean_k", "mean of daily u (plan)"), ("u_rms", "unc_rms_k", "RMS of daily u"),
            ("u_tau15", "u_tau15", "AR(1) errors, tau = 15 days"), ("u_tau7", "u_tau7", "AR(1) errors, tau = 7 days"),
            ("u_tau3", "u_tau3", "AR(1) errors, tau = 3 days"), ("u_indep", "u_indep", "independent daily errors"),
            ("u_log", "u_log", "log of mean daily u"), ("u_rank", "u_rank", "rank within station and month")]
    rows = []
    for lab, col, desc in defs:
        S = setup(d, col)
        b, _, _ = stats_of(S, S["zr"])
        sd_within = float(np.std(S["xr"], ddof=S["k"] - 1))
        pt = permutation_tests(S, a.n_perm, np.random.default_rng(SEED))
        p_wcr, _, _ = wcr(S, L.WEBB, np.random.default_rng(SEED), a.n_perm)
        rows.append(dict(definition=lab, description=desc, mean_value=float(d[col].mean()),
                         sd_within=sd_within, beta_per_sd_within=b * sd_within,
                         corr_with_u_mean_within=float(np.corrcoef(S["xr"], setup(d, "unc_mean_k")["xr"])[0, 1]),
                         t_classical=pt["t_classical"], t_cluster=pt["t_cluster"],
                         p_perm_classical=pt["p_perm_classical"], p_perm_cluster_t=pt["p_perm_cluster_t"],
                         p_wcr_webb=p_wcr))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, "test1_u_definitions.csv"), index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

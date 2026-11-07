"""Step 2: rolling-origin one-month-ahead hindcast, 2015-2024, four methods.

Input: station_monthly_sst.csv from Step 1b (date + one column per station, degC).

For each test year Y (default 2015..2024) and each station:
  * training data = Jan 2000 .. Dec (Y-1); nothing after Dec (Y-1) is used for fitting
  * LSTM   : Manuscript 1 architecture unchanged (1 LSTM layer, 64 units, Dense(1),
             Adam 1e-3, MSE, 100 epochs, batch 12, lookback 60, no shuffling,
             Min-Max scaling fitted on the training data only)
  * SARIMA : (1,1,1)(1,1,1)12 fitted by ML on the training data; parameters then held
             fixed while the Kalman filter runs through the observed record, which gives
             the same rolling one-step-ahead forecasts as Manuscript 1
  * climatology : training-period calendar-month mean
  * persistence : observed SST of the previous month
All forecasts for month t use observed SST up to t-1 only.

Output (per station-month): observations, four forecasts, and sd_train_calmonth, the
inter-annual SD of that calendar month in the fold's training years. Step 5 uses it to
standardise errors so that seasonal differences in predictability do not masquerade as
differences in data support.

  python step2_rolling_hindcast.py --input out/step1b/station_monthly_sst.csv --out out/step2
  python step2_rolling_hindcast.py --input ... --replicate-ms1      # check vs MS1 Table 5
  python step2_rolling_hindcast.py --input ... --seeds 42 43 44 45 46   # seed ensemble
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

import config as C

LOOKBACK = 60


REPO_CODES = {"Saint Martin's Island": "SaintMartins", "Teknaf": "Teknaf", "Cox's Bazar": "CoxsBazar",
              "Kutubdia": "Kutubdia", "Kuakata": "Kuakata", "Dublar Char": "DublarChar",
              "Nijhum Dwip": "NijhumDwip"}


def load_repo_daily(data_dir):
    """Monthly means from the sst_<code>.csv daily files of the sst-lstm-bangladesh repository."""
    cols = {}
    for name, code in REPO_CODES.items():
        d = pd.read_csv(os.path.join(data_dir, f"sst_{code}.csv"), parse_dates=["time"])
        daily = d.set_index("time")["SST_C"].sort_index().loc["2000-01-01":"2024-12-31"]
        if len(daily) != 9132 or daily.isna().any():
            raise ValueError(f"{code}: incomplete daily series")
        cols[name] = daily.resample("MS").mean()
    df = pd.DataFrame(cols)
    df.index.name = "date"
    return df[C.STATION_NAMES]


def load_series(path):
    if os.path.isdir(path):
        return load_repo_daily(path)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.to_period("M").to_timestamp()
    missing = [s for s in C.STATION_NAMES if s not in df.columns]
    if missing:
        raise KeyError(f"Missing station columns: {missing}")
    return df[C.STATION_NAMES]


# ---------------------------------------------------------------- LSTM
def lstm_forecasts(series, train_end, test_index, seeds, epochs):
    """series: pd.Series (monthly). Returns array (n_seeds, n_test)."""
    import random
    import tensorflow as tf
    train = series.loc[:train_end].values.astype("float64")
    xmin, xmax = train.min(), train.max()
    scale = lambda v: (v - xmin) / (xmax - xmin)   # same as sklearn MinMaxScaler(0, 1)
    full_scaled = scale(series.values.astype("float64"))
    tr = scale(train)
    X = np.stack([tr[i:i + LOOKBACK] for i in range(len(tr) - LOOKBACK)])[..., None]
    y = tr[LOOKBACK:]
    pos = [series.index.get_loc(t) for t in test_index]
    Xt = np.stack([full_scaled[p - LOOKBACK:p] for p in pos])[..., None]

    out = []
    for seed in seeds:
        tf.keras.backend.clear_session()
        os.environ["PYTHONHASHSEED"] = str(seed)          # same seeding as sst_forecast.models.set_seed
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        model = tf.keras.Sequential([tf.keras.layers.Input((LOOKBACK, 1)),
                                     tf.keras.layers.LSTM(64),
                                     tf.keras.layers.Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        model.fit(X, y, epochs=epochs, batch_size=12, shuffle=False, verbose=0)
        p = model(Xt, training=False).numpy().ravel()  # avoids tf.function retracing
        out.append(p * (xmax - xmin) + xmin)
    return np.array(out)


# ---------------------------------------------------------------- SARIMA
def sarima_forecasts(series, train_end, test_index):
    """Same procedure as sst_forecast.baselines.sarima_one_step, generalised to any fold:
    fit on data up to train_end, append any months between train_end and the first test
    month without re-estimation, then forecast one step and append each observed month."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(series.loc[:train_end].values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        gap = series.loc[train_end:test_index[0]].iloc[1:-1]
        state = fit.append(gap.values, refit=False) if len(gap) else fit
        preds = []
        for obs in series.loc[test_index].values:
            preds.append(state.forecast(1)[0])
            state = state.append([obs], refit=False)
    converged = bool(fit.mle_retvals.get("converged", True)) if fit.mle_retvals else True
    return np.array(preds), converged


# ---------------------------------------------------------------- baselines
def climatology_forecasts(series, train_end, test_index):
    train = series.loc[:train_end]
    clim = train.groupby(train.index.month).mean()
    return np.array([clim[t.month] for t in test_index])


def persistence_forecasts(series, test_index):
    return np.array([series.iloc[series.index.get_loc(t) - 1] for t in test_index])


def sd_train_calmonth(series, train_end, test_index):
    train = series.loc[:train_end]
    sd = train.groupby(train.index.month).std(ddof=1)
    return np.array([sd[t.month] for t in test_index])


def run_fold(series, station, train_end, test_index, seeds, epochs, fold_label):
    lstm = lstm_forecasts(series, train_end, test_index, seeds, epochs)
    sar, conv = sarima_forecasts(series, train_end, test_index)
    df = pd.DataFrame({
        "station": station, "fold": fold_label, "date": test_index,
        "month": test_index.month, "phase": test_index.month.map(C.PHASE_OF_MONTH),
        "obs": series.loc[test_index].values,
        "pred_persistence": persistence_forecasts(series, test_index),
        "pred_climatology": climatology_forecasts(series, train_end, test_index),
        "pred_sarima": sar,
        "pred_lstm": lstm.mean(axis=0),
        "pred_lstm_seed_sd": lstm.std(axis=0, ddof=0),
        "n_seeds": len(seeds),
        "sd_train_calmonth": sd_train_calmonth(series, train_end, test_index),
        "sarima_converged": conv,
        "train_end": train_end,
    })
    return df


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def summarise(res):
    models = ["persistence", "climatology", "sarima", "lstm"]
    rows = []
    for (st, ph), g in list(res.groupby(["station", "phase"])) + [((s, "all"), g) for s, g in res.groupby("station")]:
        r = dict(station=st, phase=ph, n_months=len(g))
        for m in models:
            r[f"rmse_{m}"] = rmse(g["obs"] - g[f"pred_{m}"])
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="out/step2")
    ap.add_argument("--first-year", type=int, default=2015)
    ap.add_argument("--last-year", type=int, default=2024)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--stations", nargs="+", default=None, help="subset, for quick tests")
    ap.add_argument("--replicate-ms1", action="store_true",
                    help="train 2000-2021, test 2023-2024 and compare with MS1 Table 5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load_series(a.input)
    stations = a.stations or C.STATION_NAMES

    if a.replicate_ms1:
        test_index = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
        frames = [run_fold(data[s], s, pd.Timestamp("2021-12-01"), test_index,
                           a.seeds, a.epochs, "ms1") for s in stations]
        res = pd.concat(frames, ignore_index=True)
        res.to_csv(os.path.join(a.out, "ms1_replication_station_month.csv"), index=False)
        s = summarise(res)
        s = s[s["phase"] == "all"].set_index("station")
        rows = []
        for st in stations:
            ref = C.MS1_TABLE5[st]
            rows.append(dict(station=st,
                             persistence=round(s.loc[st, "rmse_persistence"], 3), ms1_persistence=ref[0],
                             climatology=round(s.loc[st, "rmse_climatology"], 3), ms1_climatology=ref[1],
                             sarima=round(s.loc[st, "rmse_sarima"], 3), ms1_sarima=ref[2],
                             lstm=round(s.loc[st, "rmse_lstm"], 3), ms1_lstm=ref[3]))
        chk = pd.DataFrame(rows)
        chk.to_csv(os.path.join(a.out, "ms1_replication_check.csv"), index=False)
        print(chk.to_string(index=False))
        print("\nPersistence and climatology should match exactly; SARIMA to about 0.001;"
              " LSTM may differ slightly across TensorFlow versions and hardware.")
        return

    frames = []
    for year in range(a.first_year, a.last_year + 1):
        train_end = pd.Timestamp(f"{year - 1}-12-01")
        test_index = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
        for s in stations:
            t0 = time.time()
            frames.append(run_fold(data[s], s, train_end, test_index, a.seeds, a.epochs, year))
            print(f"fold {year} | {s:22s} | {time.time() - t0:5.1f} s")
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(a.out, "hindcast_station_month.csv"), index=False)  # checkpoint

    res = pd.concat(frames, ignore_index=True)
    res.to_csv(os.path.join(a.out, "hindcast_station_month.csv"), index=False)
    summ = summarise(res)
    summ.to_csv(os.path.join(a.out, "hindcast_rmse_summary.csv"), index=False)
    print("\nRMSE by station (all months):")
    print(summ[summ["phase"] == "all"].round(3).to_string(index=False))
    print("\nNumber of station-months:", len(res))
    print(f"Outputs written to {a.out}")


if __name__ == "__main__":
    main()

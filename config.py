"""Shared configuration for the NBoB satellite-SST reliability / forecast-skill study.

Station coordinates and grid cells are copied from Manuscript 1, Table 1, so every
script uses exactly the same site-to-grid mapping as the published LSTM work.
"""

import numpy as np

# name, site_lat, site_lon, grid_lat, grid_lon  (Manuscript 1, Table 1)
STATIONS = [
    ("Saint Martin's Island", 20.630, 92.320, 20.625, 92.325),
    ("Teknaf",                20.820, 92.320, 20.825, 92.325),
    ("Cox's Bazar",           21.430, 91.930, 21.425, 91.925),
    ("Kutubdia",              21.830, 91.820, 21.825, 91.825),
    ("Kuakata",               21.780, 90.150, 21.775, 90.175),
    ("Dublar Char",           21.740, 89.500, 21.725, 89.525),
    ("Nijhum Dwip",           22.120, 91.020, 22.125, 91.025),
]
STATION_NAMES = [s[0] for s in STATIONS]

# Monsoon-phase convention used in the proposal (Section 6.6)
PHASE_OF_MONTH = {
    3: "pre-monsoon", 4: "pre-monsoon", 5: "pre-monsoon",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post-monsoon", 11: "post-monsoon",
    12: "winter", 1: "winter", 2: "winter",
}
PHASE_ORDER = ["pre-monsoon", "monsoon", "post-monsoon", "winter"]

# Regional box for the iQuam scoping (Part A). Deliberately larger than the station
# box so that offshore drifters are counted; narrow it after scoping if needed.
SCOPING_BOX = dict(lat_min=18.0, lat_max=23.0, lon_min=86.0, lon_max=93.0)

# Li et al. (JSTARS submission, 2026) thresholds, Table II and Section II.C
LI_MIN_ACCURACY = 50      # matchups per unit for A1-A3
LI_MIN_C1_MONTH = 10      # matchups per month for C1
LI_MIN_C2_SIDE = 50       # matchups per day and per night side for C2
LI_MIN_C3_CLASS = 30      # matchups per shelf / deep class for C3
SHELF_DEPTH_M = 200.0     # shelf < 200 m, deep >= 200 m
SZA_DAY_MAX = 85.0        # day: SZA < 85
SZA_NIGHT_MIN = 95.0      # night: SZA > 95

# Manuscript 1 reference values used for replication checks
MS1_TABLE3 = {  # mean, SD of the 2000-2024 monthly series (degC)
    "Saint Martin's Island": (27.97, 2.14), "Teknaf": (27.98, 2.12),
    "Cox's Bazar": (27.73, 2.27), "Kutubdia": (27.44, 2.55),
    "Kuakata": (27.27, 3.03), "Dublar Char": (27.47, 2.77),
    "Nijhum Dwip": (26.96, 3.08),
}
MS1_TABLE5 = {  # persistence, climatology, SARIMA, LSTM test RMSE 2023-2024 (degC)
    "Saint Martin's Island": (1.673, 0.453, 0.425, 0.565),
    "Teknaf": (1.690, 0.470, 0.445, 0.555),
    "Cox's Bazar": (1.774, 0.535, 0.474, 0.582),
    "Kutubdia": (1.883, 0.561, 0.467, 0.490),
    "Kuakata": (2.184, 0.695, 0.558, 0.532),
    "Dublar Char": (1.942, 0.633, 0.492, 0.534),
    "Nijhum Dwip": (2.173, 0.600, 0.530, 0.468),
}

# C3S / ESA SST CCI Level-4 product used in Manuscripts 1 and 2
L4_DATASET_ID = "C3S-GLO-SST-L4-REP-OBS-SST"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def solar_zenith_deg(times_utc, lat, lon):
    """Solar zenith angle (deg) from the NOAA general solar position equations.

    Accuracy is about 0.5 deg, which is ample for the 85/95 deg day/night split.
    times_utc: numpy datetime64 array (UTC); lat, lon in degrees.
    """
    t = np.asarray(times_utc, dtype="datetime64[s]")
    year_start = t.astype("datetime64[Y]").astype("datetime64[s]")
    secs_in_year = (t - year_start).astype(np.int64)
    doy = secs_in_year // 86400 + 1
    frac_hour = (secs_in_year % 86400) / 3600.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (frac_hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    true_solar_min = frac_hour * 60.0 + eqtime + 4.0 * np.asarray(lon)
    hour_angle = np.radians(true_solar_min / 4.0 - 180.0)
    latr = np.radians(lat)
    cosz = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arccos(np.clip(cosz, -1.0, 1.0)))


def day_night_label(sza):
    sza = np.asarray(sza)
    return np.where(sza < SZA_DAY_MAX, "day", np.where(sza > SZA_NIGHT_MIN, "night", "twilight"))

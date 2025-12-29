import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
def adaptive_savgol(x, polyorder=3, max_window=101):
    n = len(x)
    if n <= polyorder + 2:
        return x

    win = min(max_window, n if n % 2 == 1 else n - 1)

    if win <= polyorder:
        win = polyorder + 2
        if win % 2 == 0:
            win += 1

    return savgol_filter(x, window_length=win, polyorder=polyorder)

# ===============================
# 1) Load real data
# ===============================
data = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")
data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"])

# 🔴 FIX UNIT
data["IRRADIATION"] = data["IRRADIATION"] * 1000  # kW/m² → W/m²

# ===============================
# 2) Daylight filtering
# ===============================
day_data = data[data["IRRADIATION"] > 20].reset_index(drop=True)

# ===============================
# 3) Define samples
# ===============================
N = min(635, len(day_data))

G_real = day_data["IRRADIATION"].values[:N]
T_real = day_data["MODULE_TEMPERATURE"].values[:N]
DATE_TIME = day_data["DATE_TIME"].values[:N]

# ===============================
# 4) Learn trend (SAFE)
# ===============================
G_trend = adaptive_savgol(G_real, polyorder=3, max_window=101)
T_trend = adaptive_savgol(T_real, polyorder=3, max_window=101)


# ===============================
# 5) Cloud fluctuation model
# ===============================
def cloud_fluctuation(N, scale=200, rho=0.95):
    e = np.random.randn(N)
    x = np.zeros(N)
    for i in range(1, N):
        x[i] = rho * x[i-1] + e[i]
    return scale * x

# ===============================
# 6) Generate synthetic irradiance
# ===============================
G_syn = (
    G_trend
    + cloud_fluctuation(N, scale=120)
    + np.random.normal(0, 50, N)
)

G_syn = np.clip(G_syn, 0, None)

# ===============================
# 7) Generate synthetic module temperature
# ===============================
k = 0.015  # °C per W/m²

T_syn = (
    T_trend
    + k * (G_syn - G_trend)
    + np.random.normal(0, 0.3, N)
)

# ===============================
# 8) Physical ramp-rate constraints (15 min)
# ===============================
dG_max = 600    # W/m² per 15 min
dT_max = 2.0    # °C per 15 min

G_syn[1:] = np.clip(
    G_syn[1:],
    G_syn[:-1] - dG_max,
    G_syn[:-1] + dG_max
)

T_syn[1:] = np.clip(
    T_syn[1:],
    T_syn[:-1] - dT_max,
    T_syn[:-1] + dT_max
)

# ===============================
# 9) Save synthetic dataset
# ===============================
syn = pd.DataFrame({
    "DATE_TIME": DATE_TIME,
    "IRRADIATION_SYN": G_syn/1000,
    "MODULE_TEMPERATURE_SYN": T_syn
})

syn.to_csv("synthetic_weather_635samples_daylight.csv", index=False)

# ===============================
# 10) Weather variability metrics
# ===============================
G_fluc = G_syn - G_trend
T_fluc = T_syn - T_trend

std_G = np.std(G_fluc)
std_T = np.std(T_fluc)
VI = np.sum(np.abs(np.diff(G_syn))) / np.sum(G_syn)

dt = 900
ramp_G = np.abs(np.diff(G_syn)) / dt
rr95 = np.percentile(ramp_G, 95)

print("===== Weather Variability Metrics =====")
print(f"STD Irradiance fluctuation (W/m²): {std_G:.2f}")
print(f"STD Temperature fluctuation (°C): {std_T:.2f}")
print(f"Variability Index (VI): {VI:.3f}")
print(f"95th percentile Ramp Rate (W/m²/s): {rr95:.4f}")

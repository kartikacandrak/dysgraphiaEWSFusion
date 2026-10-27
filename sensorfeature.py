import streamlit as st
import numpy as np
import pandas as pd

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def safe_diff(a, t):
    a = np.array(a, dtype=np.float32)
    t = np.array(t, dtype=np.float32)

    if len(a) < 2:
        return np.zeros(1, dtype=np.float32)

    dt = np.diff(t)
    dt[dt <= 0] = 1e-3

    d = np.diff(a) / dt

    d = np.nan_to_num(
        d,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    d = np.clip(d, -1e6, 1e6)

    return d.astype(np.float32)

# =========================================================
# SURFACE FEATURE EXTRACTION (141 FEATURES)
# =========================================================
def extract_surface_features(
    x, y, t, status,
    pressure, altitude, azimuth
):
    # [0] PEN DOWN ONLY
    mask = status == 1

    x = x[mask]
    y = y[mask]
    t = t[mask]

    pressure = pressure[mask]
    altitude = altitude[mask]
    azimuth = azimuth[mask]

    if len(x) < 5:
        return np.zeros(141)

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    dt[dt <= 0] = 1e-3

    # [1] Horizontal velocity of writing (7)
    vx = safe_diff(x, t)

    # [2] Vertical velocity of writing (7)
    vy = safe_diff(y, t)

    # [3] Velocity of writing (7)
    v = np.sqrt(vx**2 + vy**2)

    # [4] Horizontal acceleration of writing (7)
    ax = safe_diff(vx, t[:-1])

    # [5] Vertical acceleration of writing (7)
    ay = safe_diff(vy, t[:-1])

    # [6] Acceleration of writing (7)
    a = np.sqrt(ax**2 + ay**2)

    # [7] Horizontal jerk of writing (7)
    jx = safe_diff(ax, t[:-2])

    # [8] Vertical jerk of writing (7)
    jy = safe_diff(ay, t[:-2])

    # [9] Jerk of writing (7)
    j = np.sqrt(jx**2 + jy**2)

    # [10] Length of segment (7)
    dx = np.diff(x)
    dy = np.diff(y)
    seg_len = np.sqrt(dx**2 + dy**2)

    # [11] Length of segment in horizontal direction (7)
    seg_h = np.abs(dx)

    # [12] Length of segment in vertical direction (7)
    seg_v = np.abs(dy)

    # [13] Duration of segment (7)
    seg_duration = dt

    # [14] Width of segment (7)
    seg_width = np.array([np.max(x) - np.min(x)])

    # [15] Height of segment (7)
    seg_height = np.array([np.max(y) - np.min(y)])

    # [16] Pressure (7)
    pres = pressure

    # [17] Altitude (7)
    alt = altitude

    # [18] Azimuth (7)
    azi = azimuth

    def stats(z):
        z = np.nan_to_num(z)
        return [
            np.mean(z),
            np.median(z),
            np.std(z),
            np.min(z),
            np.max(z),
            np.percentile(z, 5),
            np.percentile(z, 95)
        ]

    feats = []

    table_features = [
        vx, vy, v, ax, ay, a, jx, jy, j,
        seg_len, seg_h, seg_v, seg_duration,
        seg_width, seg_height, pres, alt, azi
    ]

    for f in table_features:
        feats.extend(stats(f))

    # STROKE SEGMENTATION
    stroke_end = np.where(np.diff(status.astype(int)) == -1)[0]

    stroke_y_min = []
    stroke_y_max = []
    stroke_y_mean = []
    stroke_y_median = []

    prev = 0

    for idx in stroke_end:
        sy = y[prev:idx+1]
        if len(sy) > 0:
            stroke_y_min.append(np.min(sy))
            stroke_y_max.append(np.max(sy))
            stroke_y_mean.append(np.mean(sy))
            stroke_y_median.append(np.median(sy))
        prev = idx + 1

    if len(stroke_y_mean) < 2:
        stroke_y_min = [0, 0]
        stroke_y_max = [0, 0]
        stroke_y_mean = [0, 0]
        stroke_y_median = [0, 0]

    # [19] Difference between Y Position of First and Last Stroke (4)
    diff_first_last = [
        stroke_y_min[-1] - stroke_y_min[0],
        stroke_y_median[-1] - stroke_y_median[0],
        stroke_y_mean[-1] - stroke_y_mean[0],
        stroke_y_max[-1] - stroke_y_max[0]
    ]

    feats.extend(diff_first_last)

    # [20] Variance of Y Position of Strokes (4)
    y_variance = [
        np.var(stroke_y_min),
        np.var(stroke_y_max),
        np.var(stroke_y_median),
        np.var(stroke_y_mean)
    ]

    feats.extend(y_variance)

    # [21] Number of pen lifts (1)
    pen_lifts = np.sum((status[:-1] == 1) & (status[1:] == 0))
    feats.append(pen_lifts)

    # [22] Count of velocity change (1)
    velocity_change = np.sum(np.diff(np.sign(np.diff(v))) != 0)
    feats.append(velocity_change)

    # [23] Count of acceleration change (1)
    acceleration_change = np.sum(np.diff(np.sign(np.diff(a))) != 0)
    feats.append(acceleration_change)

    # [24] Duration (1)
    total_duration = t[-1] - t[0]
    feats.append(total_duration)

    # [25] Length (1)
    total_length = np.sum(seg_len)
    feats.append(total_length)

    feats = np.nan_to_num(np.array(feats, dtype=np.float32))

    if len(feats) < 141:
        feats = np.pad(feats, (0, 141 - len(feats)))

    return feats[:141]


# =========================================================
# IN-AIR FEATURE EXTRACTION (119 FEATURES)
# =========================================================
def extract_air_features(x, y, t, status):
    # PEN UP ONLY
    mask = status == 0

    x = x[mask]
    y = y[mask]
    t = t[mask]

    if len(x) < 5:
        return np.zeros(119)

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    dt[dt <= 0] = 1e-3

    # [1] Horizontal velocity
    vx = safe_diff(x, t)

    # [2] Vertical velocity
    vy = safe_diff(y, t)

    # [3] Velocity
    v = np.sqrt(vx**2 + vy**2)

    # [4] Horizontal acceleration
    ax = safe_diff(vx, t[:-1])

    # [5] Vertical acceleration
    ay = safe_diff(vy, t[:-1])

    # [6] Acceleration
    a = np.sqrt(ax**2 + ay**2)

    # [7] Horizontal jerk
    jx = safe_diff(ax, t[:-2])

    # [8] Vertical jerk
    jy = np.diff(ay) / (dt[:-2] + 1e-6)

    # [9] Jerk
    j = np.sqrt(jx**2 + jy**2)

    # [10] Length of segment
    seg_len = np.sqrt(dx**2 + dy**2)

    # [11] Horizontal segment length
    seg_h = np.abs(dx)

    # [12] Vertical segment length
    seg_v = np.abs(dy)

    # [13] Segment duration
    seg_duration = dt

    # [14] Width of segment
    seg_width = np.array([np.max(x) - np.min(x)])

    # [15] Height of segment
    seg_height = np.array([np.max(y) - np.min(y)])

    def stats(z):
        z = np.nan_to_num(z)
        return [
            np.mean(z),
            np.median(z),
            np.std(z),
            np.min(z),
            np.max(z),
            np.percentile(z, 5),
            np.percentile(z, 95)
        ]

    feats = []

    air_features = [
        vx, vy, v, ax, ay, a, jx, jy, j,
        seg_len, seg_h, seg_v, seg_duration,
        seg_width, seg_height
    ]

    for f in air_features:
        feats.extend(stats(f))

    # STROKE FEATURES
    stroke_end = np.where(np.diff(status.astype(int)) == 1)[0]

    stroke_y_min = []
    stroke_y_max = []
    stroke_y_mean = []
    stroke_y_median = []

    prev = 0

    for idx in stroke_end:
        sy = y[prev:idx+1]
        if len(sy) > 0:
            stroke_y_min.append(np.min(sy))
            stroke_y_max.append(np.max(sy))
            stroke_y_mean.append(np.mean(sy))
            stroke_y_median.append(np.median(sy))
        prev = idx + 1

    if len(stroke_y_mean) < 2:
        stroke_y_min = [0, 0]
        stroke_y_max = [0, 0]
        stroke_y_mean = [0, 0]
        stroke_y_median = [0, 0]

    # [19]
    feats.extend([
        stroke_y_min[-1] - stroke_y_min[0],
        stroke_y_median[-1] - stroke_y_median[0],
        stroke_y_mean[-1] - stroke_y_mean[0],
        stroke_y_max[-1] - stroke_y_max[0]
    ])

    # [20]
    feats.extend([
        np.var(stroke_y_min),
        np.var(stroke_y_max),
        np.var(stroke_y_median),
        np.var(stroke_y_mean)
    ])

    # [21]
    pen_lifts = np.sum((status[:-1] == 0) & (status[1:] == 1))
    feats.append(pen_lifts)

    # [22]
    velocity_change = np.sum(np.diff(np.sign(np.diff(v))) != 0)
    feats.append(velocity_change)

    # [23]
    acceleration_change = np.sum(np.diff(np.sign(np.diff(a))) != 0)
    feats.append(acceleration_change)

    # [24]
    total_duration = t[-1] - t[0]
    feats.append(total_duration)

    # [25]
    total_length = np.sum(seg_len)
    feats.append(total_length)

    # EXTRA FEATURE FOR 119
    air_stroke_count = len(stroke_end)
    feats.append(air_stroke_count)

    feats = np.nan_to_num(np.array(feats, dtype=np.float32))

    return feats[:119]


# =========================================================
# STREAMLIT RUN FUNCTION
# =========================================================
def run():
    st.header("⚙️ Sensor Feature Extraction (Surface + In-Air)")
    st.write("Upload the sensor acquisition CSV file to extract **Surface (141 Features)** and **In-Air (119 Features)** features. The system uses column 1 for X, column 2 for Y, column 3 for time, column 4 for pressure, column 5 for altitude , and column 6 for azimuth.**")

    uploaded_csv = st.file_uploader("Upload File Sensor CSV", type=["csv", "txt"])

    if uploaded_csv is not None:
        try:
            # Membaca data CSV (mendukung pemisah spasi/koma)
            data = np.loadtxt(uploaded_csv)
            if data.shape[1] >= 7:
                x = data[:, 0]
                y = data[:, 1]
                t = data[:, 2]
                status = data[:, 3]
                pressure = data[:, 4]
                altitude = data[:, 5]
                azimuth = data[:, 6]

                # Ekstraksi independen
                surface_feats = extract_surface_features(x, y, t, status, pressure, altitude, azimuth)
                air_feats = extract_air_features(x, y, t, status)

                # Penggabungan Surface (141) + Air (119) = 260 Fitur
                combined_feats = np.concatenate([surface_feats, air_feats])

                st.success(f"Ekstraksi Berhasil! Surface ({len(surface_feats)}) + In-Air ({len(air_feats)}) = Total {len(combined_feats)} Fitur.")

                # Tampilkan Preview Dataframe
                df_feat = pd.DataFrame([combined_feats], columns=[f"Feature_{i+1}" for i in range(len(combined_feats))])
                st.dataframe(df_feat)

                # Tombol Download NPY
                npy_bytes = combined_feats.tobytes()
                st.download_button(
                    label="💾 Download Feature (.npy)",
                    data=npy_bytes,
                    file_name="online_sensor_features.npy",
                    mime="application/octet-stream"
                )
            else:
                st.error("Format CSV harus memiliki minimal 7 kolom data!")
        except Exception as e:
            st.error(f"Gagal memproses file CSV: {e}")
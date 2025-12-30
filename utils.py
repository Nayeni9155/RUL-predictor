# utils.py
import numpy as np
import pandas as pd

def read_cmapss(filepath):
    col_names = (["unit", "cycle",
                  "op_setting_1", "op_setting_2", "op_setting_3"] +
                 [f"s{i}" for i in range(1, 22)])
    return pd.read_csv(filepath, sep=r"\s+", header=None, names=col_names)


def preprocess_df_for_model(df, sensor_cols, scaler, cap_rul=125, is_train=False):
    df = df.copy()
    if "RUL" in df.columns:
        df["RUL_capped"] = df["RUL"].clip(upper=cap_rul)
    df[sensor_cols] = scaler.transform(df[sensor_cols].values)
    return df


def build_test_X_for_model(df, sensor_cols, scaler, seq_len, domain_id, num_domains):
    df_proc = preprocess_df_for_model(df, sensor_cols, scaler)
    X_test, units = [], sorted(df_proc["unit"].unique())
    domain_vec = np.zeros(num_domains, dtype=np.float32)
    domain_vec[domain_id] = 1.0

    for u in units:
        tmp = df_proc[df_proc["unit"] == u].sort_values("cycle")
        arr = tmp[sensor_cols].values

        if arr.shape[0] >= seq_len:
            arr = arr[-seq_len:]
        else:
            pad = np.vstack([arr[0]] * (seq_len - len(arr)))
            arr = np.vstack([pad, arr])

        X_test.append(np.hstack([arr, np.tile(domain_vec, (seq_len, 1))]))

    return np.array(X_test, dtype=np.float32), units


def get_alert(rul, warning=30, critical=10):
    if rul <= critical:
        return "CRITICAL"
    elif rul <= warning:
        return "WARNING"
    else:
        return "NORMAL"

"""
Prétraitement des données :
- Ingénierie des features temporelles et de lag
- Normalisation
- Création des séquences pour LSTM
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURES = [
    "pm25", "pm10", "no2", "o3", "co", "so2",
    "temperature", "humidite", "vitesse_vent", "pression",
    "heure_sin", "heure_cos", "mois_sin", "mois_cos",
    "est_weekend",
    "iqa_lag_1h", "iqa_lag_3h", "iqa_lag_6h", "iqa_lag_12h", "iqa_lag_24h", "iqa_lag_48h",
    "pm25_lag_1h", "pm25_lag_6h", "pm25_lag_24h",
    "iqa_moy_3h", "iqa_moy_6h", "iqa_moy_12h", "iqa_moy_24h",
    "pm25_moy_6h", "pm25_moy_24h",
    "iqa_std_24h",
]
CIBLE = "iqa"


def preparer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("horodatage").reset_index(drop=True)

    df["heure"]       = df["horodatage"].dt.hour
    df["jour_sem"]    = df["horodatage"].dt.dayofweek
    df["mois"]        = df["horodatage"].dt.month
    df["est_weekend"] = (df["jour_sem"] >= 5).astype(int)

    # Encodage cyclique
    df["heure_sin"] = np.sin(2 * np.pi * df["heure"] / 24)
    df["heure_cos"] = np.cos(2 * np.pi * df["heure"] / 24)
    df["mois_sin"]  = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"]  = np.cos(2 * np.pi * df["mois"] / 12)

    # Lags
    for lag in [1, 3, 6, 12, 24, 48]:
        df[f"iqa_lag_{lag}h"]  = df["iqa"].shift(lag)
        df[f"pm25_lag_{lag}h"] = df["pm25"].shift(lag)

    # Moyennes glissantes
    for fen in [3, 6, 12, 24]:
        df[f"iqa_moy_{fen}h"]  = df["iqa"].rolling(fen, min_periods=1).mean()
        df[f"pm25_moy_{fen}h"] = df["pm25"].rolling(fen, min_periods=1).mean()

    # Écart-type glissant
    df["iqa_std_24h"] = df["iqa"].rolling(24, min_periods=6).std().fillna(0)

    df.dropna(inplace=True)
    return df


def normaliser(X: np.ndarray, y: np.ndarray):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_norm = scaler_X.fit_transform(X)
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    return X_norm, y_norm, scaler_X, scaler_y


def creer_sequences(X: np.ndarray, y: np.ndarray, longueur: int = 48):
    """Transforme les données en séquences temporelles pour LSTM."""
    Xs, ys = [], []
    for i in range(longueur, len(X)):
        Xs.append(X[i - longueur: i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def split_temporel(X, y, ratio: float = 0.80):
    """Split temporel (sans shuffle pour respecter l'ordre chronologique)."""
    split = int(len(X) * ratio)
    return X[:split], X[split:], y[:split], y[split:]
"""
Modèles de prédiction :
- Gradient Boosting (baseline)
- LSTM 2 couches (modèle principal)
"""

import numpy as np
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ── Métriques communes ──────────────────────────────────────────────────────

def calculer_metriques(y_reel, y_pred) -> dict:
    return {
        "mae":  round(mean_absolute_error(y_reel, y_pred), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_reel, y_pred))), 3),
        "r2":   round(r2_score(y_reel, y_pred), 4),
    }


# ── Gradient Boosting ───────────────────────────────────────────────────────

def entrainer_gradient_boosting(X_train, y_train, X_test, y_test) -> dict:
    print("→ Gradient Boosting...")
    modele = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    metriques = calculer_metriques(y_test, y_pred)
    print(f"  MAE={metriques['mae']}  RMSE={metriques['rmse']}  R²={metriques['r2']}")
    return {"modele": modele, "predictions": y_pred, **metriques}


# ── LSTM ────────────────────────────────────────────────────────────────────

def construire_lstm(n_timesteps: int, n_features: int) -> tf.keras.Model:
    modele = Sequential([
        LSTM(128, return_sequences=True, input_shape=(n_timesteps, n_features)),
        BatchNormalization(),
        Dropout(0.2),

        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dense(1),
    ])
    modele.compile(optimizer=Adam(1e-3), loss="huber", metrics=["mae"])
    return modele


def entrainer_lstm(
    X_train_seq, y_train_seq,
    X_test_seq,  y_test_seq,
    scaler_y: MinMaxScaler,
    epochs: int = 50,
) -> dict:
    print("\n→ LSTM...")
    n_steps, n_features = X_train_seq.shape[1], X_train_seq.shape[2]
    modele = construire_lstm(n_steps, n_features)
    modele.summary()

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=4, verbose=1),
    ]
    historique = modele.fit(
        X_train_seq, y_train_seq,
        validation_split=0.15,
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    # Dénormalisation
    y_pred_n = modele.predict(X_test_seq).flatten()
    y_pred   = scaler_y.inverse_transform(y_pred_n.reshape(-1, 1)).flatten()
    y_reel   = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    metriques = calculer_metriques(y_reel, y_pred)
    print(f"  MAE={metriques['mae']}  RMSE={metriques['rmse']}  R²={metriques['r2']}")

    return {
        "modele": modele,
        "historique": historique,
        "predictions": y_pred,
        "y_reel": y_reel,
        **metriques,
    }

"""
Génération de données horaires simulées sur 2 ans.
Inclut saisonnalité journalière, hebdomadaire et annuelle.
"""

import numpy as np
import pandas as pd


def generer_donnees(n_jours: int = 730, graine: int = 42) -> pd.DataFrame:
    np.random.seed(graine)
    n_heures = n_jours * 24
    index = pd.date_range(start="2024-01-01", periods=n_heures, freq="h")

    h   = index.hour
    doy = index.dayofyear
    dow = index.dayofweek

    # Saisonnalité journalière (pics 7h-9h et 17h-19h)
    pic_matin = np.exp(-0.5 * ((h - 8) / 2) ** 2)
    pic_soir  = np.exp(-0.5 * ((h - 18) / 2) ** 2)
    cycle_jour = 20 * (pic_matin + pic_soir)

    # Saisonnalité annuelle (hiver plus pollué)
    cycle_annuel = 15 * np.cos(2 * np.pi * (doy - 15) / 365)

    # Effet week-end
    effet_semaine = np.where(dow < 5, 10, -5)

    # Variables météo
    temperature  = 12 + 10 * np.sin(2 * np.pi * (doy - 80) / 365) + np.random.normal(0, 3, n_heures)
    humidite     = 65 + 15 * np.sin(2 * np.pi * doy / 365) + np.random.normal(0, 8, n_heures)
    vitesse_vent = np.abs(np.random.normal(12, 6, n_heures))
    pression     = 1013 + np.random.normal(0, 8, n_heures)

    # Polluants
    base_pm25 = 20 + cycle_jour + cycle_annuel + effet_semaine
    pm25 = np.clip(base_pm25 - 0.3 * vitesse_vent + np.random.normal(0, 5, n_heures), 2, 200)
    pm10 = pm25 * 1.5 + np.random.normal(0, 4, n_heures)
    no2  = np.clip(30 + cycle_jour * 1.2 + effet_semaine - 0.4 * vitesse_vent + np.random.normal(0, 8, n_heures), 5, 200)
    o3   = np.clip(50 + 20 * np.sin(2 * np.pi * (doy - 120) / 365) - 0.5 * no2 / 30 * 10 + np.random.normal(0, 10, n_heures), 10, 180)
    co   = np.clip(0.5 + 0.02 * pm25 + np.random.normal(0, 0.1, n_heures), 0.1, 10)
    so2  = np.clip(10 + cycle_annuel * 0.5 + np.random.normal(0, 3, n_heures), 1, 100)

    iqa = np.clip(
        25 + 1.8 * pm25 + 0.3 * no2 + 0.1 * o3 - 0.5 * vitesse_vent + np.random.normal(0, 5, n_heures),
        10, 300
    )

    df = pd.DataFrame({
        "id_station":    1,
        "horodatage":    index,
        "pm25":          pm25.round(1),
        "pm10":          pm10.round(1),
        "no2":           no2.round(1),
        "o3":            o3.round(1),
        "co":            co.round(2),
        "so2":           so2.round(1),
        "temperature":   temperature.round(1),
        "humidite":      np.clip(humidite, 20, 100).round(1),
        "vitesse_vent":  vitesse_vent.round(1),
        "pression":      pression.round(1),
        "iqa":           iqa.round(1),
    })
    return df

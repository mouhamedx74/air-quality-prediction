"""
Visualisation des résultats :
- Prédictions vs réel
- Courbe de perte LSTM
- Nuage de points
- Cycle journalier
- Tableau de bord mensuel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COULEURS = {
    "lstm":       "#1D9E75",
    "regression": "#378ADD",
    "reel":       "#2C2C2A",
    "alerte":     "#E24B4A",
}


def afficher_resultats(res_lstm: dict, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Prédiction de la Qualité de l'Air — Analyse complète", fontsize=14, fontweight="bold")

    # 1. Prédictions vs réel
    ax = axes[0, 0]
    n = 500
    ax.plot(res_lstm["y_reel"][:n],   color=COULEURS["reel"],  lw=1.5, label="IQA réel",    alpha=0.9)
    ax.plot(res_lstm["predictions"][:n], color=COULEURS["lstm"], lw=1.5, label="LSTM prédit", alpha=0.85)
    ax.axhline(100, color=COULEURS["alerte"], lw=1, ls="--", label="Seuil alerte (100)")
    ax.set_title("LSTM — Prédictions vs Valeurs réelles (500 h)")
    ax.set_xlabel("Heure")
    ax.set_ylabel("IQA")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Courbe de perte
    ax = axes[0, 1]
    hist = res_lstm["historique"].history
    ax.plot(hist["loss"],     color=COULEURS["lstm"],   lw=1.8, label="Entraînement")
    ax.plot(hist["val_loss"], color=COULEURS["alerte"], lw=1.8, ls="--", label="Validation")
    ax.set_title("LSTM — Courbe de perte (Huber)")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Perte")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Nuage de points prédit vs réel
    ax = axes[1, 0]
    ax.scatter(res_lstm["y_reel"][:1000], res_lstm["predictions"][:1000],
               alpha=0.25, s=6, color=COULEURS["lstm"], label=f"R²={res_lstm['r2']:.3f}")
    lim = [res_lstm["y_reel"].min() * 0.9, res_lstm["y_reel"].max() * 1.05]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="Ligne parfaite")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("LSTM — Prédit vs Réel")
    ax.set_xlabel("IQA réel")
    ax.set_ylabel("IQA prédit")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Cycle journalier moyen
    ax = axes[1, 1]
    df = df.copy()
    df["heure"] = df["horodatage"].dt.hour
    g = df.groupby("heure")["iqa"].agg(["mean", "std"])
    h, m, s = g.index, g["mean"].values, g["std"].values
    ax.plot(h, m, color=COULEURS["lstm"], lw=2, label="IQA moyen")
    ax.fill_between(h, m - s, m + s, alpha=0.15, color=COULEURS["lstm"])
    ax.axhline(100, color=COULEURS["alerte"], lw=1, ls="--", label="Seuil alerte")
    ax.set_title("Cycle journalier de l'IQA")
    ax.set_xlabel("Heure")
    ax.set_ylabel("IQA")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultats_qualite_air.png", dpi=150, bbox_inches="tight")
    print("✓ Sauvegardé : resultats_qualite_air.png")
    plt.show()


def afficher_tableau_bord(df: pd.DataFrame) -> None:
    dernier_mois = df.tail(30 * 24).copy()
    iqa_quotidien = dernier_mois.groupby(dernier_mois["horodatage"].dt.date)["iqa"].mean()

    couleurs_bar = [
        "#E24B4A" if v > 100 else "#BA7517" if v > 75 else "#1D9E75"
        for v in iqa_quotidien
    ]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(iqa_quotidien.index, iqa_quotidien.values, color=couleurs_bar, edgecolor="white", linewidth=0.5)
    ax.axhline(50,  color="#1D9E75", lw=1, ls=":", alpha=0.7, label="Bon (50)")
    ax.axhline(100, color="#E24B4A", lw=1, ls="--", alpha=0.9, label="Mauvais (100)")
    ax.set_title("IQA quotidien moyen — 30 derniers jours", fontsize=12, fontweight="bold")
    ax.set_ylabel("Indice de Qualité de l'Air")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig("tableau_bord_iqa.png", dpi=150, bbox_inches="tight")
    print("✓ Sauvegardé : tableau_bord_iqa.png")
    plt.show()
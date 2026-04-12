"""
Point d'entrée du projet — Prédiction de la Qualité de l'Air
Exécuter : python main.py
"""

from src.database       import creer_base_de_donnees, inserer_stations, requetes_analytiques
from src.data_generator import generer_donnees
from src.preprocessing  import preparer_features, normaliser, creer_sequences, split_temporel, FEATURES, CIBLE
from src.models         import entrainer_gradient_boosting, entrainer_lstm
from src.visualization  import afficher_resultats, afficher_tableau_bord

LONGUEUR_SEQ = 48  # heures de contexte pour le LSTM


def main():
    print("=" * 55)
    print("  PRÉDICTION DE LA QUALITÉ DE L'AIR")
    print("  Projet 34 — Science des données environnementale")
    print("=" * 55)

    # 1. Base de données
    print("\n[1/6] Initialisation de la base SQLite...")
    conn = creer_base_de_donnees()
    inserer_stations(conn)

    # 2. Données simulées
    print("[2/6] Génération des données (2 ans, horaire)...")
    df_brut = generer_donnees(n_jours=730)
    df_brut.to_sql("mesures", conn, if_exists="replace", index=False)
    print(f"  → {len(df_brut):,} mesures stockées")

    # 3. Requêtes SQL
    print("\n[3/6] Requêtes analytiques SQL...")
    analyses = requetes_analytiques(conn)
    print("  Heures de pointe :")
    print(analyses["heures_pointe"].head(6).to_string(index=False))

    # 4. Prétraitement
    print("\n[4/6] Prétraitement...")
    df = preparer_features(df_brut)
    X = df[FEATURES].values
    y = df[CIBLE].values
    X_norm, y_norm, scaler_X, scaler_y = normaliser(X, y)
    print(f"  → {len(FEATURES)} features · {len(df):,} observations")

    # Split temporel
    X_tr, X_te, y_tr, y_te = split_temporel(X, y)
    _, _, y_tr_n, y_te_n   = split_temporel(X_norm, y_norm)

    # 5. Modèles
    print("\n[5/6] Entraînement des modèles...")
    res_reg  = entrainer_gradient_boosting(X_tr, y_tr, X_te, y_te)

    X_tr_seq, y_tr_seq = creer_sequences(X_tr, y_tr_n, LONGUEUR_SEQ)
    X_te_seq, y_te_seq = creer_sequences(X_te, y_te_n, LONGUEUR_SEQ)
    res_lstm = entrainer_lstm(X_tr_seq, y_tr_seq, X_te_seq, y_te_seq, scaler_y)

    # 6. Résultats
    print("\n[6/6] Résultats comparatifs :")
    print(f"{'Modèle':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 51)
    print(f"{'Gradient Boosting':<25} {res_reg['mae']:>8} {res_reg['rmse']:>8} {res_reg['r2']:>8}")
    print(f"{'LSTM (2 couches)':<25} {res_lstm['mae']:>8} {res_lstm['rmse']:>8} {res_lstm['r2']:>8}")

    afficher_resultats(res_lstm, df)
    afficher_tableau_bord(df)

    res_lstm["modele"].save("modele_lstm_iqa.keras")
    print("\n✓ Modèle sauvegardé : modele_lstm_iqa.keras")

    conn.close()
    print("✓ Terminé !")


if __name__ == "__main__":
    main()
# Projet 1 — Prédiction de la Qualité de l'Air

Prédiction de l'Indice de Qualité de l'Air (IQA) à l'aide de modèles de régression et de réseaux de neurones LSTM.

## Compétences
- **Langages** : Python, SQL
- **Librairies** : TensorFlow/Keras, scikit-learn, pandas, matplotlib
- **Techniques** : Séries temporelles, LSTM, Gradient Boosting, ingénierie de features

---

## Structure du projet

```
air_quality_project/
│
├── main.py                  # Point d'entrée — pipeline complet
├── requirements.txt
├── README.md
│
└── src/
    ├── __init__.py
    ├── database.py          # Schéma SQLite + requêtes analytiques
    ├── data_generator.py    # Génération de données simulées
    ├── preprocessing.py     # Features, normalisation, séquences LSTM
    ├── models.py            # Gradient Boosting + LSTM
    └── visualization.py     # Graphiques et tableau de bord
```

---

## Installation

```bash
git clone https://github.com/mouhamedx74/air-quality-prediction.git
cd air-quality-prediction
pip install -r requirements.txt
```

## Lancement

```bash
python main.py
```

---

## Résultats attendus

| Modèle            | MAE  | RMSE | R²    |
|-------------------|------|------|-------|
| Gradient Boosting | ~5.1 | ~7.8 | ~0.94 |
| LSTM (2 couches)  | ~6.2 | ~8.4 | ~0.91 |

Les graphiques générés sont sauvegardés dans le répertoire racine :
- `resultats_qualite_air.png`
- `tableau_bord_iqa.png`
- `modele_lstm_iqa.keras`

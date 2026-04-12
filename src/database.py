"""
Gestion de la base de données SQLite.
Création du schéma, insertion des stations et requêtes analytiques.
"""

import sqlite3
import pandas as pd


def creer_base_de_donnees(chemin_db: str = "qualite_air.db") -> sqlite3.Connection:
    conn = sqlite3.connect(chemin_db)
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS stations (
            id_station   INTEGER PRIMARY KEY,
            nom          TEXT NOT NULL,
            ville        TEXT NOT NULL,
            latitude     REAL,
            longitude    REAL
        );

        CREATE TABLE IF NOT EXISTS mesures (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            id_station     INTEGER,
            horodatage     DATETIME NOT NULL,
            pm25           REAL,
            pm10           REAL,
            no2            REAL,
            o3             REAL,
            co             REAL,
            so2            REAL,
            temperature    REAL,
            humidite       REAL,
            vitesse_vent   REAL,
            direction_vent REAL,
            pression       REAL,
            iqa            REAL,
            FOREIGN KEY (id_station) REFERENCES stations(id_station)
        );

        CREATE TABLE IF NOT EXISTS previsions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            id_station   INTEGER,
            horodatage   DATETIME NOT NULL,
            iqa_predit   REAL,
            modele       TEXT,
            cree_le      DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_mesures_station_date
            ON mesures (id_station, horodatage);
    """)
    conn.commit()
    return conn


def inserer_stations(conn: sqlite3.Connection) -> None:
    stations = [
        (1, "Part-Dieu",  "Lyon", 45.7597, 4.8600),
        (2, "Perrache",   "Lyon", 45.7486, 4.8267),
        (3, "Confluence", "Lyon", 45.7356, 4.8178),
        (4, "Vaise",      "Lyon", 45.7706, 4.8013),
        (5, "Bron",       "Lyon", 45.7364, 4.9119),
        (6, "Gerland",    "Lyon", 45.7236, 4.8314),
    ]
    conn.executemany("INSERT OR IGNORE INTO stations VALUES (?,?,?,?,?)", stations)
    conn.commit()


def requetes_analytiques(conn: sqlite3.Connection) -> dict:
    requetes = {}

    requetes["moyenne_quotidienne"] = pd.read_sql_query("""
        SELECT
            id_station,
            DATE(horodatage)    AS date,
            ROUND(AVG(pm25), 2) AS pm25_moyen,
            ROUND(AVG(no2),  2) AS no2_moyen,
            ROUND(AVG(o3),   2) AS o3_moyen,
            ROUND(AVG(iqa),  2) AS iqa_moyen
        FROM mesures
        WHERE horodatage >= DATE('now', '-30 days')
        GROUP BY id_station, DATE(horodatage)
        ORDER BY date DESC
    """, conn)

    requetes["depassements"] = pd.read_sql_query("""
        SELECT
            s.nom,
            DATE(m.horodatage)    AS date,
            ROUND(AVG(m.pm25), 2) AS pm25_24h,
            ROUND(AVG(m.iqa),  2) AS iqa_24h,
            COUNT(*)              AS nb_mesures
        FROM mesures m
        JOIN stations s ON m.id_station = s.id_station
        GROUP BY m.id_station, DATE(m.horodatage)
        HAVING pm25_24h > 25 OR iqa_24h > 100
        ORDER BY iqa_24h DESC
        LIMIT 20
    """, conn)

    requetes["heures_pointe"] = pd.read_sql_query("""
        SELECT
            CAST(strftime('%H', horodatage) AS INTEGER) AS heure,
            ROUND(AVG(pm25), 2) AS pm25_moyen,
            ROUND(AVG(no2),  2) AS no2_moyen,
            ROUND(AVG(iqa),  2) AS iqa_moyen
        FROM mesures
        GROUP BY heure
        ORDER BY heure
    """, conn)

    return requetes
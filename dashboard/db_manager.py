import sqlite3
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

# Définition du chemin de la base de données (à la racine du projet)
DB_PATH = Path(__file__).resolve().parent.parent / "qa_history.db"

def init_db():
    """Crée la table qa_sessions si elle n'existe pas encore."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS qa_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_analyse DATETIME DEFAULT CURRENT_TIMESTAMP,
            scanner_model TEXT,
            noise_hu REAL,
            uniformity_hu REAL,
            mtf_50 REAL,
            global_score_percent REAL,
            tube_rul_days INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def insert_qa_session(scanner_model: str, noise_hu: float, uniformity_hu: float, mtf_50: float, global_score_percent: float, tube_rul_days: int):
    """Insère une nouvelle analyse QA dans l'historique."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO qa_sessions (scanner_model, noise_hu, uniformity_hu, mtf_50, global_score_percent, tube_rul_days)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (scanner_model, noise_hu, uniformity_hu, mtf_50, global_score_percent, tube_rul_days))
    conn.commit()
    conn.close()

def get_all_sessions() -> pd.DataFrame:
    """Récupère tout l'historique sous forme de DataFrame Pandas (trié du plus récent au plus ancien)."""
    # Assurez-vous que la base est initialisée
    init_db()
    
    conn = sqlite3.connect(DB_PATH)
    # Tri descendant par date pour afficher les plus récents en haut du tableau
    df = pd.read_sql_query("SELECT * FROM qa_sessions ORDER BY date_analyse DESC", conn)
    conn.close()
    
    # Conversion de la colonne de date en type datetime
    if not df.empty:
        df['date_analyse'] = pd.to_datetime(df['date_analyse'])
        
    return df

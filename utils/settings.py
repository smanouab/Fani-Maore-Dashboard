"""
Configuration générale pour l'application d'analyse sismique - Fani Maoré
"""

import os
from pathlib import Path

# Chemins du projet (CORRECTION: partir de la racine du projet)
BASE_DIR = Path(__file__).parent.parent  # Remonte de utils/ vers la racine
DATA_DIR = BASE_DIR / "data"
PAGES_DIR = BASE_DIR / "pages"
UTILS_DIR = BASE_DIR / "utils"

# Configuration des données
DATA_FILE = DATA_DIR / "NewDataseisme.csv"  # Votre fichier principal
BACKUP_DATA_FILE = DATA_DIR / "backup_seisme.csv"

# Configuration de l'application Streamlit
APP_CONFIG = {
    'page_title': 'Fani Maoré - Surveillance Sismique',
    'page_icon': '🌋',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Configuration des colonnes de données
REQUIRED_COLUMNS = [
    'Date', 'Magnitude', 'Latitude', 'Longitude', 'Profondeur'
]

# Configuration pour le format CSV français (séparateur point-virgule, virgule décimale)
CSV_FRENCH_FORMAT = {
    'separator': ';',
    'decimal': ',',
    'columns': ["Date", "Magnitude", "Latitude", "Longitude", "Profondeur", "origine"]
}

# Configuration cache
CACHE_TTL = 3600  # 1 heure en secondes
MAX_CACHE_SIZE = 100

# Seuils et limites
MAX_DATA_POINTS = 50000
MIN_MAGNITUDE = 0.0
MAX_MAGNITUDE = 10.0
MIN_DEPTH = -10.0
MAX_DEPTH = 1000.0

# Configuration Fani Maoré
FANI_MAORE = {
    'lat': -12.8,
    'lon': 45.5,
    'name': 'Volcan Fani Maoré',
    'depth': 3500  # Profondeur sous le niveau de la mer
}

MAYOTTE = {
    'lat': -12.8275,
    'lon': 45.1662,
    'name': 'Mayotte'
}

# Configuration des couleurs pour les visualisations
COLOR_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'magnitude_colors': {
        'low': '#2ca02c',      # Vert pour magnitude < 4
        'medium': '#ff7f0e',   # Orange pour 4 <= magnitude < 6
        'high': '#d62728'      # Rouge pour magnitude >= 6
    }
}

# Messages d'erreur
ERROR_MESSAGES = {
    'no_data': "Aucune donnée sismique disponible pour Fani Maoré.",
    'file_not_found': "Fichier de données Fani Maoré non trouvé.",
    'invalid_data': "Données invalides ou corrompues.",
    'connection_error': "Erreur de connexion réseau."
}

# Messages de succès
SUCCESS_MESSAGES = {
    'data_loaded': "Données Fani Maoré chargées avec succès.",
    'data_updated': "Données mises à jour avec succès.",
    'cache_cleared': "Cache vidé avec succès."
}

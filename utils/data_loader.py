import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class DataLoader:
    """Classe pour charger et traiter les données sismiques"""
    
    def __init__(self, data_file="data/NewDataseisme.csv"):
        self.data_file = data_file
        self.debug_info = []  # Pour stocker les infos de debug
    
    def load_data(self, filename=None):
        """Charge les données depuis le fichier CSV"""
        if filename:
            file_path = filename
        else:
            file_path = self.data_file
            
        self.debug_info = []  # Reset debug
        
        try:
            if not os.path.exists(file_path):
                self.debug_info.append(f"❌ Fichier non trouvé: {file_path}")
                return None
            
            # Vérifier la taille du fichier
            file_size = os.path.getsize(file_path)
            self.debug_info.append(f"📄 Fichier trouvé: {file_path} ({file_size} octets)")
            
            if file_size == 0:
                self.debug_info.append("❌ Fichier vide !")
                return None
            
            # Essayer différents séparateurs et encodages
            for sep in [';', ',', '\t']:
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        self.debug_info.append(f"🔄 Test: sep='{sep}', encoding='{encoding}'")
                        
                        df = pd.read_csv(file_path, sep=sep, encoding=encoding, nrows=10)  # Test avec 10 lignes
                        
                        if len(df.columns) > 3 and len(df) > 0:
                            self.debug_info.append(f"✅ Succès: {len(df.columns)} colonnes, {len(df)} lignes test")
                            self.debug_info.append(f"📋 Colonnes: {list(df.columns)}")
                            
                            # Charger le fichier complet
                            df_full = pd.read_csv(file_path, sep=sep, encoding=encoding)
                            self.debug_info.append(f"📊 Fichier complet: {len(df_full)} lignes")
                            
                            return self.process_data(df_full)
                            
                    except Exception as e:
                        self.debug_info.append(f"❌ Échec sep='{sep}', encoding='{encoding}': {str(e)[:100]}")
                        continue
            
            self.debug_info.append("❌ Aucune combinaison séparateur/encodage ne fonctionne")
            return None
            
        except Exception as e:
            self.debug_info.append(f"❌ Erreur générale: {e}")
            return None
    
    def get_debug_info(self):
        """Retourne les informations de debug"""
        return self.debug_info
    
    def process_data(self, df):
        """Traite et nettoie les données"""
        try:
            self.debug_info.append(f"🔧 Traitement des données: {len(df)} lignes, {len(df.columns)} colonnes")
            
            # Créer une copie pour éviter de modifier l'original
            df_processed = df.copy()
                     
            # Afficher les premières colonnes pour debug
            self.debug_info.append(f"📋 Colonnes originales: {list(df.columns[:10])}")
            
            # Traitement des dates - essayer différents formats
            date_cols = ['Date', 'date', 'TIME', 'time']
            date_col = None
            
            for col in date_cols:
                if col in df_processed.columns:
                    date_col = col
                    self.debug_info.append(f"📅 Colonne date trouvée: {col}")
                    break
            
            if date_col:
                try:
                    # Afficher quelques exemples de dates
                    sample_dates = df_processed[date_col].head(3).tolist()
                    self.debug_info.append(f"📅 Exemples de dates: {sample_dates}")
                    
                                       # CORRECTION: Convertir en datetime avec formats multiples
                    # Essayer plusieurs formats français ET internationaux
                    formats_to_try = [
                        '%d/%m/%Y %H:%M',     # 25/06/2025 18:47
                        '%d/%m/%y %H:%M',     # 25/06/25 18:47
                        '%d/%m/%Y',           # 25/06/2025
                        '%d/%m/%y',           # 25/06/25
                        '%Y-%m-%d %H:%M:%S',  # 2025-06-25 18:47:00
                        '%Y-%m-%d %H:%M',     # 2025-06-25 18:47
                        '%Y-%m-%d',           # 2025-06-25
                        '%d-%m-%Y %H:%M',     # 25-06-2025 18:47
                        '%d-%m-%Y',           # 25-06-2025
                        '%m/%d/%Y %H:%M',     # 06/25/2025 18:47 (format US)
                        '%m/%d/%Y'            # 06/25/2025
                    ]
                    
                    # Initialiser la colonne avec des NaT
                    df_processed['Date_dt'] = pd.NaT
                    dates_converted = 0
                    
                    # Essayer chaque format
                    for fmt in formats_to_try:
                        try:
                            # Convertir seulement les dates non encore converties
                            mask_not_converted = df_processed['Date_dt'].isna()
                            if mask_not_converted.sum() == 0:
                                break  # Toutes les dates sont converties
                            
                            # Essayer le format sur les dates non converties
                            temp_dates = pd.to_datetime(
                                df_processed.loc[mask_not_converted, date_col], 
                                format=fmt, 
                                errors='coerce'
                            )
                            
                            # Compter combien de nouvelles dates converties
                            new_converted = temp_dates.notna().sum()
                            if new_converted > 0:
                                # Mettre à jour les dates converties
                                df_processed.loc[mask_not_converted, 'Date_dt'] = temp_dates
                                dates_converted += new_converted
                                self.debug_info.append(f"✅ Format {fmt}: {new_converted} dates converties")
                            
                        except Exception as e:
                            self.debug_info.append(f"❌ Format {fmt} échoué: {str(e)[:50]}")
                            continue
                    
                    # Si il reste des dates non converties, essayer dayfirst=True
                    mask_not_converted = df_processed['Date_dt'].isna()
                    remaining = mask_not_converted.sum()
                    
                    if remaining > 0:
                        self.debug_info.append(f"🔄 Essai dayfirst=True pour {remaining} dates restantes")
                        try:
                            temp_dates = pd.to_datetime(
                                df_processed.loc[mask_not_converted, date_col], 
                                dayfirst=True, 
                                errors='coerce'
                            )
                            new_converted = temp_dates.notna().sum()
                            if new_converted > 0:
                                df_processed.loc[mask_not_converted, 'Date_dt'] = temp_dates
                                dates_converted += new_converted
                                self.debug_info.append(f"✅ dayfirst=True: {new_converted} dates converties")
                        except Exception as e:
                            self.debug_info.append(f"❌ dayfirst=True échoué: {e}")
                    
                    # Rapport final
                    total_valid = df_processed['Date_dt'].notna().sum()
                    self.debug_info.append(f"📅 TOTAL dates converties: {total_valid}/{len(df_processed)} ({total_valid/len(df_processed)*100:.1f}%)")
                    
                    # Extraire les composantes temporelles
                    df_processed['Annee'] = df_processed['Date_dt'].dt.year
                    df_processed['Mois'] = df_processed['Date_dt'].dt.month
                    df_processed['Jour'] = df_processed['Date_dt'].dt.day
                    df_processed['Heure'] = df_processed['Date_dt'].dt.hour
                    df_processed['JourSemaine'] = df_processed['Date_dt'].dt.dayofweek
                    
                    # NOUVEAU: Convertir les colonnes temporelles en entiers
                    temporal_cols = ['Annee', 'Mois', 'Jour', 'Heure', 'JourSemaine']
                    for col in temporal_cols:
                        if col in df_processed.columns:
                            # Convertir en entier en gérant les valeurs manquantes
                            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0).astype(int)
                            self.debug_info.append(f"🔢 {col} converti en entier")
                    
                except Exception as e:
                    self.debug_info.append(f"⚠️ Erreur traitement dates: {e}")
                    # Si l'extraction échoue, utiliser des valeurs par défaut
                    df_processed['Date_dt'] = pd.to_datetime('2024-01-01')
                    df_processed['Annee'] = 2024
                    df_processed['Mois'] = 1
                    df_processed['Jour'] = 1
                    df_processed['Heure'] = 0
                    df_processed['JourSemaine'] = 0
            else:
                self.debug_info.append("⚠️ Aucune colonne de date trouvée")
            
            # Nettoyer les colonnes numériques
            numeric_mappings = {
                'magnitude': 'Magnitude',
                'Magnitude': 'Magnitude',
                'mag': 'Magnitude',
                'depth': 'Profondeur',
                'Profondeur': 'Profondeur',
                'latitude': 'Latitude',
                'Latitude': 'Latitude',
                'lat': 'Latitude',
                'longitude': 'Longitude',
                'Longitude': 'Longitude',
                'lon': 'Longitude'
            }
            
            # Standardiser les noms de colonnes
            for old_name, new_name in numeric_mappings.items():
                if old_name in df_processed.columns and new_name not in df_processed.columns:
                    df_processed[new_name] = df_processed[old_name]
                    self.debug_info.append(f"🔄 Renommage: {old_name} → {new_name}")
            
            # Convertir les colonnes numériques
            for col in ['Magnitude', 'Profondeur', 'Latitude', 'Longitude']:
                if col in df_processed.columns:
                    original_type = df_processed[col].dtype
                    
                    # Remplacer les virgules par des points si format français
                    if df_processed[col].dtype == 'object':
                        df_processed[col] = df_processed[col].astype(str).str.replace(',', '.')
                    
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    valid_nums = df_processed[col].notna().sum()
                    
                    self.debug_info.append(f"🔢 {col}: {original_type} → numeric, {valid_nums}/{len(df_processed)} valides")
            
            # Supprimer les lignes avec des valeurs manquantes critiques
            original_length = len(df_processed)
            critical_cols = ['Latitude', 'Longitude']
            for col in critical_cols:
                if col in df_processed.columns:
                    df_processed = df_processed.dropna(subset=[col])
            
            if len(df_processed) < original_length:
                self.debug_info.append(f"🧹 Suppression lignes incomplètes: {original_length} → {len(df_processed)}")
            
            # Ajouter une colonne origine par défaut
            if 'origine' not in df_processed.columns:
                df_processed['origine'] = 'IPGP'
                self.debug_info.append("➕ Colonne 'origine' ajoutée")
            
            self.debug_info.append(f"✅ Traitement terminé: {len(df_processed)} lignes finales")
            self.debug_info.append(f"📋 Colonnes finales: {list(df_processed.columns)}")
            
            # Afficher les années trouvées pour debug
            if 'Annee' in df_processed.columns:
                years = sorted(df_processed['Annee'].unique())
                self.debug_info.append(f"📅 Années extraites: {years}")
            
            return df_processed
            
        except Exception as e:
            self.debug_info.append(f"❌ Erreur lors du traitement: {e}")
            return df
    
    def create_sample_data_for_dashboards(self, n_samples=1000):
        """Crée des données d'exemple pour tester les dashboards"""
        self.debug_info.append(f"🎲 Génération de {n_samples} données d'exemple")
        
        np.random.seed(42)
        
        # Générer des dates sur les 2 dernières années
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
        
        # Données aléatoires mais réalistes pour Mayotte
        data = {
            'Date': dates.strftime('%Y-%m-%d'),
            'Date_dt': dates,
            'Heure': [f"{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}" for _ in range(n_samples)],
            'Latitude': np.random.uniform(-13.5, -12.0, n_samples),  # Autour de Mayotte
            'Longitude': np.random.uniform(44.5, 46.0, n_samples),   # Autour de Mayotte
            'Magnitude': np.clip(np.random.exponential(1.5) + 1, 1.0, 6.0),
            'Profondeur': np.clip(np.random.exponential(20) + 5, 1.0, 200.0),
            'Lieu': [f"Région {i%20 + 1} - Mayotte" for i in range(n_samples)],
            'origine': ['IPGP'] * n_samples,
            'Annee': dates.year,
            'Mois': dates.month,
            'Jour': dates.day,
            'JourSemaine': dates.dayofweek
        }
        
        df = pd.DataFrame(data)
        self.debug_info.append(f"✅ Données d'exemple créées: {len(df)} lignes")
        
        return df

def load_seismic_data():
    """Fonction simple pour charger les données"""
    loader = DataLoader()
    df = loader.load_data()
    
    if df is None or df.empty:
        # Créer des données d'exemple
        df = loader.create_sample_data_for_dashboards()
    
    return df
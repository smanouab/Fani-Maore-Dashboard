import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os
import warnings

# Configuration Streamlit
st.set_page_config(page_title="Fani Maoré - Surveillance Sismique", page_icon="🌋", layout="wide")

# Configuration obligatoire pour Streamlit Cloud
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'

# Imports sécurisés pour le cloud
try:
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.style.use('default')
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Erreur matplotlib: {e}")
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    sns.set_palette("husl")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Hide Streamlit's automatic file browser/navigation
st.markdown("""
<style>
/* Hide the file browser navigation */
.stApp > header {
    background-color: transparent;
}
.stApp > header > div {
    display: none !important;
}
/* Hide the "Browse files" section */
section[data-testid="stFileUploadDropzone"] {
    display: none !important;
}
/* Hide any top navigation elements */
.main .block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Ajouter utils au path
utils_path = str(Path(__file__).parent / "utils")
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

# Header Volcan Fani Maoré
st.markdown("""
<div style="background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); 
            color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
    <h1>🌋 Surveillance Sismique - Volcan Fani Maoré</h1>
    <p style="font-size: 16px; margin: 0;">
        <strong>📍 Mayotte, Océan Indien</strong> • Système volcanique sous-marin actif depuis 2018
    </p>
</div>
""", unsafe_allow_html=True)

# Fonction pour charger les données
@st.cache_data
def load_data():
    try:
        from utils.data_loader import DataLoader
        loader = DataLoader()
        df = loader.load_data()
        
        if df is None or df.empty:
            df = loader.create_sample_data_for_dashboards()
            st.info("📊 Données d'exemple utilisées")
        else:
            st.success(f"✅ {len(df)} séismes réels chargés")
        
        # Fix data types to prevent Arrow serialization issues
        df = fix_dataframe_types(df)
        
        return df
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        return None

def fix_dataframe_types(df):
    """Fix DataFrame column types to prevent PyArrow serialization errors"""
    df = df.copy()
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['Magnitude', 'Profondeur', 'Latitude', 'Longitude']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure integer columns are properly typed
    int_columns = ['Annee', 'Mois', 'Jour', 'Heure', 'JourSemaine']
    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Ensure datetime columns
    if 'Date_dt' in df.columns:
        df['Date_dt'] = pd.to_datetime(df['Date_dt'], errors='coerce')
    
    # Convert any remaining object columns with mixed types to string
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                numeric_version = pd.to_numeric(df[col], errors='coerce')
                if not numeric_version.isna().all():
                    df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(str)
            except:
                df[col] = df[col].astype(str)
    
    # Remove any rows with critical NaN values
    df = df.dropna(subset=['Magnitude', 'Profondeur'])
    
    return df

def safe_dataframe_display(df, **kwargs):
    """Safely display a dataframe by ensuring Arrow compatibility"""
    try:
        return st.dataframe(df, **kwargs)
    except Exception as e:
        try:
            df_fixed = df.copy()
            for col in df_fixed.columns:
                if df_fixed[col].dtype == 'object':
                    df_fixed[col] = df_fixed[col].astype(str)
            for col in df_fixed.columns:
                if 'Valeur' in col or 'Value' in col:
                    df_fixed[col] = df_fixed[col].astype(str)
            return st.dataframe(df_fixed, **kwargs)
        except Exception as e2:
            try:
                df_string = df.copy()
                for col in df_string.columns:
                    df_string[col] = df_string[col].astype(str)
                return st.dataframe(df_string, **kwargs)
            except Exception as e3:
                st.error(f"Cannot display dataframe due to type conflicts. Showing as text:")
                st.text(str(df))
                return None

# Charger les données
df = load_data()

if df is not None:
    # Stocker en session
    st.session_state.data = df
    st.session_state.filtered_df = df
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <h2>🌋 Fani Maoré</h2>
            <p>Dashboard Sismique</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.selectbox("📍 Navigation:", [
            "🏠 Accueil", 
            "📊 Analyse Générale",
            "🗺️ Analyse Spatio-Temporelle", 
            "📈 Analyse Tendances",
            "🔬 Analyse Caractéristiques"
        ])
        
        st.markdown("---")
        
        # Contexte volcanique
        st.markdown("""
        ### 📍 Contexte
        **Volcan:** Fani Maoré  
        **Région:** Mayotte  
        **Type:** Sous-marin  
        **Profondeur:** 3500m  
        **Surveillance:** REVOSIMA
        """)
        
        # Quick stats in sidebar
        if 'data' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 Données")
            st.metric("Total", f"{len(st.session_state.data):,}")
            st.metric("Magnitude Max", f"{st.session_state.data['Magnitude'].max():.1f}")
            st.metric("Période", f"{st.session_state.data['Annee'].min()}-{st.session_state.data['Annee'].max()}")
    
    if page == "🏠 Accueil":
        st.header("🏠 Vue d'Ensemble - Fani Maoré")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Séismes", len(df))
        with col2:
            st.metric("⚡ Magnitude Moy.", f"{df['Magnitude'].mean():.2f}")
        with col3:
            years = sorted(df['Annee'].unique())
            st.metric("📅 Période", f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0]))
        with col4:
            st.metric("🕳️ Profondeur Moy.", f"{df['Profondeur'].mean():.1f} km")
        
        # Informations sur le volcan Fani Maoré
        with st.expander("ℹ️ À propos du Volcan Fani Maoré", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🌋 Caractéristiques:**
                - **Localisation:** 50 km à l'est de Mayotte
                - **Type:** Volcan sous-marin basaltique
                - **Profondeur:** 3500m sous le niveau de la mer
                - **Découverte:** 2019
                """)
            
            with col2:
                st.markdown("""
                **📈 Surveillance:**
                - **Début activité:** Mai 2018
                - **Réseau:** REVOSIMA
                - **Stations:** 15 sismomètres
                - **Fréquence:** Monitoring 24h/7j
                """)
        
        # Aperçu des données
        st.subheader("📋 Aperçu des Données Sismiques")
        
        display_df = df.head().copy()
        for col in display_df.columns:
            if display_df[col].dtype in ['float64', 'float32']:
                display_df[col] = display_df[col].round(3)
            elif display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str)
        
        safe_dataframe_display(display_df, use_container_width=True)
        
        # Graphiques de base
        col1, col2 = st.columns(2)
        
        with col1:
            if MATPLOTLIB_AVAILABLE:
                try:
                    fig, ax = plt.subplots()
                    ax.hist(df['Magnitude'], bins=20, alpha=0.7, color='skyblue')
                    ax.set_title('Distribution des magnitudes - Fani Maoré')
                    ax.set_xlabel('Magnitude')
                    ax.set_ylabel('Fréquence')
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Erreur graphique: {e}")
                    # Fallback vers Plotly
                    try:
                        import plotly.express as px
                        fig = px.histogram(df, x='Magnitude', nbins=20, title='Distribution des magnitudes - Fani Maoré')
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.warning("📊 Graphique temporairement indisponible")
            else:
                try:
                    import plotly.express as px
                    fig = px.histogram(df, x='Magnitude', nbins=20, title='Distribution des magnitudes - Fani Maoré')
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.bar_chart(df['Magnitude'].value_counts().sort_index())
        
        with col2:
            yearly_counts = df['Annee'].value_counts().sort_index()
            
            if MATPLOTLIB_AVAILABLE:
                try:
                    fig, ax = plt.subplots()
                    ax.bar(yearly_counts.index, yearly_counts.values, alpha=0.7, color='lightcoral')
                    ax.set_title('Évolution annuelle - Fani Maoré')
                    ax.set_xlabel('Année')
                    ax.set_ylabel('Nombre de séismes')
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Erreur graphique: {e}")
                    # Fallback vers Plotly
                    try:
                        import plotly.express as px
                        fig = px.bar(x=yearly_counts.index, y=yearly_counts.values, title='Évolution annuelle - Fani Maoré')
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.bar_chart(yearly_counts)
            else:
                try:
                    import plotly.express as px
                    fig = px.bar(x=yearly_counts.index, y=yearly_counts.values, title='Évolution annuelle - Fani Maoré')
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.bar_chart(yearly_counts)
        
        # Info sur les données
        st.subheader("📊 Résumé de l'Activité Sismique")
        st.info(f"""
        **Période d'observation:** {df['Date_dt'].min().strftime('%d/%m/%Y')} - {df['Date_dt'].max().strftime('%d/%m/%Y')}
        
        **Années disponibles:** {', '.join(map(str, sorted(df['Annee'].unique())))}
        
        **Magnitude:** {df['Magnitude'].min():.1f} - {df['Magnitude'].max():.1f}
        
        **Profondeur:** {df['Profondeur'].min():.1f} - {df['Profondeur'].max():.1f} km
        
        **Zone de surveillance:** Volcan sous-marin Fani Maoré, Mayotte
        """)
        
    elif page == "📊 Analyse Générale":
        st.header("📊 Analyse Générale - Fani Maoré")
        
        if 'filtered_df' in st.session_state:
            try:
                # Ajouter pages au path
                pages_path = str(Path(__file__).parent / "pages")
                if pages_path not in sys.path:
                    sys.path.insert(0, pages_path)
                
                # Import et lancement du module
                from pages.Analyse_generale import show_analyse_generale
                st.success("✅ Module Analyse Générale chargé")
                show_analyse_generale()
                
            except ImportError as e:
                st.warning(f"⚠️ Module Analyse Générale non disponible: {e}")
                st.info("Affichage d'une analyse simplifiée...")
                
                # Analyse simplifiée de secours
                data = st.session_state.filtered_df
                
                # Sélecteur d'année
                years = sorted(data['Annee'].unique())
                selected_year = st.selectbox("Choisir l'année:", years)
                
                # Filtrer par année
                data_year = data[data['Annee'] == selected_year]
                
                st.metric("Séismes pour l'année sélectionnée", len(data_year))
                
                # Graphique mensuel
                if MATPLOTLIB_AVAILABLE:
                    try:
                        monthly_counts = data_year['Mois'].value_counts().sort_index()
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 
                                 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
                        
                        ax.bar(monthly_counts.index, monthly_counts.values, alpha=0.7, color='steelblue')
                        ax.set_title(f'Activité mensuelle Fani Maoré - {selected_year}')
                        ax.set_xlabel('Mois')
                        ax.set_ylabel('Nombre de séismes')
                        ax.set_xticks(range(1, 13))
                        ax.set_xticklabels(months)
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.error(f"Erreur graphique: {e}")
        else:
            st.error("❌ Données non trouvées")
    
    elif page == "🗺️ Analyse Spatio-Temporelle":
        st.header("🗺️ Analyse Spatio-Temporelle - Fani Maoré")
        
        if 'filtered_df' in st.session_state:
            try:
                # Ajouter pages au path
                pages_path = str(Path(__file__).parent / "pages")
                if pages_path not in sys.path:
                    sys.path.insert(0, pages_path)
                
                # Import et lancement du module
                from pages.Analyse_spatio_temporelle import show_analyse_spatio_temporelle
                st.success("✅ Module Analyse Spatio-Temporelle chargé")
                show_analyse_spatio_temporelle()
                
            except ImportError as e:
                st.warning(f"⚠️ Module Analyse Spatio-Temporelle non disponible: {e}")
                st.info("Affichage d'une analyse simplifiée...")
                
                # Analyse simplifiée de secours
                data = st.session_state.filtered_df
                
                # Carte simple avec matplotlib
                if MATPLOTLIB_AVAILABLE:
                    try:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Scatter plot des séismes
                        scatter = ax.scatter(data['Longitude'], data['Latitude'], 
                                           c=data['Magnitude'], cmap='plasma', 
                                           alpha=0.6, s=30)
                        
                        plt.colorbar(scatter, label='Magnitude')
                        ax.set_xlabel('Longitude')
                        ax.set_ylabel('Latitude')
                        ax.set_title('Localisation des séismes - Zone Fani Maoré')
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.error(f"Erreur graphique: {e}")
                
                # Statistiques spatiales
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Latitude min", f"{data['Latitude'].min():.4f}°")
                    st.metric("Latitude max", f"{data['Latitude'].max():.4f}°")
                with col2:
                    st.metric("Longitude min", f"{data['Longitude'].min():.4f}°")
                    st.metric("Longitude max", f"{data['Longitude'].max():.4f}°")
        else:
            st.error("❌ Données non trouvées")
    
    elif page == "📈 Analyse Tendances":
        st.header("📈 Analyse des Tendances - Fani Maoré")
        
        if 'filtered_df' in st.session_state:
            try:
                # Ajouter pages au path
                pages_path = str(Path(__file__).parent / "pages")
                if pages_path not in sys.path:
                    sys.path.insert(0, pages_path)
                
                # Import et lancement du module
                from pages.Analyse_tendances_sismique import show_analyse_tendances
                st.success("✅ Module Analyse Tendances chargé")
                show_analyse_tendances()
                
            except ImportError as e:
                st.warning(f"⚠️ Module Analyse Tendances non disponible: {e}")
                st.info("Affichage d'une analyse simplifiée...")
                
                # Analyse simplifiée de secours
                data = st.session_state.filtered_df
                
                if MATPLOTLIB_AVAILABLE:
                    try:
                        # Évolution temporelle
                        daily_counts = data.groupby(data['Date_dt'].dt.date).size()
                        
                        fig, ax = plt.subplots(figsize=(14, 6))
                        ax.plot(daily_counts.index, daily_counts.values, alpha=0.7)
                        ax.set_title('Évolution temporelle - Fani Maoré')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Nombre de séismes')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close()
                        
                        # Tendances par jour de la semaine
                        jours = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
                        weekly_counts = data['JourSemaine'].value_counts().sort_index()
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(range(7), [weekly_counts.get(i, 0) for i in range(7)], alpha=0.7, color='orange')
                        ax.set_title('Distribution hebdomadaire - Fani Maoré')
                        ax.set_xlabel('Jour')
                        ax.set_ylabel('Nombre de séismes')
                        ax.set_xticks(range(7))
                        ax.set_xticklabels(jours)
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.error(f"Erreur graphique: {e}")
        else:
            st.error("❌ Données non trouvées")
    
    elif page == "🔬 Analyse Caractéristiques":
        st.header("🔬 Analyse des Caractéristiques - Fani Maoré")
        
        if 'filtered_df' in st.session_state:
            try:
                # Ajouter pages au path
                pages_path = str(Path(__file__).parent / "pages")
                if pages_path not in sys.path:
                    sys.path.insert(0, pages_path)
                
                # Import et lancement du module
                from pages.Analyse_caracteristiques_sismique import show_analyse_caracteristiques
                st.success("✅ Module Analyse Caractéristiques chargé")
                show_analyse_caracteristiques()
                
            except ImportError as e:
                st.warning(f"⚠️ Module Analyse Caractéristiques non disponible: {e}")
                st.info("Affichage d'une analyse simplifiée...")
                
                # Analyse simplifiée de secours
                data = st.session_state.filtered_df
                
                if MATPLOTLIB_AVAILABLE:
                    try:
                        # Distribution magnitude vs profondeur
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                        
                        # Distribution des magnitudes
                        ax1.hist(data['Magnitude'], bins=20, alpha=0.7, color='skyblue')
                        ax1.set_title('Distribution des magnitudes - Fani Maoré')
                        ax1.set_xlabel('Magnitude')
                        ax1.set_ylabel('Fréquence')
                        
                        # Distribution des profondeurs
                        ax2.hist(data['Profondeur'], bins=20, alpha=0.7, color='lightcoral')
                        ax2.set_title('Distribution des profondeurs - Fani Maoré')
                        ax2.set_xlabel('Profondeur (km)')
                        ax2.set_ylabel('Fréquence')
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        # Relation magnitude-profondeur
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(data['Profondeur'], data['Magnitude'], alpha=0.6)
                        ax.set_xlabel('Profondeur (km)')
                        ax.set_ylabel('Magnitude')
                        ax.set_title('Relation Magnitude vs Profondeur - Fani Maoré')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.error(f"Erreur graphique: {e}")
                
                # Statistiques
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Statistiques Magnitude")
                    mag_stats = data['Magnitude'].describe()
                    stats_df = pd.DataFrame({
                        'Statistique': mag_stats.index,
                        'Valeur': [f"{val:.3f}" for val in mag_stats.values]
                    })
                    safe_dataframe_display(stats_df, hide_index=True, use_container_width=True)
                    
                with col2:
                    st.subheader("Statistiques Profondeur")
                    depth_stats = data['Profondeur'].describe()
                    stats_df = pd.DataFrame({
                        'Statistique': depth_stats.index,
                        'Valeur': [f"{val:.1f} km" for val in depth_stats.values]
                    })
                    safe_dataframe_display(stats_df, hide_index=True, use_container_width=True)
        else:
            st.error("❌ Données non trouvées")

else:
    st.error("❌ Impossible de charger les données")
    
    # Informations de debug
    st.subheader("🔧 Informations de debug")
    st.info("""
    Si vous voyez cette erreur:
    1. Vérifiez que le fichier `data/NewDataseisme.csv` existe
    2. Vérifiez que le module `data_loader` est dans `utils/`
    3. Redémarrez l'application
    """)

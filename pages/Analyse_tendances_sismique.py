"""
Analyse des Tendances Sismiques
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import calendar
from matplotlib.colors import LinearSegmentedColormap
import warnings
import sys
import os

warnings.filterwarnings('ignore')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
plt.style.use('default')
sns.set_palette("husl")

def apply_custom_css():
    """Appliquer le CSS personnalisé pour les tendances"""
    st.markdown("""
    <style>
    .intro-section {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .intro-text {
        color: #155724;
        font-size: 16px;
        line-height: 1.6;
        margin: 0;
        text-align: center;
    }
    
    .statistical-result {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2ecc71;
        color: #155724;
        font-weight: 500;
    }
    
    .statistical-result h4 {
        color: #155724;
        margin-bottom: 10px;
        font-weight: bold;
    }
    
    .statistical-result p {
        color: #155724;
        margin: 5px 0;
        font-weight: 500;
    }
    
    .trend-metric {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .filter-label {
        font-weight: 600;
        color: #495057;
        margin-bottom: 5px;
        font-size: 13px;
    }
    
    .stSlider > div > div > div > div {
        background-color: #28a745 !important;
    }
    
    .stSlider > label {
        display: none !important;
    }
    
    .stMultiSelect > label {
        display: none !important;
    }
    
    .stSelectbox > label {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

def show_analyse_tendances():
    """Fonction principale pour afficher l'analyse des tendances"""
    
    apply_custom_css()
    
    st.markdown("""
    <div class="intro-section">
        <p class="intro-text" style="text-align: center; font-weight: bold; line-height: 1.8;">
            ✅ <strong>15407 séismes réels chargés</strong><br><br>
            Ce tableau de bord permet d'analyser les <strong>tendances temporelles</strong> de l'activité sismique. 
            Vous pouvez explorer les <strong>patterns saisonniers</strong>, <strong>journaliers</strong>, les <strong>cycles hebdomadaires</strong> et les <strong>tendances à long terme</strong>. 
            📊 Utilisez les filtres ci-dessous pour personnaliser votre analyse selon vos besoins spécifiques.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'filtered_df' not in st.session_state:
        st.error("❌ Données non disponibles. Veuillez retourner à la page d'accueil.")
        return
    
    df_original = st.session_state.filtered_df.copy()
    
    if len(df_original) == 0:
        st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés.")
        return
    
    df_filtered = show_trends_filters(df_original)
    
    if df_filtered is None or len(df_filtered) == 0:
        st.info("💡 Ajustez les filtres pour voir les données.")
        return
    
    df_filtered = prepare_temporal_components(df_filtered)
    
    st.markdown("---")
    
    analysis_type = st.session_state.get('selected_trends_analysis_type', 'Tendances saisonnières')
    
    show_data_summary(df_filtered)
    
    if analysis_type == "Tendances saisonnières":
        analyser_tendances_saisonnieres(df_filtered)
    elif analysis_type == "Tendances journalières":
        analyser_tendances_journalieres(df_filtered)
    elif analysis_type == "Tendances à long terme":
        analyser_tendances_long_terme(df_filtered)
    elif analysis_type == "Cycles et périodicités":
        analyser_cycles_periodicites(df_filtered)

def show_trends_filters(df):
    """Afficher les filtres spécifiques à l'analyse des tendances"""
    
    with st.container():
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown('<div class="filter-label">📅 Période d\'analyse:</div>', unsafe_allow_html=True)
            if 'Date_dt' in df.columns:
                annee_min = int(df['Date_dt'].dt.year.min())
                annee_max = int(df['Date_dt'].dt.year.max())
            else:
                annee_min, annee_max = 2018, 2025
            
            annees_range = st.slider(
                "Sélection des années",
                min_value=annee_min,
                max_value=annee_max,
                value=(annee_min, annee_max),
                key="trends_annees_slider",
                label_visibility="collapsed"
            )
            st.caption(f"📊 {annees_range[0]} – {annees_range[1]}")
        
        with col2:
            st.markdown('<div class="filter-label">⚡ Magnitude:</div>', unsafe_allow_html=True)
            mag_min = float(df['Magnitude'].min())
            mag_max = float(df['Magnitude'].max())
            
            magnitude_range = st.slider(
                "Plage de magnitude",
                min_value=mag_min,
                max_value=mag_max,
                value=(mag_min, mag_max),
                step=0.05,
                format="%.2f",
                key="trends_mag_slider",
                label_visibility="collapsed"
            )
            st.caption(f"🎯 {magnitude_range[0]:.2f} – {magnitude_range[1]:.2f}")
        
        with col3:
            st.markdown('<div class="filter-label">🕳️ Profondeur (km):</div>', unsafe_allow_html=True)
            prof_min = float(df['Profondeur'].min())
            prof_max = float(df['Profondeur'].max())
            
            profondeur_range = st.slider(
                "Plage de profondeur",
                min_value=prof_min,
                max_value=prof_max,
                value=(prof_min, prof_max),
                step=2.0,
                format="%.1f",
                key="trends_prof_slider",
                label_visibility="collapsed"
            )
            st.caption(f"📏 {profondeur_range[0]:.1f} – {profondeur_range[1]:.1f}")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown('<div class="filter-label">🌿 Saisons:</div>', unsafe_allow_html=True)
            saisons_options = ["Printemps", "Été", "Automne", "Hiver"]
            selected_saisons = st.multiselect(
                "Sélection des saisons",
                options=saisons_options,
                default=saisons_options,
                key="trends_saisons_multiselect",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown('<div class="filter-label">📆 Mois:</div>', unsafe_allow_html=True)
            mois_options = [
                "Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"
            ]
            selected_mois = st.multiselect(
                "Sélection des mois",
                options=mois_options,
                default=mois_options,
                key="trends_mois_multiselect",
                label_visibility="collapsed"
            )
        
        with col3:
            st.markdown('<div class="filter-label">📈 Type d\'analyse:</div>', unsafe_allow_html=True)
            analysis_type = st.selectbox(
                "Choisissez le type d'analyse",
                options=[
                    "Tendances saisonnières",
                    "Tendances journalières", 
                    "Tendances à long terme",
                    "Cycles et périodicités"
                ],
                index=0,
                key="trends_analysis_type_select",
                label_visibility="collapsed"
            )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            apply_filters = st.button(
                "🔄 Appliquer les Filtres",
                key="trends_apply_filters_btn",
                use_container_width=True,
                type="primary"
            )
    
    df_filtered = apply_trends_filters_simple(
        df, annees_range, selected_mois, magnitude_range, profondeur_range, selected_saisons
    )
    
    st.session_state.trends_filtered_df = df_filtered
    st.session_state.selected_trends_analysis_type = analysis_type
    
    if len(df_filtered) == 0:
        st.error("❌ Aucune donnée ne correspond aux filtres")
    else:
        percentage = (len(df_filtered) / len(df)) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Séismes filtrés", f"{len(df_filtered):,}")
        with col2:
            st.metric("📈 Pourcentage", f"{percentage:.1f}%")
        with col3:
            st.metric("⚡ Magnitude moy.", f"{df_filtered['Magnitude'].mean():.2f}")
        with col4:
            if len(df_filtered) > 0:
                date_range = (df_filtered['Date_dt'].max() - df_filtered['Date_dt'].min()).days
                st.metric("📅 Période", f"{date_range} jours")
    
    return df_filtered

def apply_trends_filters_simple(df, annees_range, selected_mois, magnitude_range, profondeur_range, selected_saisons):
    """Appliquer les filtres de base des tendances"""
    
    df_filtered = df.copy()
    
    if 'Date_dt' not in df_filtered.columns and 'Date' in df_filtered.columns:
        df_filtered['Date_dt'] = pd.to_datetime(df_filtered['Date'])
    
    if 'Date_dt' in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered['Date_dt'].dt.year >= annees_range[0]) & 
            (df_filtered['Date_dt'].dt.year <= annees_range[1])
        ]
        
        if selected_mois:
            mois_mapping = {
                "Jan": 1, "Fév": 2, "Mar": 3, "Avr": 4,
                "Mai": 5, "Jun": 6, "Jul": 7, "Aoû": 8,
                "Sep": 9, "Oct": 10, "Nov": 11, "Déc": 12
            }
            selected_month_numbers = [mois_mapping[mois] for mois in selected_mois if mois in mois_mapping]
            if selected_month_numbers:
                df_filtered = df_filtered[df_filtered['Date_dt'].dt.month.isin(selected_month_numbers)]
        
        if selected_saisons:
            seasons_mapping = {
                'Hiver': [12, 1, 2],
                'Printemps': [3, 4, 5],
                'Été': [6, 7, 8],
                'Automne': [9, 10, 11]
            }
            
            valid_months = []
            for saison in selected_saisons:
                if saison in seasons_mapping:
                    valid_months.extend(seasons_mapping[saison])
            
            if valid_months:
                df_filtered = df_filtered[df_filtered['Date_dt'].dt.month.isin(valid_months)]
    
    df_filtered = df_filtered[
        (df_filtered['Magnitude'] >= magnitude_range[0]) & 
        (df_filtered['Magnitude'] <= magnitude_range[1])
    ]
    
    df_filtered = df_filtered[
        (df_filtered['Profondeur'] >= profondeur_range[0]) & 
        (df_filtered['Profondeur'] <= profondeur_range[1])
    ]
    
    return df_filtered

def prepare_temporal_components(df):
    """Préparer les composantes temporelles nécessaires pour l'analyse"""
    
    df = df.copy()
    
    if 'Date_dt' in df.columns:
        df['Date'] = df['Date_dt']
    
    required_columns = ['Annee', 'Mois', 'Jour', 'Heure', 'JourSemaine', 'Trimestre']
    
    for col in required_columns:
        if col not in df.columns:
            if col == 'Annee':
                df['Annee'] = df['Date'].dt.year
            elif col == 'Mois':
                df['Mois'] = df['Date'].dt.month
            elif col == 'Jour':
                df['Jour'] = df['Date'].dt.day
            elif col == 'Heure':
                df['Heure'] = df['Date'].dt.hour
            elif col == 'JourSemaine':
                df['JourSemaine'] = df['Date'].dt.dayofweek
            elif col == 'Trimestre':
                df['Trimestre'] = df['Date'].dt.quarter
    
    try:
        df['Semaine'] = df['Date'].dt.isocalendar().week
    except AttributeError:
        df['Semaine'] = df['Date'].dt.week
    
    df['JourAnnee'] = df['Date'].dt.dayofyear
    
    seasons = {
        'Hiver': [12, 1, 2],
        'Printemps': [3, 4, 5],
        'Été': [6, 7, 8],
        'Automne': [9, 10, 11]
    }
    
    def get_season(month):
        for season, months in seasons.items():
            if month in months:
                return season
        return 'Inconnu'
    
    df['Saison'] = df['Mois'].apply(get_season)
    
    return df

def show_data_summary(df):
    """Afficher un résumé des données pour l'analyse des tendances"""
    
    st.markdown("### 📋 Résumé des Données Analysées")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total séismes", len(df))
    
    with col2:
        years_span = df['Annee'].max() - df['Annee'].min() + 1
        st.metric("📅 Étendue", f"{years_span} ans")
    
    with col3:
        unique_years = df['Annee'].nunique()
        st.metric("🗓️ Années uniques", unique_years)
    
    with col4:
        period_days = (df['Date'].max() - df['Date'].min()).days
        st.metric("⏱️ Période totale", f"{period_days} jours")

def analyser_tendances_saisonnieres(df_filtered):
    """Analyser les tendances saisonnières avec tests statistiques"""
    
    st.subheader("🌸 Analyse des Tendances Saisonnières")
    
    if len(df_filtered) == 0:
        st.warning("Aucune donnée pour l'analyse saisonnière.")
        return
    
    mois_noms = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 
                 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
    
    # 1. Distribution mensuelle
    st.markdown("#### 📊 Distribution mensuelle")
    
    mois_counts = df_filtered.groupby('Mois').size()
    
    mois_dict = {i: 0 for i in range(1, 13)}
    for mois, count in mois_counts.items():
        mois_dict[mois] = count
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(1, 13), [mois_dict[i] for i in range(1, 13)], 
                  color='skyblue', alpha=0.8, edgecolor='navy')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + max(mois_dict.values()) * 0.01,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Nombre de séismes par mois (toutes années confondues)', fontsize=14, pad=20)
    ax.set_xlabel('Mois')
    ax.set_ylabel('Nombre de séismes')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(mois_noms, rotation=45)
    ax.grid(alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Test statistique Chi²
    if len(mois_counts) >= 6:
        observed_values = [mois_dict[i] for i in range(1, 13)]
        chi2, p = stats.chisquare(observed_values)
        
        st.markdown(f"""
        <div class="statistical-result">
            <h4>🧮 Test Chi² d'uniformité</h4>
            <p><strong>Chi² =</strong> {chi2:.2f}</p>
            <p><strong>p-value =</strong> {p:.4f}</p>
            <p><strong>Interprétation :</strong> {'Il existe une variation saisonnière statistiquement significative' if p < 0.05 else 'La distribution mensuelle semble uniforme (pas de tendance saisonnière significative)'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 2. Heatmap par mois et année
    if len(df_filtered['Annee'].unique()) > 1:
        st.markdown("#### 🔥 Heatmap mensuelle par année")
        
        try:
            heatmap_data = df_filtered.groupby(['Annee', 'Mois']).size().unstack(fill_value=0)
            
            for m in range(1, 13):
                if m not in heatmap_data.columns:
                    heatmap_data[m] = 0
            
            heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
            
            colors = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", 
                     "#4292c6", "#2171b5", "#08519c", "#08306b"]
            cmap = LinearSegmentedColormap.from_list("custom_blues", colors)
            
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(heatmap_data, cmap=cmap, annot=True, fmt="d", linewidths=.5, ax=ax)
            ax.set_title('Nombre de séismes par mois et par année')
            ax.set_xlabel('Mois')
            ax.set_ylabel('Année')
            ax.set_xticklabels([mois_noms[i-1] for i in heatmap_data.columns], rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.warning(f"Impossible de créer la heatmap: {e}")
    
    # 3. Magnitude moyenne par mois
    st.markdown("#### ⚡ Magnitude moyenne par mois")
    
    mag_means = df_filtered.groupby('Mois')['Magnitude'].mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    mois_avec_donnees = sorted(mag_means.index)
    bars = ax.bar(mois_avec_donnees, mag_means.values, color='orange', alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Magnitude moyenne des séismes par mois')
    ax.set_xlabel('Mois')
    ax.set_ylabel('Magnitude moyenne')
    ax.set_xticks(mois_avec_donnees)
    ax.set_xticklabels([mois_noms[int(i)-1] for i in mois_avec_donnees], rotation=45)
    ax.grid(alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 4. Analyse par saison
    st.markdown("#### 🍂 Analyse saisonnière")
    
    season_counts = df_filtered.groupby('Saison').size()
    season_order = ['Hiver', 'Printemps', 'Été', 'Automne']
    season_counts = season_counts.reindex([s for s in season_order if s in season_counts.index])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(season_counts.index, season_counts.values, 
                  color=['lightblue', 'lightgreen', 'orange', 'brown'], alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        percentage = height / len(df_filtered) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}\n({percentage:.1f}%)', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_title('Nombre de séismes par saison')
    ax.set_xlabel('Saison')
    ax.set_ylabel('Nombre de séismes')
    ax.grid(alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Statistiques par saison (fixed for Arrow compatibility)
    if len(season_counts) > 0:
        st.markdown("#### 📋 Statistiques saisonnières")
        
        stats_data = []
        for saison, count in season_counts.items():
            percentage = count / len(df_filtered) * 100
            stats_data.append({
                "Saison": str(saison),
                "Nombre de séismes": int(count),
                "Pourcentage": f"{percentage:.1f}%"
            })
        
        stats_df = pd.DataFrame(stats_data)
        # Convert to proper types for Arrow compatibility
        stats_df['Saison'] = stats_df['Saison'].astype(str)
        stats_df['Nombre de séismes'] = stats_df['Nombre de séismes'].astype(int)
        stats_df['Pourcentage'] = stats_df['Pourcentage'].astype(str)
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

def analyser_tendances_journalieres(df_filtered):
    """Analyser les tendances journalières et cycles hebdomadaires"""
    
    st.subheader("🕐 Analyse des Tendances Journalières")
    
    if len(df_filtered) == 0:
        st.warning("Aucune donnée pour l'analyse journalière.")
        return
    
    # 1. Distribution par heure
    st.markdown("#### ⏰ Distribution par heure")
    
    heure_counts = df_filtered.groupby('Heure').size()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(heure_counts.index, heure_counts.values, color='darkblue', alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Nombre de séismes par heure de la journée')
    ax.set_xlabel('Heure')
    ax.set_ylabel('Nombre de séismes')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 2. Analyse par périodes de la journée
    st.markdown("#### 🌅 Analyse par périodes")
    
    periodes = {
        'Nuit (0h-6h)': list(range(0, 6)),
        'Matin (6h-12h)': list(range(6, 12)),
        'Après-midi (12h-18h)': list(range(12, 18)),
        'Soir (18h-24h)': list(range(18, 24))
    }
    
    periode_counts = {}
    for nom, heures in periodes.items():
        periode_counts[nom] = df_filtered[df_filtered['Heure'].isin(heures)].shape[0]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['darkblue', 'gold', 'orange', 'purple']
    bars = ax.bar(periode_counts.keys(), periode_counts.values(), color=colors, alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        percentage = height / len(df_filtered) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}\n({percentage:.1f}%)', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_title('Nombre de séismes par période de la journée')
    ax.set_xlabel('Période')
    ax.set_ylabel('Nombre de séismes')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 3. Distribution par jour de la semaine
    st.markdown("#### 📅 Distribution hebdomadaire")
    
    jours_semaine = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    jour_counts = df_filtered.groupby('JourSemaine').size()
    
    jours_dict = {i: 0 for i in range(7)}
    for jour, count in jour_counts.items():
        jours_dict[jour] = count
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['lightblue' if i < 5 else 'lightcoral' for i in range(7)]
    bars = ax.bar(range(7), [jours_dict.get(i, 0) for i in range(7)], color=colors, alpha=0.8)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Nombre de séismes par jour de la semaine')
    ax.set_xlabel('Jour de la semaine')
    ax.set_ylabel('Nombre de séismes')
    ax.set_xticks(range(7))
    ax.set_xticklabels(jours_semaine)
    ax.grid(alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 4. Comparaison semaine vs weekend
    st.markdown("#### 🏢 Semaine vs Weekend")
    
    semaine = df_filtered[df_filtered['JourSemaine'] < 5].shape[0]
    weekend = df_filtered[df_filtered['JourSemaine'] >= 5].shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Jours de semaine\n(Lun-Ven)', 'Weekend\n(Sam-Dim)']
    values = [semaine, weekend]
    colors = ['lightblue', 'lightcoral']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    
    total = semaine + weekend
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = height / total * 100 if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}\n({percentage:.1f}%)', 
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('Nombre de séismes : Jours de semaine vs Weekend')
    ax.set_ylabel('Nombre de séismes')
    ax.grid(alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Statistiques semaine/weekend
    if total > 0:
        ratio_observe = weekend / semaine if semaine > 0 else 0
        ratio_attendu = 2/5  # 2 jours weekend / 5 jours semaine
        
        st.markdown(f"""
        <div class="trend-metric">
            <h4>📊 Analyse Semaine/Weekend</h4>
            <p><strong>Jours de semaine :</strong> {semaine} séismes ({semaine/total*100:.1f}%)</p>
            <p><strong>Weekend :</strong> {weekend} séismes ({weekend/total*100:.1f}%)</p>
            <p><strong>Rapport observé :</strong> {ratio_observe:.2f} (weekend/semaine)</p>
            <p><strong>Rapport attendu si uniforme :</strong> {ratio_attendu:.2f}</p>
            <p><strong>Interprétation :</strong> {'Le weekend a proportionnellement plus de séismes' if ratio_observe > ratio_attendu else 'Distribution relativement uniforme'}</p>
        </div>
        """, unsafe_allow_html=True)

def analyser_tendances_long_terme(df_filtered):
    """Analyser les tendances à long terme avec régression"""
    
    st.subheader("📊 Analyse des Tendances à Long Terme")
    
    annees_uniques = df_filtered['Annee'].unique()
    if len(annees_uniques) < 2:
        st.warning(f"Cette analyse nécessite au moins 2 années. Années disponibles : {', '.join(map(str, sorted(annees_uniques)))}")
        return
    
    # Regrouper par mois pour l'analyse des séries temporelles
    df_mensuel = df_filtered.groupby(pd.Grouper(key='Date', freq='M')).agg({
        'Magnitude': ['count', 'mean', 'max'],
        'Profondeur': 'mean'
    })
    
    df_mensuel.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_mensuel.columns.values]
    df_mensuel.rename(columns={'Magnitude_count': 'Nombre_Seismes'}, inplace=True)
    
    # 1. Évolution du nombre de séismes
    st.markdown("#### 📈 Évolution mensuelle du nombre de séismes")
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.plot(df_mensuel.index, df_mensuel['Nombre_Seismes'], 
           marker='o', linestyle='-', color='blue', linewidth=2, markersize=6,
           label='Données observées')
    
    # Tendance sur données brutes
    if len(df_mensuel) > 1:
        X = np.arange(len(df_mensuel)).reshape(-1, 1)
        y = df_mensuel['Nombre_Seismes'].values
        
        model = stats.linregress(X.flatten(), y)
        trend_line = model.slope * X.flatten() + model.intercept
        ax.plot(df_mensuel.index, trend_line, color='red', linestyle='--', linewidth=2,
               label=f'Tendance (pente={model.slope:.4f})')
        
        trend_significance = "significative" if model.pvalue < 0.05 else "non significative"
        trend_direction = "augmentation" if model.slope > 0 else "diminution"
        
        st.markdown(f"""
        <div class="statistical-result">
            <h4>🧮 Analyse de tendance (données brutes)</h4>
            <p><strong>Pente :</strong> {model.slope:.4f} séismes/mois</p>
            <p><strong>p-value :</strong> {model.pvalue:.4f}</p>
            <p><strong>R² :</strong> {model.rvalue**2:.4f}</p>
            <p><strong>Tendance :</strong> {trend_direction} {trend_significance}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Moyenne mobile si suffisamment de données
    if len(df_mensuel) >= 6:
        window_size = min(6, len(df_mensuel) // 2)
        rolling_mean = df_mensuel['Nombre_Seismes'].rolling(window=window_size, center=True).mean()
        ax.plot(df_mensuel.index, rolling_mean, color='green', linestyle='-.', linewidth=2,
               label=f'Moyenne mobile ({window_size} mois)')
        
        # Tendance sur moyenne mobile
        if len(rolling_mean.dropna()) > 1:
            rolling_clean = rolling_mean.dropna()
            X_roll = np.arange(len(rolling_clean)).reshape(-1, 1)
            y_roll = rolling_clean.values
            
            model_roll = stats.linregress(X_roll.flatten(), y_roll)
            trend_line_roll = model_roll.slope * X_roll.flatten() + model_roll.intercept
            
            trend_indices = df_mensuel.index[~rolling_mean.isna()]
            ax.plot(trend_indices, trend_line_roll, color='purple', linestyle='--', linewidth=2,
                   label=f'Tendance lissée (pente={model_roll.slope:.4f})')
            
            trend_significance_roll = "significative" if model_roll.pvalue < 0.05 else "non significative"
            trend_direction_roll = "augmentation" if model_roll.slope > 0 else "diminution"
            
            st.markdown(f"""
            <div class="trend-metric">
                <h4>📊 Analyse de tendance (moyenne mobile)</h4>
                <p><strong>Pente :</strong> {model_roll.slope:.4f} séismes/mois</p>
                <p><strong>p-value :</strong> {model_roll.pvalue:.4f}</p>
                <p><strong>R² :</strong> {model_roll.rvalue**2:.4f}</p>
                <p><strong>Tendance :</strong> {trend_direction_roll} {trend_significance_roll}</p>
            </div>
            """, unsafe_allow_html=True)
    
    ax.set_title('Évolution du nombre de séismes par mois', fontsize=16, pad=20)
    ax.set_xlabel('Date')
    ax.set_ylabel('Nombre de séismes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 2. Évolution de la magnitude moyenne
    st.markdown("#### ⚡ Évolution de la magnitude moyenne")
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df_mensuel.index, df_mensuel['Magnitude_mean'], 
           marker='o', linestyle='-', color='orange', linewidth=2, markersize=6)
    
    # Tendance de la magnitude
    if len(df_mensuel) > 1:
        X = np.arange(len(df_mensuel)).reshape(-1, 1)
        y = df_mensuel['Magnitude_mean'].values
        model_mag = stats.linregress(X.flatten(), y)
        trend_line_mag = model_mag.slope * X.flatten() + model_mag.intercept
        ax.plot(df_mensuel.index, trend_line_mag, color='red', linestyle='--', linewidth=2,
               label=f'Tendance (pente={model_mag.slope:.4f})')
        
        trend_significance = "significative" if model_mag.pvalue < 0.05 else "non significative"
        trend_direction = "augmentation" if model_mag.slope > 0 else "diminution"
        
        st.markdown(f"""
        <div class="trend-metric">
            <h4>📊 Tendance de la magnitude moyenne</h4>
            <p><strong>Pente :</strong> {model_mag.slope:.4f} magnitude/mois</p>
            <p><strong>p-value :</strong> {model_mag.pvalue:.4f}</p>
            <p><strong>Tendance :</strong> {trend_direction} {trend_significance}</p>
        </div>
        """, unsafe_allow_html=True)
    
    ax.set_title('Évolution de la magnitude moyenne par mois')
    ax.set_xlabel('Date')
    ax.set_ylabel('Magnitude moyenne')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # 3. Tableau récapitulatif annuel
    st.markdown("#### 📋 Résumé annuel de l'activité sismique")
    
    df_annuel = df_filtered.groupby('Annee').agg({
        'Magnitude': ['count', 'mean', 'max', 'min', 'std'],
        'Profondeur': ['mean', 'min', 'max', 'std']
    })
    
    df_annuel.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_annuel.columns.values]
    
    # Renommer les colonnes pour plus de clarté
    column_mapping = {
        'Magnitude_count': 'Nombre',
        'Magnitude_mean': 'Mag_Moyenne',
        'Magnitude_max': 'Mag_Max',
        'Magnitude_min': 'Mag_Min',
        'Magnitude_std': 'Mag_Écart-type',
        'Profondeur_mean': 'Prof_Moyenne',
        'Profondeur_min': 'Prof_Min',
        'Profondeur_max': 'Prof_Max',
        'Profondeur_std': 'Prof_Écart-type'
    }
    
    df_annuel = df_annuel.rename(columns=column_mapping)
    
    # Arrondir les valeurs et convertir les types pour Arrow compatibility
    for col in df_annuel.columns:
        if 'Nombre' in col:
            df_annuel[col] = df_annuel[col].astype(int)
        else:
            df_annuel[col] = df_annuel[col].round(2).astype(float)
    
    st.dataframe(df_annuel, use_container_width=True)

def calculate_autocorr(series, max_lags=50):
    """Calculer l'autocorrélation manuellement"""
    n = len(series)
    series = np.array(series)
    mean = np.mean(series)
    c0 = np.dot(series - mean, series - mean) / float(n)
    
    acf = np.ones(max_lags + 1)
    for k in range(1, max_lags + 1):
        if k < n:
            c_k = np.dot(series[:-k] - mean, series[k:] - mean) / float(n)
            acf[k] = c_k / c0
        else:
            acf[k] = 0
    
    return acf

def analyser_cycles_periodicites(df_filtered):
    """Analyser les cycles et périodicités dans les données"""
    
    st.subheader("🔄 Analyse des Cycles et Périodicités")
    
    if len(df_filtered) < 100:
        st.warning(f"Cette analyse nécessite un grand nombre de données (>100). Actuellement : {len(df_filtered)} séismes.")
        return
    
    # Créer une série temporelle journalière
    ts_daily = df_filtered.groupby(df_filtered['Date'].dt.date).size()
    date_range = pd.date_range(start=ts_daily.index.min(), end=ts_daily.index.max())
    ts_daily = ts_daily.reindex(date_range, fill_value=0)
    
    # 1. Autocorrélation
    st.markdown("#### 📊 Analyse d'autocorrélation")
    
    max_lags = min(100, len(ts_daily) - 1)
    autocorr = calculate_autocorr(ts_daily.values, max_lags)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    lags = np.arange(len(autocorr))
    ax.plot(lags, autocorr, 'b-', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Seuil +/-0.05')
    ax.axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
    ax.set_xlim(0, max_lags)
    ax.set_title('Autocorrélation du nombre de séismes par jour')
    ax.set_xlabel('Délai (jours)')
    ax.set_ylabel('Autocorrélation')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.info("""
    **Interprétation de l'autocorrélation :**
    - Les pics dans le graphique suggèrent des périodicités possibles
    - Un pic à 7 jours indiquerait un cycle hebdomadaire
    - Un pic à 30 jours suggérerait un cycle mensuel
    """)
    
    # 2. Cycle hebdomadaire
    st.markdown("#### 📅 Analyse du cycle hebdomadaire")
    
    jour_semaine_counts = df_filtered.groupby('JourSemaine').size()
    jours_dict = {i: 0 for i in range(7)}
    for jour, count in jour_semaine_counts.items():
        jours_dict[jour] = count
    
    jours_semaine = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['lightblue' if i < 5 else 'lightcoral' for i in range(7)]
    bars = ax.bar(range(7), [jours_dict.get(i, 0) for i in range(7)], color=colors, alpha=0.8)
    
    # Test de significativité du cycle hebdomadaire
    observed_values = [jours_dict.get(i, 0) for i in range(7)]
    if sum(observed_values) > 0:
        chi2, p = stats.chisquare(observed_values)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title(f'Cycle hebdomadaire (Chi²={chi2:.2f}, p={p:.4f})')
        ax.set_xlabel('Jour de la semaine')
        ax.set_ylabel('Nombre de séismes')
        ax.set_xticks(range(7))
        ax.set_xticklabels(jours_semaine)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown(f"""
        <div class="statistical-result">
            <h4>🧮 Test de cycle hebdomadaire</h4>
            <p><strong>Chi² =</strong> {chi2:.2f}</p>
            <p><strong>p-value =</strong> {p:.4f}</p>
            <p><strong>Conclusion :</strong> {'Cycle hebdomadaire significatif détecté' if p < 0.05 else 'Pas de cycle hebdomadaire significatif'}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Fonction principale à appeler depuis l'application principale"""
    show_analyse_tendances()

if __name__ == "__main__":
    main()

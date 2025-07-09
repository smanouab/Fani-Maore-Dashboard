"""
Analyse Spatio-Temporelle des Séismes

Ce module fournit une analyse spatio-temporelle complète des données sismiques incluant :
- Cartographie interactive avec Folium
- Heatmaps de densité sismique
- Analyse des distances par rapport aux points d'intérêt (Fani Maoré, Mayotte)
- Filtres temporels, de magnitude et de profondeur
- Analyses par périodes temporelles multiples

Converti depuis Jupyter notebook vers Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import matplotlib as mpl
from math import radians, cos, sin, asin, sqrt
import sys
import os

# Supprimer les avertissements
warnings.filterwarnings('ignore')

# Ajouter utils au chemin pour le chargement des données
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Import conditionnel de Folium et streamlit-folium
FOLIUM_AVAILABLE = False
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    st.warning("⚠️ streamlit-folium non installé. Cartes interactives désactivées.")
    st.info("Pour installer : pip install streamlit-folium")

# Configuration des paramètres d'affichage
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

# CSS personnalisé pour le style
def apply_custom_css():
    st.markdown("""
    <style>
    .filter-label {
        font-weight: 600;
        color: #495057;
        margin-bottom: 5px;
        font-size: 13px;
    }
    
    /* Réduire l'espacement global */
    .stSlider {
        margin-bottom: 0px !important;
    }
    
    .stMultiSelect {
        margin-bottom: 5px !important;
    }
    
    .stRadio {
        margin-bottom: 5px !important;
    }
    
    /* Masquer les labels par défaut */
    .stSlider > label {
        display: none !important;
    }
    
    .stMultiSelect > label {
        display: none !important;
    }
    
    .stRadio > label {
        display: none !important;
    }
    
    /* Compacter les éléments */
    .element-container {
        margin-bottom: 0px !important;
    }


    /* Match the green from your intro-section border */
    .stSlider > div > div > div > div {
    background-color: #2ecc71 !important;
    }

    
    /* Réduire la taille des sliders */
    .stSlider > div {
        padding-top: 0px !important;
        padding-bottom: 5px !important;
    }
    
    /* Réduire la taille des sliders */
    .stSlider > div {
        padding-top: 0px !important;
        padding-bottom: 5px !important;
    }
    
    .map-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 0px;
        margin: 10px 0;
        background-color: white;
        width: 100%;
        margin-left: 0;
    }
    
    .filter-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 25px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-left: 4px solid #2ecc71;
    }
    
    .map-legend {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #ddd;
        margin: 10px 0;
        font-size: 12px;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin: 5px 0;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 8px;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

def show_analyse_spatio_temporelle():
    """Fonction principale pour afficher l'analyse spatio-temporelle - Version compacte"""
    
    # Appliquer le style personnalisé
    apply_custom_css()
    
    # Obtenir les données filtrées depuis l'état de session
    if 'filtered_df' not in st.session_state:
        st.error("❌ Données non disponibles. Veuillez retourner à la page d'accueil.")
        return
    
    df_original = st.session_state.filtered_df.copy()
    
    if len(df_original) == 0:
        st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés.")
        return
    
    # Section des filtres compacts (retourne les données filtrées)
    df_filtered = show_spatial_filters(df_original)
    
    # Vérifier si df_filtered est None (ajout de cette vérification)
    if df_filtered is None:
        st.error("❌ Erreur lors du filtrage des données.")
        return
    
    if len(df_filtered) == 0:
        st.info("💡 Ajustez les filtres pour voir les données.")
        return
    
    # Séparation discrète
    st.markdown("---")
    
    # Afficher l'analyse selon le type sélectionné dans les filtres
    analysis_type = st.session_state.get('selected_analysis_type', 'Distribution temporelle')
    
    if analysis_type == "Distribution temporelle":
        show_temporal_distribution(df_filtered)
    elif analysis_type == "Carte des séismes":
        if FOLIUM_AVAILABLE:
            show_interactive_map(df_filtered)
        else:
            show_fallback_map(df_filtered)
    elif analysis_type == "Analyse par magnitude":
        show_magnitude_analysis(df_filtered)
    elif analysis_type == "Corrélations":
        show_correlation_analysis(df_filtered)

def show_spatial_filters(df):
    """Afficher les filtres spécifiques à l'analyse spatiale - Version compacte"""
    
    with st.container():
        
        # Layout en colonnes pour compacter
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Filtre Années
            st.markdown('<div class="filter-label">Années:</div>', unsafe_allow_html=True)
            annee_min = int(df['Date_dt'].dt.year.min()) if 'Date_dt' in df.columns else 2018
            annee_max = int(df['Date_dt'].dt.year.max()) if 'Date_dt' in df.columns else 2025
            
            annees_range = st.slider(
                "Plage d'années",
                min_value=annee_min,
                max_value=annee_max,
                value=(annee_min, annee_max),
                key="annees_slider",
                label_visibility="collapsed"
            )
            st.caption(f"{annees_range[0]} – {annees_range[1]}")
            
            # Filtre Magnitude
            st.markdown('<div class="filter-label">Magnitude:</div>', unsafe_allow_html=True)
            mag_min = float(df['Magnitude'].min())
            mag_max = float(df['Magnitude'].max())
            
            magnitude_range = st.slider(
                "Plage de magnitude",
                min_value=mag_min,
                max_value=mag_max,
                value=(mag_min, mag_max),
                step=0.01,
                format="%.2f",
                key="mag_slider_simple",
                label_visibility="collapsed"
            )
            st.caption(f"{magnitude_range[0]:.2f} – {magnitude_range[1]:.2f}")
        
        with col2:
            # Filtre Profondeur
            st.markdown('<div class="filter-label">Profondeur:</div>', unsafe_allow_html=True)
            prof_min = float(df['Profondeur'].min())
            prof_max = float(df['Profondeur'].max())
            
            profondeur_range = st.slider(
                "Plage de profondeur",
                min_value=prof_min,
                max_value=prof_max,
                value=(prof_min, prof_max),
                step=1.0,
                format="%.1f",
                key="prof_slider_simple",
                label_visibility="collapsed"
            )
            st.caption(f"{profondeur_range[0]:.1f} – {profondeur_range[1]:.1f}")
            
            # Options d'analyse compactes
            st.markdown('<div class="filter-label">Type d\'analyse:</div>', unsafe_allow_html=True)
            
            analysis_type = st.radio(
                "Type d'analyse sélectionné",
                options=[
                    "Distribution temporelle",
                    "Carte des séismes", 
                    "Analyse par magnitude",
                    "Corrélations"
                ],
                index=0,
                key="analysis_type_radio",
                label_visibility="collapsed"
            )
        
        # Filtre Mois sur toute la largeur mais compact
        st.markdown('<div class="filter-label">Mois:</div>', unsafe_allow_html=True)
        mois_options = [
            "Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
            "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"
        ]
        
        selected_mois = st.multiselect(
            "Sélection des mois",
            options=mois_options,
            default=mois_options,
            key="mois_multiselect",
            label_visibility="collapsed"
        )
        
        # Bouton et résultat sur une ligne
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            apply_filters = st.button(
                "Appliquer",
                key="apply_filters_btn",
                use_container_width=True,
                type="primary"
            )
    
    # Appliquer les filtres
    df_filtered = apply_simple_filters(df, annees_range, selected_mois, magnitude_range, profondeur_range)
    
    # Stocker les données filtrées et le type d'analyse
    st.session_state.filtered_df = df_filtered
    st.session_state.selected_analysis_type = analysis_type
    
    # Afficher les résultats de manière compacte
    if len(df_filtered) == 0:
        st.error("❌ Aucune donnée")
    else:
        percentage = (len(df_filtered) / len(df)) * 100
        st.success(f"✅ {len(df_filtered):,} séismes ({percentage:.0f}%)")
    
    return df_filtered  # Important: s'assurer que la fonction retourne toujours un DataFrame

def apply_simple_filters(df, annees_range, selected_mois, magnitude_range, profondeur_range):
    """Appliquer les filtres de manière simple"""
    
    df_filtered = df.copy()
    
    # Filtre par années
    if 'Date_dt' in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered['Date_dt'].dt.year >= annees_range[0]) & 
            (df_filtered['Date_dt'].dt.year <= annees_range[1])
        ]
        
        # Filtre par mois (noms courts)
        if selected_mois:
            mois_mapping = {
                "Jan": 1, "Fév": 2, "Mar": 3, "Avr": 4,
                "Mai": 5, "Jun": 6, "Jul": 7, "Aoû": 8,
                "Sep": 9, "Oct": 10, "Nov": 11, "Déc": 12
            }
            selected_month_numbers = [mois_mapping[mois] for mois in selected_mois if mois in mois_mapping]
            if selected_month_numbers:
                df_filtered = df_filtered[df_filtered['Date_dt'].dt.month.isin(selected_month_numbers)]
    
    # Filtre par magnitude
    df_filtered = df_filtered[
        (df_filtered['Magnitude'] >= magnitude_range[0]) & 
        (df_filtered['Magnitude'] <= magnitude_range[1])
    ]
    
    # Filtre par profondeur
    df_filtered = df_filtered[
        (df_filtered['Profondeur'] >= profondeur_range[0]) & 
        (df_filtered['Profondeur'] <= profondeur_range[1])
    ]
    
    return df_filtered

def show_interactive_map(df):
    """Afficher la carte interactive avec toutes les fonctionnalités"""
    
    if not FOLIUM_AVAILABLE:
        st.error("❌ Folium non disponible pour les cartes interactives")
        show_fallback_map(df)
        return
    
    st.subheader("🗺️ Carte Interactive des Séismes")
    
    if len(df) == 0:
        st.warning("Aucune donnée à afficher sur la carte.")
        return
    
    # Informations sur les données affichées
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Séismes affichés", len(df))
    with col2:
        st.metric("Magnitude moyenne", f"{df['Magnitude'].mean():.2f}")
    with col3:
        st.metric("Profondeur moyenne", f"{df['Profondeur'].mean():.1f} km")
    with col4:
        if 'Date_dt' in df.columns:
            period_days = (df['Date_dt'].max() - df['Date_dt'].min()).days
            st.metric("Période", f"{period_days} jours")
        else:
            st.metric("Période", "N/A")
    
    # Créer la carte avec légende intégrée
    try:
        carte = creer_carte_seismes_complete(df)
        
        # Afficher la carte avec streamlit-folium - Pleine largeur
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        map_data = st_folium(carte, width="100%", height=600, returned_objects=["last_object_clicked"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Afficher les informations sur le dernier point cliqué
        if map_data["last_object_clicked"]:
            st.info(f"📍 Dernier point cliqué: {map_data['last_object_clicked']}")
            
    except Exception as e:
        st.error(f"Erreur lors de la création de la carte: {e}")
        st.warning("Affichage d'une carte simplifiée...")
        show_fallback_map(df)

def creer_carte_seismes_complete(df_filtered):
    """Créer une carte complète des séismes avec toutes les fonctionnalités"""
    
    if not FOLIUM_AVAILABLE:
        raise ImportError("Folium non disponible")
    
    # Coordonnées précises du volcan Fani Maoré
    fanimaore = {
        'nom': 'Fani Maoré',
        'lat': -12.80,  # 12° 48′ sud
        'lon': 45.467   # 45° 28′ est
    }
    
    # Coordonnées des principales îles et villes de Mayotte
    mayotte_points = [
        # Îles principales
        {'nom': 'Grande-Terre', 'lat': -12.7817, 'lon': 45.2269, 'type': 'île', 'color': 'green'},
        {'nom': 'Petite-Terre', 'lat': -12.7892, 'lon': 45.2804, 'type': 'île', 'color': 'green'},
        
        # Villes principales
        {'nom': 'Mamoudzou', 'lat': -12.7806, 'lon': 45.2278, 'type': 'ville', 'color': 'blue'},
        {'nom': 'Dzaoudzi', 'lat': -12.7878, 'lon': 45.2814, 'type': 'ville', 'color': 'blue'},
        {'nom': 'Koungou', 'lat': -12.7333, 'lon': 45.2000, 'type': 'ville', 'color': 'blue'},
        {'nom': 'Dembéni', 'lat': -12.8333, 'lon': 45.1833, 'type': 'ville', 'color': 'blue'},
        {'nom': 'Pamandzi', 'lat': -12.8014, 'lon': 45.2881, 'type': 'ville', 'color': 'blue'},
        
        # Points géographiques importants
        {'nom': 'Aéroport de Mayotte', 'lat': -12.8047, 'lon': 45.2808, 'type': 'infrastructure', 'color': 'purple'},
        {'nom': 'Port de Longoni', 'lat': -12.7167, 'lon': 45.1833, 'type': 'infrastructure', 'color': 'purple'},
        
        # Points géologiques d'intérêt
        {'nom': 'Mont Bénara', 'lat': -12.8167, 'lon': 45.1833, 'type': 'géologique', 'color': 'brown'},
        {'nom': 'Mont Choungui', 'lat': -12.9000, 'lon': 45.1167, 'type': 'géologique', 'color': 'brown'},
    ]
    
    # Créer une carte centrée entre Mayotte et Fani Maoré
    center_lat = (fanimaore['lat'] + mayotte_points[0]['lat']) / 2
    center_lon = (fanimaore['lon'] + mayotte_points[0]['lon']) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=8,
        tiles='CartoDB positron'
    )
    
    # Ajouter la heatmap
    heat_data = [[row['Latitude'], row['Longitude'], row['Magnitude']] 
                for _, row in df_filtered.iterrows()]
    
    HeatMap(heat_data, radius=15, blur=10).add_to(m)
    
    # Créer un cluster pour les séismes
    marker_cluster = MarkerCluster(name="Séismes").add_to(m)
    
    # Ajouter des marqueurs pour chaque séisme (échantillon)
    sample_size = min(100, len(df_filtered))  # Limiter pour la performance
    for _, row in df_filtered.sample(sample_size).iterrows():
        
        # Créer le popup
        popup_html = f"""
        <div style="width: 220px; font-family: Arial; font-size: 12px;">
            <h4 style="margin: 0 0 5px 0; color: #2c3e50;">🌍 Séisme</h4>
            <hr style="margin: 2px 0; border-color: #eee;">
            <p><b>⚡ Magnitude:</b> {row['Magnitude']:.2f}</p>
            <p><b>🕳️ Profondeur:</b> {row['Profondeur']:.1f} km</p>
            <p><b>📍 Coordonnées:</b> {row['Latitude']:.4f}, {row['Longitude']:.4f}</p>
        </div>
        """
        
        marker = folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=max(3, row['Magnitude'] * 1.5),
            color='red',
            fill=True,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Magnitude {row['Magnitude']:.1f}"
        )
        
        marker.add_to(marker_cluster)
    
    # Ajouter le marqueur du volcan Fani Maoré
    folium.Marker(
        location=[fanimaore['lat'], fanimaore['lon']],
        popup=f"""
        <div style="width: 200px; font-family: Arial; font-size: 12px;">
            <h4 style="margin: 0 0 5px 0; color: #c0392b;">🌋 {fanimaore['nom']}</h4>
            <hr style="margin: 2px 0; border-color: #eee;">
            <p><b>Type:</b> Volcan sous-marin actif</p>
            <p><b>Coordonnées:</b> {fanimaore['lat']:.4f}, {fanimaore['lon']:.4f}</p>
            <p><b>Statut:</b> Surveillance continue</p>
        </div>
        """,
        tooltip="Fani Maoré - Volcan sous-marin",
        icon=folium.Icon(color='red', icon='fire', prefix='fa')
    ).add_to(m)
    
    # Ajouter tous les points d'intérêt de Mayotte
    for point in mayotte_points:
        
        # Définir l'icône selon le type
        if point['type'] == 'île':
            icon_name = 'leaf'
            icon_prefix = 'fa'
        elif point['type'] == 'ville':
            icon_name = 'home'
            icon_prefix = 'fa'
        elif point['type'] == 'infrastructure':
            icon_name = 'cog'
            icon_prefix = 'fa'
        elif point['type'] == 'géologique':
            icon_name = 'mountain'
            icon_prefix = 'fa'
        else:
            icon_name = 'info-sign'
            icon_prefix = 'glyphicon'
        
        # Créer le popup détaillé
        popup_html = f"""
        <div style="width: 180px; font-family: Arial; font-size: 12px;">
            <h4 style="margin: 0 0 5px 0; color: {point['color']};">📍 {point['nom']}</h4>
            <hr style="margin: 2px 0; border-color: #eee;">
            <p><b>Type:</b> {point['type'].title()}</p>
            <p><b>Coordonnées:</b> {point['lat']:.4f}, {point['lon']:.4f}</p>
        </div>
        """
        
        folium.Marker(
            location=[point['lat'], point['lon']],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{point['nom']} ({point['type']})",
            icon=folium.Icon(color=point['color'], icon=icon_name, prefix=icon_prefix)
        ).add_to(m)
    
    # Ajouter les cercles de distance autour de Fani Maoré avec couleurs plus foncées
    distance_colors = [
        {'radius': 5000, 'color': '#CC0000', 'opacity': 0.8, 'fill_opacity': 0.2},   # 5 km - Rouge foncé
        {'radius': 10000, 'color': '#FF6600', 'opacity': 0.7, 'fill_opacity': 0.15}, # 10 km - Orange foncé  
        {'radius': 20000, 'color': '#CCCC00', 'opacity': 0.6, 'fill_opacity': 0.12}, # 20 km - Jaune foncé
        {'radius': 50000, 'color': '#008800', 'opacity': 0.5, 'fill_opacity': 0.08}  # 50 km - Vert foncé
    ]
    
    for circle_info in distance_colors:
        folium.Circle(
            location=[fanimaore['lat'], fanimaore['lon']],
            radius=circle_info['radius'],
            color=circle_info['color'],
            weight=2,
            opacity=circle_info['opacity'],
            fill=True,
            fill_color=circle_info['color'],
            fill_opacity=circle_info['fill_opacity'],
            popup=f"Rayon de {circle_info['radius']/1000:.0f} km autour de Fani Maoré",
            tooltip=f"Distance: {circle_info['radius']/1000:.0f} km"
        ).add_to(m)
    
    # Ajouter un contrôle de couches
    folium.LayerControl().add_to(m)
    
    # Ajouter une légende directement sur la carte avec fond foncé
    legend_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 220px; height: auto; 
                background-color: rgba(40, 40, 40, 0.95); border:2px solid #333; z-index:9999; 
                font-size:12px; padding: 15px; border-radius: 12px;
                box-shadow: 0 0 20px rgba(0,0,0,0.5); color: white;
                ">
    <h4 style="margin-top: 0px; text-align: center; color: #fff; border-bottom: 1px solid #555; padding-bottom: 8px;">Magnitude des séismes</h4>
    <div style="background: linear-gradient(to right, #800080, #0000FF, #00FFFF, #00FF00, #FFFF00, #FFA500, #FF0000); 
                height: 18px; border-radius: 9px; margin: 10px 0; border: 1px solid #555;"></div>
    <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 10px; color: #ccc;">
        <span><b>Faible</b></span>
        <span><b>Forte</b></span>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 15px; color: #fff;">
        <span>{df_filtered['Magnitude'].min():.1f}</span>
        <span>{df_filtered['Magnitude'].max():.1f}</span>
    </div>
    
    <h5 style="color: #fff; margin: 15px 0 8px 0; border-bottom: 1px solid #555; padding-bottom: 5px;">Points d'intérêt</h5>
    <div style="margin: 6px 0; color: #fff;"><span style="color: #FF4444;">🔴</span> Fani Maoré (volcan)</div>
    <div style="margin: 6px 0; color: #fff;"><span style="color: #44FF44;">🟢</span> Îles principales</div>
    <div style="margin: 6px 0; color: #fff;"><span style="color: #4444FF;">🔵</span> Villes</div>
    <div style="margin: 6px 0; color: #fff;"><span style="color: #AA44AA;">🟣</span> Infrastructures</div>
    <div style="margin: 6px 0; color: #fff;"><span style="color: #8B4513;">🟤</span> Points géologiques</div>
    
    <h5 style="color: #fff; margin: 15px 0 8px 0; border-bottom: 1px solid #555; padding-bottom: 5px;">Cercles de distance</h5>
    <div style="margin: 4px 0; color: #fff;"><span style="color: #FF0000; font-size: 16px;">●</span> 5 km (Fani Maoré)</div>
    <div style="margin: 4px 0; color: #fff;"><span style="color: #FFA500; font-size: 16px;">●</span> 10 km (Fani Maoré)</div>
    <div style="margin: 4px 0; color: #fff;"><span style="color: #FFFF00; font-size: 16px;">●</span> 20 km (Fani Maoré)</div>
    <div style="margin: 4px 0; color: #fff;"><span style="color: #90EE90; font-size: 16px;">●</span> 50 km (Fani Maoré)</div>
    
    <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #555; font-size: 11px; color: #ccc;">
        <div><b style="color: #fff;">Total séismes:</b> {len(df_filtered)}</div>
        <div><b style="color: #fff;">Profondeur:</b> {df_filtered['Profondeur'].min():.0f}-{df_filtered['Profondeur'].max():.0f} km</div>
    </div>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def show_fallback_map(df):
    """Carte de secours avec matplotlib si Folium échoue"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot des séismes
    scatter = ax.scatter(
        df['Longitude'], df['Latitude'], 
        c=df['Magnitude'], 
        cmap='plasma', 
        alpha=0.6,
        s=df['Magnitude'] * 20,  # Taille proportionnelle à la magnitude
        edgecolors='black',
        linewidths=0.5
    )
    
    # Ajouter la colorbar
    plt.colorbar(scatter, label='Magnitude', ax=ax)
    
    # Marquer Fani Maoré et Mayotte
    ax.plot(45.467, -12.80, 'r^', markersize=15, label='Fani Maoré (Volcan)')
    ax.plot(45.2269, -12.7817, 'gs', markersize=12, label='Grande-Terre (Mayotte)')
    
    ax.set_title(f'Carte des séismes (Alternative simple) - {len(df)} événements')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Améliorer l'apparence
    ax.set_facecolor('#f0f8ff')
    
    st.pyplot(fig)
    plt.close()

def haversine(lat1, lon1, lat2, lon2):
    """Calculer la distance entre deux points géographiques using formule haversine"""
    # Convertir degrés en radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Formule haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Rayon de la Terre en km
    return c * r

def show_temporal_distribution(df):
    """Afficher l'analyse de distribution temporelle"""
    
    st.subheader("📊 Distribution Temporelle des Séismes")
    
    if len(df) == 0:
        st.warning("Aucune donnée pour l'analyse temporelle.")
        return
    
    # Récupérer la période sélectionnée
    period = st.session_state.get('temporal_period', 'Mois')
    
    # Assurer la compatibilité des colonnes temporelles
    if 'Date_dt' not in df.columns and 'Date' in df.columns:
        df = df.copy()
        df['Date_dt'] = pd.to_datetime(df['Date'])
    
    # Ajouter les colonnes temporelles si nécessaires
    if 'Annee' not in df.columns:
        df['Annee'] = df['Date_dt'].dt.year
    if 'Mois' not in df.columns:
        df['Mois'] = df['Date_dt'].dt.month
    if 'Jour' not in df.columns:
        df['Jour'] = df['Date_dt'].dt.day
    if 'Heure' not in df.columns:
        df['Heure'] = df['Date_dt'].dt.hour
    if 'JourSemaine' not in df.columns:
        df['JourSemaine'] = df['Date_dt'].dt.dayofweek
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if period == 'Année':
        counts = df['Annee'].value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values, color='#3498db', alpha=0.8)
        ax.set_xlabel('Année')
        ax.set_title(f'Distribution des séismes par année ({len(df)} séismes)')
        
    elif period == 'Mois':
        mois_noms = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 
                     'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        counts = df['Mois'].value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values, color='#e74c3c', alpha=0.8)
        ax.set_xlabel('Mois')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(mois_noms, rotation=45)
        ax.set_title(f'Distribution des séismes par mois ({len(df)} séismes)')
        
    elif period == 'Jour':
        counts = df['Jour'].value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values, color='#2ecc71', alpha=0.8)
        ax.set_xlabel('Jour du mois')
        ax.set_title(f'Distribution des séismes par jour du mois ({len(df)} séismes)')
        
    elif period == 'Heure':
        counts = df['Heure'].value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values, color='#f39c12', alpha=0.8)
        ax.set_xlabel('Heure de la journée')
        ax.set_xticks(range(0, 24, 2))
        ax.set_title(f'Distribution des séismes par heure ({len(df)} séismes)')
        
    elif period == 'Semaine':
        jours = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
        counts = df['JourSemaine'].value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values, color='#9b59b6', alpha=0.8)
        ax.set_xlabel('Jour de la semaine')
        ax.set_xticks(range(7))
        ax.set_xticklabels(jours, rotation=45)
        ax.set_title(f'Distribution des séismes par jour de la semaine ({len(df)} séismes)')
    
    # Ajouter des annotations sur les barres
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Nombre de séismes')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def show_magnitude_analysis(df):
    """Afficher l'analyse par magnitude"""
    
    st.subheader("📈 Analyse par Magnitude")
    
    if len(df) == 0:
        st.warning("Aucune donnée pour l'analyse des magnitudes.")
        return
    
    # Distribution des magnitudes
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(df['Magnitude'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('Distribution des magnitudes')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Fréquence')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Statistiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Statistiques des magnitudes")
        st.write(f"**Moyenne:** {df['Magnitude'].mean():.2f}")
        st.write(f"**Médiane:** {df['Magnitude'].median():.2f}")
        st.write(f"**Min:** {df['Magnitude'].min():.2f}")
        st.write(f"**Max:** {df['Magnitude'].max():.2f}")
        st.write(f"**Écart-type:** {df['Magnitude'].std():.2f}")
    
    with col2:
        st.markdown("### 🎯 Répartition par catégorie")
        
        # Catégoriser les magnitudes
        bins = [0, 2.5, 4, 5, 6, 7, float('inf')]
        labels = ['Micro', 'Faible', 'Léger', 'Modéré', 'Fort', 'Majeur+']
        df_temp = df.copy()
        df_temp['MagnitudeCategorie'] = pd.cut(df_temp['Magnitude'], bins=bins, labels=labels)
        cat_counts = df_temp['MagnitudeCategorie'].value_counts().sort_index()
        
        for category, count in cat_counts.items():
            percentage = count / len(df) * 100
            st.write(f"**{category}:** {count} ({percentage:.1f}%)")

def show_correlation_analysis(df):
    """Afficher l'analyse des corrélations"""
    
    st.subheader("🔗 Analyse des Corrélations")
    
    if len(df) == 0:
        st.warning("Aucune donnée pour l'analyse des corrélations.")
        return
    
    # Sélectionner les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Pas assez de colonnes numériques pour l'analyse des corrélations.")
        return
    
    # Calculer la matrice de corrélation
    corr_matrix = df[numeric_cols].corr()
    
    # Afficher la matrice
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                fmt='.2f', linewidths=0.5, ax=ax, center=0)
    
    ax.set_title('Matrice de corrélation')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def show_spatial_statistics(df):
    """Afficher les statistiques spatiales"""
    
    st.subheader("📋 Statistiques Spatiales")
    
    if len(df) == 0:
        st.warning("Aucune donnée pour les statistiques spatiales.")
        return
    
    # Coordonnées de référence
    fanimaore_lat, fanimaore_lon = -12.80, 45.467
    mayotte_lat, mayotte_lon = -12.7817, 45.2269
    
    # Calculer les distances pour tous les séismes
    df_temp = df.copy()
    df_temp['Distance_Fanimaore'] = df_temp.apply(
        lambda row: haversine(row['Latitude'], row['Longitude'], fanimaore_lat, fanimaore_lon), 
        axis=1
    )
    df_temp['Distance_Mayotte'] = df_temp.apply(
        lambda row: haversine(row['Latitude'], row['Longitude'], mayotte_lat, mayotte_lon), 
        axis=1
    )
    
    # Statistiques générales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌍 Étendue géographique")
        st.write(f"**Latitude min:** {df['Latitude'].min():.4f}°")
        st.write(f"**Latitude max:** {df['Latitude'].max():.4f}°")
        st.write(f"**Longitude min:** {df['Longitude'].min():.4f}°")
        st.write(f"**Longitude max:** {df['Longitude'].max():.4f}°")
        
        # Calculer l'aire approximative
        lat_range = df['Latitude'].max() - df['Latitude'].min()
        lon_range = df['Longitude'].max() - df['Longitude'].min()
        # Approximation grossière en km²
        area_approx = lat_range * 111 * lon_range * 111 * np.cos(np.radians(df['Latitude'].mean()))
        st.write(f"**Zone approximative:** {area_approx:.0f} km²")
    
    with col2:
        st.markdown("### 🌋 Distances à Fani Maoré")
        st.write(f"**Distance min:** {df_temp['Distance_Fanimaore'].min():.1f} km")
        st.write(f"**Distance max:** {df_temp['Distance_Fanimaore'].max():.1f} km")
        st.write(f"**Distance moyenne:** {df_temp['Distance_Fanimaore'].mean():.1f} km")
        st.write(f"**Distance médiane:** {df_temp['Distance_Fanimaore'].median():.1f} km")
        
        # Répartition par zones
        proche = len(df_temp[df_temp['Distance_Fanimaore'] <= 10])
        moyen = len(df_temp[(df_temp['Distance_Fanimaore'] > 10) & (df_temp['Distance_Fanimaore'] <= 50)])
        loin = len(df_temp[df_temp['Distance_Fanimaore'] > 50])
        
        st.write(f"**< 10 km:** {proche} ({proche/len(df)*100:.1f}%)")
        st.write(f"**10-50 km:** {moyen} ({moyen/len(df)*100:.1f}%)")
        st.write(f"**> 50 km:** {loin} ({loin/len(df)*100:.1f}%)")
    
    with col3:
        st.markdown("### 🏝️ Distances à Mayotte")
        st.write(f"**Distance min:** {df_temp['Distance_Mayotte'].min():.1f} km")
        st.write(f"**Distance max:** {df_temp['Distance_Mayotte'].max():.1f} km")
        st.write(f"**Distance moyenne:** {df_temp['Distance_Mayotte'].mean():.1f} km")
        st.write(f"**Distance médiane:** {df_temp['Distance_Mayotte'].median():.1f} km")
        
        # Répartition par zones
        proche_m = len(df_temp[df_temp['Distance_Mayotte'] <= 20])
        moyen_m = len(df_temp[(df_temp['Distance_Mayotte'] > 20) & (df_temp['Distance_Mayotte'] <= 100)])
        loin_m = len(df_temp[df_temp['Distance_Mayotte'] > 100])
        
        st.write(f"**< 20 km:** {proche_m} ({proche_m/len(df)*100:.1f}%)")
        st.write(f"**20-100 km:** {moyen_m} ({moyen_m/len(df)*100:.1f}%)")
        st.write(f"**> 100 km:** {loin_m} ({loin_m/len(df)*100:.1f}%)")
    
    # Graphique des distributions de distance
    st.subheader("📊 Distribution des distances")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distribution des distances à Fani Maoré
    ax1.hist(df_temp['Distance_Fanimaore'], bins=30, alpha=0.7, color='red', edgecolor='black')
    ax1.set_title('Distribution des distances à Fani Maoré')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Nombre de séismes')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(df_temp['Distance_Fanimaore'].mean(), color='darkred', linestyle='--', 
               label=f'Moyenne: {df_temp["Distance_Fanimaore"].mean():.1f} km')
    ax1.legend()
    
    # Distribution des distances à Mayotte
    ax2.hist(df_temp['Distance_Mayotte'], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Distribution des distances à Mayotte')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Nombre de séismes')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(df_temp['Distance_Mayotte'].mean(), color='darkgreen', linestyle='--',
               label=f'Moyenne: {df_temp["Distance_Mayotte"].mean():.1f} km')
    ax2.legend()
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Tableau des séismes les plus proches
    st.subheader("🎯 Séismes les plus proches des points d'intérêt")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌋 Plus proches de Fani Maoré**")
        if 'Date_dt' in df_temp.columns:
            closest_fani = df_temp.nsmallest(5, 'Distance_Fanimaore')[
                ['Date_dt', 'Magnitude', 'Distance_Fanimaore']
            ].copy()
            closest_fani['Date_dt'] = closest_fani['Date_dt'].dt.strftime('%d/%m/%Y %H:%M')
            closest_fani.columns = ['Date', 'Magnitude', 'Distance (km)']
        else:
            closest_fani = df_temp.nsmallest(5, 'Distance_Fanimaore')[
                ['Magnitude', 'Distance_Fanimaore']
            ].copy()
            closest_fani.columns = ['Magnitude', 'Distance (km)']
        st.dataframe(closest_fani, hide_index=True)
    
    with col2:
        st.markdown("**🏝️ Plus proches de Mayotte**")
        if 'Date_dt' in df_temp.columns:
            closest_mayo = df_temp.nsmallest(5, 'Distance_Mayotte')[
                ['Date_dt', 'Magnitude', 'Distance_Mayotte']
            ].copy()
            closest_mayo['Date_dt'] = closest_mayo['Date_dt'].dt.strftime('%d/%m/%Y %H:%M')
            closest_mayo.columns = ['Date', 'Magnitude', 'Distance (km)']
        else:
            closest_mayo = df_temp.nsmallest(5, 'Distance_Mayotte')[
                ['Magnitude', 'Distance_Mayotte']
            ].copy()
            closest_mayo.columns = ['Magnitude', 'Distance (km)']
        st.dataframe(closest_mayo, hide_index=True)

# Fonction principale qui peut être appelée depuis app.py
def main():
    """Fonction principale à appeler depuis l'application principale"""
    show_analyse_spatio_temporelle()

if __name__ == "__main__":
    main()

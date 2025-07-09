"""
Analyse Générale - General Analysis Dashboard

This module provides general seismic analysis including:
- Temporal filtering by year and month
- Annual distribution with trend analysis
- Monthly and daily breakdowns
- Statistical summaries and explanations

Converted from Jupyter notebook to Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Add utils to path for data loading
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))


# Configure matplotlib for Streamlit
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.2)
sns.set_palette("deep")

# Colors from your original design
COLORS = {
    'primary': '#3498db',   # Blue
    'secondary': '#e74c3c',  # Red
    'accent': '#2ecc71',    # Green
    'neutral': '#95a5a6',   # Gray
    'dark': '#2c3e50',      # Dark blue
    'light': '#ecf0f1'      # Light gray
}

# Custom CSS for styling (converted from your original CSS)
def apply_custom_css():
    st.markdown("""
    <style>
    .metric-card {
        background-color: #ecf0f1;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid #3498db;
    }
    
    .success-message {
        background-color: #eafaf1;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 5px solid #2ecc71;
    }
    
    .warning-message {
        background-color: #fdecea;
        padding: 10px;
        border-radius: 5px;
        color: #e74c3c;
        font-weight: bold;
    }
    
    .info-section {
        background-color: #eef5fb;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        border-left: 5px solid #3498db;
    }
    
    .explanation-section {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        border-left: 5px solid #3498db;
    }
    </style>
    """, unsafe_allow_html=True)

def show_analyse_generale():
    """Main function to display the general analysis dashboard"""
    
    # Apply custom styling
    apply_custom_css()
    
    # Header
    st.header("📊 Analyse Générale des Séismes")
    st.markdown("*Analyse temporelle complète avec filtrage et calcul de tendances*")
    
    # Get filtered data from session state (passed from main app)
    if 'filtered_df' not in st.session_state:
        st.error("❌ Données non disponibles. Veuillez retourner à la page d'accueil.")
        return
    
    df = st.session_state.filtered_df
    
    
    if len(df) == 0:
        st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés.")
        return
    
    # Get all required years (from your original logic)
    annees_presentes = sorted(df['Annee'].unique())
    annees_requises = list(range(min(min(annees_presentes), 2018), max(annees_presentes) + 1))
    
    # Period Analysis Section
    st.subheader("🔍 Analyse par Période")
    
    # Create columns for filters
    col1, col2 = st.columns(2)
    
    with col1:
        # Year selector - include all required years
        annee_selectionnee = st.selectbox(
            "📅 Sélectionner l'année:",
            options=annees_requises,
            index=len(annees_requises)-1 if annees_requises else 0,
            help="Choisissez l'année à analyser"
        )
    
    with col2:
        # Month selector
        mois_noms = ["Tous", "Janvier", "Février", "Mars", "Avril", "Mai", "Juin", 
                    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
        
        mois_selectionne = st.selectbox(
            "📆 Sélectionner le mois:",
            options=list(range(len(mois_noms))),
            format_func=lambda x: mois_noms[x],
            index=0,
            help="Choisissez le mois à analyser (Tous = toute l'année)"
        )
    
    # Filter data based on selections
    mask_annee = df['Annee'] == annee_selectionnee
    df_filtered = df[mask_annee].copy()
    
    # Apply month filter if not "Tous"
    if mois_selectionne > 0:
        df_filtered = df_filtered[df_filtered['Mois'] == mois_selectionne]
    
    # Display results for selected period
    show_period_results(df_filtered, annee_selectionnee, mois_selectionne, mois_noms, annees_presentes)
    
    # Annual Overview Section
    st.subheader("📈 Vue d'Ensemble Annuelle")
    show_annual_overview(df, annees_requises, annees_presentes)

def show_period_results(df_filtered, annee_selectionnee, mois_selectionne, mois_noms, annees_presentes):
    """Display results for the selected period"""
    
    nb_seismes = len(df_filtered)
    
    # Check if selected year has data
    if annee_selectionnee not in annees_presentes:
        st.markdown(f"""
        <div class="warning-message">
            Aucune donnée disponible pour l'année {annee_selectionnee}.
        </div>
        """, unsafe_allow_html=True)
        return
    elif nb_seismes == 0:
        st.markdown("""
        <div class="warning-message">
            Aucun séisme ne correspond aux critères de filtrage.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display summary information
    period_text = f"du {df_filtered['Date_dt'].min().strftime('%d/%m/%Y')} au {df_filtered['Date_dt'].max().strftime('%d/%m/%Y')}"
    
    st.markdown(f"""
    <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #28a745;">
        <span style='font-size:1.3em; font-weight:bold; color: #155724;'>{nb_seismes:,} séismes</span> correspondent aux critères sélectionnés<br>
        <span style='color: #155724; font-weight: bold;'>Période: {period_text}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if mois_selectionne == 0:  # All months selected
        show_monthly_distribution(df_filtered, annee_selectionnee, ax)
    else:  # Specific month selected
        show_daily_distribution(df_filtered, annee_selectionnee, mois_selectionne, mois_noms, ax)
    
    # Display the plot
    st.pyplot(fig)
    plt.close()

def show_monthly_distribution(df_filtered, annee_selectionnee, ax):
    """Show monthly distribution for the selected year"""
    
    # Aggregate by month
    monthly_counts = df_filtered.groupby('Mois').size()
    
    # Create index for all months
    all_months = pd.Series(range(1, 13))
    monthly_counts = monthly_counts.reindex(all_months).fillna(0)
    
    # Month names for labels
    month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 
                   'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
    
    # Create the chart
    bars = ax.bar(range(1, 13), monthly_counts.values, color=COLORS['primary'], alpha=0.8)
    
    # Add annotations on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + max(monthly_counts.values) * 0.02,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10
            )
    
    ax.set_title(f"Nombre de séismes par mois en {annee_selectionnee}", fontsize=14, pad=20)
    ax.set_xlabel("Mois", fontsize=12)
    ax.set_ylabel("Nombre de séismes", fontsize=12)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(monthly_counts.values) * 1.15 if max(monthly_counts.values) > 0 else 1)
    
    # Style the chart
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_facecolor('#f8f9fa')

def show_daily_distribution(df_filtered, annee_selectionnee, mois_selectionne, mois_noms, ax):
    """Show daily distribution for the selected month"""
    
    # Aggregate by day of month
    daily_counts = df_filtered.groupby('Jour').size()
    
    # Number of days in this month
    month_lengths = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if annee_selectionnee % 4 == 0 and (annee_selectionnee % 100 != 0 or annee_selectionnee % 400 == 0):
        month_lengths[2] = 29  # Leap year February
    
    days_in_month = month_lengths[mois_selectionne]
    
    # Create index for all days of the month
    all_days = pd.Series(range(1, days_in_month + 1))
    daily_counts = daily_counts.reindex(all_days).fillna(0)
    
    # Create the chart
    bars = ax.bar(range(1, days_in_month + 1), daily_counts.values, color=COLORS['secondary'], alpha=0.8)
    
    # Add annotations on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + max(daily_counts.values) * 0.02 if max(daily_counts.values) > 0 else 0.5,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10
            )
    
    ax.set_title(f"Nombre de séismes par jour en {mois_noms[mois_selectionne]} {annee_selectionnee}", 
                fontsize=14, pad=20)
    ax.set_xlabel("Jour du mois", fontsize=12)
    ax.set_ylabel("Nombre de séismes", fontsize=12)
    ax.set_xticks(range(1, days_in_month + 1, 2))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(daily_counts.values) * 1.15 if max(daily_counts.values) > 0 else 1)
    
    # Style the chart
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_facecolor('#f8f9fa')

def show_annual_overview(df, annees_requises, annees_presentes):
    """Show annual overview with trend analysis"""
    
    # Calculate earthquakes per year in existing data
    annual_counts = df.groupby('Annee').size()
    
    # Create DataFrame with all required years, even those without data
    all_years_df = pd.DataFrame(index=annees_requises)
    all_years_df['count'] = annual_counts
    all_years_df['count'] = all_years_df['count'].fillna(0)
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use different colors to distinguish years with/without data
    colors = []
    for year in all_years_df.index:
        if year in annees_presentes:
            colors.append(COLORS['dark'])  # Normal color for years with data
        else:
            colors.append(COLORS['neutral'])  # Gray color for years without data
    
    # Create bars
    bars = ax.bar(all_years_df.index, all_years_df['count'].values, color=colors, alpha=0.8)
    
    # Add annotations on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + max(all_years_df['count'].values) * 0.02,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold'
            )
    
    ax.set_title("Nombre total de séismes par année", fontsize=16, pad=20)
    ax.set_xlabel("Année", fontsize=14)
    ax.set_ylabel("Nombre de séismes", fontsize=14)
    ax.tick_params(axis='x', rotation=45 if len(all_years_df.index) > 10 else 0)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(all_years_df['count'].values) * 1.15 if max(all_years_df['count'].values) > 0 else 10)
    
    # Trend analysis
    years_with_data = np.array([year for year in all_years_df.index if year in annees_presentes])
    counts_with_data = np.array([all_years_df.loc[year, 'count'] for year in years_with_data])
    
    r_squared = None
    if len(years_with_data) > 1:
        # Calculate trend with numpy polyfit (linear regression)
        z = np.polyfit(years_with_data, counts_with_data, 1)
        p = np.poly1d(z)
        
        # Calculate correlation coefficient to evaluate trend quality
        correlation = np.corrcoef(years_with_data, counts_with_data)[0, 1]
        r_squared = correlation**2
        
        # Calculate trend values for all years
        trend_values = [p(year) for year in all_years_df.index]
        
        # Plot trend line
        ax.plot(all_years_df.index, trend_values, 
                linestyle='--', color=COLORS['accent'], linewidth=2, 
                label=f"Tendance: {z[0]:.1f} séismes/an")
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Determine trend reliability
        if r_squared < 0.3:
            reliability = " (fiabilité faible)"
        elif r_squared < 0.7:
            reliability = " (fiabilité moyenne)"
        else:
            reliability = " (fiabilité élevée)"
        
        # Display trend coefficient in the chart
        trend_text = f"Tendance: {z[0]:.2f} séismes/an{reliability}"
        if z[0] > 0:
            trend_text += "\n(augmentation)"
        else:
            trend_text += "\n(diminution)"
        trend_text += f"\nR² = {r_squared:.2f}"
            
        ax.text(
            all_years_df.index[len(all_years_df.index)//2],
            max(all_years_df['count'].values) * 0.9,
            trend_text,
            ha='center', va='center', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['accent'], boxstyle='round,pad=0.5')
        )
    
    plt.tight_layout()
    
    # Style the chart
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_facecolor('#f8f9fa')
    
    # Display the plot
    st.pyplot(fig)
    plt.close()
    
    # Show summary table and explanations
    show_annual_summary_table(all_years_df, annees_presentes)
    
    # Show trend explanation if available
    if len(years_with_data) > 1 and r_squared is not None:
        show_trend_explanation(z[0], r_squared, len(years_with_data))

def show_annual_summary_table(all_years_df, annees_presentes):
    """Display annual summary table - VERSION CORRIGÉE"""
    
    total_seismes = int(all_years_df['count'].sum())
    
    st.subheader("📋 Récapitulatif par année")
    
    # Créer les données du tableau
    summary_data = []
    for year, row in all_years_df.iterrows():
        count = int(row['count'])
        percentage = count / total_seismes * 100 if total_seismes > 0 else 0
        
        # Statut des données
        status = "✅ Présent" if year in annees_presentes else "❌ Absent"
        
        summary_data.append({
            "Année": str(year),  # ← Convertir en string explicitement
            "Nombre de séismes": f"{count:,}",
            "Pourcentage": f"{percentage:.1f}%",
            "Statut": status
        })
    
    # Ajouter la ligne de total
    summary_data.append({
        "Année": "🔢 TOTAL",  # Maintenant c'est cohérent (tout est string)
        "Nombre de séismes": f"{total_seismes:,}",
        "Pourcentage": "100.0%",
        "Statut": "—"
    })
    
    # Créer le DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # SOLUTION 1: Forcer tous les types en string pour éviter l'erreur PyArrow
    summary_df = summary_df.astype(str)
    
    # Afficher le tableau avec style
    st.dataframe(
        summary_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Année": st.column_config.TextColumn("📅 Année", width="small"),
            "Nombre de séismes": st.column_config.TextColumn("📊 Nombre de séismes", width="medium"),
            "Pourcentage": st.column_config.TextColumn("📈 Pourcentage", width="small"),
            "Statut": st.column_config.TextColumn("📋 Statut des données", width="medium")
        }
    )
    
    # Informations complémentaires (reste identique)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"📊 **Total**: {total_seismes:,} séismes")
    
    with col2:
        years_span = max(annees_presentes) - min(annees_presentes) + 1
        st.info(f"📅 **Période**: {years_span} années")
    
    with col3:
        avg_per_year = total_seismes / len(annees_presentes) if annees_presentes else 0
        st.info(f"📈 **Moyenne**: {avg_per_year:.0f}/an")

def show_trend_explanation(slope, r_squared, num_points):
    """Display trend analysis explanation - VERSION STREAMLIT NATIVE"""
    
    st.subheader("📊 Explication du calcul de tendance")
    
    # Determine reliability
    if r_squared < 0.3:
        reliability_text = "Faible"
        reliability_emoji = "🔴"
    elif r_squared < 0.7:
        reliability_text = "Moyenne"
        reliability_emoji = "🟡"
    else:
        reliability_text = "Élevée"
        reliability_emoji = "🟢"
    
    # Utiliser les composants Streamlit natifs
    st.info(f"""
    **📈 Méthode :** Régression linéaire (moindres carrés) sur les années disposant de données.
    
    **📊 Interprétation :** La pente de {slope:.2f} indique {'une augmentation' if slope > 0 else 'une diminution'} moyenne de {abs(slope):.2f} séismes par an.
    
    **🎯 Fiabilité (R²) :** {r_squared:.4f} - {reliability_emoji} {reliability_text}
    
    **📋 Données utilisées :** {num_points} années
    """)
    
    st.caption("💡 Le coefficient R² mesure la qualité de l'ajustement du modèle. Une valeur proche de 1 indique une forte corrélation.")

# Main function that can be called from app.py
def main():
    """Main function to be called from the main app"""
    show_analyse_generale()

if __name__ == "__main__":
    main()
# 🌋 Fani Maoré - Surveillance Sismique Dashboard

Solym M. Manou-Abi, Said Said Hachim, Sophie Dabo-Niang, Jean-Berky Nguala. A Comparative Study for Clustering Methods using KL Divergence, Rao distance, and Bregman divergence with Fani Maoré marine volcano earthquake Data. (2024) Under revision.
## 📋 Description

Dashboard interactif dédié à l'analyse de l'activité sismique du **volcan sous-marin Fani Maoré** situé à Mayotte. Cette application permet d'analyser, visualiser et comprendre les patterns sismiques de ce système volcanique actif depuis 2018.

## 🚀 Accès Direct à l'Application

**[👉 Lancer le Dashboard Fani Maoré](https://fani-maore-seismic-data.streamlit.app/)** 

## 🌋 À propos du Volcan Fani Maoré

- **📍 Localisation**: 50 km à l'est de Mayotte, Océan Indien
- **🌊 Type**: Volcan sous-marin basaltique  
- **📅 Activité**: Essaim sismique depuis mai 2018
- **📡 Surveillance**: Réseau REVOSIMA (15 stations sismiques)
- **📊 Données**: 15,407+ événements sismiques analysés
- **🌊 Profondeur**: Édifice volcanique à 3500m sous le niveau de la mer

## ✨ Fonctionnalités du Dashboard

### 📊 Analyses Disponibles
- **🏠 Vue d'Ensemble** : Métriques clés et contexte volcanique
- **📊 Analyse Générale** : Statistiques globales et distributions
- **🔬 Analyse des Caractéristiques** : Magnitudes, profondeurs, énergie libérée
- **📈 Analyse des Tendances** : Patterns temporels et cycles saisonniers
- **🗺️ Analyse Spatio-Temporelle** : Cartographie interactive de la zone

### 🔍 Filtres Interactifs
- **📅 Période d'analyse**: 2018-2024
- **⚡ Plage de magnitude**: 0.1 - 5.8
- **🕳️ Profondeur**: 0 - 50+ km
- **🌿 Filtres saisonniers**: Par mois, saison, jour de la semaine

### 📈 Visualisations Avancées
- **🗺️ Cartes interactives** avec localisation précise des séismes
- **📊 Heatmaps** de densité et d'activité temporelle
- **📈 Graphiques temporels** avec tendances et régressions
- **⚡ Distribution des énergies** et potentiel destructeur
- **🔄 Analyse des cycles** et périodicités

## 🚀 Installation Locale (Développeurs)

### Prérequis
- Python 3.8 ou plus récent
- pip (gestionnaire de packages Python)

### Installation
```bash
# Cloner le repository
git clone https://github.com/YOUR-USERNAME/seismic-dashboard.git
cd seismic-dashboard

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app_minimal.py
```

## 📁 Structure du Projet

```
seismic-dashboard/
├── app_minimal.py                   # Application principale Fani Maoré
├── requirements.txt                 # Dépendances Python
├── README.md                       # Documentation
├── pages/                          # Modules d'analyse
│   ├── Analyse_generale.py         # Analyse générale
│   ├── Analyse_caracteristiques_sismique.py # Caractéristiques physiques
│   ├── Analyse_tendances_sismique.py # Tendances temporelles
│   └── Analyse_spatio_temporelle.py # Analyse spatiale
├── utils/                          # Fonctions utilitaires
│   ├── data_loader.py             # Chargement des données
│   ├── data_collector.py          # Collecte de données
│   └── settings.py                # Configuration
├── data/                          # Données sismiques
│   ├── NewDataseisme.csv          # Base de données Fani Maoré
│   ├── backup/                    # Sauvegardes
│   └── logs/                      # Journaux
└── config/                        # Configuration
    └── settings.py                # Paramètres
```

## 📊 Données Sismiques

### Source des Données
- **Réseau REVOSIMA** (Réseau de surveillance VOlcanique et SIsmique de MAyotte)
- **IPGP** (Institut de Physique du Globe de Paris)
- **BRGM** (Bureau de Recherches Géologiques et Minières)

### Format des Données: `NewDataseisme.csv`
```csv
Date,Magnitude,Latitude,Longitude,Profondeur,Annee,Mois,Jour,Heure
10/05/2018 14:15,4.2,-12.8456,45.5123,25.3,2018,5,10,14
```

**Colonnes principales:**
- `Date` : Date et heure UTC du séisme
- `Magnitude` : Magnitude locale (ML)
- `Latitude/Longitude` : Coordonnées géographiques
- `Profondeur` : Profondeur hypocentrale (km)

## 🗺️ Zone d'Étude

### Contexte Géologique
- **Zone de surveillance**: Région est de Mayotte
- **Système volcanique**: Fani Maoré (découvert en 2019)
- **Contexte tectonique**: Rift de Madagascar - Comores
- **Phénomène**: Plus grand essaim sismique enregistré à Mayotte

### Points d'Intérêt
- **🌋 Volcan Fani Maoré**: Édifice principal
- **🏝️ Mayotte**: Référence géographique
- **📡 Stations REVOSIMA**: Réseau de surveillance

## 🔧 Fonctionnalités Techniques

### Calculs Scientifiques
- **⚡ Énergie libérée**: E = 10^(1.5×M + 4.8)
- **💥 Potentiel destructeur**: Magnitude × (1 + 70/profondeur)
- **📊 Loi de Gutenberg-Richter**: Distribution magnitude-fréquence
- **📈 Tests statistiques**: Chi², corrélations, tendances

### Visualisations Cartographiques
- **🗺️ Cartes Folium interactives**
- **🔥 Heatmaps de densité**
- **📍 Clustering des événements**
- **📏 Calcul de distances** depuis Fani Maoré

## 📈 Analyses Disponibles

### 1. 🏠 Vue d'Ensemble
- Métriques clés du volcan Fani Maoré
- Contexte géologique et volcanique
- Statistiques générales de l'activité

### 2. 📊 Analyse Générale  
- Distribution globale des événements
- Évolution temporelle de l'activité
- Comparaisons inter-annuelles

### 3. 🔬 Caractéristiques Sismiques
- Distribution des magnitudes et profondeurs
- Relation magnitude-profondeur
- Calcul du potentiel destructeur
- Analyse de l'énergie libérée

### 4. 📈 Tendances Temporelles
- Cycles saisonniers et hebdomadaires
- Tendances à long terme
- Analyse des périodicités
- Tests de significativité statistique

### 5. 🗺️ Analyse Spatio-Temporelle
- Cartographie de l'activité sismique
- Évolution spatiale dans le temps
- Distance aux points d'intérêt
- Clustering géographique

## 🌊 Impact et Surveillance

### Importance Scientifique
- **Premier volcan sous-marin** surveillé en temps réel à Mayotte
- **Phénomène géologique majeur** de l'océan Indien occidental
- **Déformation du sol** de plusieurs centimètres observée
- **Création d'un nouvel édifice volcanique** de 800m de hauteur

### Surveillance Continue
- **Surveillance 24h/24, 7j/7** par le réseau REVOSIMA
- **Alertes automatiques** pour les événements significatifs
- **Modélisation** de l'évolution du système volcanique
- **Évaluation des risques** pour la population de Mayotte

## 🆘 Support et Documentation

### En cas de problème
1. Vérifiez la connectivité internet
2. Consultez les logs d'erreur dans la console
3. Redémarrez l'application
4. Contactez l'équipe de développement

### Ressources Scientifiques
- [REVOSIMA - IPGP](http://revosima.ipgp.fr/)
- [BRGM Mayotte](https://www.brgm.fr/)
- [Observatoire Volcanologique du Piton de la Fournaise](http://www.ipgp.fr/fr/ovpf)

## 📝 Historique des Versions

- **v1.0** (2024): Version initiale Fani Maoré
- **v1.1** (2024): Ajout analyse des caractéristiques
- **v1.2** (2024): Intégration cartographie avancée
- **v1.3** (2025): Optimisation pour déploiement cloud

---

**🌋 Développé pour la surveillance du volcan Fani Maoré - Mayotte, Océan Indien**

*Application scientifique dédiée à la compréhension de l'activité sismique volcanique sous-marine*

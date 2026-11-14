# 🌋 Fani Maoré – Seismic Monitoring Dashboard

**Scientific reference:**  
Solym M. Manou-Abi and Said Said Hachim. *A Data-Driven and Scalable Computational Framework for Spatio-Temporal Modeling of Fani Maoré Seismicity*. Preprint, 2026.

## 📋 Description

An interactive Python-based dashboard dedicated to the analysis of seismic activity associated with the **Fani Maoré submarine volcano**, located off the coast of Mayotte in the Indian Ocean.

This application provides tools to explore, analyze, and visualize seismic patterns associated with this active volcanic system, which has been monitored since 2018.

## 🚀 Access the Dashboard

**[👉 Launch the Fani Maoré Seismic Monitoring Dashboard](https://seismic-dashboard-final.streamlit.app/)**

## 🌋 About the Fani Maoré Volcano

- **📍 Location:** Approximately 50 km east of Mayotte, Indian Ocean
- **🌊 Type:** Submarine basaltic volcano
- **📅 Activity:** Seismic swarm ongoing since May 2018
- **📡 Monitoring:** REVOSIMA monitoring network (15 seismic stations)
- **📊 Dataset:** More than 15,407 seismic events analyzed
- **🌊 Depth:** Volcanic edifice located approximately 3,500 m below sea level

## ✨ Dashboard Features

### 📊 Available Analyses

- **🏠 Overview:** Key indicators and geological context of the volcanic system.
- **📊 General Analysis:** Global statistics and distributions of seismic events.
- **🔬 Seismic Characteristics:** Analysis of magnitudes, depths, and released energy.
- **📈 Trend Analysis:** Temporal patterns, long-term trends, and seasonal cycles.
- **🗺️ Spatio-Temporal Analysis:** Interactive mapping and exploration of seismic activity across the study area.

### 🔍 Interactive Filters

- **📅 Analysis period:** 2018–2024.
- **⚡ Magnitude range:** 0.1–5.8.
- **🕳️ Depth range:** 0–50+ km.
- **🌿 Seasonal filters:** Month, season, and day of the week.

### 📈 Advanced Visualizations

- **🗺️ Interactive maps** displaying the geographical locations of seismic events.
- **📊 Heatmaps** illustrating seismic density and temporal activity.
- **📈 Time-series plots** with trend analysis and regression models.
- **⚡ Energy distribution analysis** and indicators of potential seismic impact.
- **🔄 Cyclic behavior and periodicity analysis.**

## 🚀 Local Installation (Developers)

### Requirements

- Python 3.8 or later
- pip (Python package manager)

### Installation Instructions

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/seismic-dashboard.git
cd seismic-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app_minimal.py
```

## 📁 Project Structure

```text
seismic-dashboard/
├── app_minimal.py                   # Main Fani Maoré dashboard
├── requirements.txt                 # Python dependencies
├── README.md                         # Project documentation
├── pages/                            # Analysis modules
│   ├── Analyse_generale.py           # General analysis
│   ├── Analyse_caracteristiques_sismique.py
│   │                                  # Seismic characteristics
│   ├── Analyse_tendances_sismique.py # Temporal trends
│   └── Analyse_spatio_temporelle.py  # Spatial analysis
├── utils/                            # Utility functions
│   ├── data_loader.py                # Data loading
│   ├── data_collector.py              # Data collection
│   └── settings.py                    # Configuration
├── data/                              # Seismic datasets
│   ├── NewDataseisme.csv              # Fani Maoré seismic database
│   ├── backup/                        # Data backups
│   └── logs/                          # Application logs
└── config/                            # Configuration files
    └── settings.py
```

## 📊 Seismic Data

### Data Sources

The seismic dataset is based on information provided by the following institutions and monitoring networks:

- **REVOSIMA:** Mayotte Volcanological and Seismological Monitoring Network (*Réseau de surveillance volcanologique et sismologique de Mayotte*).
- **IPGP:** Institut de Physique du Globe de Paris.
- **BRGM:** French Geological Survey (*Bureau de Recherches Géologiques et Minières*).

### Data Format: `NewDataseisme.csv`

```csv
Date,Magnitude,Latitude,Longitude,Depth,Year,Month,Day,Hour
10/05/2018 14:15,4.2,-12.8456,45.5123,25.3,2018,5,10,14
```

**Main variables:**

- `Date`: Date and time of the seismic event (UTC).
- `Magnitude`: Local magnitude (ML).
- `Latitude/Longitude`: Geographic coordinates of the seismic event.
- `Depth`: Hypocentral depth (km).

## 🗺️ Study Area

### Geological Context

- **Monitoring area:** Eastern region of Mayotte.
- **Volcanic system:** Fani Maoré submarine volcano, discovered in 2019.
- **Tectonic setting:** Madagascar–Comoros region.
- **Geological phenomenon:** The largest seismic swarm recorded in the Mayotte region.

### Key Geographical Features

- **🌋 Fani Maoré Volcano:** Main volcanic edifice.
- **🏝️ Mayotte:** Geographical reference point.
- **📡 REVOSIMA Stations:** Seismic monitoring network.

## 🔧 Technical Features

### Scientific Computations

The dashboard incorporates several quantitative methods and statistical indicators:

- **⚡ Released seismic energy:**

  \[
  E = 10^{1.5M+4.8}
  \]

- **💥 Potential impact indicator:** A magnitude- and depth-based indicator:

  \[
  I = M\left(1+\frac{70}{d}\right)
  \]

- **📊 Gutenberg–Richter law:** Analysis of the magnitude–frequency relationship.
- **📈 Statistical tests:** Chi-square tests, correlation analysis, and trend detection.

### Mapping and Spatial Visualization

- **🗺️ Interactive Folium maps.**
- **🔥 Seismic density heatmaps.**
- **📍 Spatial clustering of seismic events.**
- **📏 Distance calculations** from the Fani Maoré volcanic edifice.

## 📈 Detailed Analysis Modules

### 1. 🏠 Overview

- Key indicators of Fani Maoré volcanic activity.
- Geological and volcanic background.
- General statistics describing seismic activity.

### 2. 📊 General Seismic Analysis

- Overall distribution of seismic events.
- Temporal evolution of seismic activity.
- Year-to-year comparisons.

### 3. 🔬 Seismic Characteristics

- Magnitude and depth distributions.
- Magnitude–depth relationships.
- Potential impact indicators.
- Analysis of released seismic energy.

### 4. 📈 Temporal Trend Analysis

- Seasonal and weekly cycles.
- Long-term temporal trends.
- Periodicity analysis.
- Statistical significance testing.

### 5. 🗺️ Spatio-Temporal Analysis

- Geographic mapping of seismic activity.
- Spatial evolution of seismic events over time.
- Distance analysis relative to key geographical locations.
- Geographic clustering and spatial patterns.

## 🌊 Scientific Impact and Monitoring

### Scientific Significance

- **First submarine volcanic system** monitored in real time in the Mayotte region.
- **Major geological phenomenon** in the western Indian Ocean.
- **Ground deformation:** Several centimeters of deformation have been observed.
- **Formation of a new volcanic edifice:** Approximately 800 m in height.

### Continuous Monitoring

- **24/7 monitoring** through the REVOSIMA network.
- **Automated alerts** for significant seismic events.
- **Volcanic system modeling** to investigate its evolution.
- **Risk assessment** to support the monitoring and protection of Mayotte's population.

## 🆘 Support and Documentation

### Troubleshooting

If you encounter an issue while using the dashboard:

1. Check your internet connection.
2. Review error messages in the application console.
3. Restart the application.
4. Contact the development team.

### Scientific Resources

- [REVOSIMA – IPGP](http://revosima.ipgp.fr/)
- [BRGM – French Geological Survey](https://www.brgm.fr/)
- [Piton de la Fournaise Volcanological Observatory](http://www.ipgp.fr/fr/ovpf)

## 📝 Version History

- **v1.0 (2024):** Initial release of the Fani Maoré dashboard.
- **v1.1 (2024):** Introduction of seismic characteristics analysis.
- **v1.2 (2024):** Integration of advanced mapping functionalities.
- **v1.3 (2025):** Optimization for cloud deployment.

---

---

**🌋 Developed for the statistical exploration, visualization, and spatio-temporal analysis of seismic activity associated with the Fani Maoré submarine volcano**

*Application scientifique dédiée à la compréhension de l'activité sismique volcanique sous-marine*

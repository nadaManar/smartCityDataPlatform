##  Objectif du projet

L’objectif de ce projet est de construire un **tableau de bord prédictif** capable d’afficher la pollution fine (PM2.5) et de la **prédire automatiquement** à partir des variables les plus importantes du dataset (PM10, CO, NO2, SO2, humidité, pression).

Pour cela, nous avons :

1. **Exploré les données** afin d’identifier les variables les plus influentes sur la pollution PM2.5.
2. **Testé plusieurs modèles de Machine Learning et Deep Learning** (Régression Linéaire, Random Forest, XGBoost, MLP) pour sélectionner le modèle le plus performant.
3. **Entraîné un modèle Random Forest**, qui a obtenu un R² de 0.948 et une erreur MAE d’environ 1.7 µg/m³.
4. **Créé un dashboard interactif avec Streamlit** permettant :
   - de prédire la pollution PM2.5 via des sliders,
   - d’afficher l’importance des variables,
   - de visualiser les données de manière intuitive.

### Installation des dépendances
```bash
python -m venv venv
venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn streamlit pyspark pip install xgboost

```
### Lancer le dashboard 
streamlit run notebooks/app.py

### Configuration du thème Streamlit
mkdir .streamlit

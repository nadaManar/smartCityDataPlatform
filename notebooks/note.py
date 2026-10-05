import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# Chargement du jeu de données météo et pollution
df = pd.read_csv("data/openweather_weather.csv", sep=",", header=0, low_memory=False)

# Sélection des variables explicatives et de la cible (PM2.5)
features = df[['humidity', 'pressure', 'temp_category', 'comfort_proxy',
               'has_rain', 'has_snow', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3']]
target = df['target_pm25']  # Cible : pollution fine (PM2.5)

# Vérification des valeurs manquantes
print(df.isnull().sum())  # Vérifie s’il y a des données manquantes

# Visualisation de la relation entre chaque variable et la cible
for col in features.columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(features[col], target, alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('PM2.5')
    plt.title(f'Relation entre {col} et PM2.5')
    # plt.show()  # Décommenter pour afficher les graphiques

# Fonction pour tester la relation linéaire entre une variable et la cible
def test_linear(var):
    X = sm.add_constant(df[var])
    Y = df['target_pm25']
    model = sm.OLS(Y, X).fit()

    R2 = model.rsquared
    Ir = np.sum(model.resid**2)
    It = np.sum((Y - np.mean(Y))**2)
    Im = It - Ir

    print(f"\nVariable : {var}")
    print("R² :", R2)
    print("Ir :", Ir)
    print("Im :", Im)

# Test de linéarité pour quelques variables
for var in ['humidity', 'so2', 'no2', 'co']:
    test_linear(var)

# Fonction pour tester un modèle linéaire multivarié
def test_multi(vars_list, label):
    X = sm.add_constant(df[vars_list])
    Y = df['target_pm25']
    model = sm.OLS(Y, X).fit()

    R2 = model.rsquared
    Ir = np.sum(model.resid**2)
    It = np.sum((Y - np.mean(Y))**2)
    Im = It - Ir

    print(f"\n{label}")
    print("Variables :", vars_list)
    print("R² :", R2)
    print("Ir :", Ir)
    print("Im :", Im)

# Comparaison de plusieurs combinaisons de variables
test_multi(['pm10', 'co', 'no2', 'so2', 'humidity', 'pressure'], "combined")
test_multi(['pm10', 'co', 'no2', 'so2', 'humidity'], "combined without pressure")
test_multi(['pm10', 'co', 'no2', 'so2', 'pressure'], "combined without humidity")
test_multi(['pm10', 'co', 'no2', 'so2'], "combined without humidity or pressure")

# Le modèle OLS donne un R² faible → les phénomènes physiques sont non linéaires
# On passe donc à des modèles non linéaires (Random Forest, XGBoost, MLP)

# Modèle Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

features1 = df[['pm10', 'co', 'no2', 'so2', 'humidity', 'pressure']]
target1 = df['target_pm25']

X_train, X_test, y_train, y_test = train_test_split(features1, target1, test_size=0.2, random_state=42)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae1 = mean_absolute_error(y_pred, y_test)
rmse1 = np.sqrt(mean_squared_error(y_test, y_pred))
r2_1 = r2_score(y_test,y_pred)

print("Random Forest Results:")
print("MAE:", mae1)
print("RMSE:", rmse1)
print("R²:", r2_1)
# Résultat : le modèle explique ~95.2 % de la variance du PM2.5
# Erreur moyenne ≈ 1.7 unités → très bon modèle

#Modèle XGBoost
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("XGBoost Results:")
print("MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
print("R²:", r2_score(y_test, y_pred_xgb))

# Modèle MLP

from sklearn.neural_network import MLPRegressor

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train_std, y_train)
y_pred_mlp = mlp.predict(X_test_std)

mae = mean_absolute_error(y_pred_mlp, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
r2 = r2_score( y_test,y_pred_mlp)

print("MLP Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

#le meilleur modèle est Random Forest
# On sauvegarde le modèle pour une utilisation future
import joblib
joblib.dump(rf, "models/rf_model.pkl")

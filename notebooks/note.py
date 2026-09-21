import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


#chargement de donnees
df=pd.read_csv("data/openweather_weather.csv", sep=",", header=0,low_memory=False)
#print(df.head(5))
 
#print(df.columns.tolist())#pour savoir les noms des columns
features=df[['humidity', 'pressure', 'temp_category', 'comfort_proxy',
               'has_rain', 'has_snow', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3']]
target=df['target_pm25']# Cible : pollution fine

#let's clean data first 
print(df.isnull().sum)#nothing empty

#on  visualise maintenant relation entre chaque  variable et target variable 
#plt.figure(figsize=(10,6))
for col in features.columns:
    plt.figure(figsize=(6,4))
    plt.scatter(features[col], target, alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('PM25')
    plt.title(f'Relation entre {col} et PM25')
    #plt.show()

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
for var in ['humidity', 'so2', 'no2', 'co']:
    test_linear(var)

#essayer avec plusiers 

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

test_multi(['pm10', 'co', 'no2', 'so2', 'humidity', 'pressure'], "combined")
test_multi(['pm10', 'co', 'no2', 'so2', 'humidity'], "combined without pressure")
test_multi(['pm10', 'co', 'no2', 'so2', 'pressure'], "combined without humidity")
test_multi(['pm10', 'co', 'no2', 'so2'], "combined without humidity or pressure")

#best one is ['pm10', 'co', 'no2', 'so2', 'humidity', 'pressure']


#le phenomene n est pas linear parceque 
#La relation entre X et Y n'est une droite
# Le modèle OLS donne un R² qui n est pas  élevé 
#Les résidus montrent une structure
#Les phénomènes physiques sont connus pour être non linéaires
#DU COUP JUQU A MNT ON A FAIT JUSTE L EXPLORATION

#Now that we know that it's not lenear so we will use non linear models and shoose the best 

#random forest 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
features1=df[['pm10', 'co', 'no2', 'so2', 'humidity', 'pressure']]

target1=df['target_pm25']# Cible : pollution fine
X_train,X_test ,y_train,y_test=train_test_split(features1,target1,test_size=0.2,random_state=42)
rf=RandomForestRegressor( n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1)#nb of arbre dept of arbre n_jobs=vitess
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
mae1=mean_absolute_error(y_pred,y_test)
rmse1 = np.sqrt(mean_squared_error(y_test, y_pred))
r2_1 = r2_score(y_pred,y_test)

print("MAE:", mae1)
print("RMSE:", rmse1)
print("R²:", r2_1)
#resultat : Le modèle se trompe en moyenne de 1.7 unités de PM2.5.Même les grosses erreurs restent faibles.
# ce modèle explique 94.8% de la variance du PM2.5.


#XGB
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)#% de lignes par arbre l autre ce qu elle peut voir de l arbre d avant .
#% de colonnes par arbre

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

#print("XGB MAE:", mean_absolute_error(y_test, y_pred_xgb))
#print("XGB RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
#print("XGB R²:", r2_score(y_test, y_pred_xgb))


#let's try with mlp now 
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_std=scaler.fit_transform(X_train)
X_test_std=scaler.fit_transform(X_test)
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)

# Train the model
mlp.fit(X_train_std, y_train)

# Make predictions
y_pred = mlp.predict(X_test_std)
mae=mean_absolute_error(y_pred,y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_pred,y_test)
#print("mlp")
#print("MAE:", mae)
#print("RMSE:", rmse)
#print("R²:", r2)

#meilleur resultat c est avec random forest model donc on va
#  l'utiliser pour notre prediction
#now that we know our model let's create our dashboard
#first let's save the model random forest
import joblib
joblib.dump(rf,"models/rf_model.pkl")
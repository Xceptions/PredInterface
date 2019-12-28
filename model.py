import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error as mse

df = pd.read_csv('./Wat Dataset.csv')
OIL_SAMPLE_NUMBER = df['OIL_SAMPLE_NUMBER'].copy()
df = df.drop(['OIL_SAMPLE_NUMBER'], axis=1)

# feature engineering

density = df['DENSITY'].copy()
molecular_weight = df['MOLECULAR_WEIGHT'].copy()
pressure = [14.504 for x in density]
AE = df['ACTIVATION_ENERGY'].copy()
enthalpy = df['ENTHALPY_CHANGE'].copy()
kinematic_visc = df['KINEMATIC_VISCOSITY'].copy()

temp = [80 for x in density]
df['R_constant'] = (density * temp) / (pressure * molecular_weight) # generated feature
# R_const = df['R_constant'].copy()
df['dynamic_visc'] = kinematic_visc * density # generated feature
df['another_visc'] = kinematic_visc * molecular_weight
df['density_per_pressure'] = pressure / density
dynamic_visc = df['dynamic_visc'].copy()
kinematic_visc = df['KINEMATIC_VISCOSITY'].copy()
df['pressure_per_dynamic_visc'] = pressure / dynamic_visc
df['pressure_per_kinematic_visc'] = pressure / kinematic_visc
# print(df.head())
# df['AE'] = enthalpy + (R_const*temp)
# df['pressure'] = pressure
# df['temp'] = temp



target = df['WAT'].values
df = df.drop(['WAT'], axis=1)

data = df.values

sc = StandardScaler()

data = sc.fit_transform(data)

clf = LinearRegression(normalize=True).fit(data, target)
R_value = clf.score(data, target)
predictions = clf.predict(data)
data_statistics = pd.DataFrame(predictions).describe()
mse = mse(target, predictions)


result = pd.DataFrame({'OIL_SAMPLE_NUMBER': OIL_SAMPLE_NUMBER, 'PREDICTIONS': predictions})
result.to_csv('Model Predictions.csv', index=False)

other_data = [[R_value], [mse]]
other_csv = pd.DataFrame(other_data, index=["R_value", "Mean Squared Error"])
other_csv.to_csv("Accuracy Data.csv", index=True)

stat_data = pd.DataFrame(data_statistics)
stat_data.to_csv("Data Statistics.csv", index=True)

filename = "finalized_model.sav"
joblib.dump(clf, filename)



"""
 what is left to do: arrange the predictions in a csv file, save the r value then get some graphs
 your score: 0.9983862403302834
"""
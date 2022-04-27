import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, scale
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

sheets = pd.ExcelFile('agro_tech.xlsx')
df1 = pd.read_excel(sheets, 'plants')
df2 = pd.read_excel(sheets, 'flight dates')
df3 = pd.read_excel(sheets, 'planting')
df4 = pd.read_excel(sheets, 'weather')

df1.drop(['Head Weight (g)', 'Diameter Ratio', 'Density (kg/L)', 'Leaf Area (cm^2)', 'Square ID', 'Check Date', 'Remove'], axis=1, inplace=True)

rename = {'Batch Number' : 'batch', 'Plant Date' : 'planted', 'Class' : 'class', 'Fresh Weight (g)' : 'fresh_w', 'Radial Diameter (mm)' : 'radial_dia', 'Polar Diameter (mm)' : 'polar_dia', 'Leaves' : 'leaves', 'Check Date' : 'checked', 'Flight Date' : 'flighted'}

df1.rename(columns = rename, inplace=True)

median = df1['fresh_w'].median()
df1['fresh_w'].fillna(median, inplace=True)


median = df1['radial_dia'].median()
df1['radial_dia'].fillna(median, inplace=True)

median = df1['polar_dia'].median()
df1['polar_dia'].fillna(median, inplace=True)

median = df1['leaves'].median()
df1['leaves'].fillna(median, inplace=True)

df1['grown_up'] = df1['flighted'] - df1['planted']
df1.drop(['planted', 'flighted'], axis=1, inplace=True)
df1['grown_up'] = df1['grown_up'].astype(int)
df1['grown_up'] = df1['grown_up'].replace(-9223372036854775808, 0)

x = df1.drop(['fresh_w'], axis=1).values
y = df1['fresh_w'].values.T

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def get_r2_score (model, x_train, x_test, y_train, y_test):
  #kFold do the same thing as cross_val_score()
  k_fold = KFold(n_splits=11, shuffle=True, random_state=42)
  #fit() actually Trains the Model
  model.fit(x_train, y_train)
  #Cross validation
  cross_val_score(model, x_train, y_train, cv=k_fold, scoring='r2').mean()
  y_predict = model.predict(x_test)

  return  y_predict, r2_score(y_test, y_predict)
  
y_pred_lr, scor_lr = get_r2_score(LinearRegression(), x_train, x_test, y_train, y_test)
y_pred_dt, scor_dt = get_r2_score(DecisionTreeRegressor(), x_train, x_test, y_train, y_test)

print('r2_score(LinearRegressor):',scor_lr)
print('r2_score(DecisionTreeRegressor):',scor_dt)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))

print('RMSE(LinearRegressor):',rmse_lr)
print('RMSE(DecisionTreeRegressor):',rmse_dt)

plt.title('Using LinearRegressor')
plt.xlabel('Actual value (y_test)')
plt.ylabel('Predicted value')
plt.scatter(y_test, y_pred_lr)
plt.show()

plt.title('Using DecisionTreeRegressor')
plt.xlabel('Actual value (y_test)')
plt.ylabel('Predicted value')
plt.scatter(y_test, y_pred_dt)
plt.show()

def get_variation(model):
  model.fit(x_train, y_train)
  return scale(model.coef_), model.intercept_
  
m, b = get_variation(LinearRegression())
print('Coefficient:',m)
print('Intercept:',b)


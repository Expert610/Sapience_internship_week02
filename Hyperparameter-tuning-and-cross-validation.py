import pandas as pd
import seaborn as sb
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

#Load the Dataset
df = pd.read_csv('car data.csv')
# print(df.head())

#Cleaning Data
# print(df.isnull().sum())
# print(df.describe())
# print(df.info())

#droping useless column

df.drop('Car_Name', axis=1 ,inplace=True)
#creating a new featureb'car_age'
df['Car_Age'] = 2025 - df['Year']
df.drop('Year',axis=1,inplace=True)
# print(df.head())

# Encode Catorgical variable

df= pd.get_dummies(df, columns=['Fuel_Type','Selling_type','Transmission'] ,drop_first=True)
df[df.select_dtypes(include='bool').columns] = df.select_dtypes(include='bool').astype(int)
# print(df.head())

X =df.drop('Selling_Price', axis=1)

# ouput target

Y = df['Selling_Price']

dt_model = DecisionTreeRegressor(random_state=42)

dt_params ={
    'max_depth':[5,10,15],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]

}

dt_grid = GridSearchCV(estimator=dt_model,param_grid=dt_params, cv=5 ,scoring='r2' , n_jobs=-1)

dt_grid.fit(X,Y)

print(" Best Decision tree Parameters:", dt_grid.best_params_)
print(" Best R² Score from CV:", dt_grid.best_score_)


rf_model = RandomForestRegressor(random_state=42)

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(estimator=rf_model, param_grid=rf_params,
                       cv=5, scoring='r2', n_jobs=-1)
rf_grid.fit(X, Y)

print(" Best Random Forest Parameters:", rf_grid.best_params_)
print(" Best R² Score from CV:", rf_grid.best_score_)

from sklearn.model_selection import cross_val_score

# Use best estimator from grid search
best_rf = rf_grid.best_estimator_
best_dt = dt_grid.best_estimator_

rf_scores = cross_val_score(best_rf, X, Y, cv=5, scoring='r2')
dt_scores = cross_val_score(best_dt, X, Y, cv=5, scoring='r2')


print(" Decision Tree R² (CV average):", dt_scores.mean())
print(" Random Forest R² (CV average):", rf_scores.mean())

import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

df = pd.read_csv('car data.csv')

df['car_age']= 2025-df['Year']

df.drop(['Year','Car_Name'],axis=1,inplace=True)

df = pd.get_dummies(df,drop_first=True)
df[df.select_dtypes(include='bool').columns] = df.select_dtypes(include='bool').astype(int)
 
x =df.drop('Selling_Price',axis=1)
y= df['Selling_Price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)

print("🎯 Random Forest Results:")
print(f"MSE: {mse: .2f}")
print(f"R2 : { r2: .2f}")
print(f"MAE : { mae: .2f}")

importance = model.feature_importances_
features = x.columns

imp_df = pd.Series(importance , index=features).sort_values(ascending=True)

plt.figure(figsize=(10,6))
imp_df.plot(kind='barh',color='teal')
plt.xlabel('importance score')
plt.title("Random forest regressor features importance")
plt.tight_layout()
plt.show()
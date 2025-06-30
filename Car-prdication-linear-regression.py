import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
print(df.head())
 
#  input feature
X =df.drop('Selling_Price', axis=1)

# ouput target

Y = df['Selling_Price']

# Train and Test Split model
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2 ,random_state=42)

# Training The model

model = LinearRegression()
model.fit(X_train,Y_train)

# Predication
y_pred =model.predict(X_test)
mae = mean_absolute_error(Y_test,y_pred)

print(f"Mean Square Error:{mean_squared_error(Y_test,y_pred): .2f}")
print(f"R2 Score: {r2_score(Y_test,y_pred): .2f}")
print(f"MAE : { mae: .2f}")

# Visualize Graph

plt.figure(figsize=(8,6))
sb.scatterplot(x=Y_test,y=y_pred,color='blue')
plt.xlabel("Actual Selling price of Car")
plt.ylabel("Predict Selling Price of Car")
plt.title("Actual Selling price VS  Predicated Selling price")
plt.grid(True)
plt.show()


# Analyze Feature most contribute to predication

# Make Dataframe

coeff_df = pd.DataFrame({
    'Features': X.columns,
    'Co-efficient':model.coef_
})


coeff_df['Absolute'] = coeff_df['Co-efficient'].abs()
coeff_df.sort_values(by='Absolute',ascending=False,inplace=True)
coeff_df.drop('Absolute',axis=1,inplace=True)

print(coeff_df)
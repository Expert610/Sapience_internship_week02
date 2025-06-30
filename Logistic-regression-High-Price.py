import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

# Dataset Load
df = pd.read_csv('car data.csv')
# Drop Car_Name column if present
if 'Car_Name' in df.columns:
    df.drop('Car_Name', axis=1, inplace=True)

df['Car_Age'] = 2025-df['Year']    

df.drop('Year',axis=1,inplace=True)
df = pd.get_dummies(df,drop_first=True)
df[df.select_dtypes(include='bool').columns]=df.select_dtypes(include='bool').astype(int)

Price_median = df['Selling_Price'].median()
df['High_Price'] =(df['Selling_Price']> Price_median).astype(int)

df = df.astype(float)

x = df.drop(['Selling_Price','High_Price'],axis=1 )
y = df['High_Price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2 ,random_state=42)

model= LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_predict = model.predict(x_test)

accuracy = accuracy_score(y_test,y_predict)
precision = precision_score(y_test,y_predict)
recall = recall_score(y_test,y_predict)
f1 = f1_score(y_test,y_predict)


print(f"Accuracy : { accuracy: .2f}")
print(f"Precision : { precision: .2f}")
print(f"Recall : {recall: .2f}")
print(f"F1 : {f1 : .2f}")
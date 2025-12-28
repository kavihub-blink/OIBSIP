import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#load dataset
df=pd.read_csv(r"E:\sales prediction using python\dataset\Advertising.csv")
#display basic info
print(df.head())
print(df.info())
#features and target
X=df[['TV', 'Radio', 'Newspaper']]
y=df['Sales']
#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#model training
model = LinearRegression()
model.fit(X_train, y_train)
#predictions
y_pred = model.predict(X_test)
#evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:",r2_score(y_test, y_pred))
#predict new sales
new_data=pd.DataFrame({'TV':[150],'Radio':[25],'Newspaper':[10]})
predicted_sales=model.predict(new_data)
print("Predicted Sales: ",predicted_sales[0])
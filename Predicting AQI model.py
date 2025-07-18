import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

data = pd.read_csv('air_quality_data.csv')
print(data.head())
data = data.dropna()
data.columns =[col.strip().lower() for col in data.columns]

sns.pairplot(data)
plt.show()

corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')

X = data[['co aqi value', 'ozone aqi value', 'no2 aqi value', 'pm2.5 value']]
y = data['aqi value']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42) #Model training by Random forest ML ALGO multiple decision trees 
model.fit(X_train, y_train)
#No we do model evaluation 
y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))  
print("R2 Score:",r2_score(y_test,y_pred))  
#Now Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual AQI')
plt.plot(y_pred, label='Predicted AQI',alpha=0.7)
plt.title('Actual vs Predicted AQI')
plt.legend()
plt.show()
# Save the model
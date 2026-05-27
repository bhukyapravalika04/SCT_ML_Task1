import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


train_data = pd.read_csv('train.csv')

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']

X = train_data[features]


y = train_data['SalePrice']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)


predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("Model Trained Successfully!")
print("Mean Squared Error:", mse)


print("\nSample Predictions:")
print(predictions[:10])

import matplotlib.pyplot as plt

# Scatter plot
plt.scatter(y_test, predictions)

# Labels
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

# Title
plt.title("Actual Prices vs Predicted Prices")

# Show graph
plt.show()
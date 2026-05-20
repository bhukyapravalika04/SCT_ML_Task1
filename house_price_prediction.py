import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
train_data = pd.read_csv('train.csv')

# Select features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']

# Input data
X = train_data[features]

# Target variable
y = train_data['SalePrice']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict house prices
predictions = model.predict(X_test)

# Calculate error
mse = mean_squared_error(y_test, predictions)

print("Model Trained Successfully!")
print("Mean Squared Error:", mse)

# Display sample predictions
print("\nSample Predictions:")
print(predictions[:10])
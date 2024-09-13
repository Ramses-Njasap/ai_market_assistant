import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Create 'model' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load the dataset
data = pd.read_csv('/home/ramses/Desktop/real_estate_ai/data/real_estate_data .csv')

# Features and target variable
X = data[['Size', 'Age', 'Rooms', 'Bathrooms']]
y = data['Price']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'models/real_estate_price_model.pkl')

print("Model trained and saved.")

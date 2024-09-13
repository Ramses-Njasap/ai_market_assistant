import joblib
import numpy as np

# Load the trained model
model = joblib.load('/home/ramses/Desktop/real_estate_ai/models/real_estate_price_model.pkl')

def predict_price(size, age, rooms, bathrooms):
    """Predicts the price based on the input features."""
    input_data = np.array([[size, age, rooms, bathrooms]])
    predicted_price = model.predict(input_data)[0]
    return predicted_price

if __name__ == "__main__":
    # Get user input
    size = float(input("Enter the property size (in square feet): "))
    age = float(input("Enter the age of the property (in years): "))
    rooms = int(input("Enter the number of rooms: "))
    bathrooms = int(input("Enter the number of bathrooms: "))

    # Make prediction
    predicted_price = predict_price(size, age, rooms, bathrooms)
    print(f"The predicted price for the property is: ${predicted_price:.2f}")

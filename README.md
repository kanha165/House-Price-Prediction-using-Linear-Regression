🏠 House Price Prediction using Linear Regression
📘 Project Overview

This project predicts house prices based on multiple features such as area (sqft), number of bedrooms, bathrooms, age of the house, and distance from the city center.
It uses a Linear Regression model to establish a relationship between the features and the price.
The main goal is to create a simple, interpretable, and accurate regression model that can predict house prices for new data points.

📊 Dataset Description

The dataset used for this project is house_data.csv and contains the following columns:

Feature Description
sqft Area of the house in square feet
bedrooms Number of bedrooms
bathrooms Number of bathrooms
age Age of the house in years
distance_to_city_km Distance from the city center (in kilometers)
price Price of the house (target variable)

The dataset is clean, structured, and numerical — ideal for regression-based prediction models.

🧠 Model Training (Linear Regression)

Data Preprocessing:

Imported required libraries (pandas, numpy, matplotlib, sklearn).

Loaded the dataset and handled missing values if any.

Split the dataset into training and testing sets using train_test_split().

Model Building:

Used LinearRegression() from sklearn.linear_model.

Trained the model on input features (X) and output (y).

Saved the trained model as my_model.pkl for future use.

Evaluation Metrics:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R² Score

These metrics help measure the accuracy and performance of the model.

⚙️ How to Run

Install dependencies:

pip install pandas numpy scikit-learn matplotlib joblib

Run the training notebook:
Open train.ipynb to:

Load dataset

Train Linear Regression model

Save it as my_model.pkl

Run the test notebook:
Open test_model.ipynb to:

Load the trained model

Test it with sample inputs

Visualize actual vs predicted prices

💾 How to Use Saved Model

You can use the saved .pkl model for predictions without retraining.

import joblib

# Load saved model

model = joblib.load("my_model.pkl")

# Input new data

sqft = 1800
bedrooms = 3
bathrooms = 2
age = 5
distance_to_city_km = 8

# Predict price

features = [[sqft, bedrooms, bathrooms, age, distance_to_city_km]]
predicted_price = model.predict(features)
print(f"🏠 Predicted House Price: ₹{predicted_price[0]:,.2f}")

📈 Results and Accuracy

Model Used: Linear Regression

Evaluation Metrics (example):

MAE ≈ X

MSE ≈ Y

R² Score ≈ Z

The scatter plot between Actual vs Predicted Prices shows a near-linear pattern, confirming good model accuracy.

🚀 Future Improvements

Use advanced models like Random Forest or XGBoost.

Add categorical features such as location or type of property.

Build a Flask or Streamlit web interface for real-time predictions.

Apply feature scaling and normalization to improve performance.

🧑‍💻 Author

Kanha Patidar
B.Tech (CSIT) — Chamelidevi Group of Institutions, Indore
Project: House Price Prediction using Linear Regression

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('./datasets/nutrients_csvfile.csv')
    
    # Convert relevant columns to numeric, handling non-numeric values
    numeric_columns = ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].replace('t', '0'), errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna(subset=numeric_columns)
    return df

# Train ML model
@st.cache_resource
def train_model(df):
    # Prepare features and target
    X = df[['Protein', 'Carbs', 'Fat', 'Fiber']]
    y = df['Calories']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get coefficients and intercept
    coefficients = pd.DataFrame({
        'Feature': ['Protein', 'Carbs', 'Fat', 'Fiber'],
        'Coefficient': model.coef_
    })
    coefficients = coefficients.sort_values('Coefficient', ascending=False)
    
    return model, scaler, mse, r2, coefficients

# Load data and train model
df = load_and_preprocess_data()
model, scaler, mse, r2, coefficients = train_model(df)

# Title and description
st.title("Nutrition Calorie Predictor")
st.write("Predict calorie content based on nutritional values using Linear Regression")

# Add prediction interface
st.sidebar.header("Enter Nutritional Values")
pred_protein = st.sidebar.number_input("Protein (g):", min_value=0.0, max_value=100.0, value=20.0)
pred_carbs = st.sidebar.number_input("Carbs (g):", min_value=0.0, max_value=200.0, value=50.0)
pred_fat = st.sidebar.number_input("Fat (g):", min_value=0.0, max_value=100.0, value=15.0)
pred_fiber = st.sidebar.number_input("Fiber (g):", min_value=0.0, max_value=50.0, value=5.0)

# Make prediction
input_features = np.array([[pred_protein, pred_carbs, pred_fat, pred_fiber]])
input_scaled = scaler.transform(input_features)
predicted_calories = model.predict(input_scaled)[0]

# Display prediction and model metrics
st.write("### Prediction Results")
st.write(f"**Predicted Calories:** {predicted_calories:.1f} kcal")

st.write("\n### Model Performance Metrics")
st.write(f"- Mean Squared Error: {mse:.2f}")
st.write(f"- R² Score: {r2:.3f}")

# Display coefficients
st.write("\n### Feature Coefficients")
st.write("These coefficients show how much each nutrient affects the calorie content:")
st.dataframe(coefficients)

# Visualization of actual vs predicted values
st.write("\n### Model Analysis")
X_all_scaled = scaler.transform(df[['Protein', 'Carbs', 'Fat', 'Fiber']])
all_predictions = model.predict(X_all_scaled)

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['Calories'],
    y=all_predictions,
    mode='markers',
    name='Predictions',
    marker=dict(size=8, opacity=0.6)
))

# Add diagonal line for perfect predictions
perfect_line = np.linspace(df['Calories'].min(), df['Calories'].max(), 100)
fig.add_trace(go.Scatter(
    x=perfect_line,
    y=perfect_line,
    mode='lines',
    name='Perfect Prediction',
    line=dict(dash='dash')
))

fig.update_layout(
    title='Actual vs Predicted Calories',
    xaxis_title='Actual Calories',
    yaxis_title='Predicted Calories',
    showlegend=True
)

st.plotly_chart(fig)

# Add explanation of the model
st.write("""
### How Linear Regression Works
Linear regression predicts calories by finding the best linear combination of nutrients:

```
Calories = (coefficient₁ × Protein) + (coefficient₂ × Carbs) + (coefficient₃ × Fat) + (coefficient₄ × Fiber) + intercept
```

The coefficients shown above indicate how much each nutrient contributes to the calorie content. A positive coefficient means 
that increasing that nutrient increases calories, while a negative coefficient means the opposite.
""")

# Dataset Statistics
st.write("### Dataset Statistics")
st.write("Summary statistics of the nutritional values in our dataset:")
st.dataframe(df[['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber']].describe())

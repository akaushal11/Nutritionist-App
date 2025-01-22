import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess the Indian food dataset
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('./datasets/indian_food.csv')
    
    # Convert relevant columns to numeric, handling any non-numeric values
    numeric_columns = ['Total Calories', 'Total Carbs', 'Total Fats', 'Total Protein', 'Total Sugar', 'Total Sodium']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna(subset=numeric_columns)
    return df

# Train ML model to predict calories
@st.cache_resource
def train_model(df):
    # Prepare features and target
    X = df[['Total Carbs', 'Total Fats', 'Total Protein', 'Total Sugar', 'Total Sodium']]
    y = df['Total Calories']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mse, r2

# Load data and train model
df = load_and_preprocess_data()
model, scaler, mse, r2 = train_model(df)

# Streamlit UI
st.title('Indian Food Analysis Dashboard 🍛')

# Sidebar for filters
st.sidebar.header('Filters')
selected_state = st.sidebar.multiselect('Select States', options=sorted(df['State'].unique()), default=[])
selected_type = st.sidebar.multiselect('Select Food Type', options=sorted(df['Type'].unique()), default=[])

# Filter data based on selections
filtered_df = df.copy()
if selected_state:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_state)]
if selected_type:
    filtered_df = filtered_df[filtered_df['Type'].isin(selected_type)]

# Display key metrics
st.header('Key Metrics')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Average Calories', f"{filtered_df['Total Calories'].mean():.1f}")
with col2:
    st.metric('Average Protein', f"{filtered_df['Total Protein'].mean():.1f}g")
with col3:
    st.metric('Average Carbs', f"{filtered_df['Total Carbs'].mean():.1f}g")

# Visualizations
st.header('Visualizations')

# 1. Distribution of Food Types by State
type_counts = pd.DataFrame(filtered_df['Type'].value_counts()).reset_index()
type_counts.columns = ['Type', 'Count']
fig_types = px.bar(
    type_counts,
    x='Type',
    y='Count',
    title='Distribution of Food Types'
)
st.plotly_chart(fig_types)

# 2. Average Calories by State
calories_by_state = filtered_df.groupby('State')['Total Calories'].mean().sort_values(ascending=True)
fig_calories = px.bar(
    calories_by_state,
    orientation='h',
    title='Average Calories by State',
    labels={'value': 'Average Calories', 'State': 'State'}
)
st.plotly_chart(fig_calories)

# 3. Nutrient Composition
st.subheader('Nutrient Composition Analysis')
nutrient_cols = ['Total Carbs', 'Total Fats', 'Total Protein']
avg_nutrients = filtered_df[nutrient_cols].mean()

fig_nutrients = go.Figure(data=[
    go.Pie(labels=nutrient_cols, values=avg_nutrients, hole=.3)
])
fig_nutrients.update_layout(title='Average Nutrient Distribution')
st.plotly_chart(fig_nutrients)

# 4. Scatter plot of Calories vs Protein with Food Type
fig_scatter = px.scatter(
    filtered_df,
    x='Total Protein',
    y='Total Calories',
    color='Type',
    hover_data=['Food Name', 'State'],
    title='Calories vs Protein Content by Food Type'
)
st.plotly_chart(fig_scatter)

# Calorie Prediction Section
st.header('Calorie Prediction')
st.write('Predict calories based on nutrient content:')

col1, col2 = st.columns(2)
with col1:
    input_carbs = st.number_input('Carbohydrates (g)', min_value=0.0, value=30.0)
    input_fats = st.number_input('Fats (g)', min_value=0.0, value=10.0)
    input_protein = st.number_input('Protein (g)', min_value=0.0, value=8.0)
with col2:
    input_sugar = st.number_input('Sugar (g)', min_value=0.0, value=5.0)
    input_sodium = st.number_input('Sodium (mg)', min_value=0.0, value=300.0)

if st.button('Predict Calories'):
    input_features = np.array([[input_carbs, input_fats, input_protein, input_sugar, input_sodium]])
    input_scaled = scaler.transform(input_features)
    predicted_calories = model.predict(input_scaled)[0]
    st.success(f'Predicted Calories: {predicted_calories:.1f}')

# Diet Recipe Recommendation System
st.header('Personalized Diet Recipe Recommendations 🍽️')
st.write('Get personalized Indian food recommendations based on your profile')

def calculate_bmr(weight, height, age, gender):
    if gender == 'Male':
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        'Sedentary (little or no exercise)': 1.2,
        'Lightly active (1-3 days/week)': 1.375,
        'Moderately active (3-5 days/week)': 1.55,
        'Very active (6-7 days/week)': 1.725,
        'Super active (physical job or 2x training)': 1.9
    }
    return bmr * activity_multipliers[activity_level]

def calculate_target_calories(tdee, goal):
    if goal == 'Lose weight':
        return tdee - 500  # Create a 500 calorie deficit
    elif goal == 'Gain weight':
        return tdee + 500  # Create a 500 calorie surplus
    else:
        return tdee

def get_meal_recommendations(df, target_calories, state=None):
    # Filter by state if specified
    if state:
        df = df[df['State'] == state]
    
    # Ensure we have some data to work with
    if len(df) == 0:
        df = load_and_preprocess_data()  # Reset to full dataset if state filter leaves no options
    
    # Calculate calories per meal
    breakfast_cals = target_calories * 0.5  # 30% of daily calories
    lunch_cals = target_calories * 0.3     # 40% of daily calories
    dinner_cals = target_calories * 0.2     # 30% of daily calories
    
    def get_meals(calories, meal_type, n_meals=2):
        # First try: Look for meals within a wider range
        lower_bound = calories * 0.5  # 50% below target
        upper_bound = calories * 1.5  # 50% above target
        
        possible_meals = df[
            (df['Total Calories'] >= lower_bound) & 
            (df['Total Calories'] <= upper_bound)
        ]
        
        # If no meals found, find the closest matches
        if len(possible_meals) < n_meals:
            # Calculate how far each meal is from target calories
            df['calorie_diff'] = abs(df['Total Calories'] - calories)
            # Sort by difference and get the closest matches
            possible_meals = df.nsmallest(n_meals, 'calorie_diff')
            # Drop the temporary column
            df.drop('calorie_diff', axis=1, inplace=True)
        else:
            # If we have enough meals in range, sample from them
            possible_meals = possible_meals.sample(n=min(n_meals, len(possible_meals)))
        
        return possible_meals

    # Get recommendations for each meal
    breakfast = get_meals(breakfast_cals, "Breakfast")
    lunch = get_meals(lunch_cals, "Lunch")
    dinner = get_meals(dinner_cals, "Dinner")
    
    return breakfast, lunch, dinner

def display_meals(meals, meal_type, target_calories):
    if not meals.empty:
        st.write(f"**{meal_type} Options:** (Target: {target_calories:.0f} calories)")
        for _, meal in meals.iterrows():
            calorie_diff = meal['Total Calories'] - target_calories
            diff_text = f"({calorie_diff:+.0f} cal)" if abs(calorie_diff) > 50 else "(on target)"
            
            st.write(f"- {meal['Food Name']} ({meal['State']}) - {meal['Total Calories']:.0f} calories {diff_text}")
            st.write(f"  • Carbs: {meal['Total Carbs']:.1f}g | Protein: {meal['Total Protein']:.1f}g | Fats: {meal['Total Fats']:.1f}g")
            if 'Allergic Ingredients' in meal and meal['Allergic Ingredients'] != 'None':
                st.write(f"  •  Contains: {meal['Allergic Ingredients']}")
            if 'Vitamin Content' in meal and pd.notna(meal['Vitamin Content']):
                st.write(f"  •  Vitamins: {meal['Vitamin Content']}")
    else:
        st.write(f"No {meal_type.lower()} recommendations found.")

# Input form for user details
col1, col2 = st.columns(2)
with col1:
    weight = st.number_input('Current Weight (kg)', min_value=30.0, max_value=200.0, value=70.0)
    height = st.number_input('Height (cm)', min_value=100.0, max_value=250.0, value=170.0)
    age = st.number_input('Age', min_value=15, max_value=100, value=30)
    
with col2:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    activity_level = st.selectbox('Activity Level', [
        'Sedentary (little or no exercise)',
        'Lightly active (1-3 days/week)',
        'Moderately active (3-5 days/week)',
        'Very active (6-7 days/week)',
        'Super active (physical job or 2x training)'
    ])
    goal = st.selectbox('Goal', ['Maintain weight', 'Lose weight', 'Gain weight'])
    preferred_state = st.selectbox('Preferred State Cuisine', ['All States'] + sorted(df['State'].unique().tolist()))

if st.button('Get Diet Recommendations'):
    # Calculate daily calorie needs
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(bmr, activity_level)
    target_calories = calculate_target_calories(tdee, goal)
    
    st.subheader('Your Daily Calorie Target')
    st.info(f'Based on your profile, your recommended daily calorie intake is: {target_calories:.0f} calories')
    
    # Get meal recommendations
    state_filter = None if preferred_state == 'All States' else preferred_state
    breakfast, lunch, dinner = get_meal_recommendations(df, target_calories, state_filter)
    
    # Display recommendations
    st.subheader('Your Personalized Meal Plan')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        display_meals(breakfast, "Breakfast", target_calories * 0.3)
    with col2:
        display_meals(lunch, "Lunch", target_calories * 0.4)
    with col3:
        display_meals(dinner, "Dinner", target_calories * 0.3)
    
    # Display total nutritional information
    all_meals = pd.concat([breakfast, lunch, dinner])
    if not all_meals.empty:
        st.subheader('Total Daily Nutritional Information')
        total_calories = all_meals['Total Calories'].sum()
        total_carbs = all_meals['Total Carbs'].sum()
        total_protein = all_meals['Total Protein'].sum()
        total_fats = all_meals['Total Fats'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Total Calories', f'{total_calories:.0f}', 
                   f"{total_calories - target_calories:+.0f} from target")
        col2.metric('Total Carbs', f'{total_carbs:.1f}g')
        col3.metric('Total Protein', f'{total_protein:.1f}g')
        col4.metric('Total Fats', f'{total_fats:.1f}g')
        
        # Add daily balance information
        if abs(total_calories - target_calories) > 100:
            if total_calories > target_calories:
                st.warning(f"This meal plan is {total_calories - target_calories:.0f} calories above your daily target. Consider smaller portions or alternative dishes if following strictly.")
            else:
                st.warning(f"This meal plan is {target_calories - total_calories:.0f} calories below your daily target. Consider adding healthy snacks or slightly larger portions.")
        else:
            st.success("This meal plan is well-balanced with your daily calorie target! 🎯")

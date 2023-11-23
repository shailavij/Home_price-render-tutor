# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle  # Import the pickle library

# Function to train a linear regression model
def train_model(data):
    X = data[['Income', 'House_age','Rooms','Area']]
    y = data['Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Display model performance
    st.write(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')

    # Save the trained model to a file using pickle
    with open('trained_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    return model

# Function to predict house price based on user input
def predict_price(model, Income, House_age, Rooms, Area):
    input_data = pd.DataFrame({
        'Income': [Income],
        'House_age': [House_age],
        'Rooms': [Rooms],
        'Area': [Area]
    })

    prediction = model.predict(input_data)

    return prediction[0]

# Streamlit app

def main():
    st.title('House Price Prediction App')

    # Load example data
    example_data = pd.read_csv('USA_Housing.csv')

    # Train the model
    trained_model = train_model(example_data)

    # User input for prediction
    st.sidebar.header('User Input')
    Income = st.sidebar.slider('Income', min_value=500, max_value=500000, value=15000)
    House_age = st.sidebar.slider('House_age', min_value=1, max_value=20, value=5)
    Rooms = st.sidebar.slider('Rooms', min_value=1, max_value=5, value=2)
    Area = st.sidebar.slider('Area', min_value=500, max_value=5000, value=1500)


    if st.button("Predict"):

        # Predict house price
        predicted_price = predict_price(trained_model, Income, House_age, Rooms, Area)

        st.subheader('Predicted House Price')
        st.write(f'${predicted_price:,.2f}')

        # Plot example data
        st.subheader('Example Data')
        st.line_chart(example_data['Rooms'])

if __name__ == '__main__':
    main()

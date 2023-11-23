import pickle
import streamlit as st


def main():
    st.title('House Price Prediction App')

    # User input for prediction
    st.sidebar.header('User Input')
    Income = st.sidebar.slider('Income', min_value=500, max_value=500000, value=15000)
    House_age = st.sidebar.slider('House_age', min_value=1, max_value=20, value=5)
    Rooms = st.sidebar.slider('Rooms', min_value=1, max_value=5, value=2)
    Area = st.sidebar.slider('Area', min_value=500, max_value=5000, value=1500)

    if st.button("Predict"):

        # Load the trained model from the pickle filecd
        with open('trained_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        # Now you can use the loaded_model for making predictions
        def predict_price(model, Income, House_age, Rooms, Area):
            input_data = [[Income, House_age, Rooms, Area]]
            prediction = model.predict(input_data)
            return prediction[0]


        predicted_price = predict_price(loaded_model, Income, House_age, Rooms, Area)

        #print(f'Predicted House Price: ${predicted_price:,.2f}')
        st.subheader('Predicted House Price')
        st.write(f'${predicted_price:,.2f}')



if __name__ == '__main__':
    main()
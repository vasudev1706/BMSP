
import pandas as pd 
import numpy as np
import pickle 
import joblib
import os
import streamlit as st

def main():
    # Title
    st.title('BigMart Sales Prediction')

    # Input fields
    Item_Weight = st.text_input('Enter Item Weight')
    Item_Fat_Content = st.selectbox('Select Item Fat Content', ['Low Fat', 'Regular'])
    Item_Visibility = st.text_input('Enter Item Visibility')
    Item_Type = st.text_input('Enter Item Type')
    Item_MRP = st.text_input('Enter Item MRP')
    Outlet_Establishment_Year = st.slider('Select Outlet Establishment Year', min_value=1985, max_value=2010)
    Outlet_Size = st.selectbox('Select Outlet Size', ['Small', 'Medium', 'High'])
    Outlet_Location_Type = st.selectbox('Select Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])
    Outlet_Type = st.selectbox('Select Outlet Type', ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

    if st.button('Predict'):
        try:
            # Preprocess input data
            input_data = {
                'Item_Weight': float(Item_Weight),
                'Item_Fat_Content': 1 if Item_Fat_Content == 'Regular' else 0,
                'Item_Visibility': float(Item_Visibility),
                'Item_Type': Item_Type,
                'Item_MRP': float(Item_MRP),
                'Outlet_Establishment_Year': Outlet_Establishment_Year,
                'Outlet_Size': ['Small', 'Medium', 'High'].index(Outlet_Size),
                'Outlet_Location_Type': ['Tier 1', 'Tier 2', 'Tier 3'].index(Outlet_Location_Type),
                'Outlet_Type': ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'].index(Outlet_Type)
            }

            # Load scaler
            sc = joblib.load('sc.sav')

            # Transform input data
            X_train_std = sc.transform([list(input_data.values())])

            # Load model
            loaded_model = joblib.load('lr.sav')

            # Make prediction
            prediction = loaded_model.predict(X_train_std)

            # Display prediction
            st.success(f'Predicted Sales: {float(prediction[0])}')

        except Exception as e:
            st.error(f'An error occurred: {e}')

    

if __name__ == "__main__":
    main()
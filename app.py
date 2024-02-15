import pandas as pd 
import numpy as np
import pickle 
import joblib
import os
import streamlit as st

def main():
    # title
    st.title('BigMart Sales Prediction')

    Item_Weight = float(st.text_input('ITEM WEIGHT'))
    Item_Fat_Content = st.sidebar.radio('ITEM FAT CONTENT', ['Regular', 'Low Fat'])
    Item_Visibility = float(st.text_input('ITEM VISIBILITY'))
    Item_Type = st.text_input('ITEM TYPE')
    Item_MRP = float(st.text_input('ITEM MRP'))
    Outlet_Establishment_Year = st.sidebar.slider('OUTLET ESTABLISHMENT YEAR', min_value=1985, max_value=2010)
    Outlet_Size = st.sidebar.radio('OUTLET SIZE', ['Small', 'Medium', 'High'])
    Outlet_Location_Type = st.sidebar.radio('OUTLET LOCATION TYPE', ['Tier 1', 'Tier 2', 'Tier 3'])
    Outlet_Type = st.sidebar.radio('OUTLET TYPE', ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])

    if st.button('PREDICT'):
        # Load the scaler
        with open('sc.sav', 'rb') as f:
            sc = pickle.load(f)

        # Load the model
        loaded_model = joblib.load('lr.sav')

        # Encode categorical variables
        Item_Fat_Content_encoded = 1 if Item_Fat_Content == 'Regular' else 0
        Item_Type_encoded = 0  # Implement proper encoding for Item_Type

        # Create input array
        X = np.array([Item_Weight, Item_Fat_Content_encoded, Item_Visibility, Item_Type_encoded, Item_MRP, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type]).reshape(1, -1)

        # Transform the input data
        X_train_std = sc.transform(X)

        # Make predictions
        Y_pred = loaded_model.predict(X_train_std)

        # Display the prediction
        st.success(f'Prediction: {float(Y_pred[0])}')

if __name__ == "__main__":
    main()

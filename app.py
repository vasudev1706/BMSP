
import pandas as pd 
import numpy as np
import pickle 
import os
import streamlit as st


def main():
    # Title
    st.title('BigMart Sales Prediction')

    Item_Weight = st.text_input('ITEM WEIGHT')
    Item_Fat_Content = st.sidebar.radio('ITEM FAT CONTENT', ['Low Fat', 'Regular'])
    Item_Visibility = st.text_input('ITEM VISIBILITY')
    Item_Type = st.text_input('ITEM TYPE')
    Item_MRP = st.text_input('ITEM MRP')
    Outlet_Establishment_Year = st.sidebar.slider('OUTLET ESTABLISHMENT YEAR', min_value=1985, max_value=2010)
    Outlet_Size = st.sidebar.radio('OUTLET SIZE', ['Small', 'Medium', 'High'])
    Outlet_Location_Type = st.sidebar.radio('OUTLET LOCATION TYPE', ['Tier 1', 'Tier 2', 'Tier 3'])
    Outlet_Type = st.sidebar.radio('OUTLET TYPE', ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

    if st.button('PREDICT'):
        try:
            # Convert input values to appropriate data types
            Item_Weight = float(Item_Weight) if Item_Weight else 0.0
            Item_Visibility = float(Item_Visibility) if Item_Visibility else 0.0
            Item_MRP = float(Item_MRP) if Item_MRP else 0.0

            # Map categorical values to numeric values
            fat_content_map = {'Low Fat': 0, 'Regular': 1}
            outlet_size_map = {'Small': 0, 'Medium': 1, 'High': 2}
            location_type_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
            outlet_type_map = {'Supermarket Type1': 0, 'Supermarket Type2': 1, 'Supermarket Type3': 2, 'Grocery Store': 3}

            Item_Fat_Content = fat_content_map.get(Item_Fat_Content, 0)
            Outlet_Size = outlet_size_map.get(Outlet_Size, 0)
            Outlet_Location_Type = location_type_map.get(Outlet_Location_Type, 0)
            Outlet_Type = outlet_type_map.get(Outlet_Type, 0)

            # Create input array
            X = np.array([Item_Weight, Item_Fat_Content, Item_Visibility, Item_MRP, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type])

            # Load the scaler
            with open('sc.pkl', 'rb') as f:
                sc = pickle.load(f)

            # Transform the input data
            X_train_std = sc.transform(X.reshape(1, -1))

            # Load the model
            with open('RF.pkl', 'rb') as f:
                loaded_model = pickle.load(f)

            # Make predictions
            Y_pred = loaded_model.predict(X_train_std)

            # Display the prediction
            st.success(f'Prediction: {float(Y_pred[0])}')

        except Exception as e:
            st.error(f'An error occurred: {e}')
    

if __name__ == "__main__":
    main()

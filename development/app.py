import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

# Load the saved XGBoost model
loaded_model = xgb.Booster()
loaded_model.load_model("xgb_model.bin")

# Define the Streamlit app
def main():

    # Create the header image
    header_image = 'paragon-corp.png'
    st.image(header_image, use_column_width=True)

    # Create the title heading
    st.write('Paragon Corp Sales Forecast Prediction')
    st.write('Hactiv8 Bootcamp Talent Fair 2023')
    st.write('Determine the weeks ahead you want to predict the sales for')

    # Create the form
    with st.form(key= 'form_a'):
        st.markdown('##### **Forecast Sales Category A**')
        input_a = st.number_input('Week Ahead', min_value=1, max_value=48, value=2 ,step=1)
        st.write('###### **Mean Absolute Percentage Error :** ', '10.006 %')
        submitted_a = st.form_submit_button('Predict')

        if submitted_a:
            # Perform prediction with loaded XGBoost model
            test_data = pd.DataFrame({'week_ahead': [input_a]})  # Create test data with input value
            test_pred = loaded_model.predict(xgb.DMatrix(test_data))  # Perform prediction
            st.write('##### **Sales Prediction for Category A**')
            st.write('Weeks Ahead:', input_a)
            st.write('Sales Prediction:', test_pred[0])  # Display the predicted sales value

# Run the Streamlit app
if __name__ == '__main__':
    main()

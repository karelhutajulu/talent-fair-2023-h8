import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objs as go
import plotly.express as px

# Create the header image
header_image = 'paragon-corp.png'
st.image(header_image, use_column_width=True)

# Create the title heading
st.write('Paragon Corp Sales Forecast Prediction')
st.write('Hactiv8 Bootcamp Talent Fair 2023')
st.write('Determine the weeks ahead you want to predict the sales for')
# Load the trained XGBoost model from a file
reg = xgb.XGBRegressor()
reg.load_model('xgb_model.bin')

# Load the training data from a CSV file
train = pd.read_csv('train.csv', index_col='week_start_date', parse_dates=['week_start_date'])

# Define a function to generate features for the specified date
def generate_features(date, train):
    # Create a DataFrame with the specified date as the index
    df = pd.DataFrame(index=pd.DatetimeIndex([date]))
    
    # Add columns for each product_item (use the mean value from the training set)
    for col in train.columns:
        if col != 'quantity':
            df[col] = train[col].mean()
    
    # Add the last 4 total sales quantity features
    for i in range(1, 5):
        # Get the date for the current lag
        lag_date = date - pd.DateOffset(weeks=i)
        
        # Check if the date is in the training set
        if lag_date in train.index:
            # Use the value from the training set
            df[f'last {i} quantity'] = train.loc[lag_date, 'quantity']
        else:
            # Use the mean value from the training set
            df[f'last {i} quantity'] = train['quantity'].mean()
    
    # Add the quarter, month, year, and weekofyear features
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    
    return df

st.title('Sales Prediction')

# Get user input for the number of weeks to predict
weeks_to_predict = st.number_input('Enter the number of weeks to predict:', min_value=1)

# Generate predictions for the specified number of weeks using a rolling forecast approach
start_date = pd.to_datetime('2023-01-02')
dates = pd.date_range(start_date, periods=weeks_to_predict, freq='W-MON')
predictions = []
for date in dates:
    # Generate features for the current date
    X = generate_features(date, train)
    
    # Make a prediction using the trained XGBoost model
    y_pred = reg.predict(X)
    
    # Store the prediction and date
    predictions.append((date, y_pred[0]))
    
    # Update the training data with the predicted value
    train.loc[date, 'quantity'] = y_pred[0]

# Convert the predictions to a DataFrame and display it
predictions_df = pd.DataFrame(predictions, columns=['Date', 'Predicted Sales'])
st.write(predictions_df)

# Create a line chart to visualize the predicted sales
fig = px.line(predictions_df, x='Date', y='Predicted Sales', title='Predicted Sales')
st.plotly_chart(fig)


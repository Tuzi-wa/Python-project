import streamlit as st
import pandas as pd
import numpy as np
from src.data import get_stock_data, preprocess_data
from src.methods import StockLSTM, train_model, predict_future, plot_candlestick



st.title("Stock Prediction Application")


ticker = st.text_input("Enter Stock Ticker Symbol:", value="AAPL")
start_date = st.date_input("Select Start Date:", value=pd.to_datetime("2023-01-01"))
end_date = st.date_input("Select End Date:", value=pd.Timestamp.now())


if st.button("Load Data"):
    try:
        
        st.write(f"Fetching data for {ticker}...")
        df = get_stock_data(ticker, start=start_date, end=end_date)
        st.write("Stock Data Preview:")
        st.write(df.head())

        
        sequence_length = 10

        
        st.write("Preprocessing the data...")
        X, y, scaler = preprocess_data(df, sequence_length)

        
        st.write(f"X shape: {X.shape}, y shape: {y.shape}")

        
        model = StockLSTM(input_size=5, output_size=4)
        st.write("Training the model...")
        train_model(model, X, y, epochs=10, batch_size=32, learning_rate=0.001)

        st.write("Predicting future stock prices...")
        future_predictions = predict_future(model, X[-1], 30, scaler, sequence_length, 5)

        st.write("Future Predictions (Candlestick Chart):")
        fig = plot_candlestick(future_predictions, df)  
        st.pyplot(fig)  

    except Exception as e:
     st.error(f"An error occurred: {e}")
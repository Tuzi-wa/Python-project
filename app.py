import streamlit as st
import pandas as pd
import numpy as np
from src.data import get_stock_data, preprocess_data
from src.methods import StockLSTM, train_model, predict_future, plot_candlestick

# Streamlit 应用标题

st.title("Stock Prediction Application")

# 用户输入股票代码和日期范围
ticker = st.text_input("Enter Stock Ticker Symbol:", value="AAPL")
start_date = st.date_input("Select Start Date:", value=pd.to_datetime("2023-01-01"))
end_date = st.date_input("Select End Date:", value=pd.Timestamp.now())

# 处理用户输入并下载数据
if st.button("Load Data"):
    try:
        # 获取股票数据
        st.write(f"Fetching data for {ticker}...")
        df = get_stock_data(ticker, start=start_date, end=end_date)
        st.write("Stock Data Preview:")
        st.write(df.head())

        # 设置序列长度
        sequence_length = 10

        # 数据预处理
        st.write("Preprocessing the data...")
        X, y, scaler = preprocess_data(df, sequence_length)

        # 检查数据形状
        st.write(f"X shape: {X.shape}, y shape: {y.shape}")

        # 定义和训练模型
        model = StockLSTM(input_size=5, output_size=4)
        st.write("Training the model...")
        train_model(model, X, y, epochs=10, batch_size=32, learning_rate=0.001)

        st.write("Predicting future stock prices...")
        future_predictions = predict_future(model, X[-1], 30, scaler, sequence_length, 5)

        st.write("Future Predictions (Candlestick Chart):")
        fig = plot_candlestick(future_predictions, df)  # 调用修改后的绘图函数，返回 Figure 对象
        st.pyplot(fig)  # 使用 Streamlit 显示 Matplotlib 图形

    except Exception as e:
     st.error(f"An error occurred: {e}")
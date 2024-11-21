import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mplfinance as mpf  
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta 


def process_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        try:
            return datetime.strptime(date_str, "%d-%m-%Y")
        except ValueError:
            print("Invalid date format. Using default dates.")
            return datetime.now() 

def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  

    return df

def preprocess_data(df, sequence_length):
    df_pct_change = df[['Open', 'High', 'Low', 'Close']].pct_change().dropna()
    df_pct_change['Volume'] = df['Volume'].iloc[1:].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_pct_change.values)

    x, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        x.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length, :4])  

    return np.array(x), np.array(y), scaler

class StockLSTM(tf.keras.Model):
        def __init__(self, input_size=5, hidden_layer_size=128, num_layers=3, output_size=4):
         super(StockLSTM, self).__init__()
         self.lstm_layers = [tf.keras.layers.LSTM(hidden_layer_size, return_sequences=True) for _ in range(num_layers)]
         self.dropout = tf.keras.layers.Dropout(0.2)
         self.fc = tf.keras.layers.Dense(output_size)

        def call(self, x):
         for lstm in self.lstm_layers:
            x = lstm(x)
         x = self.dropout(x)
         x = self.fc(x[:, -1, :])  
         return x


def train_model(model, X_train, y_train, epochs, batch_size, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)



def predict_future(model, data, steps, scaler, sequence_length, input_size):
    predictions = []

    for _ in range(steps):
        seq = np.expand_dims(data[-sequence_length:], axis=0)  
        pred = model.predict(seq)[0]  
        predictions.append(pred)
        pred_with_volume = np.append(pred, 0) 
        new_seq = np.expand_dims(pred_with_volume, axis=0)
        data = np.vstack((data, new_seq))  

    predictions = np.array(predictions) 
    
    if predictions.shape[1] == 4:
        predictions_with_volume = np.concatenate([predictions, np.zeros((predictions.shape[0], 1))], axis=1)
    else:
        predictions_with_volume = predictions

    predictions_rescaled = scaler.inverse_transform(predictions_with_volume)[:, :4] 
    return predictions_rescaled

def plot_candlestick(predictions, df, last_days=30):
    future_dates = pd.date_range(df.index[-1], periods=predictions.shape[0] + 1, freq='B')[1:]  
    future_df = pd.DataFrame(predictions, index=future_dates, columns=['Open', 'High', 'Low', 'Close'])
    full_df = pd.concat([df[-last_days:], future_df])  
    full_df.index = pd.to_datetime(full_df.index)
    if "Volume" not in df.columns:
        df['Volume'] = 0
    if "Volume" not in future_df.columns:
        future_df['Volume'] = 0
    future_df = future_df.reindex(columns=df.columns, fill_value=0)

    
    print("Last days data (df):")
    print(df[-last_days:])
    print("Future predictions data (future_df):")
    print(future_df)

    
    try:
        full_df = pd.concat([df[-last_days:], future_df])
    except Exception as e:
        print("Concat error:", e)
        return

    mc = mpf.make_marketcolors(up='darkseagreen', down='darksalmon', edge='tan', wick='tan', volume='gray')
    s = mpf.make_mpf_style(marketcolors=mc)

    mpf.plot(
        full_df,
        type = 'candle',
        style = mpf.make_mpf_style(),
        title = '30-Day Stock Price Prediction with Historical Data',
        mav = (3, 6, 9),
        volume = True,
        returnfig = True
        )


if __name__ == "__main__":
    ticker = input("enter the stock ticker(default is AAPL.):").strip() or "AAPL"
    start_date = input("Enter the start date (YYYY-MM-DD):") or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = input("Enter the end date (YYYY-MM-DD):") or datetime.now().strftime("%Y-%m-%d")
    start_date = process_date(start_date).strftime("%Y-%m-%d")
    end_date = process_date(end_date).strftime("%Y-%m-%d")
   
   
    print(f"Stock Ticker: {ticker}")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
   
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = get_stock_data(ticker, start_date, end_date)

    if df.empty:
        print(f"No data available for {ticker} in the selected date range.")
    else:
        print(f"Data fetched successfully for {ticker}.")


    sequence_length = 10
    X, y, scaler = preprocess_data(df, sequence_length)


    model = StockLSTM(input_size=5, output_size=4)
    train_model(model, X, y, epochs=100, batch_size=32, learning_rate=0.001)

   
    future_predictions = predict_future(model, X[-1], 30, scaler, sequence_length, 5)

    plot_candlestick(future_predictions, df)

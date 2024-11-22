import numpy as np
import tensorflow as tf
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd

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

    mc = mpf.make_marketcolors(up='green', down='red', edge='white', wick='black', volume='gray')
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

mpf.show()
        
    
   
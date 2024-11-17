Requirements

		yfinance
		numpy
		pandas
		tensorflow
		matplotlib
		mplfinance
	 scikit-learn
  The project requires the following Python libraries:
	•	yfinance: To fetch historical stock data.
	•	numpy: For numerical operations.
	•	pandas: For data manipulation.
	•	tensorflow: For building and training the LSTM model.
	•	matplotlib & mplfinance: For data visualization.
	•	scikit-learn: For data normalization (MinMaxScaler).


 The main functions in the code:
	•	process_date(date_str): Parses the input date string and handles different formats.
	•	get_stock_data(ticker, start, end): Downloads the stock data from Yahoo Finance.
	•	preprocess_data(df, sequence_length): Normalizes the data and prepares it for training.
	•	StockLSTM: Defines the LSTM neural network model.
	•	train_model(model, X_train, y_train, epochs, batch_size, learning_rate): Trains the LSTM model.
	•	predict_future(model, data, steps, scaler, sequence_length, input_size): Predicts future stock prices.
	•	plot_candlestick(predictions, df): Visualizes the results using a candlestick chart.

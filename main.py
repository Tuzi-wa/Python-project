from src.data import get_stock_data, preprocess_data
from src.methods import StockLSTM, train_model, predict_future, plot_candlestick
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

if __name__ == "__main__":
    ticker = input("Enter the stock ticker (default is AAPL): ").strip() or "AAPL"
    start_date = input("Enter the start date (YYYY-MM-DD): ") or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = input("Enter the end date (YYYY-MM-DD): ") or datetime.now().strftime("%Y-%m-%d")
    start_date = process_date(start_date).strftime("%Y-%m-%d")
    end_date = process_date(end_date).strftime("%Y-%m-%d")
   
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



    #需要继续改的地方： 把下载的数据保留到小数点后两位
    #运行完结果的图一闪而过
    #继续找出sreamlit上运行出错的原因
    # init的文件写完
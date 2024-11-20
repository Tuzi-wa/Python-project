import numpy as np
import pandas as pd
import tensorflow as tf
import mplfinance as mpf

# 验证安装
print("Numpy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("TensorFlow version:", tf.__version__)

# 简单绘图测试
data = pd.DataFrame({
    'Open': [100, 102, 103],
    'High': [105, 107, 108],
    'Low': [95, 97, 98],
    'Close': [102, 104, 107],
    'Volume': [1000, 1200, 1300]
}, index=pd.date_range(start="2023-01-01", periods=3))

mpf.plot(data, type='candle', title='Test Candlestick Chart', volume=True)
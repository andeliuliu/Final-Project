# final_proj/data_processing.py

import pandas as pd
from final_proj.models import StockData

def load_stock_data(ticker):
    # Load data from the database into a Pandas DataFrame
    qs = StockData.objects.filter(ticker=ticker).order_by('date')
    data = pd.DataFrame.from_records(qs.values())
    
    # Feature Engineering
    data['MA10'] = data['close'].rolling(window=10).mean()
    data['MA50'] = data['close'].rolling(window=50).mean()
    
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['BB_upper'] = data['close'].rolling(window=20).mean() + (data['close'].rolling(window=20).std() * 2)
    data['BB_lower'] = data['close'].rolling(window=20).mean() - (data['close'].rolling(window=20).std() * 2)
    
    data['target'] = data['close'].shift(-1)
    data.dropna(inplace=True)  # Remove rows with NaN values after feature engineering
    
    return data
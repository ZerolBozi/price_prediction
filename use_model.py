import torch
import pandas as pd
import matplotlib.pyplot as plt

from processer import create_indicators, scale_datasets, inverse_scale_datasets, convert_to_lstm_format
from models import Model

def use_model():
    pass

if __name__ == "__main__":
    ticker = 'binance_matic_kline_1h__spot_'
    data = pd.read_csv(f'./datas/{ticker}.csv')

    data_processed = create_indicators(
        data=data,
        save=False
    )

    data_scaled_np, scalers = scale_datasets(
        data=data_processed
    )

    time_window_size = 1
    x_data, y_data = convert_to_lstm_format(data_scaled_np, time_window_size)

    model_obj = Model()
    model = model_obj.get_model('lstm_binance_eth_kline_1h__spot_')

    with torch.no_grad():
        output = model(x_data)

    predictions = output.cpu().numpy()

    real = ['close']
    targets =  ['close', 'low', 'high']

    real_data = inverse_scale_datasets(data_scaled_np[time_window_size:], scalers, data_processed.columns)
    predicted_data = inverse_scale_datasets(predictions, scalers, targets)

    plt.figure(figsize=(12, 6))

    for column in real:
        plt.plot(real_data[column], label=f'Real {column} Price')

    for column in targets:
        plt.plot(predicted_data[column], label=f'Predicted {column} Price', alpha=0.7)

    plt.title(ticker)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
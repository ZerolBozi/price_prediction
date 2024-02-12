import torch
import pandas as pd
import matplotlib.pyplot as plt

from processer import create_indicators, scale_datasets, inverse_scale_datasets, convert_to_lstm_format, get_y_idx
from models import Model

REAL_COLS = ['close']
TARGET_COLS =  ['close', 'low', 'high']

def use_model(ticker: str, data: pd.DataFrame, model_name: str):
    data_processed = create_indicators(data)
    data_scaled_np, scalers = scale_datasets(data_processed)

    time_window_size = 1
    y_idx_list = get_y_idx(data_processed.columns, TARGET_COLS)

    # y_data用來評估模型的預測能力, 如R2, MSE, 這部分沒有寫
    x_data, _ = convert_to_lstm_format(data_scaled_np, time_window_size, y_idx_list)

    model_obj = Model()
    model = model_obj.get_model(model_name)

    with torch.no_grad():
        output = model(x_data)

    predictions = output.cpu().numpy()

    real_data = inverse_scale_datasets(data_scaled_np[time_window_size:], scalers, data_processed.columns)
    predicted_data = inverse_scale_datasets(predictions, scalers, TARGET_COLS)

    plot_results(ticker, real_data, predicted_data)

def plot_results(ticker: str, real_data: pd.DataFrame, predicted_data: pd.DataFrame):
    plt.figure(figsize=(12, 6))

    for column in REAL_COLS:
        plt.plot(real_data[column], label=f'Real {column} Price')

    for column in TARGET_COLS:
        plt.plot(predicted_data[column], label=f'Predicted {column} Price', alpha=0.7)

    plt.title(ticker)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ticker = 'binance_sol_kline_1h_spot'
    data = pd.read_csv(f'./datas/{ticker}.csv')

    use_model(ticker, data, f'lstm_stock_crypto')
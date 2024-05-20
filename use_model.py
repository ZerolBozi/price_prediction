import argparse

import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # fix RuntimeError
from matplotlib import pyplot as plt
import os
from processer import create_indicators, scale_datasets, inverse_scale_datasets, convert_to_lstm_format, get_y_idx
from TradingEnvironment import TradingEnvironment, Action
from strategy import trend_strategy
from models import Model
from decimal import Decimal
DATA_DEFAULT_PATH = "./datas"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_parser():
    parser = argparse.ArgumentParser(description="use model to predict stock or crypto price")
    parser.add_argument("--ticker", type=str, help="stock or crypto data file name", required=True)
    parser.add_argument("--model_name", type=str, help="model name", required=True)
    parser.add_argument("--time_window_size", type=int, help="time window size", required=True)
    parser.add_argument("--data_path", type=str, help="model params", default=None)
    parser.add_argument("--real_cols", nargs='+', help="real cols", default=['close'])
    parser.add_argument("--target_cols", nargs='+', help="target cols", default=['close', 'low', 'high'])
    parser.add_argument("--save_csv", type=bool, help="save csv", default=False)
    parser.add_argument("--output_path", type=str, help="output path", default=None)
    return parser

def use_model(ticker: str, model_name: str, model_params: dict, data:pd.DataFrame = None, data_path: str = None, save_csv: bool = False, output_path: str = None,show_plot:bool=False):
    """
    model_params: dict
        :key 'time_window_size', type: int
        :key 'real_cols', type: list
        :key 'target_cols', type: list
        :key 'model_type', type: str
    """
    if not all(key in model_params.keys() for key in ['time_window_size', 'real_cols', 'target_cols']):
        raise Exception("model_params should have keys: time_window_size, real_cols, target_cols")
    if data is None:
        if data_path is None:
            data = pd.read_csv(f'{DATA_DEFAULT_PATH}/{ticker}.csv')
        else:
            data = pd.read_csv(data_path)

    data_processed = create_indicators(data)
    data_scaled_np, scalers = scale_datasets(data_processed)

    time_window_size = model_params.get('time_window_size')
    y_idx_list = get_y_idx(data_processed.columns, model_params.get('target_cols'))

    # y_data用來評估模型的預測能力, 如R2, MSE, 這部分沒有寫
    x_data, _ = convert_to_lstm_format(data_scaled_np, time_window_size, y_idx_list)

    model_obj = Model()
    model = model_obj.get_model(model_name)

    with torch.no_grad():
        output = model(x_data)

    predictions = output.cpu().numpy()

    real_data = inverse_scale_datasets(data_scaled_np[time_window_size:], scalers, data_processed.columns)
    predicted_data = inverse_scale_datasets(predictions, scalers, model_params.get('target_cols'))

    if (save_csv) and (output_path is not None):
        predicted_data.to_csv(output_path + '.csv', index=False)
    if show_plot:
        plot_results(ticker, real_data, predicted_data, model_params)
    return real_data, predicted_data

def use_DQNmodel(ticker: str, model_name: str=None, model_params: dict=None, data:dict=None):
    """
    model_params: dict
        :key 'window_size', type: int
        :key 'trading_strategy', type: callable
        :key 'position_size_ratio', type: Decimal
        :key 'initial_balance', type: Decimal
        :key 'model_type', type: str
        :key 'trade_side', type: list

    data: dict
        :key 'original_data', type: pd.DataFrame
        :key 'predict_data', type: pd.DataFrame
    """
    if not all(key in model_params.keys() for key in ['window_size', 'trading_strategy']):
        raise Exception("model_params should have keys: window_size, trading_strategy")

    if  not all(key in data.keys() for key in ['original_data', 'predict_data']):
        raise Exception("data should original_data and predicted_data")
    
    if model_name is None:
        model_name = f'{model_params.get("model_type", "DQN").lower()}_{ticker}'

    original_data = data.get('original_data', None)
    predict_data = data.get('predict_data', None)
    if original_data is None or predict_data is None:
        raise Exception("original_data and predicted data not None")

    net_obj = Model()
    net = net_obj.get_model(model_name)

    env = TradingEnvironment(
        ticker=ticker,
        original_data=original_data,
        predict_data=predict_data,
        initial_balance=model_params.get('initial_balance', Decimal(100000)),
        position_size_ratio=model_params.get('position_size_ratio', Decimal(0.5)),
        window_size=model_params.get('window_size', 25),
        action_size=Action.space,
        trading_strategy=model_params.get('trading_strategy',trend_strategy),
        model_type=model_params.get('model_type', 'DQN'),
        trade_side=model_params.get('trade_side',[Action.long, Action.close_long])
    )

    done = False
    total_reward = 0
    state = env.reset()
    print("Start Trading")
    while not done:
        action=None 
        with torch.no_grad():
            _state = np.array(state)
            state_tensor = torch.tensor(_state, dtype=torch.float32).to(device)
            q_values = net(state_tensor).detach().cpu().squeeze(0)
            action = torch.argmax(q_values).item()
        next_state, total_reward, done = env.step(action)
        state = next_state
    print(f"Total Reward: {total_reward}")
    env.render()
    
def plot_results(ticker: str, real_data: pd.DataFrame, predicted_data: pd.DataFrame, model_params: dict):
    chart_path = f'./records/{ticker}/chart'

    plt.figure(figsize=(12, 6))
    plt.plot(real_data['close'], label=f'Real close Price')
    plt.plot(predicted_data['close'], label=f'Predicted close Price', alpha=0.7)
    plt.title(ticker)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(f'{chart_path}/{model_params.get("model_type","LSTM")}_{ticker}_predicted_2lines.png',
        transparent=False,
        bbox_inches='tight',
        pad_inches=1
    )
    plt.close()

    plt.figure(figsize=(12, 6))

    for column in model_params.get('real_cols'):
        plt.plot(real_data[column], label=f'Real {column} Price')

    for column in model_params.get('target_cols'):
        plt.plot(predicted_data[column], label=f'Predicted {column} Price', alpha=0.7)

    plt.title(ticker)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(f'{chart_path}/{model_params.get("model_type","LSTM")}_{ticker}_predicted_4lines.png',
        transparent=False,
        bbox_inches='tight',
        pad_inches=1
    )
    plt.close()

def main(args):
    use_model(
        ticker=args.ticker,
        model_name=args.model_name,
        model_params={
            'time_window_size': args.time_window_size,
            'real_cols': args.real_cols,
            'target_cols': args.target_cols
        },
        data_path=args.data_path,
        save_csv=args.save_csv,
        output_path=args.output_path
    )

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
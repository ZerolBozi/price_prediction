from strategy import trend_strategy
from train import TrainDQN
from use_model import use_model

def main():
    original_data, predict_data = use_model(
        'binance_btc_kline_1h_spot',
        'lstm_stock_crypto',
        {
            'time_window_size': 1,
            'real_cols': ['close'],
            'target_cols': ['close', 'high', 'low']
        },
    )

    train_dqn = TrainDQN(
        ticker='binance_btc_kline_1h_spot',
        original_data=original_data,
        predict_data=predict_data,
        model_params={
            'trading_strategy': trend_strategy,
            'window_size': 5
        },
        checkpoint_name='dqn_btc_1h',
        
    )

    train_dqn.run()

if __name__ == "__main__":
    main()
from strategy import trend_strategy,daytrading_strategy
from train import TrainDQN
from use_model import use_model

def main():
    original_data, predict_data = use_model(
        'NVDA',
        'lstm_NVDA',
        {
            'time_window_size': 1,
            'real_cols': ['close'],
            'target_cols': ['close', 'high', 'low']
        },
    )

    train_dqn = TrainDQN(
        ticker='NVDA',
        original_data=original_data,
        predict_data=predict_data,
        model_params={
            'trading_strategy': trend_strategy,
            'window_size': 30
        },
        should_save_model=True,
        model_name='dqn_NVDA',
        checkpoint_name='dqn_NVDA',
   
    )

    train_dqn.run()

if __name__ == "__main__":
    main()
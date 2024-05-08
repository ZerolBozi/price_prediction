import pandas as pd
from processer import create_indicators, scale_datasets, split_datasets, convert_to_lstm_format, get_y_idx
from train import Train
from strategy import trend_strategy, regression_strategy
from train import TrainDQN
from use_model import use_model,use_DQNmodel
from datetime import datetime
from downloader import downloader
from decimal import Decimal
from Indicators import Indicators

def main(
    ticker:str,
    trade_side:list,
    strategy_func: callable,
    predict_model_type:str="LSTM",
    predict_model_name:str=None,
    predict_train:bool=True,
    dqn_model_type:str="DQN",
    dqn_model_name:str=None,
    dqn_train:bool=True,
    download:bool=True
):
    """
    market: str -> market type
    ticker: str -> ticker
    predict_model_type: str -> model type for prediction
    predict_model_name: str -> model name for prediction
    predict_train: bool -> should train model for prediction. if False,use the model with predict_model_name 
    dqn_model_type: str -> model type for DQN.
    dqn_model_name: str -> model name for DQN. if None,use the model name is dqn_model_type and ticker
    dqn_train: bool -> should train model for DQN. if False,use the model with dqn_model_name
    download: bool -> download data
    """
    data = pd.read_csv(f'./history_datas/{ticker}.csv')
    if 'DateTime' in data.columns:  
        data.drop(["Date", "Time"], axis=1, inplace=True)
        data.rename(columns={"DateTime": "datetime", "Open": "open", "High":"high", "Low":"low", "Close": "close", "Volume": "volume", "Amount": "amount"}, inplace=True)
        
    if "USDT" not in ticker:
        data = Indicators.resample_klines(data,'1h')
        
    # output_path = f'./datas/{ticker}_processed'
    if ".US" in ticker:
        time_window_size = 40
    elif "USDT" in ticker:
        time_window_size = 120
    else:
        time_window_size = 25
        
    if predict_train:
        data_processed = create_indicators(
            data=data,
            save=False,
        )

        save_dataset: bool=False
        save_scalers: bool=False
        dataset_path: str='./dataset_scale'
        scalers_path: str='./data_scale'

        data_scaled_np, scalers = scale_datasets(
            data=data_processed,
            save_dataset=save_dataset,
            save_scalers=save_scalers,
            dataset_path=dataset_path,
            scalers_path=scalers_path
        )

        targets = ['close', 'high', 'low']

        y_idx_list = get_y_idx(data_processed.columns, targets)
        
        train_np, test_np = split_datasets(data_scaled_np, 0.6)

        batch_size = 32
        epoches = 2500

        original_data = {
            'datas': test_np[time_window_size:],
            'scalers': scalers,
            'real': ['close'],
            'predict': targets,
            'plot': targets
        }

        model_params = {
            'input_size': len(data_processed.columns),
            'output_size': len(targets),
            'hidden_size': 128,
            'fc_size': 128,
            'num_layers': 2,
            'dropout_prob': 0.10,
            'batch_first': True
        }

        train_data_loader = convert_to_lstm_format(train_np, time_window_size, y_idx_list, batch_size=batch_size)
        test_x_data, test_y_data = convert_to_lstm_format(test_np, time_window_size, y_idx_list)

        train_obj = Train(
            ticker=ticker,
            train_loader=train_data_loader,
            test_x_data=test_x_data,
            test_y_data=test_y_data,
            epoches=epoches,
            original_data=original_data,
            model_params=model_params,
            should_save_model=True,
            model_name=f"{predict_model_type.lower()}_{ticker}",
            checkpoint_name=f"{predict_model_type.lower()}_{ticker}",
            model_type=predict_model_type,
            use_early_stopping=True,
            patience=500
        )

        train_obj.run()
    
    if predict_model_name is None:
        predict_model_name = f"{predict_model_type.lower()}_{ticker}"

    original_data, predict_data = use_model(
        ticker=ticker,
        model_name=predict_model_name,
        model_params={
            'time_window_size': time_window_size,
            'real_cols': ['close'],
            'target_cols': ['close', 'high', 'low'],
            'model_type': predict_model_type
        },
        data=data,
        show_plot=True
    )
    
    if dqn_model_name is None or dqn_train:
        dqn_model_name = f"{dqn_model_type.lower()}_{ticker}"

    if dqn_train:
        train_dqn = TrainDQN(
            ticker=ticker,
            original_data=original_data,
            predict_data=predict_data,
            model_params={
                'trading_strategy': strategy_func,
                'window_size': time_window_size,
                'position_size_ratio': Decimal(0.5),
            },
            episodes=200,
            batch_size=2048,
            should_save_model=True,
            model_name=dqn_model_name,
            checkpoint_name=dqn_model_name,
            use_early_stopping=True,
            trade_side=trade_side
        )

        train_dqn.run()
    else:
        use_DQNmodel(
            ticker=ticker,
            model_name=dqn_model_name,
            model_params={
                'window_size': time_window_size,
                'trading_strategy': strategy_func,
                'position_size_ratio': Decimal(0.5),
                'model_type': dqn_model_type
            },
            data={
                'original_data': original_data,
                'predict_data': predict_data
            }
        )

if __name__ == "__main__":
    # markets={
    #     "tw":["2317", "2330", "2454", "2888"],
    #     # "tw":["2317","2330","2308","2454","2603","2356"],
    #     "us":["AAPL.US","JPM.US","MSFT.US","NLFX.US"],
    #     # "weighted_index":["^TWII"],
    #     #"crypto":["BTC","ETH","BNB","SOL","OKB","XRP"]
    # }
    import os
    from TradingEnvironment import Action
    
    tickers = os.listdir("./history_datas")
    tickers = [ticker.replace('.csv', '') for ticker in tickers]
    trade_side = [Action.long, Action.close_long]
    # change strategy
    strategy_func = regression_strategy
    
    ticker = "JPM.US"
    # for ticker in tickers:
    main(ticker, predict_model_type="BiGRU", trade_side=trade_side, strategy_func=strategy_func)
import pandas as pd
from datetime import datetime

from downloader import downloader
from processer import create_indicators, scale_datasets, split_datasets, convert_to_lstm_format, get_y_idx
from train import Train

if __name__ == "__main__":
    # market_tpye = 'tw'
    # ticker = '2330.TW'
    # interval = '1d'
    # start = datetime(2018,1,1)
    # end = datetime(2024,1,20)
    # output_path = f'./datas/{ticker}'

    # data = downloader(
    #     market_tpye=market_tpye, 
    #     ticker=ticker,
    #     interval=interval,
    #     start=start,
    #     end=end,
    #     save=True,
    #     output_path=output_path
    # )
    checkpoint_name = f"checkpoint3"
    ticker = "NVDA"

    data = pd.read_csv(f'./datas/{ticker}.csv')
    
    # output_path = f'./datas/{ticker}_processed'
    output_path = None

    data_processed = create_indicators(
        data=data,
        save=False,
        output_path=output_path
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

    time_window_size = 1
    batch_size = 32
    epoches = 750

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
        model_name=f"lstm_NVDA",
        checkpoint_name=checkpoint_name,
        use_early_stopping=True
    )

    train_obj.run()
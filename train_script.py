import os
import argparse

import pandas as pd
from datetime import datetime

from processer import create_indicators, scale_datasets, split_datasets, convert_to_lstm_format, get_y_idx
from train import Train

def get_parser():
    parser = argparse.ArgumentParser(description="train stock or crypto model")
    parser.add_argument("--ticker", type=str, help="stock or crypto data file name", required=True)
    parser.add_argument("--time_window_size", type=int, help="time window size", default=1)
    parser.add_argument("--split_ratio", type=float, help="split ratio", default=0.6)
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--hidden_size", type=int, help="hidden size", default=128)
    parser.add_argument("--fc_size", type=int, help="fc size", default=128)
    parser.add_argument("--num_layers", type=int, help="num layers", default=2)
    parser.add_argument("--dropout_prob", type=float, help="dropout prob", default=0.10)
    parser.add_argument("--batch_first", type=bool, help="batch first", default=True)
    parser.add_argument("--epoches", type=int, help="epoches", default=750)
    parser.add_argument("--model_name", type=str, help="model name", default="lstm_market")
    parser.add_argument("--save_model", type=bool, help="if save model == false, then save checkpoint", default=False)
    parser.add_argument("--checkpoint_name", type=str, help="checkpoint name", default="checkpoint")
    parser.add_argument("--use_early_stopping", type=str, help="use early stopping", default=True)
    parser.add_argument("--learning_rate", type=str, help="learning rate", default=0.00001)
    parser.add_argument("--model_type", type=str, help="model type", default="LSTM")
    parser.add_argument("--patience", type=str, help="patience", default=250)
    parser.add_argument("--min_delta", type=str, help="min delta", default=0.001)
    parser.add_argument("--targets", nargs='+', help="targets", default=['close', 'high', 'low'])
    return parser

def main(args):
    # init args
    ticker = args.ticker
    time_window_size = args.time_window_size
    split_ratio = args.split_ratio
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    fc_size = args.fc_size
    num_layers = args.num_layers
    dropout_prob = args.dropout_prob
    batch_first = args.batch_first
    epoches = args.epoches
    model_name = args.model_name
    save_model = args.save_model
    checkpoint_name = args.checkpoint_name
    use_early_stopping = args.use_early_stopping
    learning_rate = args.learning_rate
    model_type = args.model_type
    patience = args.patience
    min_delta = args.min_delta
    targets = args.targets

    # check data file exists
    if not os.path.exists(f'./datas/{ticker}.csv'):
        raise Exception(f"file {ticker}.csv not found")

    data = pd.read_csv(f'./datas/{ticker}.csv')

    # create indicators and scale datasets
    data_processed = create_indicators(data=data)
    data_scaled_np, scalers = scale_datasets(data=data_processed)

    y_idx_list = get_y_idx(data_processed.columns, targets)

    train_np, test_np = split_datasets(data_scaled_np, split_ratio)

    # Train class需要使用到的資料
    original_data = {
        'datas': test_np[time_window_size:],
        'scalers': scalers,
        'real': ['close'],
        'predict': targets,
        'plot': targets
    }

    # model參數
    model_params = {
        'input_size': len(data_processed.columns),
        'output_size': len(targets),
        'hidden_size': hidden_size,
        'fc_size': fc_size,
        'num_layers': num_layers,
        'dropout_prob': dropout_prob,
        'batch_first': batch_first
    }

    # convert to lstm format
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
        should_save_model=save_model,
        model_name=model_name,
        checkpoint_name=checkpoint_name,
        use_early_stopping=use_early_stopping,
        learning_rate=learning_rate,
        model_type=model_type,
        patience=patience,
        min_delta=min_delta
    )

    train_obj.run()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
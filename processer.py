import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from Indicators import Indicators

# 前77是0 這個有點奇怪, 暫時先這樣
MINIMUM_DATA_SIZE = 77

def create_indicators(data:pd.DataFrame, save: bool = False, output_path: str = None, data_type: str = 'csv'):
    ret_data = data.copy()

    # drop datetime
    if 'datetime' in ret_data.columns:
        ret_data.drop('datetime', axis=1, inplace=True)
    
    # drop unix
    if 'unix' in ret_data.columns:
        ret_data.drop('unix', axis=1, inplace=True)

    indicator_functions = [
        ('sma', Indicators.sma, 'close', ['sma']),
        ('ema', Indicators.ema, 'close', ['ema']),
        ('macd', Indicators.macd, 'close', ['macd', 'macd_signal']),
        ('rsi', Indicators.rsi, 'close', ['rsi']),
        ('stoch', Indicators.stoch, ['high', 'low', 'close'], ['stoch_k', 'stoch_d']),
        ('bb', Indicators.bb, 'close', ['bb_upper', 'bb_middle', 'bb_lower']),
        ('ichimoku', Indicators.ichimoku, ['high', 'low'], ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']),
        ('stoch_rsi', Indicators.stoch_rsi, 'close', ['stoch_rsi_k', 'stoch_rsi_d']),
        ('volume_ema', Indicators.ema, 'volume', ['volume_ema']),
        ('volume_sma', Indicators.sma, 'volume', ['volume_sma']),
        ('volume_bb', Indicators.bb, 'volume', ['volume_bb_upper', 'volume_bb_middle', 'volume_bb_lower'])
    ]

    for _, indicator_func, data_cols, ret_cols in indicator_functions:
        ret_values = indicator_func(data[data_cols])

        if type(ret_values) != tuple: ret_values = [ret_values]

        for idx, col in enumerate(ret_cols):
            ret_data[col] = ret_values[idx]

    # 把有0的橫排移除
    ret_data = ret_data[MINIMUM_DATA_SIZE:]

    if (save) and (output_path is not None):
        output_data(ret_data, data_type, output_path)

    return ret_data

def output_data(data: pd.DataFrame, data_type: str, output_path: str):
    """
    :param datatype: 資料類型, (csv, json, excel)
    :type datatype: str
    """
    if data_type not in ('csv', 'json', 'excel'):
        raise ValueError('datatype must be one of csv, json, excel')
    
    output_func = {
        'csv': (data.to_csv, '.csv'),
        'json': (data.to_json, '.json'),
        'excel': (data.to_excel, '.xlsx')
    }

    func, endwith = output_func.get(data_type)

    func(output_path + endwith, index=False)

def scale_datasets(data: pd.DataFrame, save_dataset: bool=False, save_scalers: bool=False, dataset_path: str=None, scalers_path: str=None):
    """
    數據縮放

    :return Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]
    """
    scalers = {}
    data_scaled = data.copy()

    for column in data.columns:
        scalers[column] = MinMaxScaler(feature_range=(0, 1))
        data_scaled[column] = scalers[column].fit_transform(data[[column]])
    
    data_scaled_np = data_scaled.to_numpy()

    if (save_dataset) and (dataset_path is not None):
        np.save(dataset_path + '.npy', data_scaled_np)

    if (save_scalers) and (scalers_path is not None):
        with open(scalers_path + '.pkl', 'wb') as file:
            pickle.dump(scalers, file)

    return data_scaled_np, scalers

def inverse_scale_datasets(scaled_data: np.ndarray, scalers: dict, original_columns: list) -> pd.DataFrame:
    """
    反轉縮放 (還原資料)

    :param scaled_data: The scaled data as a NumPy array.
    :param scalers: A dictionary of MinMaxScaler objects used for the original scaling.
    :param original_columns: List of column names in the original data.
    :return: DataFrame containing the inverse scaled data.
    """
    inverse_data = pd.DataFrame(scaled_data, columns=original_columns)

    for column in original_columns:
        scaler = scalers[column]
        inverse_data[column] = scaler.inverse_transform(inverse_data[[column]])

    return inverse_data

def split_datasets(data: np.ndarray, train_test_split_ratio: float):
    """
    根據參數ratio分割資料成訓練集與測試集

    :param data: NumPy array containing the data to be split.
    :param train_test_split_ratio: Ratio of the data to be used for training.
    :return: Tuple containing the training data and testing data.
    """
    train_size = int(len(data) * train_test_split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def convert_to_lstm_format(data: np.ndarray, time_window_size: int, y_idx_list: list, shuffle: bool=True, batch_size=None):
    """
    把np資料轉成LSTM格式

    :param data: NumPy array containing the time-series data.
    :param time_window_size: Size of the time window used for creating the sequences.
    :param future_window_size: Size of the future window used for creating the sequences.
    :param y_idx_list: List of indices of the columns to be used as targets.
    :param shuffle: Whether to shuffle the data.
    :param batch_size: Batch size to be used if a DataLoader is required.
    :return: A DataLoader if batch_size is specified, else a tuple of torch.Tensor (features, targets).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_data, y_data = [], []
    for i in range(len(data) - time_window_size):
        x_data.append(data[i:(i + time_window_size), :])
        y_data.append([data[i + time_window_size, y_idx] for y_idx in y_idx_list])
    
    x_data_np_arr = np.array(x_data)
    y_data_np_arr = np.array(y_data)

    x_data_tensor = torch.tensor(x_data_np_arr, dtype=torch.float32).to(device)
    y_data_tensor = torch.tensor(y_data_np_arr, dtype=torch.float32).to(device)

    if batch_size is not None:
        dataset = TensorDataset(x_data_tensor, y_data_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        return x_data_tensor, y_data_tensor
    
def get_y_idx(data_columns:pd.core.indexes.base.Index, targets: list) -> list:
    """
    :param data: 資料
    :param targets: 目標
    :return: 目標的index
    """
    if set(targets).issubset(set(data_columns)):
        return [data_columns.get_loc(target) for target in targets]
    return []

def dict_convert_to_tensor(data: dict, keys: list):
    """
    把字典轉成tensor

    :param data: 字典
    :param keys: 字典的key (可以自己設定順序)

    :return: list tensor
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return [torch.tensor(data[key], dtype=torch.float32).to(device) for key in keys]
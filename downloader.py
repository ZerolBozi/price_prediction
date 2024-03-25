import os
from time import sleep
from datetime import datetime

import ccxt
import pandas as pd
import shioaji as sj
import yfinance as yf

def downloader(
        market_tpye: str,
        ticker: str,
        interval: str,
        start: datetime,
        end: datetime,
        save: bool = False,
        output_path: str = None,
        data_type: str = 'csv',
        crypto_api_path: str = None,
        sj_api_path: str = None
    ):
    """
    :param market_tpye: 市場類型, (tw, us, crypto)
    :type market_tpye: str

    :param ticker: 商品名稱, (AAPL.US, 2330, BTC/USDT)
    :type ticker: str

    :param interval: 資料間隔, (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
    :type interval: str

    :param start: 資料開始時間, (datetime)
    :type start: datetime

    :param end: 資料結束時間, (datetime)
    :type end: datetime

    :param save: 是否儲存資料, (True, False)
    :type save: bool
    
    :param output_path: 輸出路徑, (None, 'data')
    :type output_path: str

    :param data_type: 資料類型, (csv, json, excel)
    :type data_type: str

    :param crypto_api_path: 加密貨幣API Key路徑, (None, 'api.txt')
    :type crypto_api_path: str

    :param sj_api_path: 台股API Key路徑, (None, 'api.txt')
    :type sj_api_path: str

    :return: 資料, 如果錯誤會返回None
    :rtype: pd.DataFrame
    """
    if (market_tpye == 'tw') and (interval in ['1d', '1w']): ticker = ticker + '.TW'
        
    data = download(market_tpye, ticker, interval, start, end, crypto_api_path, sj_api_path)
    
    if data is None:
        return None
    
    ret_data = data_format(data)

    if save and output_path is not None:
        output_data(ret_data, data_type, output_path)

    return ret_data

def download(market_tpye: str, ticker: str, interval: str, start: datetime, end: datetime, crypto_api_path: str = None, sj_api_path: str = None):
    """
    :param market_tpye: 市場類型, (tw, us, crypto, weighted_index)
    :type market_tpye: str

    :param ticker: 商品名稱, (AAPL.US, 2330, BTCUSDT, ^TWII)
    :type ticker: str

    :param interval: 資料間隔, (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
    :type interval: str

    :param start: 資料開始時間, (datetime)
    :type start: datetime

    :param end: 資料結束時間, (datetime)
    :type end: datetime

    :param crypto_api_path: 加密貨幣API Key路徑, (None, 'api.txt')
    :type crypto_api_path: str

    :param sj_api_path: 台股API Key路徑, (None, 'api.txt')
    :type sj_api_path: str

    :return: 資料, 如果錯誤會返回None
    :rtype: pd.DataFrame
    """
    # 檢查市場類型是否符合
    if market_tpye not in ('tw', 'us', 'crypto','weighted_index'):
        raise ValueError('market_tpye must be one of tw, us, crypto, weighted_index')
    
    # 檢查資料間隔是否符合
    if interval not in ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'):
        raise ValueError('interval must be one of 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w')
    
    # 使用yfinance下載資料, 僅限1d, 1w
    if (market_tpye == 'tw' or market_tpye == 'us' or market_tpye == 'weighted_index') and (interval in ['1d', '1w']):
        return download_from_yfinance(ticker, interval, start, end)
    
    # 台股如果使用永豐API, 只有1m資料, 如果要其他interval, 需要自行組合kbars
    if (market_tpye == 'tw') and (interval not in ['1d', '1w']):
        return download_from_shioaji(ticker, start, end, sj_api_path)
    
    if market_tpye == 'crypto':
        return download_from_binance(ticker, interval, start, end, crypto_api_path)

def download_from_yfinance(ticker: str, interval: str, start: datetime, end: datetime):
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        data.drop('Adj Close', axis=1, inplace=True)
        return data
    except Exception as e:
        print(e)
        return None
    
def download_from_shioaji(ticker: str, start: datetime, end: datetime, api_path: str):
    """
    下載台股1m資料需要永豐金證券的API, 這部分涉及到自己的銀行帳戶, 請用txt保存在自己本地的電腦內, 切勿將自己的API公開在網路上

    ex: c://api.txt
    content:
        api_key
        secret

    說明: 
        txt文件內第一行為api_key, 第二行為secret, 請按照這個規則設定, 否則將無法正常使用
    """
    if os.path.exists(api_path) is False:
        return None
    
    if start < datetime(2020, 3, 2):
        return None
    
    with open(api_path, 'r') as f:
        api_key = f.readline().strip()
        secret = f.readline().strip()
    
    sj_obj = init_sj(api_key, secret)
    
    if sj_obj is None:
        return None
    
    try:
        kbars = sj_obj.kbars(
            contract=sj_obj.Contracts.Stocks[ticker],
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
        )

        df = pd.DataFrame({**kbars})
        df.ts = pd.to_datetime(df.ts)
        df = df.reindex(columns=["ts", "Open", "High", "Low", "Close", "Volume", "Amount"])
        df.rename(columns={'ts': 'datetime'}, inplace=True)
        return df
    except Exception as e:
        print(e)
        return None

def download_from_binance(ticker: str, interval: str, start: datetime, end: datetime, api_path: str, market: str = 'spot'):
    """
    下載加密貨幣資料需要API Key跟Secret, 自己設定API相關的檔案路徑, 請用txt保存在自己本地的電腦內

    ex: c://api.txt
    content:
        api_key
        secret

    說明: 
        txt文件內第一行為api_key, 第二行為secret, 請按照這個規則設定, 否則將無法正常使用
    """
    if os.path.exists(api_path) is False:
        return None
    
    with open(api_path, 'r') as f:
        api_key = f.readline().strip()
        secret = f.readline().strip()

    binance_obj = init_binance(api_key, secret, market)
    
    if binance_obj is None:
        return None
    
    df_all_data = pd.DataFrame()
    _start = int(start.timestamp() * 1000)
    _end = int(end.timestamp() * 1000)

    timeframe_dict = {
        '1m': 60000,
        '5m': 300000,
        '15m': 900000,
        '30m': 1800000,
        '1h': 3600000,
        '4h': 14400000,
        '1d': 86400000,
        '1w': 604800000
    }

    while _start <= _end:
        data = binance_obj.fetch_ohlcv(ticker, timeframe=interval, since=_start, limit=1000)
        
        if len(data) < 0:
            break

        df = pd.DataFrame(data,columns=['unix','open','high','low','close','volume'])
        df['datetime'] = pd.to_datetime(df['unix'],unit='ms') + pd.TimedeltaIndex(hours=8)
        df = df.reindex(columns = ['datetime','unix','open','high','low','close', 'volume'])

        df_all_data = pd.concat([df_all_data, df], ignore_index=True)
        interval_ms = timeframe_dict.get(interval)
        _start = int(df.iloc[-1]['unix']) + interval_ms
        
        sleep(1)

    df_filtered = df_all_data[df_all_data['unix'] <= _end]
    return df_filtered

def init_sj(api_key: str, secret: str):
    sj_obj = sj.Shioaji()
    try:
        sj_obj.login(api_key, secret)
        return sj_obj
    except Exception as e:
        print(e)
        return None

def init_binance(api_key: str, secret: str, market: str):
    binance_obj = ccxt.binance({
        'apiKey': api_key,
        'secret': secret,
        'timeout': 15000,
        'enableRateLimit': True,
        'options': {
            'defaultType': market
        }
    })
    try:
        binance_obj.load_markets()
        return binance_obj
    except Exception as e:
        print(e)
        return None

def data_format(data: pd.DataFrame):
    ret_data = data.copy()
    
    if isinstance(ret_data.index, pd.DatetimeIndex):
        ret_data.reset_index(inplace=True)

    ret_data.columns = ret_data.columns.str.lower()

    if 'date' in ret_data.columns:
        ret_data.rename(columns={'date': 'datetime'}, inplace=True)

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

if __name__ == '__main__':
    save = True
    ticker = '^TWII'
    output_path = f'./datas/{ticker}'

    data = downloader(
        'weighted_index', 
        ticker,
        '1d',
        datetime(2024, 1, 1),
        datetime(2024, 1, 20),
        save= True,
        output_path=output_path,
    )

    # data = downloader(
    #     'crypto', 
    #     'BTC/USDT',
    #     '4h',
    #     datetime(2020, 1, 1),
    #     datetime(2020, 12, 31),
    #     save= True,
    #     output_path=output_path,
    #     api_path='./api.txt'
    # )
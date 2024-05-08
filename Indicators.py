import numpy as np
import pandas as pd

class Indicators:
    @staticmethod
    def verify_series(data:pd.Series, periods: list):
        """
        If verify successful, return the original data, else raise ValueError
        """
        if type(periods) != list:
            periods = [periods]

        for period in periods:
            if period > len(data) or period < 1:
                raise ValueError(f'Period {period} is invalid for the given data')
        
        return data
    
    @staticmethod
    def sma(close_data:pd.Series, period:int= 14)->pd.Series:
        """
        Simple Moving Average

        :param close_data: Series of close prices
        :type close_data: pd.Series

        :param period: Period of the moving average, defaults to 14
        :type period: int, optional

        :return: Series of the moving average
        """
        close_data = Indicators.verify_series(close_data, period)
        return close_data.rolling(period).mean().fillna(0)
    
    @staticmethod
    def ema(close_data:pd.Series, period:int = 14)->pd.Series:
        """
        Exponential Moving Average

        :param close_data: Series of close prices
        :type close_data: pd.Series

        :param period: Period of the moving average, defaults to 14
        :type period: int, optional

        :return: Series of the moving average
        """
        close_data = Indicators.verify_series(close_data, period)
        
        alpha = 2 / (period + 1)
        ema_values = [close_data.iloc[0]]

        for i in range(1, len(close_data)):
            ema_today = alpha * close_data.iloc[i] + (1 - alpha) * ema_values[-1]
            ema_values.append(ema_today)

        return pd.Series(ema_values, index=close_data.index)
    
    @staticmethod
    def macd(close_data:pd.Series, ma_type:str = 'ema',short_period:int = 12, long_period:int = 26, signal_period:int = 9)->tuple:
        """
        Moving Average Convergence Divergence

        :param close_data: Series of close prices
        :type close_data: pd.Series

        :param ma_type: Type of moving average to use, defaults to 'ema'
        :type ma_type: str, optional

        :param short_period: Period of the short moving average, defaults to 12
        :type short_period: int, optional

        :param long_period: Period of the long moving average, defaults to 26
        :type long_period: int, optional

        :param signal_period: Period of the signal line, defaults to 9
        :type signal_period: int, optional

        :return: Tuple of the macd and signal line
        """
        close_data = Indicators.verify_series(close_data, [short_period, long_period, signal_period])
        if short_period > len(close_data) or short_period < 1 or long_period > len(close_data) or long_period < 1 or signal_period > len(close_data) or signal_period < 1:
            raise ValueError('Short period cannot be greater than the length of the data')
        
        if short_period > long_period:
            tmp = short_period
            short_period = long_period
            long_period = tmp
        
        if ma_type == 'sma':
            short_ma = Indicators.sma(close_data, short_period)
            long_ma = Indicators.sma(close_data, long_period)
        else:
            short_ma = Indicators.ema(close_data, short_period)
            long_ma = Indicators.ema(close_data, long_period)

        macd = short_ma - long_ma
        signal = Indicators.ema(macd, signal_period)
        return macd, signal
    
    def _rma(close_data:pd.Series, period:int = 14)->pd.Series:
        """
        Relative Moving Average

        :param close_data: Series of close prices
        :type close_data: pd.Series

        :param period: Period of the moving average, defaults to 14
        :type period: int, optional

        :return: Series of the moving average
        """
        close_data = Indicators.verify_series(close_data, period)
        alpha = (1.0 / period) if period > 0 else 0.5
        rma = close_data.ewm(alpha=alpha, min_periods=period).mean()
        rma = rma.fillna(0)
        return rma
    
    @staticmethod
    def rsi(close_data:pd.Series, period:int = 14, simplify:bool = False)->pd.Series:
        """
        Relative Strength Index

        :param close_data: Series of close prices
        :type close_data: pd.Series

        :param period: Period of the moving average, defaults to 14
        :type period: int, optional

        :return: Series of the RSI
        """
        close_data = Indicators.verify_series(close_data, period)

        negative = close_data.diff(1)
        positive = negative.copy()
        negative[negative > 0] = 0
        positive[positive < 0] = 0
        positive_avg = Indicators._rma(positive, period)
        negative_avg = Indicators._rma(negative, period).abs()

        rsi = 100 * positive_avg / (positive_avg + negative_avg.abs())
        rsi = rsi.fillna(0)

        if simplify:
            rsi = rsi.apply(lambda x: 1 if x < 40 and x != 0 else 0)

        return rsi
    
    @staticmethod
    def bb(close_data:pd.Series, period:int = 20, std_mult:int = 2)->tuple:
        """
        Bollinger Bands

        :param close_data: Series of close prices
        :type close_data: pd.Series

        :param period: Period of the moving average, defaults to 20
        :type period: int, optional

        :param std: Standard deviation multiplier, defaults to 2
        :type std: int, optional

        :return: Tuple of the upper, middle, and lower bands
        """
        close_data = Indicators.verify_series(close_data, period)
        
        middle = Indicators.sma(close_data, period)
        std = close_data.rolling(period).std().fillna(0)
        upper = middle + (std * std_mult)
        lower = middle - (std * std_mult)
        return upper, middle, lower
    
    @staticmethod
    def ichimoku(data:pd.DataFrame, tenkan_period:int = 9, kijun_period:int = 26, senkou_b_period:int = 52)->tuple:
        """
        Ichimoku Cloud

        :param data: Dataframe of high and low prices
        :type data: pd.DataFrame

        :param tenkan_period: Period of the Tenkan-sen, defaults to 9
        :type tenkan_period: int, optional

        :param kijun_period: Period of the Kijun-sen, defaults to 26
        :type kijun_period: int, optional

        :param senkou_b_period: Period of the Senkou Span B, defaults to 52
        :type senkou_b_period: int, optional

        :return: Dataframe of the Tenkan-sen, Kijun-sen, Senkou Span A, and Senkou Span B
        """
        data = Indicators.verify_series(data, [tenkan_period, kijun_period, senkou_b_period])
        high = data.get('high', data.get('High', None))
        low = data.get('low', data.get('Low', None))

        if high is None or low is None:
            raise ValueError('Dataframe must contain a high and low column')
        
        tenkan_sen = (high.rolling(window=tenkan_period).max() +
                  low.rolling(window=tenkan_period).min()) / 2

        kijun_sen = (high.rolling(window=kijun_period).max() +
                    low.rolling(window=kijun_period).min()) / 2

        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)

        senkou_span_b = ((high.rolling(window=senkou_b_period).max() +
                        low.rolling(window=senkou_b_period).min()) / 2).shift(kijun_period)

        tenkan_sen = tenkan_sen.fillna(0)
        kijun_sen = kijun_sen.fillna(0)
        senkou_span_a = senkou_span_a.fillna(0)
        senkou_span_b = senkou_span_b.fillna(0)

        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

    @staticmethod
    def smi(data:pd.DataFrame, length:int = 10, smooth_length:int = 3, double_smooth_length:int = 3, simplify:bool = False)->pd.Series:
        """
        Stochastic Momentum Index

        :param data: Dataframe of high, low, and close prices
        :type data: pd.DataFrame

        :param length: Length of the SMI, defaults to 10
        :type length: int, optional

        :param smooth_length: Length of the smoothing, defaults to 3
        :type smooth_length: int, optional

        :param double_smooth_length: Length of the double smoothing, defaults to 3
        :type double_smooth_length: int, optional

        :return: Series of the SMI
        """    
        data = Indicators.verify_series(data, [length, smooth_length, double_smooth_length])
        high = data.get('high', data.get('High', None))
        low = data.get('low', data.get('Low', None))

        if high is None or low is None:
            raise ValueError('High and Low columns must be present in the dataframe')

        highest = high.rolling(window=length).max().fillna(0)
        lowest = low.rolling(window=length).min().fillna(0)
        _add = 0.5 * (highest + lowest)
        _sub = (highest - lowest)
        smi = 100 * Indicators.ema(Indicators.ema(data['Close'] - _add, smooth_length), double_smooth_length) / (0.5 * Indicators.ema(Indicators.ema(_sub, smooth_length), double_smooth_length))
        smi = smi.replace([np.inf, -np.inf], 0)

        if simplify:
            smi = smi.apply(lambda x: 1 if x < -30 and x != 0 else 0)
        
        return smi
    
    def _stoch(source:pd.Series, high:pd.Series, low:pd.Series, length:int = 14)->pd.Series:
        highest = high.rolling(window=length).max().fillna(0)
        lowest = low.rolling(window=length).min().fillna(0)
        _stoch = 100 * (source - lowest) / (highest - lowest)
        return _stoch
    
    @staticmethod
    def stoch(data:pd.DataFrame, period_k: int = 14, smooth_k: int = 1, period_d: int = 3, simplify:bool = False)->tuple:
        """
        Stochastic Oscillator

        :param data: Dataframe of high, low, and close prices
        :type data: pd.DataFrame

        :param period_k: Period of the %K, defaults to 14
        :type period_k: int, optional

        :param smooth_k: Period of the smoothing, defaults to 1
        :type smooth_k: int, optional

        :param period_d: Period of the %D, defaults to 3
        :type period_d: int, optional

        :return: Tuple of the %K and %D
        """
        data = Indicators.verify_series(data, [period_k, smooth_k, period_d])
        close = data.get('close', data.get('Close', None))
        high = data.get('high', data.get('High', None))
        low = data.get('low', data.get('Low', None))

        if close is None or high is None or low is None:
            raise ValueError('High and Low and Close columns must be present in the dataframe')
        
        _stoch = Indicators._stoch(close, high, low, period_k)
        k = Indicators.sma(_stoch, smooth_k)
        d = Indicators.sma(k, period_d)

        if simplify:
            k = k.apply(lambda x: 1 if x < 30 and x != 0 else 0)
            d = d.apply(lambda x: 1 if x < 30 and x != 0 else 0)

        return k, d

    @staticmethod
    def stoch_rsi(close_data:pd.Series, smooth_k: int = 3, smooth_d: int = 3, rsi_length:int = 14, stoch_length:int = 14, simplify:bool = False)->tuple:
        """
        Stochastic RSI

        :param close_data: Series of close prices
        :type close_data: pd.Series

        :param smooth_k: Period of the smoothing, defaults to 3
        :type smooth_k: int, optional

        :param smooth_d: Period of the smoothing, defaults to 3
        :type smooth_d: int, optional

        :param rsi_length: Period of the RSI, defaults to 14
        :type rsi_length: int, optional

        :param stoch_length: Period of the %K, defaults to 14
        :type stoch_length: int, optional

        :return: Series of the Stochastic RSI
        """
        close_data = Indicators.verify_series(close_data, [smooth_k, smooth_d, rsi_length, stoch_length])
        rsi = Indicators.rsi(close_data, rsi_length)
        k = Indicators.sma(Indicators._stoch(rsi, rsi, rsi, stoch_length), smooth_k)
        d = Indicators.sma(k, smooth_d)

        if simplify:
            k = k.apply(lambda x: 1 if x < 30 and x != 0 else 0)
            d = d.apply(lambda x: 1 if x < 30 and x != 0 else 0)

        return k, d

    @staticmethod
    def target(close_data: pd.Series) -> pd.Series:
        """
        Create a target column based on price changes for a given series.

        :param close_data: Series of close prices
        :type close_data: pd.Series

        :return: Series with target values
        """
        target = pd.Series(0, index=close_data.index)  # Initialize target series with 0 values

        for i in range(1, len(close_data)):
            if close_data.iloc[i] > close_data.iloc[i - 1]:
                target.iloc[i] = 1
            else:
                target.iloc[i] = 0

        return target
    @staticmethod
    def resample_klines(klines:pd.DataFrame, interval:str)->pd.DataFrame:
        """
        Resample klines to a different interval

        :param klines: Klines
        :type klines: pd.DataFrame

        :param interval: New interval
        :type interval: str

        :return: Resampled klines
        :rtype: pd.DataFrame
        """
        interval = interval.replace('m', 'min')
        klines['datetime'] = pd.to_datetime(klines['datetime'])
        klines.set_index('datetime', inplace=True)
        resampled_klines = klines.resample(interval).agg(
            {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }
        )
        resampled_klines = resampled_klines.reset_index()
        resampled_klines = resampled_klines.dropna(how='all').query('volume != 0')
        return resampled_klines

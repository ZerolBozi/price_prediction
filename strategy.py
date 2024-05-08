import pandas as pd
import numpy as np

from TradingEnvironment import Action

def trend_strategy_prob(state: dict):
    _balance = state['balance']
    _curr_original_data = state['curr_original_data'].copy()
    _curr_predict_data = state['curr_predict_data'].copy()
    _curr_positions= state['current_positions']
    
    prob = np.array([0,0,0,0,0])

    # 沒有多單
    if _curr_positions.get('long') is None:
        if _balance < _curr_original_data['open']:
            prob[0] = 0.6
            
        if _curr_original_data['open'] > _curr_predict_data['close']:
            prob[1] += 0.2
        
        if _curr_original_data['open'] > _curr_predict_data['high']:
            prob[1] += 0.2
            
        if _curr_original_data['open'] > _curr_predict_data['low']:
            prob[1] += 0.2
    # 有多單        
    else:
        if _curr_original_data['open'] < _curr_predict_data['close']:
            prob[3] += 0.2
            
        if _curr_original_data['open'] < _curr_predict_data['high']:
            prob[3] += 0.2
            
        if _curr_original_data['open'] < _curr_predict_data['low']:
            prob[3] += 0.2
            
    # 沒有空單
    if _curr_positions.get('short') is None:
        if _balance < _curr_original_data['open']:
            prob[0] = 0.6
            
        if _curr_original_data['open'] < _curr_predict_data['close']:
            prob[2] += 0.2
        
        if _curr_original_data['open'] < _curr_predict_data['high']:
            prob[2] += 0.2
            
        if _curr_original_data['open'] < _curr_predict_data['low']:
            prob[2] += 0.2
    # 有多單        
    else:
        if _curr_original_data['open'] > _curr_predict_data['close']:
            prob[4] += 0.2
            
        if _curr_original_data['open'] > _curr_predict_data['high']:
            prob[4] += 0.2
            
        if _curr_original_data['open'] > _curr_predict_data['low']:
            prob[4] += 0.2
    
    prob[0] = 0.6
    return prob


def trend_strategy(state: dict):
    _balance = state['balance']
    _curr_original_data = state['curr_original_data'].copy()
    _curr_predict_data = state['curr_predict_data'].copy()
    _curr_positions= state['current_positions']

    # 餘額不足時, 不進行交易

    if _curr_positions.get('long') is None:
        if _balance < _curr_original_data['close']:
            return Action.hold
        # 相差的門檻值先不設定
        if (_curr_original_data['open'] > _curr_predict_data['close'] and
            _curr_original_data['open'] > _curr_predict_data['high'] and
            _curr_original_data['open'] > _curr_predict_data['low']):
            # 當真實價格大於所有的預測價格時, 則開啟long
            return Action.long
    else:
        if (_curr_original_data['open'] < _curr_predict_data['close'] or
            _curr_original_data['open'] < _curr_predict_data['high'] or
            _curr_original_data['open'] < _curr_predict_data['low']):
            # 當真實價格小於任一預測價格時, 則關閉long
            return Action.close_long
    
    if _curr_positions.get('short') is None:
        if _balance < _curr_original_data['open']:
            return Action.hold
        if (_curr_original_data['open'] < _curr_predict_data['close'] and
            _curr_original_data['open'] < _curr_predict_data['high'] and
            _curr_original_data['open'] < _curr_predict_data['low']):
            # 當真實價格小於所有的預測價格時, 則開啟short
            return Action.short
    else:
        if (_curr_original_data['open'] > _curr_predict_data['close'] or
            _curr_original_data['open'] > _curr_predict_data['high'] or
            _curr_original_data['open'] > _curr_predict_data['low']):
            # 當真實價格大於任一預測價格時, 則關閉short
            return Action.close_short
        
    return Action.hold

def regression_strategy(state: dict):
    _balance = state['balance']
    _curr_original_data = state['curr_original_data'].copy()
    _curr_predict_data = state['curr_predict_data'].copy()
    _curr_positions= state['current_positions']

    # 餘額不足時, 不進行交易
    if _curr_positions.get('short') is None:
        if _balance < _curr_original_data['close']:
            return Action.hold
        # 相差的門檻值先不設定
        if (_curr_original_data['open'] > _curr_predict_data['close'] and
            _curr_original_data['open'] > _curr_predict_data['high'] and
            _curr_original_data['open'] > _curr_predict_data['low']):
            # 當真實價格大於所有的預測價格時, 則開啟long
            return Action.short
    else:
        if (_curr_original_data['open'] < _curr_predict_data['close'] or
            _curr_original_data['open'] < _curr_predict_data['high'] or
            _curr_original_data['open'] < _curr_predict_data['low']):
            # 當真實價格小於任一預測價格時, 則關閉long
            return Action.close_short
    
    if _curr_positions.get('long') is None:
        if _balance < _curr_original_data['open']:
            return Action.hold
        if (_curr_original_data['open'] < _curr_predict_data['close'] and
            _curr_original_data['open'] < _curr_predict_data['high'] and
            _curr_original_data['open'] < _curr_predict_data['low']):
            # 當真實價格小於所有的預測價格時, 則開啟short
            return Action.long
    else:
        if (_curr_original_data['open'] > _curr_predict_data['close'] or
            _curr_original_data['open'] > _curr_predict_data['high'] or
            _curr_original_data['open'] > _curr_predict_data['low']):
            # 當真實價格大於任一預測價格時, 則關閉short
            return Action.close_long
        
    return Action.hold

def daytrading_strategy(state: dict):
    _balance = state['balance']
    _curr_original_data = state['curr_original_data'].copy()
    _curr_predict_data = state['curr_predict_data'].copy()
    _curr_positions= state['current_positions']
    _curr_positions_oid = state['current_positions_oid']

    # 餘額不足時, 不進行交易
    if _balance < _curr_original_data['close']:
        return Action.hold
    
    if _curr_positions.get('long') is None:
        # 相差的門檻值先不設定
        if (_curr_original_data['open'] < _curr_predict_data['close']):
            # 當真實價格大於所有的預測價格時, 則開啟long
            return Action.long
    else:
        if (_curr_original_data['open'] > _curr_predict_data['close'] or 
            _curr_original_data['open'] > _curr_predict_data['high']):
            # 當真實價格大於任一預測價格時, 則關閉long
            return Action.close_long
    
    if _curr_positions.get('short') is None:
        if (_curr_original_data['open'] > _curr_predict_data['close']):
            # 當真實價格小於所有的預測價格時, 則開啟short
            return Action.short
    else:
        _short_oid= [_curr_positions_oid[oid] for oid in _curr_positions_oid if _curr_positions_oid[oid].side == 'short']
        if (_curr_original_data['open'] < _curr_predict_data['close'] and
            _curr_original_data['open'] < _short_oid[0].price):
            # 當真實價格大於任一預測價格時, 則關閉short
            return Action.close_short
        
    return Action.hold
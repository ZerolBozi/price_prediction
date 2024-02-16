from time import time
from decimal import Decimal
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Action:
    space = 5
    hold = 0
    long = 1
    short = 2
    close_long = 3
    close_short = 4

@dataclass
class TradeInfo:
    """
    TradeInfo Structure:
        - int order_id:
            a unique identifier for each order
        - str order_type:
            type of order, it's position state, only two options ['open', 'close']
        - str: ticker:
            the ticker of the stock or crypto or any items
        - int current_step:
            current step of the window
        - str side:
            side of the order, only two options ['long', 'short']
        - Decimal price:
            entry price of the order
        - Decimal size:
            size of the order
        - Decimal amount:
            price * size
        - Decimal fee:
            fee of the order
        - Decimal tax:
            tax of the order
        - Decimal cost:
            amount + fee + tax
        - int from_position:
            the order_id of the position that the order is from (only for close position order)
        - Decimal profit:
            profit of the order (only for close position order)
        - Decimal return_rate:
            return rate of the order (only for close position order)
    """
    order_id: int
    order_type: str
    ticker: str
    current_step: int
    side: str
    price: Decimal
    size: Decimal
    amount: Decimal
    fee: Decimal
    tax: Decimal
    cost: Decimal
    from_position: int = -1
    profit: Decimal = Decimal(0)
    return_rate: Decimal = Decimal(0)

class TradingEnvironment:
    def __init__(
            self,
            ticker: str,
            original_data: pd.DataFrame,
            predict_data: pd.DataFrame,
            initial_balance: Decimal,
            position_size_ratio: Decimal,
            window_size: int,
            action_size: int,
            trading_strategy: callable
        ):
        self.ticker = ticker
        self.original_data = original_data
        self.predict_data = predict_data
        self.initial_balance = initial_balance
        self.position_size_ratio = position_size_ratio
        self.window_size = window_size
        self.action_size = action_size

        # 交易策略function
        self.trading_strategy = trading_strategy

        self.reset()

    def get_state_keys(self):
        return ['balance']
        # return ['balance', 'trading_profits', 'trading_returns']
    
    def reset(self):
        self.balance = self.initial_balance
        self.current_positions = {'long': None, 'short': None}
        self.current_positions_oid = {}
        self.history_positions = []
        self.history_close_positions = []
        self.trading_profits = []
        self.trading_returns = []
        self.current_step = 0
        return self.get_state()

    def get_state(self):
        """
        return {
            'balance': float,
            'trading_profits': list,
            'trading_returns': list,
        }
        """
        return self.balance

    def step(self, action: int):
        state = self.get_state()

        _state = {'balance': state}
        _state['curr_original_data'] = self.original_data.iloc[self.current_step]
        _state['curr_predict_data'] = self.predict_data.iloc[self.current_step]
        _state['current_positions'] = self.current_positions
        
        _action = self.trading_strategy(_state)
        price = self.original_data['close'].iloc[self.current_step]
        free_balance = self.balance * self.position_size_ratio
        size = free_balance // Decimal(price)
        reward = 0.0

        if (action == Action.long) and (_action == Action.long):
            self.open_position('long', price, size, self.current_step)
            reward = 1.0
        
        elif (action == Action.short) and (_action == Action.short):
            self.open_position('short', price, size, self.current_step)
            reward = 1.0

        elif (action == Action.close_long) and (_action == Action.close_long):
            _, _reward = self.close_position(self.current_positions['long'].order_id, price, size, self.current_step)
            reward = float(_reward)

        elif (action == Action.close_short) and (_action == Action.close_short):
            _, reward = self.close_position(self.current_positions['short'].order_id, price, size, self.current_step)

        self.current_step += 1

        done = self.current_step == self.window_size

        return self.get_state(), action, reward, done
        
    def open_position(self, side: str, price: Decimal, size: Decimal, c_step: int) -> int:
        """
        fee default is 0, open position no need tax

        return open position order_id
        """
        fee =  Decimal(0)
        tax = Decimal(0)
        order_id = int(time())

        amount = price * size

        o_trade_info = TradeInfo(
            order_id=order_id,
            order_type='open',
            ticker=self.ticker,
            current_step=c_step,
            side=side,
            price=price,
            size=size,
            amount=amount,
            fee=fee,
            tax=tax,
            cost=amount
        )

        self.current_positions[side] = o_trade_info
        self.current_positions_oid[order_id] = o_trade_info
        self.history_positions.append(o_trade_info)

        # amount = cost
        self.balance -= amount

        return o_trade_info.order_id

    def close_position(self, oid: int, price: Decimal, size: Decimal, c_step: int) -> tuple:
        """
        fee and tax default is 0

        return close position order_id
        """
        fee = Decimal(0)
        tax = Decimal(0)
        order_id = int(time())

        cost = self.current_positions[oid].cost
        amount = price * size

        if self.current_positions[oid].side == "long":
            side = "short"
            profit = (self.current_positions[oid].price - price) * size
        elif self.current_positions[oid].side == "short":
            side = "long"
            profit = (price - self.current_positions[oid].price) * size
        else:
            profit = Decimal(0) 
            
        return_rate = profit / self.current_positions[oid].cost

        c_trade_info = TradeInfo(
            order_id=order_id,
            order_type='close',
            ticker=self.ticker,
            current_step=c_step,
            side=side,
            price=price,
            size=size,
            amount=amount,
            fee=fee,
            tax=tax,
            cost=amount,
            from_position=oid,
            profit=profit,
            return_rate=return_rate
        )

        self.current_positions[side] = None
        self.current_positions_oid.pop(oid)
        self.history_positions.append(c_trade_info)
        self.history_close_positions.append(c_trade_info)
        self.trading_profits.append(profit)
        self.trading_returns.append(return_rate)

        # update blance
        self.balance = self.balance + cost + profit

        return c_trade_info.order_id, profit

    def render(self):
        # show trading record
        plt.figure(figsize=(12, 6))

        close_price = self.original_data['close']
        plt.plot(close_price, 'close price')

        marker_dict = {
            'open': {
                'long': {'marker': '^', 'color': 'g'},
                'short': {'marker': 'v', 'color': 'r'}
            },
            'close': {
                'long': {'marker': '^', 'color': 'b'},
                'short': {'marker': 'v', 'color': 'b'}
            }
        }

        for trade in self.history_positions:
            plt.plot(
                close_price, 
                marker_dict[trade.order_type][trade.side]['marker'], 
                markersize=10, 
                color=marker_dict[trade.order_type][trade.side]['color']
            )

        plt.title(f"{self.ticker} trading record")
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

        # show trading profits
        plt.figure(figsize=(12, 6))
        plt.plot(self.trading_profits, label='Trading Profits')
        plt.title(f"{self.ticker} trading profits")
        plt.xlabel('Time')
        plt.ylabel('Profits')
        plt.legend()
        plt.show()

        # show trading returns
        plt.figure(figsize=(12, 6))
        plt.plot(self.trading_returns, label='Trading Returns')
        plt.title(f"{self.ticker} trading returns")
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.legend()
        plt.show()
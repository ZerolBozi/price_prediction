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
    
    def reset(self)->np.ndarray:
        self.balance = self.initial_balance
        self.current_positions = {'long': None, 'short': None}
        self.current_positions_oid = {}
        self.history_positions = []
        self.history_close_positions = []
        self.total_profits = Decimal(0)
        self.trading_profits = []
        self.trading_returns = []
        self.current_step = 0
        self.total_trading_profits = []
        return self.get_state()

    def get_state(self)->np.ndarray:
        # datas.append(float(self.balance))
        # datas.append(float(self.total_profits))

        # for col in self.original_data.columns:
        #     datas.append(self.original_data[col][self.current_step])
        
        # for col in self.predict_data.columns:
        #     datas.append(self.predict_data[col][self.current_step])

        # return np.hstack(datas).astype(np.float64)
        datas = []
        
        window_size = self.window_size + 1
        d = self.current_step - window_size + 1 #判斷當前step是否足夠window_size
        block = []
        if d<0:
            for i in range(-d):
                block.append(self.original_data['close'][0]) #取第一筆資料填充不足window_size的部分
            for i in range(self.current_step+1):
                block.append(self.original_data['close'][i]) #填充剩餘的部分
        else:
            block = list(self.original_data['close'][d : self.current_step + 1]) #取得當前的window_size的資料
                
        for i in range(window_size-1):
            datas.append((block[i + 1] - block[i])/(block[i]+0.0001)) #輸出每天的漲跌幅作為狀態輸入

        return np.hstack(datas).astype(np.float64)

    def step(self, action: int)->tuple[np.ndarray, float, bool]:
        _state = {
            'balance': self.balance,
            'curr_original_data': self.original_data.iloc[self.current_step],
            'curr_predict_data': self.predict_data.iloc[self.current_step],
            'current_positions': self.current_positions,
            'current_positions_oid': self.current_positions_oid,
            'current_step': self.current_step,
        }
        
        trading_strategy_action = self.trading_strategy(_state)
        price = Decimal(self.original_data['close'].iloc[self.current_step])
        free_balance = self.balance * self.position_size_ratio
        size = free_balance // price
        reward = 0.0

        if (
            (action == Action.long) and 
            (trading_strategy_action == Action.long) and 
            (self.current_positions['long'] is None)
        ):
            self.open_position('long', price, size)
            reward = 1.0
        
        elif (
            (action == Action.short) and 
            (trading_strategy_action == Action.short) and 
            (self.current_positions['short'] is None)
        ):
            self.open_position('short', price, size)
            reward = 1.0

        elif (
            (action == Action.close_long) and 
            (trading_strategy_action == Action.close_long) and 
            (self.current_positions['long'] is not None)
        ):
            _, _reward = self.close_position(self.current_positions['long'].order_id, price, size)
            reward = float(_reward)

        elif (
            (action == Action.close_short) and 
            (trading_strategy_action == Action.close_short) and 
            (self.current_positions['short'] is not None)
        ):
            _, reward = self.close_position(self.current_positions['short'].order_id, price, size)

        self.current_step += 1

        done = self.current_step == len(self.original_data['open']) - 2

        return self.get_state(), reward, done
        
    def open_position(self, side: str, price: Decimal, size: Decimal) -> int:
        """
        fee default is 0, open position no need tax

        return open position order_id
        """
        fee =  Decimal(0)
        tax = Decimal(0)
        c_step = self.current_step
        order_id = int(time()) + c_step

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

        # amount = cost
        self.balance -= amount

        return o_trade_info.order_id

    def close_position(self, oid: int, price: Decimal, size: Decimal) -> tuple:
        """
        fee and tax default is 0

        return close position order_id
        """
        fee = Decimal(0)
        tax = Decimal(0)
        c_step = self.current_step
        order_id = int(time()) + c_step

        o_trade_info = self.current_positions_oid[oid]

        cost = o_trade_info.cost
        amount = price * size

        if o_trade_info.side == "long":
            side = "short"
            profit = (price - o_trade_info.price) * size
        elif o_trade_info.side == "short":
            side = "long"
            profit = (o_trade_info.price - price) * size
        else:
            profit = Decimal(0) 
            
        return_rate = profit / o_trade_info.cost

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

        self.current_positions[o_trade_info.side] = None
        self.current_positions_oid.pop(oid)
        self.history_positions.append((o_trade_info, c_trade_info))
        self.history_close_positions.append(c_trade_info)
        self.total_profits += profit
        self.trading_profits.append(profit)
        self.trading_returns.append(return_rate)

        # update blance
        self.balance = self.balance + cost + profit
        self.total_trading_profits.append(self.total_profits)

        return c_trade_info.order_id, profit

    def render(self):
        with open('./test.txt', 'w+') as f:
            for o_trade, c_trade in self.history_positions:
                f.write("OPEN TRADE INFO\n")
                f.write(f"order_id: {o_trade.order_id}\n")
                f.write(f"order_type: {o_trade.order_type}\n")
                f.write(f"ticker: {o_trade.ticker}\n")
                f.write(f"current_step: {o_trade.current_step}\n")
                f.write(f"side: {o_trade.side}\n")
                f.write(f"price: {o_trade.price}\n")
                f.write(f"size: {o_trade.size}\n")
                f.write(f"cost: {o_trade.cost}\n")
                f.write(f"from_position: {o_trade.from_position}\n")
                f.write(f"profit: {o_trade.profit}\n")
                f.write(f"return_rate: {o_trade.return_rate}\n")
                f.write("------------------------------------\n")
                f.write("CLOSE TRADE INFO\n")
                f.write(f"order_id: {c_trade.order_id}\n")
                f.write(f"order_type: {c_trade.order_type}\n")
                f.write(f"ticker: {c_trade.ticker}\n")
                f.write(f"current_step: {c_trade.current_step}\n")
                f.write(f"side: {c_trade.side}\n")
                f.write(f"price: {c_trade.price}\n")
                f.write(f"size: {c_trade.size}\n")
                f.write(f"cost: {c_trade.cost}\n")
                f.write(f"from_position: {c_trade.from_position}\n")
                f.write(f"profit: {c_trade.profit}\n")
                f.write(f"return_rate: {c_trade.return_rate}\n")
                f.write("------------------------------------\n")

        # show trading record
        plt.figure(figsize=(12, 6))

        close_price = self.original_data['close']
        plt.plot(close_price, label='close price')

        marker_dict = {
            'open': {
                'long': {'marker': '^', 'color': 'g'},
                'short': {'marker': 'v', 'color': 'r'}
            },
            'close': {
                'long': {'marker': '^', 'color': 'b'},
                'short': {'marker': 'v', 'color': 'c'}
            }
        }

        for o_trade, c_trade in self.history_positions:
            plt.plot(
                close_price, 
                marker_dict[o_trade.order_type][o_trade.side]['marker'], 
                markersize=10, 
                color=marker_dict[o_trade.order_type][o_trade.side]['color'],
                markevery=o_trade.current_step
            )
            plt.plot(
                close_price, 
                marker_dict[c_trade.order_type][c_trade.side]['marker'], 
                markersize=10, 
                color=marker_dict[c_trade.order_type][c_trade.side]['color'],
                markevery=c_trade.current_step
            )


        plt.title(f"{self.ticker} trading record")
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))

        plt.plot(close_price, label='close price')

        for o_trade, c_trade in self.history_positions:
            plt.scatter(
                o_trade.current_step,
                o_trade.price,
                s=100,
                marker=marker_dict[o_trade.order_type][o_trade.side]['marker'],
                color=marker_dict[o_trade.order_type][o_trade.side]['color'],
            )
            plt.scatter(
                c_trade.current_step,
                c_trade.price,
                s=100,
                marker=marker_dict[c_trade.order_type][c_trade.side]['marker'],
                color=marker_dict[c_trade.order_type][c_trade.side]['color'],
            )
            

        plt.title(f"{self.ticker} trading record")
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

        # show trading profits
        plt.figure(figsize=(12, 6))
        plt.plot(self.total_trading_profits, label='Trading Profits')
        plt.title(f"{self.ticker} trading profits")
        plt.xlabel('Time')
        plt.ylabel('Profits')
        plt.legend()
        plt.show()

        # show trading returns
        # plt.figure(figsize=(12, 6))
        # plt.plot(self.trading_returns, label='Trading Returns')
        # plt.title(f"{self.ticker} trading returns")
        # plt.xlabel('Time')
        # plt.ylabel('Returns')
        # plt.legend()
        # plt.show()
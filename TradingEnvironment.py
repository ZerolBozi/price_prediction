from time import time
from decimal import Decimal
from dataclasses import dataclass

import numpy as np
import pandas as pd
import csv
import os

import matplotlib
matplotlib.use('Agg') # fix RuntimeError
from matplotlib import pyplot as plt

from Indicators import Indicators

US_WINDOW_SIZE = 40
TW_WINDOW_SIZE = 25
CRYPTO_WINDOW_SIZE = 120

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
            trading_strategy: callable,
            model_type: str='DQN',
            trade_side: list = [Action.long, Action.close_long]
        ):
        self.ticker = ticker
        self.original_data = original_data
        self.predict_data = predict_data
        self.initial_balance = initial_balance
        self.position_size_ratio = position_size_ratio
        self.window_size = window_size
        self.action_size = action_size
        self.chart_path = f'./records/{self.ticker}/chart'
        if not os.path.exists(self.chart_path): os.mkdir(self.chart_path)
        # 交易策略function
        self.trading_strategy = trading_strategy
        self.model_type = model_type
        self.trade_side = trade_side

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
        self.total_trading_profits = []
        self.positive_trading_profits = []
        self.nagative_trading_profits = []
        self.current_step = 0
        return self.get_state()

    def get_state(self)->np.ndarray:
        datas = []

        # datas.append(float(self.balance))
        # datas.append(float(self.total_profits))

        # for col in self.original_data.columns:
        #     datas.append(self.original_data[col][self.current_step])
        
        # for col in self.predict_data.columns:
        #     datas.append(self.predict_data[col][self.current_step])

        # return np.hstack(datas).astype(np.float64)
        
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
        size = round(free_balance / price, 4) if "USDT" in self.ticker else free_balance // price
        _reward = 0.0

        if (
            (action == Action.long) and 
            (trading_strategy_action == Action.long) and 
            (self.current_positions['long'] is None) and 
            action in self.trade_side
        ):
            self.open_position('long', price, size)
        
        elif (
            (action == Action.short) and 
            (trading_strategy_action == Action.short) and 
            (self.current_positions['short'] is None) and
            action in self.trade_side
        ):
            self.open_position('short', price, size)

        elif (
            (action == Action.close_long) and 
            (trading_strategy_action == Action.close_long) and 
            (self.current_positions['long'] is not None)
        ):
            _, _, return_rate = self.close_position(self.current_positions['long'].order_id, price, self.current_positions['long'].size)
            _reward = float(return_rate)
            
        elif (
            (action == Action.close_short) and 
            (trading_strategy_action == Action.close_short) and 
            (self.current_positions['short'] is not None)
        ):
            _, _, return_rate = self.close_position(self.current_positions['short'].order_id, price, self.current_positions['short'].size)
            _reward = float(return_rate)

        # 隔天漲跌幅
        next_day_ratio = (self.original_data['close'][self.current_step + 1] - self.original_data['close'][self.current_step]) / self.original_data['close'][self.current_step]

        if (self.current_positions['long'] is not None) or (action == Action.close_long):
            reward = (next_day_ratio * -1) + _reward if action == Action.close_long else next_day_ratio
        elif (self.current_positions['short'] is not None) or (action == Action.close_short):
            reward = (next_day_ratio * -1) + _reward
        else:
            reward = (next_day_ratio * -1) * 0.1

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

        if profit > 0:
            self.positive_trading_profits.append(profit)
        else:
            self.nagative_trading_profits.append(profit)

        return c_trade_info.order_id, profit, return_rate

    def render(self):
        if not os.path.exists(f'./records/{self.ticker}'):
            os.mkdir(f'./records/{self.ticker}')
        
        with open(f'./records/{self.ticker}/{self.ticker}.csv', 'a+',newline='') as f:
            field = ['ticker','order_id','order_type','current_step','side','price','size','cost','from_position','profit','return_rate']
            f.seek(0)
            csvWriter = csv.DictWriter(f, fieldnames = field) #建立Writer物件
            csvReader = list(csv.reader(f))
            try:
                if csvReader[0] != field:
                    csvWriter.writeheader() #寫入標題
            except:
                    csvWriter.writeheader()
            for o_trade, c_trade in self.history_positions:
                csvWriter.writerow({
                    'ticker': o_trade.ticker,
                    'order_id': o_trade.order_id,
                    'order_type': o_trade.order_type,
                    'current_step': o_trade.current_step,
                    'side': o_trade.side,
                    'price': o_trade.price,
                    'size': o_trade.size,
                    'cost': o_trade.cost,
                    'from_position': o_trade.from_position,
                    'profit': o_trade.profit,
                    'return_rate': o_trade.return_rate
                })
                csvWriter.writerow({
                    'ticker': c_trade.ticker,
                    'order_id': c_trade.order_id,
                    'order_type': c_trade.order_type,
                    'current_step': c_trade.current_step,
                    'side': c_trade.side,
                    'price': c_trade.price,
                    'size': c_trade.size,
                    'cost': c_trade.cost,
                    'from_position': c_trade.from_position,
                    'profit': c_trade.profit,
                    'return_rate': c_trade.return_rate
                })

        records = pd.read_csv(f"./records/{self.ticker}/{self.ticker}.csv")
        price_data = pd.read_csv(f"./history_datas/{self.ticker}.csv")
        
        if "Date" in price_data.columns:
            price_data.drop(["Date", "Time"], axis=1, inplace=True)
            price_data.rename(columns={"DateTime": "datetime", "Open": "open", "High":"high", "Low":"low", "Close": "close", "Volume": "volume", "Amount": "amount"}, inplace=True)
       
        if 'unix' in price_data.columns:
            price_data['datetime'] = pd.to_datetime(price_data['unix'], unit='ms')
            price_data = price_data[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            
        price_data = Indicators.resample_klines(price_data,'1h')
        
        if ".US" in self.ticker:
            tmp_time_window_size = US_WINDOW_SIZE
        elif "USDT" in self.ticker:
            tmp_time_window_size = CRYPTO_WINDOW_SIZE
        else:
            tmp_time_window_size = TW_WINDOW_SIZE
        # records.to_csv(f"./records/{self.ticker}/{self.ticker}_mix.csv", index=False)
        
        price_data['order_type'] = ''
        price_data['side'] = ''
        price_data['open_price'] = ''
        price_data['cost'] = ''
        price_data['from_position'] = ''
        price_data['profit'] = ''
        price_data['return_rate'] = ''
        
        price_data = price_data.reset_index(drop=True)
        
        for idx, record in records.iterrows():
            price_data.loc[record['current_step'] + tmp_time_window_size + 77, ['order_type', 'side', 'open_price', 'size', 'cost', 'from_position', 'profit', 'return_rate']] = [record['order_type'], record['side'], record['price'], record['size'], record['cost'], record['from_position'], record['profit'], record['return_rate']]

        price_data.to_csv(f"./records/{self.ticker}/trading_records.csv", index=False)
        
        with open(f'./records/records.csv', 'a+',newline='') as f:
            field = ['ticker','order_id','order_type','current_step','side','price','size','cost','from_position','profit','return_rate']
            f.seek(0)
            csvWriter = csv.DictWriter(f, fieldnames = field) #建立Writer物件
            csvReader = list(csv.reader(f))
            try:
                if csvReader[0] != field:
                    csvWriter.writeheader() #寫入標題
            except:
                    csvWriter.writeheader()
            for o_trade, c_trade in self.history_positions:
                csvWriter.writerow({
                    'ticker': o_trade.ticker,
                    'order_id': o_trade.order_id,
                    'order_type': o_trade.order_type,
                    'current_step': o_trade.current_step,
                    'side': o_trade.side,
                    'price': o_trade.price,
                    'size': o_trade.size,
                    'cost': o_trade.cost,
                    'from_position': o_trade.from_position,
                    'profit': o_trade.profit,
                    'return_rate': o_trade.return_rate
                })
                csvWriter.writerow({
                    'ticker': c_trade.ticker,
                    'order_id': c_trade.order_id,
                    'order_type': c_trade.order_type,
                    'current_step': c_trade.current_step,
                    'side': c_trade.side,
                    'price': c_trade.price,
                    'size': c_trade.size,
                    'cost': c_trade.cost,
                    'from_position': c_trade.from_position,
                    'profit': c_trade.profit,
                    'return_rate': c_trade.return_rate
                })

        # show trading record
        if not os.path.isdir(self.chart_path):
            os.makedirs(self.chart_path)

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
        close_price = self.original_data['close']
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
        plt.savefig(f'{self.chart_path}/{self.model_type}_{self.ticker}_record.png',
            transparent=False,
            bbox_inches='tight',
            pad_inches=1
        )
        plt.close()
        #plt.show()

        # show trading profits
        plt.figure(figsize=(12, 6))
        plt.plot(self.total_trading_profits, label='Trading Profits')
        plt.title(f"{self.ticker} trading profits")
        plt.xlabel('Time')
        plt.ylabel('Profits')
        plt.legend()
        plt.savefig(f'{self.chart_path}/{self.model_type}_{self.ticker}_profits.png',
            transparent=False,
            bbox_inches='tight',
            pad_inches=1
        )
        plt.close()
        
        # process mdd
        # peak = self.total_trading_profits[0]
        max_drawdown_percent = Decimal("0")
        max_drawdown = self.balance
        max_drawdown_list = []
        
        for i in range(len(self.total_trading_profits)):
            tmp_list = self.total_trading_profits[0:i+1]
            peak = max(tmp_list)
            peak = max(peak, Decimal("1"))
            drawdown = (self.total_trading_profits[i] - peak)
            drawdown_percent = (self.total_trading_profits[i] - peak) / peak
            
            max_drawdown_list.append(drawdown)
            
            if drawdown < max_drawdown:
                max_drawdown = drawdown
                max_drawdown_percent = drawdown_percent

        tmp_df = pd.DataFrame({
            'Balance': self.total_trading_profits,
            'MDD': max_drawdown_list
        })

        plt.figure(figsize=(12, 6))
        plt.fill_between(tmp_df.index, tmp_df['MDD'], step="pre", alpha=0.4)
        plt.title('Maximum Drawdown (MDD) over Transactions')
        plt.xlabel('Transaction Number')
        plt.ylabel('Maximum Drawdown (MDD)')
        plt.savefig(f'{self.chart_path}/{self.model_type}_{self.ticker}_mdd_area.png',
            transparent=False,
            bbox_inches='tight',
            pad_inches=1
        )
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.total_trading_profits, label='Trading Profits')
        plt.fill_between(tmp_df.index, tmp_df['MDD'], step="pre", alpha=0.4)
        plt.title(f"{self.ticker} trading profits & MDD area")
        plt.xlabel('Time')
        plt.ylabel('Profits & MDD')
        plt.legend()
        plt.savefig(f'{self.chart_path}/{self.model_type}_{self.ticker}.png',
            transparent=False,
            bbox_inches='tight',
            pad_inches=1
        )
        plt.close()

        win_rate = len(self.positive_trading_profits) / len(self.trading_profits)
        avg_positive = np.mean(self.positive_trading_profits)
        avg_negative = np.mean(self.nagative_trading_profits)
        odds = Decimal(avg_positive) / np.abs(Decimal(avg_negative))
        # expected_value = win_rate * odds - (1 - win_rate)
        expected_value = Decimal(win_rate) * (1 + odds) - 1
        max_loss = np.min(self.trading_profits)
        mar = np.sum(self.trading_profits) / np.abs(max_drawdown)
        sqn = Decimal(np.mean(self.trading_profits) / np.std(self.trading_profits)) * Decimal(np.sqrt(len(self.trading_profits)))

        print(f"{self.ticker} trading record")
        print(self.initial_balance)
        print("total profits: ", self.total_profits)
        print(f"win rate: {win_rate}")
        print(f"avg positive: {avg_positive}")
        print(f"avg negative: {avg_negative}")
        print(f"odds: {odds}")
        print(f"expected value: {expected_value}")
        print(f"max loss: {max_loss}")
        print(f"mdd: {max_drawdown}")
        print(f"mdd percentage': {max_drawdown_percent}")
        print(f"mar: {mar}")
        print(f"sqn: {sqn}")

        with open(f'./records/{self.ticker}/results.csv', 'a+',newline='') as f:
            field = ['ticker','initial_balance','total_profits','win_rate','avg_positive','avg_negative','odds','expected_value','max_loss','mdd','mdd_percentage','mar','sqn']
            f.seek(0)
            csvWriter = csv.DictWriter(f, fieldnames = field) #建立Writer物件
            csvReader = list(csv.reader(f))
            try:
                if csvReader[0] != field:
                    csvWriter.writeheader() #寫入標題
            except:
                    csvWriter.writeheader()
            csvWriter.writerow({
                'ticker': self.ticker,
                'initial_balance': self.initial_balance,
                'total_profits': self.total_profits,
                'win_rate': win_rate,
                'avg_positive': avg_positive,
                'avg_negative': avg_negative,
                'odds': odds,
                'expected_value': expected_value,
                'max_loss': max_loss,
                'mdd': max_drawdown,
                'mdd_percentage': max_drawdown_percent,
                'mar': mar,
                'sqn': sqn
            })


        # show trading returns
        # plt.figure(figsize=(12, 6))
        # plt.plot(self.trading_returns, label='Trading Returns')
        # plt.title(f"{self.ticker} trading returns")
        # plt.xlabel('Time')
        # plt.ylabel('Returns')
        # plt.legend()
        # plt.show()
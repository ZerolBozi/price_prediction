import numpy as np
import pandas as pd

class TradingEnvironment:
    def __init__(
            self,
            ticker: str,
            original_data: pd.DataFrame,
            predict_data: pd.DataFrame,
            initial_balance: float,
            n_actions: int,
            window_size: int,
            trading_strategy: callable = None
        ):
        self.ticker = ticker
        self.original_data = original_data
        self.predict_data = predict_data
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.n_actions = n_actions
        self.window_size = window_size

        # 交易策略function
        self.trading_strategy = trading_strategy

        self.shares_held = 0
        self.current_step = 0

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0

    def get_state(self):
        pass

    def step(self, action:int, codition:bool=False):
        pass

    def render(self):
        pass
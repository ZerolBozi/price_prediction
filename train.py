import os
import random
from decimal import Decimal
from collections import deque

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics  
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from processer import inverse_scale_datasets
from TradingEnvironment import TradingEnvironment
from models import LSTM, GRU, DQN, DoubleDQN, DuelingDQN

class RepalyBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return torch.stack(state), torch.tensor(action), torch.tensor(reward), torch.stack(next_state), torch.tensor(done)

    def __len__(self):
        return len(self.buffer)
    
class TrainDQN:
    def __init__(
        self,
        ticker: str,
        original_data: pd.DataFrame,
        predict_data: pd.DataFrame,
        model_params: dict,
        model_type: str='DQN',
        epoches: int=1000,
        learning_rate: float=0.00001,
        should_save_model: bool=False,
        model_path: str="./models",
        model_name: str=None,
    ):
        """
        model_params: dict
            :key 'initial_balance', type: Decimal
            :key 'position_size_ratio', type: Decimal
            :key 'window_size', type: int
            :key 'memory_size', type: int
            :key 'batch_size', type: int
            :key 'reward_decay', type: Decimal
            :key 'e_greedy', type: Decimal
            :key 'replace_target_iter', type: int
            :key 'e_greedy_increment', type: Decimal
            :key 'trading_strategy', type: callable
            :key 'dqn_units', int

        """
        if not all(key in model_params.keys() for key in ['trading_strategy']):
            raise Exception("model_params should have keys: trading_strategy")
        
        self.ticker = ticker
        self.original_data = original_data
        self.predict_data = predict_data
        self.model_params = model_params
        self.model_type = model_type.upper()
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.should_save_model = should_save_model
        self.model_path = model_path
        self.model_name = model_name

        self.env = TradingEnvironment(
            ticker=self.ticker,
            original_data=self.original_data,
            predict_data=self.predict_data,
            initial_balance=self.model_params.get('initial_balance', Decimal(10000)),
            position_size_ratio=self.model_params.get('position_size_ratio', Decimal(0.5)),
            window_size=self.model_params.get('window_size', 10),
            trading_strategy=self.model_params.get('trading_strategy')
        )

        model_dict = {
            'DQN': DQN(
                input_size=self.env.window_size,
                output_size=self.env.action_size,
                units=self.model_params.get('dqn_units', 128)
            ),
            'DoubleDQN': DoubleDQN(
                input_size=self.env.window_size,
                output_size=self.env.action_size,
                units=self.model_params.get('dqn_units', 128)
            ),
            'DuelingDQN': DuelingDQN(
                input_size=self.env.window_size,
                output_size=self.env.action_size,
                units=self.model_params.get('dqn_units', 128)
            )
        }

        self.model = model_dict.get(self.model_type, "DQN")
    
    def run(self):
        pass

    def save_model(self):
        pass

class Train:
    def __init__(
        self,
        ticker: str,
        train_loader: DataLoader,
        test_x_data: tuple,
        test_y_data: tuple,
        epoches: int,
        original_data: dict,
        model_params: dict,
        should_save_model: bool=False,
        model_path: str="./models",
        checkpoint_path: str="./checkpoints",
        model_name: str=None,
        checkpoint_name: str=None,
        learning_rate: float=0.00001,
        model_type: str='LSTM',
        use_early_stopping: bool=True,
        patience=250, 
        min_delta=0.001,
    ):
        """
        :original_data
        :key 'datas', type: np.ndarray
        :key 'scalers', type: dict
        :key 'real', type: list[str] (column name)
        :key 'predict', type: list[str] (column name)
        :key 'plot', type: list[str] (column name)
        """
        # check original_data keys
        if not all(key in original_data.keys() for key in ['datas', 'scalers', 'real', 'predict', 'plot']):
            raise Exception("original_data should have keys: datas, scalers, real, predict, plot")
        
        if not all(key in model_params.keys() for key in ['input_size', 'output_size']):
            raise Exception("model_params should have keys: input_size, output_size")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.ticker = ticker
        self.train_loader = train_loader
        self.test_x_data = test_x_data
        self.test_y_data = test_y_data
        self.epoches = epoches
        self.original_data = original_data
        self.model_params = model_params
        self.model_type = model_type.upper()

        self.should_save_model = should_save_model
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.checkpoint_name = checkpoint_name

        # input_size = next(iter(self.train_loader))[0].size(2)

        # output_size從original_data裡面的predict欄位數量來決定
        model_dict = {
            'LSTM': LSTM(
                input_size=self.model_params.get('input_size'), 
                output_size=self.model_params.get('output_size'),
                hidden_size=self.model_params.get('hidden_size', 128),
                fc_size=self.model_params.get('fc_size', 128),
                num_layers=self.model_params.get('num_layers', 2),
                dropout_prob=self.model_params.get('dropout_prob', 0.10),
                batch_first=self.model_params.get('batch_first', True)
            ),
            'GRU': GRU(
                input_size=self.model_params.get('input_size'), 
                output_size=self.model_params.get('output_size'),
                hidden_size=self.model_params.get('hidden_size', 128),
                fc_size=self.model_params.get('fc_size', 128),
                num_layers=self.model_params.get('num_layers', 2),
                dropout_prob=self.model_params.get('dropout_prob', 0.10),
                batch_first=self.model_params.get('batch_first', True)
            )
        }

        self.model = model_dict.get(self.model_type, "LSTM")
        self.model.to(self.device)

        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []
        self.predict_result = None

        # early stopping init
        self.use_early_stopping = use_early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        self.load_checkpoint()

    def load_checkpoint(self):
        if (self.checkpoint_path is None) or (self.checkpoint_name is None):
            return
        
        if not os.path.exists(f"{self.checkpoint_path}/{self.checkpoint_name}.pth"):
            return
        
        checkpoint = torch.load(f"{self.checkpoint_path}/{self.checkpoint_name}.pth")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['val_loss']
        # self.epochs_no_improve = checkpoint['epoch']

        print('load checkpoint successed!')

    def run(self):
        for epoch in range(self.epoches):
            self.model.train()
            train_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)

            # val
            output, val_loss = self.verify()
            self.val_losses.append(val_loss)
            self.predict_result = output

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epoches}], Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")

            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            # check early stoppping
            if (self.use_early_stopping) and (self.epochs_no_improve >= self.patience):
                print("Early stopping triggered")
                break

        self.show_evaluate()
        self.save_model(epoch, val_loss)

    def verify(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            outputs = self.model(self.test_x_data)
            loss = self.criterion(outputs, self.test_y_data)
            val_loss = loss.item()

        return outputs.cpu().numpy(), val_loss

    def show_evaluate(self):
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.legend()
        plt.show()

        self.calc_metrics()

    def calc_metrics(self):
        real_data_np = self.original_data['datas']

        original_columns = list(self.original_data['scalers'].keys())

        real_data = inverse_scale_datasets(real_data_np, self.original_data['scalers'], original_columns)
        predicted_data = inverse_scale_datasets(self.predict_result, self.original_data['scalers'], self.original_data.get('predict'))
        
        # 取出real list裡面與predict list相同的欄位
        common_columns = list(set(self.original_data.get('real')).intersection(set(self.original_data.get('predict'))))

        for column in common_columns:
            mse = mean_squared_error(real_data[column], predicted_data[column])
            r2_score = metrics.r2_score(real_data[column], predicted_data[column])
            print(f"MSE ({column}):", mse)
            print(f"R^2 Score ({column}):", r2_score)

        plt.figure(figsize=(12, 6))

        for column in common_columns:
            if column in self.original_data.get('plot'):
                plt.plot(real_data[column], label=f'Real {column} Price')

        for column in self.original_data.get('predict'):
            if column in self.original_data.get('plot'):
                plt.plot(predicted_data[column], label=f'Predicted {column} Price', alpha=0.7)

        plt.title(self.ticker)
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    def save_model(self, epoch: int, val_loss:float):
        if (self.model_path is None) or (self.model_name is None):
            return
        
        if (self.should_save_model):
            torch.save(self.model, f'{self.model_path}/{self.model_name}.pt')
        else:
            self.save_checkpoint(epoch, val_loss)

    def save_checkpoint(self, epoch: int, val_loss:float):
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, f'{self.checkpoint_path}/{self.checkpoint_name}_epoch_{epoch}.pth')
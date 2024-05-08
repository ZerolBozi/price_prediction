import os
import random
from decimal import Decimal
from collections import deque

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics  
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg') # fix RuntimeError
from matplotlib import pyplot as plt

from TradingEnvironment import TradingEnvironment, Action
from models import LSTM, GRU, BiLSTM, BiGRU, CNNLSTM, CNNGRU, DQN, DoubleDQN, DuelingDQN
from processer import inverse_scale_datasets

class ReplayBuffer:
    def __init__(self, capacity: int, device: str):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.vstack(states).astype(np.float32)
        actions = np.array(actions).astype(np.int64)
        rewards = np.array(rewards).astype(np.float32)
        next_states = np.vstack(next_states).astype(np.float32)
        dones = np.array(dones).astype(np.float32)

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        return states, actions, rewards, next_states, dones

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
        episodes: int=1000,
        reward_decay: Decimal=0.9,
        batch_size: int=32,
        learning_rate: float=0.00001,
        e_greedy: float=0.1,
        should_save_model: bool=False,
        model_path: str="./models",
        model_name: str=None,
        checkpoint_path: str="./checkpoints",
        checkpoint_name: str=None,
        use_early_stopping: bool=True,
        patience: int=100,
        trade_side: list = [Action.long, Action.close_long, Action.short, Action.close_short]
    ):
        """
        model_params: dict
            :key 'initial_balance', type: Decimal
            :key 'position_size_ratio', type: Decimal
            :key 'window_size', type: int
            :key 'memory_size', type: int
            :key 'trading_strategy', type: callable
            :key 'dqn_units', int

            # DQN params (not used)
            :key 'e_greedy', type: Decimal
            :key 'replace_target_iter', type: int
            :key 'e_greedy_increment', type: Decimal
        """
        if not all(key in model_params.keys() for key in ['trading_strategy']):
            raise Exception("model_params should have keys: trading_strategy")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.ticker = ticker
        self.original_data = original_data
        self.predict_data = predict_data
        self.model_params = model_params
        self.model_type = model_type
        self.episodes = episodes
        self.reward_decay = reward_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.should_save_model = should_save_model
        self.model_path = model_path
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_name
        self.e_greedy = e_greedy
        
        # early stopping
        self.use_early_stopping = use_early_stopping
        self.patience = patience
        self.best_reward = float('-inf')
        self.counter = 0
        
        self.trade_side = trade_side
        
        self.env = TradingEnvironment(
            ticker=self.ticker,
            original_data=self.original_data,
            predict_data=self.predict_data,
            initial_balance=self.model_params.get('initial_balance', Decimal(100000)),
            position_size_ratio=self.model_params.get('position_size_ratio', Decimal(0.5)),
            window_size=self.model_params.get('window_size', 1),
            action_size=Action.space,
            trading_strategy=self.model_params.get('trading_strategy'),
            model_type=self.model_type,
            trade_side=trade_side
        )

        model_dict = {
            'DQN': DQN(
                n_observations=len(self.env.get_state()),
                n_actions=self.env.action_size,
                units=self.model_params.get('dqn_units', 128)
            ),
            'DoubleDQN': DoubleDQN(
                input_size=self.env.window_size,
                n_actions=self.env.action_size,
                units=self.model_params.get('dqn_units', 128)
            ),
            'DuelingDQN': DuelingDQN(
                input_size=self.env.window_size,
                n_actions=self.env.action_size,
                units=self.model_params.get('dqn_units', 128)
            )
        }

        self.net = model_dict.get(self.model_type, "DQN")
        self.target_net = model_dict.get(self.model_type, "DQN")

        self.net.to(self.device)
        self.target_net.to(self.device)

        self.criteria = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.model_params.get('memory_size', 10000), device=self.device)

        self.load_checkpoint()

    def load_checkpoint(self):
        if (self.checkpoint_path is None) or (self.checkpoint_name is None):
            return
        
        if not os.path.exists(f"{self.checkpoint_path}/{self.checkpoint_name}.pth"):
            return
        
        checkpoint = torch.load(f"{self.checkpoint_path}/{self.checkpoint_name}.pth")

        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.replay_buffer.buffer = checkpoint['replay_buffer']

        print('load checkpoint successed!')

    def select_action(self, state: np.ndarray):
        # 因為會出現waring, 所以要先轉np.array()
        _state = np.array(state)
        state_tensor = torch.tensor(_state, dtype=torch.float32).to(self.device)

        if np.random.uniform() < self.e_greedy:
            action = np.random.choice(self.env.action_size)
        else:
            with torch.no_grad():
                q_values = self.net(state_tensor).detach().cpu().squeeze(0)
                action = torch.argmax(q_values).item()
        
        return action
    
    def optimize_model(self, batch: tuple):

        self.net.train()
        self.target_net.train()
        if len(self.replay_buffer) % 300 == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        # self.target_net.load_state_dict(self.net.state_dict())

        state, action, reward, next_state, done = batch

        torch.as_tensor(state, dtype=torch.float32).to(self.device)
        torch.as_tensor(action, dtype=torch.int64).to(self.device)
        torch.as_tensor(reward, dtype=torch.float32).to(self.device)
        torch.as_tensor(next_state, dtype=torch.float32).to(self.device)
        torch.as_tensor(done, dtype=torch.float32).to(self.device)

        current_q_values = self.net(state).gather(1, action.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(1)[0]
            target_q_values = reward + self.reward_decay * next_q_values * (1 - done)

        loss = self.criteria(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def run(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                total_reward += float(reward)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                    
                if len(self.replay_buffer) > self.batch_size and len(self.replay_buffer) % 5 == 0:
                    batch = self.replay_buffer.sample(self.batch_size)
                    self.optimize_model(batch)
            
            print(f"Episode {episode}, Total Reward: {total_reward}")

            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.counter = 0
            else:
                self.counter += 1

            if (self.use_early_stopping) and (self.counter >= self.patience):
                print("Early stopping triggered")
                break
        self.save_model(episode, total_reward)            
        self.env.render()

    def save_model(self, episode: int, total_reward: float):
        if (self.model_path is None) or (self.model_name is None):
            return
        
        if (self.should_save_model):
            torch.save(self.net, f'{self.model_path}/{self.model_name}.pt')
            self.save_checkpoint(episode, total_reward)
        else:
            self.save_checkpoint(episode, total_reward)

    def save_checkpoint(self, episode: int, total_reward:float):
        checkpoint = {
            'episode': episode,
            'total_reward': total_reward,
            'net_state_dict': self.net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer': self.replay_buffer.buffer
        }
        torch.save(checkpoint, f'{self.checkpoint_path}/{self.checkpoint_name}_episode_{episode}.pth')

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

        if ('CNN' in model_type) and ('input_channels' not in model_params.keys()):
            raise Exception("model_params should have keys: input_channels")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.ticker = ticker
        self.train_loader = train_loader
        self.test_x_data = test_x_data
        self.test_y_data = test_y_data
        self.epoches = epoches
        self.original_data = original_data
        self.model_params = model_params
        self.model_type = model_type

        self.should_save_model = should_save_model
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.checkpoint_name = checkpoint_name
        self.chart_path = f'./records/{self.ticker}/chart'
        if not os.path.exists(f'./records/{self.ticker}'): os.mkdir(f'./records/{self.ticker}')
        if not os.path.exists(self.chart_path): os.mkdir(self.chart_path)

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
            ),
            'BiLSTM': BiLSTM(
                input_size=self.model_params.get('input_size'), 
                output_size=self.model_params.get('output_size'),
                hidden_size=self.model_params.get('hidden_size', 128),
                fc_size=self.model_params.get('fc_size', 128),
                num_layers=self.model_params.get('num_layers', 2),
                dropout_prob=self.model_params.get('dropout_prob', 0.10),
                batch_first=self.model_params.get('batch_first', True)
            ),
            'BiGRU': BiGRU(
                input_size=self.model_params.get('input_size'), 
                output_size=self.model_params.get('output_size'),
                hidden_size=self.model_params.get('hidden_size', 128),
                fc_size=self.model_params.get('fc_size', 128),
                num_layers=self.model_params.get('num_layers', 2),
                dropout_prob=self.model_params.get('dropout_prob', 0.10),
                batch_first=self.model_params.get('batch_first', True)
            ),
            'CNNLSTM': CNNLSTM(
                input_channels=self.model_params.get('input_channels', 1),
                output_size=self.model_params.get('output_size'),
                seq_length=self.model_params.get('input_size'),
                feture_size=self.model_params.get('feture_size', 1),
                hidden_size=self.model_params.get('hidden_size', 128),
                output_channels=self.model_params.get('output_channels', 16),
                kernel_size=self.model_params.get('kernel_size', 3),
                cnn_out_size=self.model_params.get('cnn_out_size', 32),
                pool_kernel_size=self.model_params.get('pool_kernel_size', 2),
                lstm_out_size=self.model_params.get('lstm_out_size', 64),
                fc_size=self.model_params.get('fc_size', 128),
                num_layers=self.model_params.get('num_layers', 2),
                dropout_prob=self.model_params.get('dropout_prob', 0.10),
                batch_first=self.model_params.get('batch_first', True)
            ),
            'CNNGRU': CNNGRU(
                input_channels=self.model_params.get('input_channels', 1),
                output_size=self.model_params.get('output_size'),
                seq_length=self.model_params.get('input_size'),
                feture_size=self.model_params.get('feture_size', 1),
                hidden_size=self.model_params.get('hidden_size', 128),
                output_channels=self.model_params.get('output_channels', 16),
                kernel_size=self.model_params.get('kernel_size', 3),
                cnn_out_size=self.model_params.get('cnn_out_size', 32),
                pool_kernel_size=self.model_params.get('pool_kernel_size', 2),
                lstm_out_size=self.model_params.get('lstm_out_size', 64),
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
        # self.best_val_loss = checkpoint['val_loss']
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
        plt.title(self.ticker)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.legend()
        plt.savefig(f'{self.chart_path}/{self.model_type}_{self.ticker}_loss.png',
            transparent=False,
            bbox_inches='tight',
            pad_inches=1
        )
        plt.close()
        #plt.show()

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
            is_first = not os.path.exists("./predict_records.csv")
            with open("./predict_records.csv", "a+") as f:
                if is_first:
                    f.write("Ticker,Model,MSE,R2_Score\n")
                f.write(f"{self.ticker},{self.model_type},{mse},{r2_score}\n")
                
            if not os.path.exists(f'./records/{self.ticker}'):
                os.mkdir(f'./records/{self.ticker}')
            
            is_first = not os.path.exists(f'./records/{self.ticker}/{self.ticker}_predict_record.csv')
            with open(f'./records/{self.ticker}/{self.ticker}_predict_record.csv', "a+") as f:
                if is_first:
                    f.write("Ticker,Model,MSE,R2_Score\n")
                f.write(f"{self.ticker},{self.model_type},{mse},{r2_score}\n")
                
            print(f"MSE ({column}):", mse)
            print(f"R^2 Score ({column}):", r2_score)

        # plot real_close, predicted_close, predicted_high, predicted_low
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
        plt.savefig(f'{self.chart_path}/{self.model_type}_{self.ticker}_predicted_4lines.png',
            transparent=False,
            bbox_inches='tight',
            pad_inches=1
        )
        plt.close()
        #plt.show()

        # plot real_close, predicted_close
        plt.figure(figsize=(12, 6))

        plt.plot(real_data['close'], label=f'Real close Price')
        plt.plot(predicted_data['close'], label=f'Predicted close Price', alpha=0.7)

        plt.title(self.ticker)
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(f'{self.chart_path}/{self.model_type}_{self.ticker}_predicted_2lines.png',
            transparent=False,
            bbox_inches='tight',
            pad_inches=1
        )
        plt.close()
        #plt.show()

    def save_model(self, epoch: int, val_loss:float):
        if (self.model_path is None) or (self.model_name is None):
            return
        
        if (self.should_save_model):
            torch.save(self.model, f'{self.model_path}/{self.model_name}.pt')
        
        self.save_checkpoint(epoch, val_loss)

    def save_checkpoint(self, epoch: int, val_loss:float):
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, f'{self.checkpoint_path}/{self.checkpoint_name}_epoch_{epoch}.pth')
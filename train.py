import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics  
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from models import LSTM, GRU
from processer import inverse_scale_datasets

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
        model_path: str=None,
        learning_rate: float=0.00001,
        model_name: str='LSTM',
        patience=250, 
        min_delta=0.001
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
        self.model_name = model_name.upper()

        self.should_save_model = should_save_model
        self.model_path = model_path

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
            'GRU': GRU(input_size=self.model_params.get('input_size'), output_size=self.model_params.get('output_size'))
        }

        self.model = model_dict.get(self.model_name)
        self.model.to(self.device)

        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []
        self.predict_result = None

        # early stopping init
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

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
                print(f"Epoch [{epoch+1}/{self.epoches}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            # check early stoppping
            if self.epochs_no_improve >= self.patience:
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
        if self.model_path is None:
            return
        
        if (self.should_save_model):
            torch.save(self.model, self.model_path + '.pt')
        else:
            self.save_checkpoint(epoch, val_loss)

    def save_checkpoint(self, epoch: int, val_loss:float):
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, f'{self.model_path}_checkpoint_epoch_{epoch}.pth')
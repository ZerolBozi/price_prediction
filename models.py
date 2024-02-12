import torch
import torch.nn as nn

class Model:
    def __init__(self, model_path:str='./models/'):
        self.model_path = model_path
    
    def get_model(self, model_name:str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(self.model_path + model_name + '.pt', map_location=device)
        model.eval()
        return model
    
class DQN(nn.Module):
    def __init__(self, input_shape:int, n_actions:int, units: int=32):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape, units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(units, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LSTM(nn.Module):
    def __init__(
            self, 
            input_size:int, 
            output_size:int, 
            hidden_size:int=128,
            fc_size: int=128,
            num_layers:int=2,
            dropout_prob:float=0.10, 
            batch_first:bool=True
        ):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob,
            batch_first=batch_first
        )
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class GRU(nn.Module):
    def __init__(
            self, 
            input_size:int, 
            output_size:int, 
            hidden_size:int=128,
            fc_size: int=128,
            num_layers:int=2,
            dropout_prob:float=0.10, 
            batch_first:bool=True
        ):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob,
            batch_first=batch_first
        )
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, output_size)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        x = gru_out[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
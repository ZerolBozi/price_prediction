import torch
import torch.nn as nn

"""
時間序列 參數說明:
    input_size: 輸入的特徵數量
    output_size: 輸出的特徵數量
    hidden_size: 隱藏層的cell數量
    fc_size: 全連接層的cell數量
    num_layers: 模型的層數
    dropout_prob: dropout的比例, 防止過擬合, 訓練過程中丟掉一些cell
    batch_first: 輸入的格式, True: (batch, seq, feature), False: (seq, batch, feature),
                    因為convert_to_lstm_format函數的輸出是(batch, seq, feature), 所以這邊設為True

深度強化學習 參數說明:
    input_size: 輸入的特徵數量
    n_actions: 輸出的動作數量
    units: 全連接層的cell數量
"""

class Model:
    def __init__(self, model_path:str='./models/'):
        self.model_path = model_path
    
    def get_model(self, model_name:str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(self.model_path + model_name + '.pt', map_location=device)
        model.eval()
        return model
    
class DQN(nn.Module):
    def __init__(self, n_observations:int, n_actions:int, units: int=32):
        super(DQN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(n_observations, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, n_actions)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class Net(nn.Module):
    def __init__(self, n_observations:int, n_actions:int, units: int=32):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(n_observations, units)
        self.fc2 = nn.Linear(units, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
# Dueling DQN
class DuelingDQN(nn.Module):
    def __init__(self, input_size:int, n_actions:int, units: int=32):
        super(DuelingDQN, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, units),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, n_actions)
        )

    def forward(self, x):
        x = self.feature_layer(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage to get Q values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

# Double DQN 
class DoubleDQN(DQN):
    def __init__(self, input_size:int, n_actions:int, units: int=32):
        super(DoubleDQN, self).__init__(input_size, n_actions, units)

        self.target_fc1 = nn.Linear(input_size, units)
        self.target_fc2 = nn.Linear(units, n_actions)

    def forward(self, x):
        x = super(DoubleDQN, self).forward(x)
        
        target_x = self.target_fc1(x)
        target_x = self.relu(target_x)
        target_x = self.target_fc2(target_x)

        return x, target_x
    
    def update_target_network(self):
        self.target_fc1.load_state_dict(self.fc1.state_dict())
        self.target_fc2.load_state_dict(self.fc2.state_dict())

class CNNLSTM(nn.Module):
    def __init__(
            self,
            input_channels:int,
            output_size:int,
            seq_length:int,
            feture_size:int,
            hidden_size:int,
            output_channels:int=16,
            kernel_size:int=3,
            cnn_out_size:int=32,
            pool_kernel_size:int=2,
            lstm_out_size:int=64,
            fc_size: int=128,
            num_layers:int=2,
            dropout_prob:float=0.10, 
            batch_first:bool=True
        ):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size
        )
        self.conv2 = nn.Conv1d(
            in_channels=output_channels,
            out_channels=cnn_out_size,
            kernel_size=kernel_size
        )
        self.conv3 = nn.Conv1d(
            in_channels=cnn_out_size,
            out_channels=cnn_out_size,
            kernel_size=kernel_size
        )
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
        self.lstm = nn.LSTM(cnn_out_size, lstm_out_size, batch_first=True)
        self.fc1 = nn.Linear(lstm_out_size, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, output_size)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1, x.size(1))
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CNNGRU(nn.Module):
    def __init__(
            self,
            input_channels:int,
            output_size:int,
            seq_length:int,
            feture_size:int,
            hidden_size:int,
            output_channels:int=16,
            kernel_size:int=3,
            cnn_out_size:int=32,
            pool_kernel_size:int=2,
            lstm_out_size:int=64,
            fc_size: int=128,
            num_layers:int=2,
            dropout_prob:float=0.10, 
            batch_first:bool=True
        ):
        super(CNNGRU, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size
        )
        self.conv2 = nn.Conv1d(
            in_channels=output_channels,
            out_channels=cnn_out_size,
            kernel_size=kernel_size
        )
        self.conv3 = nn.Conv1d(
            in_channels=cnn_out_size,
            out_channels=cnn_out_size,
            kernel_size=kernel_size
        )
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
        self.gru = nn.GRU(cnn_out_size, lstm_out_size, batch_first=True)
        self.fc1 = nn.Linear(lstm_out_size, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, output_size)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1, x.size(1))
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class BiLSTM(nn.Module):
    def __init__(
            self,
            input_size:int,
            output_size:int,
            hidden_size:int,
            fc_size: int=128,
            num_layers:int=2,
            dropout_prob:float=0.10, 
            batch_first:bool=True
        ):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob,
            batch_first=batch_first,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_size*2, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class BiGRU(nn.Module):
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
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob,
            batch_first=batch_first,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_size*2, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        x = gru_out[:, -1, :]
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
import torch.nn as nn


class CNNRNNHybrid(nn.Module):
    """
    Hybrid CNN-RNN model for trajectory prediction.
    See src/models/CNNRNNHybridEvaluator.py for running the model.
    See src/models/CNNRNNHybridTrainer.py for how the model is trained.

    Architecture:
    1. 1D CNN layers for local features
    2. LSTM layers for temporal sequence modeling
    3. Fully connected layers for final prediction
    """
    
    def __init__(self, input_dim: int = 4, cnn_channels: list = [64, 128, 64], lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.2, output_dim: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        
        # 1D CNN for feature extraction
        cnn_layers = []
        in_channels = input_dim
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected output layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, output_dim)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, cnn_channels[-1])
        
        # LSTM
        x, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden * 2)
        
        # Fully connected
        x = self.fc(x)  # (batch, seq_len, output_dim)
        
        return x

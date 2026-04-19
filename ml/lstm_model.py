import torch
import torch.nn as nn


class LSTMRiskModel(nn.Module):
    """
    PyTorch LSTM for sequential financial risk prediction.

    Input:  sequence of daily [sentiment_compound, risk_score, keyword_risk]
            shape: (batch_size, sequence_length, input_size=3)

    Output: class probabilities for price_movement
            shape: (batch_size, num_classes)

    Why LSTM over XGBoost:
        XGBoost sees one article in isolation.
        LSTM sees the last N days as a sequence and learns
        that sustained negative sentiment is a stronger signal
        than a single bad article.
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 64,
                 num_layers: int = 2, num_classes: int = 3, dropout: float = 0.2):
        super(LSTMRiskModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # input shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)

        # Take only the last time step output
        last_out = lstm_out[:, -1, :]

        out = self.dropout(last_out)
        out = self.fc(out)
        return out

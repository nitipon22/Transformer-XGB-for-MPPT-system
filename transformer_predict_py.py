# transformer_predict_py.py
import torch
import torch.nn as nn
import numpy as np

def predict(seq):
    """
    seq: numpy array [10 x 2]
    return: numpy array [2,]
    """

    class TransformerModel(nn.Module):
        def __init__(self, input_size=2, hidden_size=64, output_size=2):
            super().__init__()
            self.input_proj = nn.Linear(input_size, hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fc = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            x = self.input_proj(x)
            out = self.encoder(x)
            return self.fc(out[:, -1, :])

    model = TransformerModel()
    state_dict = torch.load("transformer_ts_weights.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        y = model(x)
    return y.squeeze(0).numpy()

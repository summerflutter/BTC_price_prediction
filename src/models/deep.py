from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, X, y, lookback: int):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.L = lookback
        self.n = len(X)

    def __len__(self):
        return self.n - self.L

    def __getitem__(self, idx):
        xseq = self.X[idx:idx+self.L]  # (L, F)
        target = self.y[idx+self.L]    # predict next-step return
        return torch.from_numpy(xseq), torch.tensor(target)

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        z = self.inp(x)
        z = self.pos(z)
        z = self.enc(z)
        last = z[:, -1, :]
        return self.head(last).squeeze(-1)

def train_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)

def eval_epoch(model, loader, device, criterion):
    model.eval()
    total = 0.0
    preds = []
    ys = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total += loss.item() * len(xb)
            preds.append(pred.cpu().numpy())
            ys.append(yb.cpu().numpy())
    import numpy as np
    preds = np.concatenate(preds) if preds else np.array([])
    ys = np.concatenate(ys) if ys else np.array([])
    return total / len(loader.dataset), preds, ys

def fit_deep(model_name: str, X_train, y_train, X_val, y_val, params: Dict, device: str):
    lookback = int(params.get("lookback", 32))
    batch_size = int(params.get("batch_size", 64))
    lr = float(params.get("lr", 1e-3))
    epochs = int(params.get("epochs", 50))
    patience = int(params.get("patience", 8))

    input_dim = X_train.shape[1]

    if model_name == "lstm":
        model = LSTMRegressor(input_dim, hidden_size=int(params.get("hidden_size",64)),
                              num_layers=int(params.get("num_layers",1)), dropout=float(params.get("dropout",0.0)))
    elif model_name == "transformer":
        model = TransformerRegressor(input_dim, d_model=int(params.get("d_model",128)),
                                     nhead=int(params.get("nhead",8)), num_layers=int(params.get("num_layers",2)),
                                     dim_feedforward=int(params.get("dim_feedforward",256)), dropout=float(params.get("dropout",0.1)))
    else:
        raise ValueError("Unknown deep model")

    device = device or "cpu"
    model.to(device)
    train_ds = SeqDataset(X_train, y_train, lookback)
    val_ds = SeqDataset(X_val, y_val, lookback)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()

    best_state = None
    best_val = float("inf")
    patience_ctr = 0

    for ep in range(epochs):
        tr_loss = train_epoch(model, train_loader, device, opt, crit)
        val_loss, _, _ = eval_epoch(model, val_loader, device, crit)
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, lookback

def predict_deep(model, X, y, lookback: int, device: str):
    from torch.utils.data import DataLoader
    ds = SeqDataset(X, y, lookback)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
    import numpy as np
    preds = np.concatenate(preds) if preds else np.array([])
    return preds

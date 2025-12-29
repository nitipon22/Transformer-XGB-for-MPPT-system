import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# -----------------------------
# 1. อ่านข้อมูลและลบ NaN
# -----------------------------
df = pd.read_csv("solar_single_source_15min.csv")
print("Original data shape:", df.shape)

df = df.dropna()
print("After dropna:", df.shape)

# -----------------------------
# 2. Feature Engineering
# -----------------------------
features = df[['IRRADIATION', 'MODULE_TEMPERATURE']]
target   = df[['IRRADIATION', 'MODULE_TEMPERATURE']]
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

# ตัด feature ที่ variance = 0
features = features.loc[:, features.var() != 0]

# Normalize
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(features)

target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(target)

# -----------------------------
# 3. Create Time Sequences
# -----------------------------
def create_sequences(X, y, seq_length=10):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

SEQ_LENGTH = 10

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)

# target time (n+1)
time_seq = df['DATE_TIME'].iloc[SEQ_LENGTH:].reset_index(drop=True)

N = len(X_seq)

print("\n=== Time Sequence Info ===")
print(f"Timesteps (time period): {SEQ_LENGTH}")
print(f"Total samples: {N}")
print(f"Input shape: {X_seq.shape}")
print(f"Target shape: {y_seq.shape}")

# -----------------------------
# 4. Time-based split (70/10/20)
# -----------------------------
train_end = int(0.7 * N)
val_end   = int(0.8 * N)

X_train = X_seq[:train_end]
y_train = y_seq[:train_end]

X_val = X_seq[train_end:val_end]
y_val = y_seq[train_end:val_end]

X_test = X_seq[val_end:]
y_test = y_seq[val_end:]

time_train = time_seq[:train_end]
time_val   = time_seq[train_end:val_end]
time_test  = time_seq[val_end:]

print("\n=== Split Info ===")
print(f"Train samples: {len(X_train)}")
print(f"  → From {time_train.iloc[0]} to {time_train.iloc[-1]}")

print(f"Val samples  : {len(X_val)}")
print(f"  → From {time_val.iloc[0]} to {time_val.iloc[-1]}")

print(f"Test samples : {len(X_test)}")
print(f"  → From {time_test.iloc[0]} to {time_test.iloc[-1]}")

# เก็บข้อมูลดิบสำหรับคำนวณ MASE
y_train_raw = target.iloc[SEQ_LENGTH:train_end+SEQ_LENGTH].values

# -----------------------------
# 5. แปลงเป็น Tensor
# -----------------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# -----------------------------
# 6. DataLoader
# -----------------------------
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)
val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)


# ==== Models ====
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.encoder(x)
        return self.fc(out[:, -1, :])


class TCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TCNModel, self).__init__()
        self.tcn = nn.Conv1d(
            input_size, hidden_size,
            kernel_size=3, padding=1, dilation=1
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = self.relu(out)
        out = out[:, :, -1]
        out = self.fc(out)
        return out

def train(model, train_loader, val_loader, epochs=20, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    best_val_loss = np.inf
    best_model_state = None
    wait = 0

    for epoch in range(epochs):
        # ===== Training =====
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ===== Validation =====
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        # ===== Early stopping =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)


def calculate_mase(y_true, y_pred, y_train_raw):
    """
    คำนวณ MASE (Mean Absolute Scaled Error)
    MASE = MAE / MAE_naive
    โดย MAE_naive = mean(|y_t - y_{t-1}|) จาก training set
    """
    # คำนวณ MAE ของโมเดล
    mae = np.mean(np.abs(y_true - y_pred))
    
    # คำนวณ MAE ของ naive forecast (y_t = y_{t-1})
    naive_errors = np.abs(np.diff(y_train_raw, axis=0))
    mae_naive = np.mean(naive_errors)
    
    # ป้องกันหารด้วยศูนย์
    if mae_naive == 0:
        return np.inf
    
    mase = mae / mae_naive
    return mase


def calculate_mbe(y_true, y_pred):
    """
    คำนวณ MBE (Mean Bias Error)
    MBE = mean(y_pred - y_true)
    บอกว่าโมเดล over-predict (MBE > 0) หรือ under-predict (MBE < 0)
    """
    mbe = np.mean(y_pred - y_true)
    return mbe


def evaluate(model, X_test, y_test, y_train_raw, feature_names):
    """
    ประเมินโมเดลและคืนค่า metrics แยกตาม feature
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
    
    y_true = y_test.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Inverse transform
    y_true = target_scaler.inverse_transform(y_true)
    y_pred = target_scaler.inverse_transform(y_pred)
    
    metrics_per_feature = {}
    
    for i, feature in enumerate(feature_names):
        y_true_f = y_true[:, i]
        y_pred_f = y_pred[:, i]
        
        mae = mean_absolute_error(y_true_f, y_pred_f)
        rmse = np.sqrt(mean_squared_error(y_true_f, y_pred_f))
        mape = np.mean(np.abs((y_true_f - y_pred_f) / (y_true_f + 1e-8))) * 100
        mase = calculate_mase(y_true_f, y_pred_f, y_train_raw[:, i])
        mbe = calculate_mbe(y_true_f, y_pred_f)
        
        metrics_per_feature[feature] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'MASE': mase,
            'MBE': mbe
        }
    
    return metrics_per_feature, y_true, y_pred


# ==== Training ====
input_size = X_train_tensor.shape[2]
feature_names = ['IRRADIATION', 'MODULE_TEMPERATURE']

models = {
    'LSTM': LSTMModel(input_size=input_size, hidden_size=64, output_size=2),
    'GRU': GRUModel(input_size=input_size, hidden_size=64, output_size=2),
    'Transformer': TransformerModel(input_size=input_size, hidden_size=64, output_size=2),
    'TCN': TCNModel(input_size=input_size, hidden_size=64, output_size=2)
}

trained_models = {}

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"Training {name}...")
    print(f"{'='*70}")
    train(model, train_loader, val_loader, epochs=20)
    trained_models[name] = model

# ==== Evaluation ====
results = {}

for name, model in trained_models.items():
    print(f"\n{'='*70}")
    print(f"Evaluating {name}")
    print(f"{'='*70}")

    # Evaluate on training set
    metrics_train, _, _ = evaluate(
        model, X_train_tensor, y_train_tensor, y_train_raw, feature_names
    )

    # Evaluate on test set
    metrics_test, y_true_test, y_pred_test = evaluate(
        model, X_test_tensor, y_test_tensor, y_train_raw, feature_names
    )

    results[name] = {
        'Train': metrics_train,
        'Test': metrics_test,
        'y_true': y_true_test,
        'y_pred': y_pred_test
    }

# ==== Print Results ====
print("\n" + "="*70)
print("MODEL PERFORMANCE SUMMARY")
print("="*70)

for name, data in results.items():
    print(f"\n{name} Results:")
    print(f"{'─'*70}")
    
    for feature in feature_names:
        print(f"\n{feature}:")
        print("  Training Set:")
        for metric_name, value in data['Train'][feature].items():
            if metric_name == 'MAPE':
                print(f"    {metric_name}: {value:.2f}%")
            else:
                print(f"    {metric_name}: {value:.4f}")
        
        print("  Test Set:")
        for metric_name, value in data['Test'][feature].items():
            if metric_name == 'MAPE':
                print(f"    {metric_name}: {value:.2f}%")
            else:
                print(f"    {metric_name}: {value:.4f}")

# ==== Plot Transformer Results ====
print("\n" + "=" * 70)
print("PLOTTING TRANSFORMER PREDICTIONS")
print("="*70)

transformer_results = results['Transformer']
y_true = transformer_results['y_true']
y_pred = transformer_results['y_pred']

# Plot สำหรับแต่ละ feature
for i, feature_name in enumerate(feature_names):
    plt.figure(figsize=(14, 6))
    
    # Plot ค่าจริง vs ค่าพยากรณ์
    plt.subplot(1, 2, 1)
    plt.plot(time_test.values, y_true[:, i], label='Actual', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(time_test.values, y_pred[:, i], label='Predicted', color='red', alpha=0.7, linewidth=1.5)
    plt.xlabel('Date Time', fontsize=12)
    plt.ylabel(feature_name, fontsize=12)
    plt.title(f'Transformer: Actual vs Predicted - {feature_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_true[:, i], y_pred[:, i], color='black', alpha=0.5, s=20)
    
    # เส้น ideal (y=x)
    min_val = min(y_true[:, i].min(), y_pred[:, i].min())
    max_val = max(y_true[:, i].max(), y_pred[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', linewidth=2, label='Ideal')
    
    plt.xlabel(f'Actual {feature_name}', fontsize=12)
    plt.ylabel(f'Predicted {feature_name}', fontsize=12)
    plt.title(f'Transformer: Scatter Plot - {feature_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()

# Plot รวม 2 features
plt.figure(figsize=(16, 6))

for i, feature_name in enumerate(feature_names):
    plt.subplot(1, 2, i+1)
    
    # เลือกช่วงเวลาแสดง (เช่น 500 samples แรก)
    n_display = min(500, len(time_test))
    
    plt.plot(range(n_display), y_true[:n_display, i], 
             label='Actual', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(range(n_display), y_pred[:n_display, i], 
             label='Predicted', color='red', alpha=0.7, linewidth=1.5)
    
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel(feature_name, fontsize=12)
    plt.title(f'Transformer Predictions - {feature_name} (First {n_display} samples)', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==== Error Distribution ====
print("\n" + "=" * 70)
print("TRANSFORMER ERROR ANALYSIS")
print("="*70)

plt.figure(figsize=(14, 5))

for i, feature_name in enumerate(feature_names):
    errors = y_pred[:, i] - y_true[:, i]
    
    plt.subplot(1, 2, i+1)
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Error Distribution - {feature_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SCALER INFORMATION")
print("="*70)
print("===== FEATURE SCALER =====")
print("X_min =", feature_scaler.data_min_)
print("X_max =", feature_scaler.data_max_)

print("\n===== TARGET SCALER =====")
print("y_min =", target_scaler.data_min_)
print("y_max =", target_scaler.data_max_)

# Save Transformer model
transformer_model = trained_models['Transformer']
torch.save(transformer_model.state_dict(), "transformer_ts_weights.pt")
print("\n✓ Transformer model saved to 'transformer_ts_weights.pt'")

state_dict = torch.load("transformer_ts_weights.pt")
print("\nModel state dict keys:", list(state_dict.keys()))

print("\n" + "="*70)
print("ALL TASKS COMPLETED!")
print("="*70)
import matplotlib.pyplot as plt
import numpy as np

# ==== COMPREHENSIVE MODEL COMPARISON PLOT ====
print("\n" + "=" * 70)
print("GENERATING MULTI-MODEL COMPARISON PLOTS")
print("="*70)

# Create a comprehensive comparison for each feature
for i, feature_name in enumerate(feature_names):
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'Model Comparison: {feature_name} Predictions on Test Set', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Select a representative window (e.g., first 500 samples for clarity)
    n_display = min(500, len(time_test))
    time_window = time_test.values[:n_display]
    
    model_names = ['LSTM', 'GRU', 'Transformer', 'TCN']
    colors = ['orange', 'green', 'red', 'purple']
    
    for idx, (model_name, color) in enumerate(zip(model_names, colors)):
        ax = axes[idx // 2, idx % 2]
        
        # Get predictions for this model
        y_true_model = results[model_name]['y_true'][:n_display, i]
        y_pred_model = results[model_name]['y_pred'][:n_display, i]
        
        # Plot actual (ground truth)
        ax.plot(time_window, y_true_model, 
                label='Ground Truth', color='blue', alpha=0.7, linewidth=2)
        
        # Plot prediction
        ax.plot(time_window, y_pred_model, 
                label=f'{model_name} Prediction', color=color, 
                alpha=0.7, linewidth=1.5, linestyle='--')
        
        # Add metrics as text
        metrics = results[model_name]['Test'][feature_name]
        metrics_text = (f"MAE: {metrics['MAE']:.4f}\n"
                       f"RMSE: {metrics['RMSE']:.4f}\n"
                       f"MAPE: {metrics['MAPE']:.2f}%\n"
                       f"MASE: {metrics['MASE']:.4f}")
        
        ax.text(0.02, 0.98, metrics_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        ax.set_xlabel('Date Time', fontsize=11)
        ax.set_ylabel(feature_name, fontsize=11)
        ax.set_title(f'{model_name} Model', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# ==== ALL MODELS IN ONE PLOT ====
print("\n" + "=" * 70)
print("GENERATING UNIFIED COMPARISON PLOT")
print("="*70)

for i, feature_name in enumerate(feature_names):
    plt.figure(figsize=(18, 8))
    
    n_display = len(time_test)
    time_window = time_test.values[:n_display]
    
    # Plot ground truth
    y_true_baseline = results['Transformer']['y_true'][:n_display, i]
    plt.plot(time_window, y_true_baseline, 
             label='Ground Truth', color='black', 
             linewidth=2.5, alpha=0.8, zorder=5)
    
    # Plot all model predictions
    model_colors = {
        'LSTM': 'orange',
        'GRU': 'green', 
        'Transformer': 'red',
        'TCN': 'purple'
    }
    
    for model_name, color in model_colors.items():
        y_pred_model = results[model_name]['y_pred'][:n_display, i]
        plt.plot(time_window, y_pred_model, 
                label=f'{model_name}', color=color, 
                alpha=0.6, linewidth=1.5, linestyle='--')
    
    plt.xlabel('Date Time', fontsize=13)
    plt.ylabel(feature_name, fontsize=13)
    plt.title(f'All Models Comparison: {feature_name} Predictions', 
              fontsize=15, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==== ZOOMED-IN VIEW (High Detail) ====
print("\n" + "=" * 70)
print("GENERATING ZOOMED-IN DETAIL VIEW")
print("="*70)

for i, feature_name in enumerate(feature_names):
    plt.figure(figsize=(18, 8))
    
    # Select a smaller window for detailed view (e.g., 100 samples)
    zoom_start = 0
    zoom_end = min(100, len(time_test))
    time_zoom = time_test.values[zoom_start:zoom_end]
    
    # Plot ground truth
    y_true_zoom = results['Transformer']['y_true'][zoom_start:zoom_end, i]
    plt.plot(time_zoom, y_true_zoom, 
             label='Ground Truth', color='black', 
             linewidth=3, alpha=0.9, marker='o', markersize=4, zorder=5)
    
    # Plot all model predictions
    for model_name, color in model_colors.items():
        y_pred_zoom = results[model_name]['y_pred'][zoom_start:zoom_end, i]
        plt.plot(time_zoom, y_pred_zoom, 
                label=f'{model_name}', color=color, 
                alpha=0.7, linewidth=2, linestyle='--', 
                marker='x', markersize=4)
    
    plt.xlabel('Date Time', fontsize=13)
    plt.ylabel(feature_name, fontsize=13)
    plt.title(f'Detailed View: {feature_name} Predictions (Samples {zoom_start}-{zoom_end})', 
              fontsize=15, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==== ERROR COMPARISON ACROSS MODELS ====
print("\n" + "=" * 70)
print("GENERATING ERROR COMPARISON PLOT")
print("="*70)

for i, feature_name in enumerate(feature_names):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Prediction Errors Across Models: {feature_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    n_display = min(500, len(time_test))
    time_window = time_test.values[:n_display]
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx // 2, idx % 2]
        
        y_true_model = results[model_name]['y_true'][:n_display, i]
        y_pred_model = results[model_name]['y_pred'][:n_display, i]
        errors = y_pred_model - y_true_model
        
        # Plot errors over time
        ax.plot(time_window, errors, color=colors[idx], alpha=0.7, linewidth=1)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.fill_between(time_window, errors, 0, alpha=0.3, color=colors[idx])
        
        # Add statistics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        mean_error = np.mean(errors)
        
        stats_text = (f"MAE: {mae:.4f}\n"
                     f"RMSE: {rmse:.4f}\n"
                     f"Mean Error: {mean_error:.4f}")
        
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7),
                fontsize=10)
        
        ax.set_xlabel('Date Time', fontsize=11)
        ax.set_ylabel('Prediction Error', fontsize=11)
        ax.set_title(f'{model_name} Model Errors', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# ==== SCATTER PLOT COMPARISON ====
print("\n" + "=" * 70)
print("GENERATING SCATTER PLOT COMPARISON")
print("="*70)

for i, feature_name in enumerate(feature_names):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Scatter Plot Comparison: {feature_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx // 2, idx % 2]
        
        y_true_model = results[model_name]['y_true'][:, i]
        y_pred_model = results[model_name]['y_pred'][:, i]
        
        # Scatter plot
        ax.scatter(y_true_model, y_pred_model, 
                  color=colors[idx], alpha=0.5, s=20)
        
        # Ideal line (y=x)
        min_val = min(y_true_model.min(), y_pred_model.min())
        max_val = max(y_true_model.max(), y_pred_model.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                color='red', linestyle='--', linewidth=2, label='Ideal (y=x)')
        
        # Calculate R²
        correlation_matrix = np.corrcoef(y_true_model, y_pred_model)
        r_squared = correlation_matrix[0, 1]**2
        
        # Add metrics
        metrics = results[model_name]['Test'][feature_name]
        metrics_text = (f"R²: {r_squared:.4f}\n"
                       f"MAE: {metrics['MAE']:.4f}\n"
                       f"RMSE: {metrics['RMSE']:.4f}")
        
        ax.text(0.05, 0.95, metrics_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=10)
        
        ax.set_xlabel(f'Actual {feature_name}', fontsize=11)
        ax.set_ylabel(f'Predicted {feature_name}', fontsize=11)
        ax.set_title(f'{model_name} Model', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("\n" + "="*70)
print("ALL COMPARISON PLOTS GENERATED!")
print("="*70)

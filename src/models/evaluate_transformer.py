import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from torch.amp import autocast
import glob

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "parquet"
MODEL_DIR = ROOT_DIR / "src" / "models" / "transformer_models"

def load_parquet_threaded(file_paths):
    def read_parquet(p): 
        return pd.read_parquet(p)
    
    with ThreadPoolExecutor(max_workers=min(8, len(file_paths))) as ex:
        dfs = list(ex.map(read_parquet, file_paths))
    
    return pd.concat(dfs, ignore_index=True)

class NFLDataset:
    def __init__(self, input_path, output_path=None, seq_len=15, is_train=True, data_fraction=1.0):
        self.is_train = is_train
        self.seq_len = seq_len
        
        input_path = Path(input_path)
        if input_path.is_dir():
            parquet_files = sorted(input_path.glob("*.parquet"))
            self.df_input = load_parquet_threaded(parquet_files)
        else:
            self.df_input = pd.read_parquet(input_path)
        
        if is_train and output_path:
            output_path = Path(output_path)
            if output_path.is_dir():
                parquet_files = sorted(output_path.glob("*.parquet"))
                self.df_output = load_parquet_threaded(parquet_files)
            else:
                self.df_output = pd.read_parquet(output_path)
            
            self.df_output = self.df_output.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
            last_frames = self.df_output.groupby(['game_id', 'play_id', 'nfl_id']).last().reset_index()
            last_frames = last_frames[['game_id', 'play_id', 'nfl_id', 'x', 'y']]
            last_frames.columns = ['game_id', 'play_id', 'nfl_id', 'target_x', 'target_y']
            
            self.df_input = self.df_input.merge(last_frames, on=['game_id', 'play_id', 'nfl_id'], how='left')
        
        self.df_input = self.df_input.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
        self.feature_cols = ['x', 'y', 's', 'a', 'dir', 'o', 'absolute_yardline_number']
        self.groups = list(self.df_input.groupby(['game_id', 'play_id', 'nfl_id']))
        
        if data_fraction < 1.0:
            num_samples = max(1, int(len(self.groups) * data_fraction))
            self.groups = self.groups[:num_samples]
    
    def __len__(self):
        return len(self.groups)
    
    def __getitem__(self, idx):
        _, g = self.groups[idx]
        X = g[self.feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(X) < self.seq_len:
            X = np.vstack([np.zeros((self.seq_len - len(X), X.shape[1]), dtype=np.float32), X])
        else:
            X = X[-self.seq_len:]
        
        X = torch.tensor(X, dtype=torch.float32)
        
        if self.is_train:
            target = g[['target_x', 'target_y']].iloc[-1].values.astype(np.float32)
            target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            return X, torch.tensor(target, dtype=torch.float32)
        
        return X

class MotionTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_layers=3, num_heads=4, dropout=0.1, output_dim=2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.1)
        nn.init.zeros_(self.input_projection.bias)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, output_dim)
        )
        
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.01)
        nn.init.zeros_(self.head[-1].bias)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))

def calculate_rmse(x_true, y_true, x_pred, y_pred):
    N = len(x_true)
    squared_errors = (x_true - x_pred)**2 + (y_true - y_pred)**2
    rmse = np.sqrt(np.sum(squared_errors) / (2 * N))
    return rmse

def evaluate_model(model_path, data_fraction=1.0, device='cuda'):
    
    print(f"\n{'='*80}")
    print(f"Evaluating model: {Path(model_path).name}")
    print(f"{'='*80}")
    
    #load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    print(f"Model trained for {checkpoint['epoch']} epochs")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    print(f"Data fraction used: {data_fraction*100:.1f}%")
    
    #load validation data with ground truth
    train_input_dir = DATA_DIR / "train_input"
    train_output_dir = DATA_DIR / "train_output"
    
    print("\nLoading validation data...")
    full_data = NFLDataset(
        input_path=train_input_dir,
        output_path=train_output_dir,
        seq_len=config['seq_len'],
        is_train=True,
        data_fraction=data_fraction
    )
    
    #split into train and validation (same as training)
    train_size = int(config['train_val_split'] * len(full_data))
    val_size = len(full_data) - train_size
    _, val_data = torch.utils.data.random_split(
        full_data, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  #use seed for reproducibility
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    #initialize model and load weights
    model = MotionTransformer(
        input_dim=len(full_data.feature_cols),
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    #collect predictions and ground truth
    all_preds = []
    all_targets = []
    
    print("\nGenerating predictions...")
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device, non_blocking=True)
            with autocast(device):
                pred = model(X)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())
    
    #concatenate all batches
    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    #extract x and y coordinates
    x_pred = predictions[:, 0]
    y_pred = predictions[:, 1]
    x_true = targets[:, 0]
    y_true = targets[:, 1]
    
    #calculate RMSE using the custom formula
    rmse = calculate_rmse(x_true, y_true, x_pred, y_pred)
    
    #calculate individual coordinate RMSEs for comparison
    rmse_x = np.sqrt(np.mean((x_true - x_pred)**2))
    rmse_y = np.sqrt(np.mean((y_true - y_pred)**2))
    
    #calculate mean absolute errors
    mae_x = np.mean(np.abs(x_true - x_pred))
    mae_y = np.mean(np.abs(y_true - y_pred))
    
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Number of validation samples: {len(x_true):,}")
    print("\nCustom RMSE Formula:")
    print("  RMSE = sqrt(1/(2N) * sum((x_true - x_pred)^2 + (y_true - y_pred)^2))")
    print(f"  RMSE: {rmse:.4f} yards")
    print("\nCoordinate-wise Metrics:")
    print(f"  RMSE X: {rmse_x:.4f} yards")
    print(f"  RMSE Y: {rmse_y:.4f} yards")
    print(f"  MAE X:  {mae_x:.4f} yards")
    print(f"  MAE Y:  {mae_y:.4f} yards")
    print("\nPrediction Statistics:")
    print(f"  X - Mean: {x_pred.mean():.2f}, Std: {x_pred.std():.2f}, Range: [{x_pred.min():.2f}, {x_pred.max():.2f}]")
    print(f"  Y - Mean: {y_pred.mean():.2f}, Std: {y_pred.std():.2f}, Range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print("\nGround Truth Statistics:")
    print(f"  X - Mean: {x_true.mean():.2f}, Std: {x_true.std():.2f}, Range: [{x_true.min():.2f}, {x_true.max():.2f}]")
    print(f"  Y - Mean: {y_true.mean():.2f}, Std: {y_true.std():.2f}, Range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"{'='*80}\n")
    
    return {
        'model_name': Path(model_path).name,
        'rmse': rmse,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'n_samples': len(x_true),
        'val_loss': checkpoint['val_loss'],
        'epochs': checkpoint['epoch']
    }

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    #find all model files in the transformer_models directory
    model_files = sorted(glob.glob(str(MODEL_DIR / "motion_transformer_*.pt")))
    
    if not model_files:
        print(f"\nNo model files found in {MODEL_DIR}")
        print("Please train a model first using the training script.")
        exit(1)
    
    print(f"\nFound {len(model_files)} model(s) to evaluate:")
    for i, model_file in enumerate(model_files, 1):
        print(f"  {i}. {Path(model_file).name}")
    
    #evaluate all models
    results = []
    for model_path in model_files:
        try:
            result = evaluate_model(model_path, data_fraction=0.25, device=device)
            results.append(result)
        except Exception as e:
            print(f"\nError evaluating {Path(model_path).name}: {e}")
            continue
    
    #summary comparison if multiple models
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY: All Models Comparison")
        print(f"{'='*80}")
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('rmse')
        print(df_results.to_string(index=False))
        print(f"\nBest model: {df_results.iloc[0]['model_name']} with RMSE: {df_results.iloc[0]['rmse']:.4f}")
        print(f"{'='*80}\n")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from torch.amp import GradScaler, autocast
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "parquet"
MODEL_DIR = ROOT_DIR / "src" / "models" / "transformer_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    #data settings
    'data_fraction': 1.0,      #fraction of data to use (0.0 to 1.0). 0.25 = 25%, 1.0 = 100%
    'train_val_split': 0.8,     #train/validation split ratio (0.8 = 80% train, 20% val)
    'seq_len': 15,              #sequence length for temporal modeling
    
    #model architecture
    'model_dim': 256,           #transformer model dimension
    'num_layers': 4,            # of transformer encoder layers
    'num_heads': 8,             # of attention heads
    'dropout': 0.1,             #dropout rate
    
    #training hyperparameters
    'batch_size': 256,          # Batch size for training
    'epochs': 100,               # of training epochs
    'learning_rate': 1e-4,      # Initial learning rate
    'weight_decay': 1e-5,       # Weight decay for regularization
    'max_grad_norm': 1.0,       # Maximum gradient norm for clipping
    
    #learning rate scheduler
    'lr_scheduler_factor': 0.5, # Factor to reduce LR by
    'lr_scheduler_patience': 5, # Epochs to wait before reducing LR
    
    #system settings
    'num_workers': None,        # DataLoader workers (None = auto-detect)
}

def load_parquet_threaded(file_paths):

    def read_parquet(p): 
        return pd.read_parquet(p)
    
    with ThreadPoolExecutor(max_workers=min(8, len(file_paths))) as ex:
        dfs = list(ex.map(read_parquet, file_paths))

    return pd.concat(dfs, ignore_index=True)

class NFLDataset(Dataset):
    def __init__(self, input_path, output_path=None, seq_len=15, is_train=True, data_fraction=1.0):

        self.is_train = is_train
        self.seq_len = seq_len
        
        #load input data
        input_path = Path(input_path)

        if input_path.is_dir():

            parquet_files = sorted(input_path.glob("*.parquet"))
            print(f"[INFO] Loading {len(parquet_files)} parquet files from {input_path}")
            self.df_input = load_parquet_threaded(parquet_files)

        else:
            self.df_input = pd.read_parquet(input_path)
        
        print(f"[INFO] Loaded input data: {len(self.df_input):,} rows")
        
        #load output data for training
        if is_train and output_path:
            output_path = Path(output_path)
            if output_path.is_dir():
                parquet_files = sorted(output_path.glob("*.parquet"))
                print(f"[INFO] Loading {len(parquet_files)} output parquet files")
                self.df_output = load_parquet_threaded(parquet_files)
            else:
                self.df_output = pd.read_parquet(output_path)
            print(f"[INFO] Loaded output data: {len(self.df_output):,} rows")
            
            #merge to get final frame targets
            #get the last frame for each trajectory
            self.df_output = self.df_output.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
            last_frames = self.df_output.groupby(['game_id', 'play_id', 'nfl_id']).last().reset_index()
            last_frames = last_frames[['game_id', 'play_id', 'nfl_id', 'x', 'y']]
            last_frames.columns = ['game_id', 'play_id', 'nfl_id', 'target_x', 'target_y']
            
            #merge with input data
            self.df_input = self.df_input.merge(last_frames, on=['game_id', 'play_id', 'nfl_id'], how='left')
        
        #sort by trajectory
        self.df_input = self.df_input.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
        
        #define feature columns
        self.feature_cols = ['x', 'y', 's', 'a', 'dir', 'o', 'absolute_yardline_number']
        
        #group by trajectory
        self.groups = list(self.df_input.groupby(['game_id', 'play_id', 'nfl_id']))
        
        #use specified fraction of the data
        if data_fraction < 1.0:
            num_samples = max(1, int(len(self.groups) * data_fraction))
            self.groups = self.groups[:num_samples]
            print(f"[INFO] Created {len(self.groups):,} trajectories ({data_fraction*100:.1f}% subset)")
        else:
            print(f"[INFO] Created {len(self.groups):,} trajectories (100% of data)")
        
        #print data statistics for debugging
        if is_train:
            print("[DEBUG] Feature ranges:")
            for col in self.feature_cols:
                vals = self.df_input[col].dropna()
                print(f"  {col}: [{vals.min():.2f}, {vals.max():.2f}], mean={vals.mean():.2f}")
            print("[DEBUG] Target ranges:")
            print(f"  target_x: [{self.df_input['target_x'].min():.2f}, {self.df_input['target_x'].max():.2f}]")
            print(f"  target_y: [{self.df_input['target_y'].min():.2f}, {self.df_input['target_y'].max():.2f}]")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        _, g = self.groups[idx]
        
        #extract features
        X = g[self.feature_cols].values.astype(np.float32)
        
        #handle NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        #pad or truncate to seq_len
        if len(X) < self.seq_len:
            X = np.vstack([np.zeros((self.seq_len - len(X), X.shape[1]), dtype=np.float32), X])
        else:
            X = X[-self.seq_len:]
        
        X = torch.tensor(X, dtype=torch.float32)
        
        if self.is_train:
            #get target from merged data
            target = g[['target_x', 'target_y']].iloc[-1].values.astype(np.float32)
            target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            return X, torch.tensor(target, dtype=torch.float32)
        
        return X

class MotionTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_layers=3, num_heads=4, dropout=0.1, output_dim=2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        #initialize with smaller weights
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.1)
        nn.init.zeros_(self.input_projection.bias)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  #pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, output_dim)
        )
        
        #initialize output layer with small weights
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))

def train_model(model, train_loader, val_loader, config, device='cuda', save_path=None):
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.MSELoss()
    scaler = GradScaler(device)
    
    #learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['lr_scheduler_factor'], 
        patience=config['lr_scheduler_patience']
    )

    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        #training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device):
                pred = model(X)
                loss = criterion(pred, y)
            
            #check for NaN loss
            if torch.isnan(loss):
                print("[WARNING] NaN loss detected in training, skipping batch")
                continue
            
            scaler.scale(loss).backward()
            
            #gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        #validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                with autocast(device):
                    pred = model(X)
                    loss = criterion(pred, y)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        lr_changed = " [LR REDUCED]" if new_lr < current_lr else ""
        print(f"[EPOCH {epoch+1}/{config['epochs']}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {new_lr:.2e}{lr_changed}")
        
        #save best model based on validation loss
        if save_path and (avg_val_loss < best_val_loss or not torch.isnan(torch.tensor(avg_val_loss))):
            best_val_loss = avg_val_loss
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,
            }
            torch.save(save_dict, save_path)
            print(f"[CHECKPOINT] Saved best model with validation loss: {avg_val_loss:.6f}")
        
        #save final model at last epoch as backup
        if save_path and epoch == config['epochs'] - 1 and not Path(save_path).exists():
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,
            }
            torch.save(save_dict, save_path)
            print("[CHECKPOINT] Saved final model (no best model was saved)")
    
    return model

def predict(model, loader, device='cuda'):
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.eval()
    preds = []
    
    with torch.no_grad():
        for X in loader:
            X = X.to(device, non_blocking=True)
            with autocast(device):
                out = model(X).cpu().numpy()
            preds.append(out)
    
    return np.vstack(preds)

if __name__ == "__main__":
    # Load configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Data fraction: {CONFIG['data_fraction']*100:.1f}%")
    print(f"[CONFIG] Train/Val split: {CONFIG['train_val_split']*100:.0f}/{(1-CONFIG['train_val_split'])*100:.0f}")
    print(f"[CONFIG] Sequence length: {CONFIG['seq_len']}")
    print(f"[CONFIG] Batch size: {CONFIG['batch_size']}")
    print(f"[CONFIG] Epochs: {CONFIG['epochs']}")
    print(f"[CONFIG] Learning rate: {CONFIG['learning_rate']}")
    print(f"[CONFIG] Model dim: {CONFIG['model_dim']}, Layers: {CONFIG['num_layers']}, Heads: {CONFIG['num_heads']}")
    
    #paths
    train_input_dir = DATA_DIR / "train_input"
    train_output_dir = DATA_DIR / "train_output"
    test_input_path = DATA_DIR / "test_input.parquet"
    
    #create dataset
    print("\n[DATA] Loading training data")
    full_data = NFLDataset(
        input_path=train_input_dir,
        output_path=train_output_dir,
        seq_len=CONFIG['seq_len'],
        is_train=True,
        data_fraction=CONFIG['data_fraction']
    )
    
    #split into train and validation
    train_size = int(CONFIG['train_val_split'] * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])
    
    print(f"[DATA] Training samples: {len(train_data):,}")
    print(f"[DATA] Validation samples: {len(val_data):,}")
    
    #create dataloaders with optimizations
    num_workers = CONFIG['num_workers'] if CONFIG['num_workers'] is not None else min(8, os.cpu_count() or 4)
    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    print("\n[MODEL] Building transformer architecture")
    model = MotionTransformer(
        input_dim=len(full_data.feature_cols),
        model_dim=CONFIG['model_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        dropout=CONFIG['dropout']
    )
    
    #model save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = MODEL_DIR / f"motion_transformer_{timestamp}.pt"
    
    print("\n[TRAINING] Starting model training")
    print(f"[TRAINING] Checkpoints will be saved to: {model_save_path}")
    
    model = train_model(
        model,
        train_loader,
        val_loader,
        config=CONFIG,
        device=device,
        save_path=model_save_path
    )
    
    #load best model for inference
    print("\n[INFERENCE] Loading best model checkpoint")
    
    if not model_save_path.exists():
        print(f"[ERROR] No model checkpoint found at {model_save_path}")
        print("[ERROR] Training may have failed. Please check the logs above.")
        exit(1)
    
    checkpoint = torch.load(model_save_path)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    #predict on test set
    print("\n[INFERENCE] Running predictions on test set")
    test_data = NFLDataset(
        input_path=test_input_path,
        seq_len=CONFIG['seq_len'],
        is_train=False,
        data_fraction=CONFIG['data_fraction']
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    preds = predict(model, test_loader, device=device)
    
    #create submission
    print("\n[OUTPUT] Creating submission file")

    #get the actual test trajectories (not all unique ones, but the ones we predicted on)
    test_df = pd.read_parquet(test_input_path)

    #get the groups in the same order as test_data.groups
    test_groups_info = [(gid, pid, nid) for (gid, pid, nid), _ in test_data.groups]
    
    submission = pd.DataFrame({
        'game_id': [g[0] for g in test_groups_info],
        'play_id': [g[1] for g in test_groups_info],
        'nfl_id': [g[2] for g in test_groups_info],
        'x': preds[:, 0],
        'y': preds[:, 1]
    })
    
    submission_path = ROOT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"[SUCCESS] Submission saved to: {submission_path}")
    print(f"[SUCCESS] Model saved to: {model_save_path}")
    print("[COMPLETE] Training and inference completed successfully")
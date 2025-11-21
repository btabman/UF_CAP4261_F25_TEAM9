"""
Data loading and preprocessing for NFL trajectory prediction.
"""
from pathlib import Path
from typing import Tuple, Optional
import polars as pl
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class NFLDataset:
    """Handles loading and preprocessing of NFL trajectory data."""
    
    def __init__(self, data_dir: Path, seq_len: int = 100):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.x_max = 120.0
        self.y_max = 53.3
        
    def load_and_preprocess(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data from parquet files.
        """
        # Load data
        df_in = pl.read_parquet(self.data_dir / "train_input.parquet")
        df_out = pl.read_parquet(self.data_dir / "train_output.parquet")
        
        print(f"Loaded inputs: {df_in.shape}, outputs: {df_out.shape}")
        
        # Join input and output data
        df = df_in.join(
            df_out.select(["game_id", "play_id", "nfl_id", "frame_id", "x", "y"]),
            on=["game_id", "play_id", "nfl_id", "frame_id"],
            how="inner",
            suffix="_label"
        )
        
        # Normalize labels
        df = df.with_columns([
            (pl.col("x_label") / self.x_max).alias("x_norm"),
            (pl.col("y_label") / self.y_max).alias("y_norm")
        ])
        
        # Extract sequences
        sequences, targets = self._extract_sequences(df, max_samples)
        
        return sequences, targets
    
    def _extract_sequences(self, df: pl.DataFrame, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Extract fixed-length sequences from dataframe."""
        sequences, targets = [], []
        
        # Filter to players we need to predict
        df_filtered = df.filter(pl.col("player_to_predict") == True)
        
        for _, group in df_filtered.group_by(["game_id", "play_id", "nfl_id"]):
            group = group.sort("frame_id")
            
            # Extract features and labels
            features = group.select(["s", "a", "dir", "o"]).to_numpy()
            labels = group.select(["x_norm", "y_norm"]).to_numpy()
            
            # Pad or truncate to fixed length
            features, labels = self._pad_or_truncate(features, labels)
            
            sequences.append(features)
            targets.append(labels)
            
            if max_samples and len(sequences) >= max_samples:
                break
        
        X = np.stack(sequences)
        y = np.stack(targets)
        
        print(f"Extracted {len(sequences)} sequences with shape: {X.shape}")
        return X, y
    
    def _pad_or_truncate(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pad or truncate sequences to fixed length."""
        if len(features) < self.seq_len:
            pad = self.seq_len - len(features)
            features = np.pad(features, ((0, pad), (0, 0)))
            labels = np.pad(labels, ((0, pad), (0, 0)))
        else:
            features = features[:self.seq_len]
            labels = labels[:self.seq_len]
        
        return features, labels
    
    def create_dataloaders(self,X: np.ndarray,y: np.ndarray,batch_size: int = 32, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split data into train/val/test and create DataLoaders.
        """
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), \
            "Ratios must sum to 1.0"
        
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Shuffle indices
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        # Calculate split points
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Split indices
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        print(f"Data split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.tensor(X[train_idx], dtype=torch.float32),
            torch.tensor(y[train_idx], dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X[val_idx], dtype=torch.float32),
            torch.tensor(y[val_idx], dtype=torch.float32)
        )
        test_dataset = TensorDataset(
            torch.tensor(X[test_idx], dtype=torch.float32),
            torch.tensor(y[test_idx], dtype=torch.float32)
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
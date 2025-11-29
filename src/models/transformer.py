import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "parquet"
MODEL_DIR = ROOT_DIR / "src" / "models" / "transformer_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "seq_len": 15,
    "model_dim": 64,
    "num_layers": 2,
    "dropout": 0.05,
    "batch_size": 64,
    "epochs": 15,
    "lr": 3e-4,
    "train_val_split": 0.8,
}

def build_player_id_map(input_path, output_path):
    input_files = sorted(Path(input_path).glob("*.parquet"))
    output_files = sorted(Path(output_path).glob("*.parquet"))
    players = set()
    for f in input_files:
        players.update(pd.read_parquet(f, columns=["nfl_id"])["nfl_id"].unique())
    for f in output_files:
        players.update(pd.read_parquet(f, columns=["nfl_id"])["nfl_id"].unique())
    return {pid: i for i, pid in enumerate(sorted(players))}

PLAYER_ID_MAP = build_player_id_map(DATA_DIR / "train_input", DATA_DIR / "train_output")
NUM_PLAYERS = len(PLAYER_ID_MAP)

class NFLDataset(Dataset):
    feature_cols = ["x", "y", "s", "a", "dir", "o", "absolute_yardline_number"]

    def __init__(self, input_path, output_path, seq_len=15, is_train=True):
        self.seq_len = seq_len
        self.is_train = is_train

        df_in = pd.concat([
            pd.read_parquet(f)
            for f in sorted(Path(input_path).glob("*.parquet"))
        ])
        df_in = df_in.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])

        df_in[self.feature_cols] = df_in[self.feature_cols].fillna(0)

        if is_train:
            df_out = pd.concat([
                pd.read_parquet(f)
                for f in sorted(Path(output_path).glob("*.parquet"))
            ])
            df_out = df_out.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])

            last = df_out.groupby(["game_id", "play_id", "nfl_id"], as_index=False).last()
            last = last[["game_id", "play_id", "nfl_id", "x", "y"]]
            last.columns = ["game_id", "play_id", "nfl_id", "tx", "ty"]

            last["tx"] = last["tx"].astype(float)
            last["ty"] = last["ty"].astype(float)

            df_in = df_in.merge(last, on=["game_id", "play_id", "nfl_id"], how="left")

        self.feature_mean = df_in[self.feature_cols].mean().values.astype(np.float32)
        self.feature_std = df_in[self.feature_cols].std().values.astype(np.float32)
        self.feature_std[self.feature_std < 1e-6] = 1e-6  # avoid divide-by-zero

        if is_train:
            target_vals = df_in[["tx", "ty"]].dropna().values.astype(np.float32)
            self.target_mean = target_vals.mean(0)
            self.target_std = target_vals.std(0)
            self.target_std[self.target_std < 1e-6] = 1e-6
        else:
            self.target_mean = None
            self.target_std = None

        self.inputs = []
        self.targets = []
        self.pids = []

        for (_, _, pid), g in df_in.groupby(["game_id", "play_id", "nfl_id"]):

            if is_train:
                tx = g["tx"].iloc[-1]
                ty = g["ty"].iloc[-1]

                if np.isnan(tx) or np.isnan(ty):
                    continue

            arr = g[self.feature_cols].values.astype(np.float32)
            arr = (arr - self.feature_mean) / self.feature_std

            if len(arr) < seq_len:
                pad = np.zeros((seq_len - len(arr), arr.shape[1]), np.float32)
                arr = np.vstack([pad, arr])
            else:
                arr = arr[-seq_len:]

            self.inputs.append(arr)
            self.pids.append(PLAYER_ID_MAP.get(int(pid), 0))

            if is_train:
                t = np.array([
                    (tx - self.target_mean[0]) / self.target_std[0],
                    (ty - self.target_mean[1]) / self.target_std[1]
                ], dtype=np.float32)

                self.targets.append(t)

        if is_train:
            self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        pid = torch.tensor(self.pids[idx], dtype=torch.long)

        if self.is_train:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y, pid

        return x, pid


class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
        queries = self.queries(query)
        keys = self.keys(key)
        values = self.values(value)
        energy = torch.bmm(queries, keys.transpose(1, 2))
        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=-1)
        out = torch.bmm(attention, values)
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, ff_mult=2, dropout=0.05):
        super().__init__()
        self.attn = SimpleSelfAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_mult * embed_size),
            nn.ReLU(),
            nn.Linear(ff_mult * embed_size, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, depth=2,
                 num_players=1000, dropout=0.05):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pid_embed = nn.Embedding(num_players, 16)
        self.merge = nn.Linear(model_dim + 16, model_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(model_dim, ff_mult=2, dropout=dropout)
            for _ in range(depth)
        ])
        self.out = nn.Linear(model_dim, 2)

    def forward(self, x, pid):
        x = self.input_proj(x)
        pid = self.pid_embed(pid).unsqueeze(1).repeat(1, x.size(1), 1)
        x = self.merge(torch.cat([x, pid], dim=-1))
        for layer in self.layers:
            x = layer(x)
        return self.out(x[:, -1])

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = NFLDataset(DATA_DIR / "train_input",
                    DATA_DIR / "train_output",
                    CONFIG["seq_len"],
                    True)

    n = int(CONFIG["train_val_split"] * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n, len(ds) - n])

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = Transformer(
        input_dim=len(ds.feature_cols),
        model_dim=CONFIG["model_dim"],
        depth=CONFIG["num_layers"],
        num_players=NUM_PLAYERS,
        dropout=CONFIG["dropout"]
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scaler = torch.amp.GradScaler('cuda')
    loss_fn = nn.MSELoss()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0
        for X, y, pid in tqdm(train_loader, desc=f"Epoch {epoch+1}/train"):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pid = pid.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda'):
                pred = model(X, pid)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y, pid in tqdm(val_loader, desc=f"Epoch {epoch+1}/val"):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pid = pid.to(device, non_blocking=True)

                with torch.amp.autocast(device_type='cuda'):
                    pred = model(X, pid)
                    loss = loss_fn(pred, y)

                val_loss += loss.item()

        print(f"Epoch {epoch+1}: train={train_loss:.3f}, val={val_loss:.3f}")

    save_path = MODEL_DIR / "transformer.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "PLAYER_ID_MAP": PLAYER_ID_MAP,
        "num_players": NUM_PLAYERS,
        "feature_mean": ds.feature_mean,
        "feature_std": ds.feature_std,
        "target_mean": ds.target_mean,
        "target_std": ds.target_std
    }, save_path)

    print("Saved:", save_path)

if __name__ == "__main__":
    train()

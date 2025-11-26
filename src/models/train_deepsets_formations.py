#%%
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, List, Any, Dict, Optional, Callable

import polars as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

#%% --- project root bootstrap (works from notebook or script) ---
proj = Path.cwd()
if (proj / "src").exists():
    root = proj
elif (proj.parent / "src").exists():
    root = proj.parent
else:
    root = next(p for p in [proj, *proj.parents] if (p / "src").exists())
os.chdir(root)

#%% ---- imports from your codebase ----
from src.features.play_features import FormationDataset, compute_formation_targets  # noqa: F401
from src.models.deepsets import DeepSets

#%% ---- collate: stack tensors, keep metas in a list ----
def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]]):
    Xs, Ms, Ys, Metas = [], [], [], []
    for X, M, Y, meta in batch:
        Xs.append(X)   # [N, D]
        Ms.append(M)   # [N]
        Ys.append(Y)   # [5]
        Metas.append(meta)
    X = torch.stack(Xs, dim=0).float()  # [B, N, D]
    M = torch.stack(Ms, dim=0).float()  # [B, N]
    Y = torch.stack(Ys, dim=0).float()  # [B, 5]
    return X, M, Y, Metas

#%% ---- simple frame selectors (local to this file) ----
def only_side(side: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    side = side.lower()
    def _sel(df: pl.DataFrame) -> pl.DataFrame:
        if "player_side" not in df.columns:
            raise ValueError("player_side column missing")
        return df.filter(pl.col("player_side").str.to_lowercase() == side)
    return _sel

def last_k_frames_per_team(k: int = 1) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """
    Keep the last K frames for each (game_id, play_id, player_side).
    """
    def _sel(df: pl.DataFrame) -> pl.DataFrame:
        keys = ["game_id", "play_id", "player_side"]
        if "frame_id" not in df.columns:
            raise ValueError("frame_id column missing")
        maxf = df.group_by(keys).agg(pl.col("frame_id").max().alias("_maxf"))
        df2 = df.join(maxf, on=keys, how="inner")
        return df2.filter(pl.col("frame_id") >= pl.col("_maxf") - (k - 1)).drop("_maxf")
    return _sel

#%% ---- one-model training ----
def train_one_model(
    parquet_path: Path,
    side: str,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    n_max: int = 11,
    save_to: Path | None = None,
):
    if isinstance(device, str):
        device = torch.device(device)

    # Compose selectors: filter by side -> keep last K frames
    selector = last_k_frames_per_team(k=1)
    def frame_selector(df):
        return selector(only_side(side)(df))

    ds = FormationDataset(str(parquet_path), frame_selector=frame_selector, n_max=n_max)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model = DeepSets().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for X, M, Y, _meta in dl:
            X, M, Y = X.to(device), M.to(device), Y.to(device)
            opt.zero_grad()
            _emb, pred = model(X, M)
            loss = loss_fn(pred, Y)
            loss.backward()
            opt.step()
            total += loss.item() * X.size(0)
        avg = total / len(ds)
        print(f"[{side}] epoch {epoch}/{epochs} - MSE: {avg:.6f}")

    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_to)
        print(f"Saved: {save_to}")

    return model

#%% ---- embeddings export (optional) ----
@torch.no_grad()
def export_embeddings(
    parquet_path: Path,
    side: str,
    weights_path: Path,           
    out_parquet: Path,
    n_max: int = 11,
    batch_size: int = 256,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FormationDataset(str(parquet_path), frame_selector=only_side(side), n_max=n_max)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = DeepSets().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    import numpy as np
    import polars as pl

    rows: list[dict[str, Any]] = []
    for X, M, _Y, metas in dl:
        X, M = X.to(device), M.to(device)
        emb, _ = model(X, M)  # [B, E]
        emb = emb.cpu().numpy()
        for vec, meta in zip(emb, metas):
            game_id, play_id, frame_id, player_side = meta
            row = {"game_id": game_id, "play_id": play_id, "frame_id": frame_id, "player_side": player_side}
            for i, v in enumerate(vec):
                row[f"emb_{i:03d}"] = float(v)
            rows.append(row)

    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    import polars as pl
    pl.DataFrame(rows).write_parquet(out_parquet)
    print(f"Embeddings saved to: {out_parquet}")

#%% ---- orchestration ----
def train_deepsets_formations(
    parquet: str = "data/processed/players_test.parquet",
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    n_max: int = 11,
    save_dir: str = "src/models",
    export_embeds: bool = False,
) -> Dict[str, Path]:
    parquet_path = Path(parquet)
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    off_path = save_dir_path / "deepsets_offense.pt"
    def_path = save_dir_path / "deepsets_defense.pt"

    train_one_model(parquet_path, "offense", epochs, batch_size, lr, n_max=n_max, save_to=off_path)
    train_one_model(parquet_path, "defense", epochs, batch_size, lr, n_max=n_max, save_to=def_path)

    if export_embeds:
        export_embeddings(parquet_path, "offense", weights_path=off_path,
                          out_parquet="data/processed/embeddings_offense.parquet", n_max=n_max)
        export_embeddings(parquet_path, "defense", weights_path=def_path,
                          out_parquet="data/processed/embeddings_defense.parquet", n_max=n_max)

    return {"offense": off_path, "defense": def_path}

# %%
paths = train_deepsets_formations(
    parquet="data/processed/players_test.parquet",  # or your train parquet
    epochs=3,
    batch_size=128,
    lr=1e-3,
    n_max=11,
    save_dir="src/models",
    export_embeds=True,
)
# %%

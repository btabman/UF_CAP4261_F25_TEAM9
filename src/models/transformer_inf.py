import torch
import pandas as pd
import numpy as np
from pathlib import Path

from src.models.transformer import Transformer 

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT_DIR / "src/models/transformer_models/transformer.pt"

INPUT_PARQUET = ROOT_DIR / "data/parquet/train_input/input_w01.parquet"

SEQ_LEN = 15
FEATURE_COLS = ["x","y","s","a","dir","o","absolute_yardline_number"]


def load_model():
    ckpt = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    model = Transformer(
        input_dim=7,
        model_dim=64,
        depth=2,
        num_players=ckpt["num_players"],
        dropout=0.05
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return (
        model,
        ckpt["PLAYER_ID_MAP"],
        ckpt["feature_mean"],
        ckpt["feature_std"],
        ckpt["target_mean"],
        ckpt["target_std"],
    )


def prepare_sequences(df, player_id_map, feature_mean, feature_std, seq_len):
    df = df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
    grouped = df.groupby(["game_id", "play_id", "nfl_id"])
    items = []

    for (gid, pid, nid), g in grouped:
        arr = g[FEATURE_COLS].values.astype(np.float32)

        arr = (arr - feature_mean) / feature_std

        if len(arr) < seq_len:
            pad = np.zeros((seq_len - len(arr), arr.shape[1]), np.float32)
            arr = np.vstack([pad, arr])
        else:
            arr = arr[-seq_len:]

        pid_idx = player_id_map.get(int(nid), 0)

        items.append(((gid, pid, nid), arr, pid_idx))

    return items


def load_ground_truth(df_out):

    df_out = df_out.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])

    last = df_out.groupby(["game_id", "play_id", "nfl_id"], as_index=False).last()
    last = last[["game_id", "play_id", "nfl_id", "x", "y"]]
    last.columns = ["game_id", "play_id", "nfl_id", "gt_x", "gt_y"]
    
    return last


def run_prediction(_=None):

    parquet_in = INPUT_PARQUET
    parquet_out = Path(str(parquet_in).replace("input", "output"))

    df_in = pd.read_parquet(parquet_in)
    df_out = pd.read_parquet(parquet_out)

    (
        model,
        player_id_map,
        feat_mean,
        feat_std,
        tgt_mean,
        tgt_std,
    ) = load_model()

    valid_players = set(df_out["nfl_id"].unique())
    df_in = df_in[df_in["nfl_id"].isin(valid_players)]

    items = prepare_sequences(df_in, player_id_map, feat_mean, feat_std, SEQ_LEN)

    predictions = []

    for (gid, pid, nid), arr, pid_idx in items:
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        pid_tensor = torch.tensor([pid_idx], dtype=torch.long)

        with torch.no_grad():
            pred_norm = model(x, pid_tensor)[0].numpy()

        pred_x = pred_norm[0] * tgt_std[0] + tgt_mean[0]
        pred_y = pred_norm[1] * tgt_std[1] + tgt_mean[1]

        predictions.append({
            "game_id": gid,
            "play_id": pid,
            "nfl_id": nid,
            "pred_x": float(pred_x),
            "pred_y": float(pred_y),
        })

    pred_df = pd.DataFrame(predictions)
    gt_df = load_ground_truth(df_out)

    merged = pred_df.merge(gt_df, on=["game_id", "play_id", "nfl_id"], how="inner")

    merged["dx"] = merged["pred_x"] - merged["gt_x"]
    merged["dy"] = merged["pred_y"] - merged["gt_y"]
    merged["error"] = np.sqrt(merged["dx"]**2 + merged["dy"]**2)

    rmse = float(np.sqrt((merged["error"]**2).mean()))

    return merged, rmse


if __name__ == "__main__":
    merged, rmse = run_prediction()

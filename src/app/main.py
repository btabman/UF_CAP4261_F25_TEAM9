import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import gradio as gr
import pandas as pd
import numpy as np
import torch
import polars as pl

from src.models.transformer_inf import run_prediction

from src.models.MiniMax import run_minimax_and_rmse
from src.models.CNNRNN.CNNRNNHybrid import CNNRNNHybrid
from src.models.CNNRNN.CNNRNNHybridTrainer import Trainer
from src.data.NFLDataset import NFLDataset

from src.models.physics_transformer.physics_transformer import PhysicsTransformerFrames, TransformerFramesConfig
from src.models.physics_transformer.player_dataset import join_teamframe, FrameSeqTestDataset, FrameSeqDataset
from src.models.physics_transformer.train_player_model import FEATURE_COLS

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "parquet"
TRANSFORMER_PATH = ROOT / "src" / "models" / "transformer_models" / "transformer_normalized.pt"
CNNRNN_PATH = ROOT / "src" / "models" / "CNNRNN_model" / "cnn_rnn_best_model.pt"

PHYSICS_TRANSFORMER_DIR = ROOT / "src" / "models" / "physics_transformer"

CHECKPOINT_PATH = PHYSICS_TRANSFORMER_DIR / "player_physics_transformer_frames_agentic.pt"
PLAYERS_PARQUET = PHYSICS_TRANSFORMER_DIR / "players_test.parquet"
TEAMFRAME_PARQUET = PHYSICS_TRANSFORMER_DIR / "teamframe_test.parquet"
OUTPUT_PREDICTIONS = PHYSICS_TRANSFORMER_DIR / "frame_level_predictions_from_checkpoint.csv"

COMBINED_CSV_PATH = ROOT / "src" / "models" / "combined_input_output.csv"

def list_datasets():
    return sorted([p.name for p in (DATA_DIR / "train_input").glob("*.parquet")])

def cnnrnn_evaluate():
    if not CNNRNN_PATH.exists():
        return "cnn_rnn_best_model.pt missing!", None
    dataset = NFLDataset(DATA_DIR, seq_len=100)
    X, y = dataset.load_and_preprocess()
    _, _, test_loader = dataset.create_dataloaders(X, y, batch_size=32)
    model = CNNRNNHybrid(
        input_dim=4,
        cnn_channels=[64,128,128,64],
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.3,
        output_dim=2
    )
    trainer = Trainer(
        model=model,
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        x_max=dataset.x_max,
        y_max=dataset.y_max
    )
    trainer.load_best_model(CNNRNN_PATH)
    results = trainer.test(test_loader)
    preds = results["predictions"]
    targets = results["targets"]
    df = pd.DataFrame({
        "x_true": targets[:,:,0].reshape(-1),
        "y_true": targets[:,:,1].reshape(-1),
        "x_pred": preds[:,:,0].reshape(-1),
        "y_pred": preds[:,:,1].reshape(-1),
    })
    return f"CNN-RNN RMSE: {results['rmse']:.4f}", df

def minimax_run():
    result = run_minimax_and_rmse()
    return f"RMSE: {result['rmse']:.4f}", result["merged_df"]

def load_physics_transformer(checkpoint_path: Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt["cfg"]
    cfg = TransformerFramesConfig(**cfg_dict)
    model = PhysicsTransformerFrames(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    player_id_map = ckpt["player_id_map"]
    return model, cfg, player_id_map

def load_physics_parquet(players_path: Path, teamframe_path: Path):
    players = pl.read_parquet(players_path)
    teamframe = pl.read_parquet(teamframe_path)
    players_joined = join_teamframe(players, teamframe)
    missing = [c for c in FEATURE_COLS if c not in players_joined.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return players_joined

@torch.no_grad()
def physics_predict_frames(model, players_joined, player_id_map):
    device = next(model.parameters()).device
    dataset = FrameSeqTestDataset(
        df=players_joined,
        player_id_map=player_id_map,
        feature_cols=FEATURE_COLS,
        max_seq_len=32,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    all_rows = []
    for game_id, play_id, nfl_id, frame_ids, x, pid in loader:
        x = x.to(device)
        pid = pid.to(device)
        preds = model(x, pid).cpu().numpy()
        frame_ids = frame_ids.numpy()
        for g, p, n, fr, yhat in zip(game_id, play_id, nfl_id, frame_ids, preds):
            for frame_id, (px, py) in zip(fr, yhat):
                all_rows.append({
                    "game_id": int(g),
                    "play_id": int(p),
                    "nfl_id": int(n),
                    "frame_id": int(frame_id),
                    "pred_x": float(px),
                    "pred_y": float(py)
                })
    return pd.DataFrame(all_rows)

def run_physics_transformer(checkpoint_name, players_file, teamframe_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = PHYSICS_TRANSFORMER_DIR / checkpoint_name
    players_path = PHYSICS_TRANSFORMER_DIR / players_file
    teamframe_path = PHYSICS_TRANSFORMER_DIR / teamframe_file

    model, cfg, player_id_map = load_physics_transformer(checkpoint_path, device)

    players_joined = load_physics_parquet(players_path, teamframe_path)

    preds_df = physics_predict_frames(model, players_joined, player_id_map)

    return preds_df

def compute_physics_predictions_with_truth(model, players_joined, player_id_map):
    device = next(model.parameters()).device

    ds = FrameSeqDataset(
        df=players_joined,
        player_id_map=player_id_map,
        feature_cols=FEATURE_COLS,
        max_seq_len=32,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    rows = []

    model.eval()
    with torch.no_grad():
        for game_id, play_id, nfl_id, x, y, pid in loader:
            x = x.to(device)
            pid = pid.to(device)

            preds = model(x, pid).cpu().numpy()
            y_true = y.numpy()

            frame_ids = np.arange(y_true.shape[1])

            for gi, pi, ni, true_seq, pred_seq in zip(game_id, play_id, nfl_id, y_true, preds):
                for f, (txy, pxy) in enumerate(zip(true_seq, pred_seq)):
                    rows.append({
                        "game_id": int(gi),
                        "play_id": int(pi),
                        "nfl_id": int(ni),
                        "frame_idx": int(frame_ids[f]),
                        "true_x": float(txy[0]),
                        "true_y": float(txy[1]),
                        "pred_x": float(pxy[0]),
                        "pred_y": float(pxy[1]),
                    })

    df = pd.DataFrame(rows)

    rmse = float(np.sqrt(((df["true_x"] - df["pred_x"])**2 + (df["true_y"] - df["pred_y"])**2).mean()))
    return rmse, df


def run_physics_rmse(checkpoint_name, players_file, teamframe_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = PHYSICS_TRANSFORMER_DIR / checkpoint_name
    players_path = PHYSICS_TRANSFORMER_DIR / players_file
    teamframe_path = PHYSICS_TRANSFORMER_DIR / teamframe_file

    model, cfg, player_id_map = load_physics_transformer(checkpoint_path, device)

    players_joined = load_physics_parquet(players_path, teamframe_path)

    rmse, df = compute_physics_predictions_with_truth(model, players_joined, player_id_map)
    return rmse, df

def run_gbm(csv_file, n_priors):
    df = pd.read_csv(csv_file)

    if "x_y" in df.columns and "y_y" in df.columns:
        df = df.rename(columns={"x_y": "x", "y_y": "y"})

    df = df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"]).copy()
    groups = df.groupby(["game_id", "play_id", "nfl_id"])

    for k in range(1, n_priors + 1):
        df[f"x_prior{k}"] = groups["x"].shift(k)
        df[f"y_prior{k}"] = groups["y"].shift(k)

    df = df.dropna(subset=[f"x_prior{n_priors}", f"y_prior{n_priors}"]).reset_index(drop=True)

    feature_cols = []
    for k in range(1, n_priors + 1):
        feature_cols.append(f"x_prior{k}")
        feature_cols.append(f"y_prior{k}")

    extra_cols = ["s", "a", "dir", "o"]
    for c in extra_cols:
        if c in df.columns:
            feature_cols.append(c)

    X = df[feature_cols]
    y_x = df["x"]
    y_y = df["y"]

    X_train, X_test, yx_train, yx_test, yy_train, yy_test, df_train, df_test = train_test_split(
        X, y_x, y_y, df, test_size=0.2, random_state=42
    )

    params = dict(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        feature_pre_filter=False,
        random_state=42,
    )

    gbx = LGBMRegressor(**params)
    gby = LGBMRegressor(**params)

    gbx.fit(X_train, yx_train)
    gby.fit(X_train, yy_train)

    px = gbx.predict(X_test)
    py = gby.predict(X_test)

    df_test = df_test.reset_index(drop=True)
    df_test["pred_x_gbm"] = px
    df_test["pred_y_gbm"] = py

    rmse = float(np.sqrt(((df_test["x"] - df_test["pred_x_gbm"])**2 + (df_test["y"] - df_test["pred_y_gbm"])**2).mean()))
    return rmse, df_test

def transformer_eval():
    merged, rmse = run_prediction()
    return merged, float(rmse)

def build_ui():
    with gr.Blocks(title="Model Evaluator") as ui:
        gr.Markdown("Evaluate different prediction models")

        with gr.Tab("CNN-RNN Evaluator"):
            cnn_btn = gr.Button("Run CNN-RNN Evaluation", variant="primary", size="lg")
            cnn_rmse = gr.Textbox(label="Results", lines=2)
            cnn_df = gr.DataFrame(label="Predictions vs Ground Truth")
            cnn_btn.click(cnnrnn_evaluate, outputs=[cnn_rmse, cnn_df])

        with gr.Tab("MiniMax Evaluator"):
            mm_btn = gr.Button("Run MiniMax Prediction", variant="primary", size="lg")
            mm_rmse = gr.Textbox(label="Results", lines=2)
            mm_df = gr.DataFrame(label="Predictions vs Ground Truth")
            mm_btn.click(minimax_run, outputs=[mm_rmse, mm_df])

        with gr.Tab("Physics Transformer"):
            physics_ckpt = gr.Dropdown(choices=[f.name for f in PHYSICS_TRANSFORMER_DIR.glob("*.pt")])
            physics_players = gr.Dropdown(choices=[f.name for f in PHYSICS_TRANSFORMER_DIR.glob("players_*.parquet")])
            physics_teamframe = gr.Dropdown(choices=[f.name for f in PHYSICS_TRANSFORMER_DIR.glob("teamframe_*.parquet")])

            rmse_button = gr.Button("Compute RMSE")
            rmse_output = gr.Number()
            rmse_df_output = gr.DataFrame()

            rmse_button.click(
                fn=run_physics_rmse,
                inputs=[physics_ckpt, physics_players, physics_teamframe],
                outputs=[rmse_output, rmse_df_output]
            )

        with gr.Tab("LightGBM Model"):
            gbm_priors = gr.Number(label="Number of Prior Frames", value=5)
            gbm_btn = gr.Button("Run LightGBM")
            gbm_rmse = gr.Number(label="RMSE")
            gbm_df = gr.DataFrame(label="Predictions")

            gbm_btn.click(
                fn=lambda priors: run_gbm(str(COMBINED_CSV_PATH), int(priors)),
                inputs=[gbm_priors],
                outputs=[gbm_rmse, gbm_df]
            )

        with gr.Tab("Transformer"):
            st_btn = gr.Button("Run Simple Transformer", variant="primary")

            rmse = gr.Number(label="RMSE")
            merged = gr.DataFrame(label="Predictions")

            st_btn.click(
                fn=transformer_eval,
                inputs=[],
                outputs=[merged, rmse]
            )

    return ui

if __name__ == "__main__":
    app = build_ui()
    app.launch(share=False)

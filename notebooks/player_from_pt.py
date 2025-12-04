import torch
import polars as pl

# -------------------------------------------------------------------
# 🔧 USER-EDITABLE PATHS
# -------------------------------------------------------------------
CHECKPOINT_PATH = "models/player_physics_transformer_frames_agentic.pt"   # <- path to .pt file
PLAYERS_PARQUET = "data/processed/players_test.parquet"                   # <- players_* file (train OR test)
TEAMFRAME_PARQUET = "data/processed/teamframe_test.parquet"               # <- matching teamframe_* file
OUTPUT_PREDICTIONS = "models/frame_level_predictions_from_checkpoint.csv" # <- where to save predictions
# -------------------------------------------------------------------

# Imports from your project (assumes the same src/ layout you already have)
from src.models.physics_transformer import PhysicsTransformerFrames, TransformerFramesConfig
from src.models.player_dataset import join_teamframe
from src.models.train_player_model import FEATURE_COLS, predict_frames_on_test


def load_model_and_predict(
    checkpoint_path: str,
    players_path: str,
    teamframe_path: str,
    out_path: str,
):
    """
    Load a saved frame-level PhysicsTransformer model from .pt and
    generate per-frame predictions from parquet inputs.
    """

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 1: Load checkpoint & rebuild model

    ckpt = torch.load(checkpoint_path, map_location=device)

    # cfg was saved via `asdict(best_agentic_frames["cfg"])`
    cfg_dict = ckpt["cfg"]
    cfg = TransformerFramesConfig(**cfg_dict)

    model = PhysicsTransformerFrames(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # player_id_map: dict[nfl_id -> integer index] used for embeddings
    player_id_map = ckpt["player_id_map"]
    print(f"Loaded model with model_dim={cfg.model_dim}, "
          f"num_layers={cfg.num_layers}, num_heads={cfg.num_heads}")
    print(f"Player embedding size: {len(player_id_map)} players")


    # 2: Load parquet data & join teamframe features

    print("Loading parquet data...")
    players = pl.read_parquet(players_path)
    teamframe = pl.read_parquet(teamframe_path)

    # join_teamframe adds formation / team-level features per frame
    players_joined = join_teamframe(players, teamframe)
    print(f"Joined players+teamframe shape: {players_joined.shape}")

    # Optional: ensure all training feature columns are present
    missing = [c for c in FEATURE_COLS if c not in players_joined.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns in joined data: {missing}")


    # 3: Run predictions & save

    print("Running frame-level predictions...")
    # predict_frames_on_test will:
    #   - build FrameSeqTestDataset from players_joined
    #   - iterate with a DataLoader
    #   - write CSV with game_id, play_id, nfl_id, frame_id, pred_x, pred_y
    predict_frames_on_test(model, players_joined, player_id_map, out_path)

    print(f"Predictions saved to: {out_path}")


# Run with the user-provided paths above
load_model_and_predict(
    CHECKPOINT_PATH,
    PLAYERS_PARQUET,
    TEAMFRAME_PARQUET,
    OUTPUT_PREDICTIONS,
)

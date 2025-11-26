# physics_transformer.py

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class TransformerConfig:
    input_dim: int
    model_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_players: int = 10_000  # overridden at runtime
    dropout: float = 0.1


class PhysicsTransformer(nn.Module):
    """
    Player-level transformer that takes a sequence of motion/context features and
    a player_id, and predicts a 2D target (e.g. ball_land_x, ball_land_y or future x,y).
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.cfg = cfg

        # Player identity embedding (e.g. per-nfl_id)
        self.player_embed = nn.Embedding(cfg.num_players, cfg.model_dim * 2)

        # Map physical features to model space
        self.input_proj = nn.Linear(cfg.input_dim, cfg.model_dim)

        # Normalize embedding space for stability
        self.embed_norm = nn.LayerNorm(cfg.model_dim * 2)

        # Fuse motion + identity (model_dim + 2*model_dim → model_dim)
        self.input_fusion = nn.Linear(cfg.model_dim * 3, cfg.model_dim)

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            batch_first=True,
            dropout=cfg.dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # Output head → 2D (x, y)
        self.output_fusion = nn.Sequential(
            nn.Linear(cfg.model_dim * 3, cfg.model_dim),
            nn.ReLU(),
            nn.Linear(cfg.model_dim, 2),
        )

    def forward(self, x, player_ids):
        """
        x: float tensor of shape (batch, seq_len, input_dim)
        player_ids: long tensor of shape (batch,)
        """
        # Project input features
        x_proj = self.input_proj(x)  # (B, T, model_dim)

        # Lookup & normalize player embedding
        player_emb = self.player_embed(player_ids)        # (B, 2*model_dim)
        player_emb = self.embed_norm(player_emb)          # (B, 2*model_dim)

        # Expand across sequence
        player_expanded = player_emb.unsqueeze(1).expand(-1, x_proj.size(1), -1)
        # (B, T, 2*model_dim)

        # Combine motion and identity
        fused_input = torch.cat([x_proj, player_expanded], dim=-1)  # (B, T, 3*model_dim)
        fused_input = self.input_fusion(fused_input)                # (B, T, model_dim)

        # Transformer over time
        out = self.transformer(fused_input)  # (B, T, model_dim)

        # Use last time step representation + player embedding
        last_state = out[:, -1, :]                 # (B, model_dim)
        fused_out = torch.cat([last_state, player_emb], dim=-1)  # (B, 3*model_dim)

        # Predict 2D output
        return self.output_fusion(fused_out)       # (B, 2)

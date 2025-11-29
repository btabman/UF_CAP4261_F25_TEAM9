
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class TransformerFramesConfig:
    input_dim: int
    model_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_players: int = 10_000
    dropout: float = 0.1


class PhysicsTransformerFrames(nn.Module):
    """
    Sequence-to-sequence variant: predicts (x, y) for *each frame*.

    Input:
        x: (B, T, input_dim)
        player_ids: (B,)
    Output:
        preds: (B, T, 2)   # per-frame predictions
    """

    def __init__(self, cfg: TransformerFramesConfig):
        super().__init__()
        self.cfg = cfg

        self.player_embed = nn.Embedding(cfg.num_players, cfg.model_dim * 2)
        self.input_proj = nn.Linear(cfg.input_dim, cfg.model_dim)
        self.embed_norm = nn.LayerNorm(cfg.model_dim * 2)
        self.input_fusion = nn.Linear(cfg.model_dim * 3, cfg.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            batch_first=True,
            dropout=cfg.dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # Now the head is applied per time-step:
        self.output_head = nn.Sequential(
            nn.Linear(cfg.model_dim * 3, cfg.model_dim),
            nn.ReLU(),
            nn.Linear(cfg.model_dim, 2),
        )

    def forward(self, x, player_ids):
        """
        x: (B, T, input_dim)
        player_ids: (B,)
        """
        x_proj = self.input_proj(x)  # (B, T, D)

        player_emb = self.player_embed(player_ids)        # (B, 2D)
        player_emb = self.embed_norm(player_emb)
        player_expanded = player_emb.unsqueeze(1).expand(-1, x_proj.size(1), -1)  # (B,T,2D)

        fused_input = torch.cat([x_proj, player_expanded], dim=-1)  # (B,T,3D)
        fused_input = self.input_fusion(fused_input)                # (B,T,D)

        out = self.transformer(fused_input)                         # (B,T,D)

        # Per-frame fusion: concat hidden state + player embedding per frame
        fused_out = torch.cat([out, player_expanded], dim=-1)       # (B,T,3D)

        preds = self.output_head(fused_out)                         # (B,T,2)
        return preds

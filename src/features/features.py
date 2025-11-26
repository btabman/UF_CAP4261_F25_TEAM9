# src/features/funcs.py
from __future__ import annotations
import math
from datetime import date, datetime
from typing import Iterable, Tuple, Optional
import polars as pl

# -------------------------------
# Basic parsing / scalar features
# -------------------------------

def height_to_inches(h: Optional[str]) -> float:
    """'6-1' -> 73.0 ; returns NaN on failure."""
    if not h:
        return float("nan")
    try:
        ft, inch = str(h).split("-")
        return int(ft) * 12 + int(inch)
    except Exception:
        return float("nan")

def age_years(birth_date_str: Optional[str], ref: date = date(2023, 9, 1)) -> float:
    """Age in years at a reference date. birth_date_str like '1997-02-15'."""
    if not birth_date_str:
        return float("nan")
    try:
        dob = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
        return (ref - dob).days / 365.25
    except Exception:
        return float("nan")

def bmi(weight_lbs: Optional[float], height_in: Optional[float]) -> float:
    """BMI = (lbs * 703) / in^2 ; NaN if missing/invalid."""
    if not weight_lbs or not height_in or height_in <= 0:
        return float("nan")
    return (float(weight_lbs) * 703.0) / (float(height_in) ** 2)

# -------------------------------
# Angles / motion (scalar)
# -------------------------------

def angle_deg_to_rad(deg: Optional[float]) -> float:
    return float("nan") if deg is None else math.radians(deg)

def angle_wrap_deg(deg: Optional[float]) -> float:
    """Wrap degrees to [0, 360)."""
    if deg is None:
        return float("nan")
    return deg % 360.0

def angle_sin_cos_deg(deg: Optional[float]) -> Tuple[float, float]:
    """Return (sin, cos) from degrees."""
    if deg is None:
        return float("nan"), float("nan")
    r = math.radians(deg)
    return math.sin(r), math.cos(r)

def velocity_components(speed: Optional[float], dir_deg: Optional[float]) -> Tuple[float, float]:
    """Resolve speed along (x,y) using heading (dir_deg)."""
    if speed is None or dir_deg is None:
        return float("nan"), float("nan")
    r = math.radians(dir_deg)
    return speed * math.cos(r), speed * math.sin(r)

def acceleration_components(accel: Optional[float], dir_deg: Optional[float]) -> Tuple[float, float]:
    """Resolve acceleration along (x,y) using heading (dir_deg)."""
    if accel is None or dir_deg is None:
        return float("nan"), float("nan")
    r = math.radians(dir_deg)
    return accel * math.cos(r), accel * math.sin(r)

# -------------------------------
# Ball estimation (scalar)
# -------------------------------

def estimate_ball_xy_linear(
    x0: Optional[float],
    y0: Optional[float],
    x_land: Optional[float],
    y_land: Optional[float],
    rel_frame: Optional[int],
    num_frames_output: Optional[int],
) -> Tuple[float, float]:
    """
    Linear interpolate ball position from (x0,y0) at rel_frame=0 to (x_land,y_land) at rel_frame=num_frames_output.
    """
    if None in (x0, y0, x_land, y_land, rel_frame, num_frames_output):
        return float("nan"), float("nan")
    if num_frames_output <= 0:
        return float("nan"), float("nan")
    t = max(0.0, min(1.0, rel_frame / float(num_frames_output)))
    return x0 + t * (x_land - x0), y0 + t * (y_land - y0)

def distance_and_bearing_to_point(
    x: Optional[float], y: Optional[float], tx: Optional[float], ty: Optional[float]
) -> Tuple[float, float]:
    """
    Euclidean distance and bearing (radians) from (x,y) to (tx,ty).
    Bearing is arctan2(dy, dx) in radians.
    """
    if None in (x, y, tx, ty):
        return float("nan"), float("nan")
    dx, dy = tx - x, ty - y
    return math.hypot(dx, dy), math.atan2(dy, dx)

# -------------------------------
# Field normalization (scalar)
# -------------------------------

FIELD_LENGTH = 120.0
FIELD_WIDTH  = 160.0 / 3.0  # ~53.3333333333

def normalize_rightward(
    x: Optional[float],
    y: Optional[float],
    dir_deg: Optional[float],
    o_deg: Optional[float],
    play_direction: Optional[str],
    field_length: float = FIELD_LENGTH,
    field_width: float = FIELD_WIDTH,
) -> Tuple[float, float, float, float]:
    """
    Mirror coordinates so offense always moves to the right.
    If play_direction == 'left': x' = L - x, y' = W - y, angles += 180° (wrapped).
    Returns (x_norm, y_norm, dir_norm, o_norm).
    """
    flip = (str(play_direction).lower() == "left")
    if flip:
        xn = (field_length - x) if x is not None else float("nan")
        yn = (field_width  - y) if y is not None else float("nan")
        dir_n = angle_wrap_deg((dir_deg or 0.0) + 180.0) if dir_deg is not None else float("nan")
        o_n   = angle_wrap_deg((o_deg   or 0.0) + 180.0) if o_deg   is not None else float("nan")
    else:
        xn, yn = x, y
        dir_n, o_n = angle_wrap_deg(dir_deg) if dir_deg is not None else float("nan"), \
                     angle_wrap_deg(o_deg)   if o_deg   is not None else float("nan")
    return xn, yn, dir_n, o_n

# -------------------------------
# Simple relative aggregates (scalar helper)
# -------------------------------

def team_opponent_means_for_player(
    x_norm: Iterable[float],
    y_norm: Iterable[float],
    side: Iterable[str],
    player_index: int,
) -> Tuple[float, float, float, float]:
    """
    Given all players in a frame (lists of x_norm, y_norm, side),
    return (team_avg_x, team_avg_y, opp_avg_x, opp_avg_y) for the player at player_index.
    """
    xs, ys, ss = list(x_norm), list(y_norm), list(side)
    if player_index < 0 or player_index >= len(xs):
        return float("nan"), float("nan"), float("nan"), float("nan")
    me_side = ss[player_index]
    team_pts = [(xs[i], ys[i]) for i in range(len(xs)) if ss[i] == me_side and xs[i] is not None and ys[i] is not None]
    opp_pts  = [(xs[i], ys[i]) for i in range(len(xs)) if ss[i] != me_side and xs[i] is not None and ys[i] is not None]
    if team_pts:
        tx = sum(p[0] for p in team_pts) / len(team_pts)
        ty = sum(p[1] for p in team_pts) / len(team_pts)
    else:
        tx = ty = float("nan")
    if opp_pts:
        ox = sum(p[0] for p in opp_pts) / len(opp_pts)
        oy = sum(p[1] for p in opp_pts) / len(opp_pts)
    else:
        ox = oy = float("nan")
    return tx, ty, ox, oy




# ---------------------------------------------------------------------
# Coverage / proximity features via per-frame self-join
#    Adds: distance_to_nearest_teammate, distance_to_nearest_opponent,
#          opponents_within_{r}yds, teammates_within_{r}yds, coverage_density
# Needs: game_id, play_id, frame_id, player_side, nfl_id, x, y
# Note: Self-join is O(n^2) per frame; still fine for baselines.
# ---------------------------------------------------------------------
def add_coverage_features(
    df: pl.DataFrame,
    radii=(3.0, 5.0, 7.0),
    game="game_id",
    play="play_id",
    frame="frame_id",
    side="player_side",
    pid="nfl_id",
    x="x",
    y="y",
) -> pl.DataFrame:
    # rename columns to suffix _1 / _2 after join
    left_cols  = [game, play, frame, side, pid, x, y]
    right_cols = [pl.col(c).alias(f"{c}__r") for c in [game, play, frame, side, pid, x, y]]

    base = df.select(left_cols)
    other = df.select(right_cols)

    joined = (
        base.join(
            other,
            left_on=[game, play, frame],
            right_on=[f"{game}__r", f"{play}__r", f"{frame}__r"],
            how="inner"
        )
        # exclude self-pairs
        .filter(pl.col(pid) != pl.col(f"{pid}__r"))
        .with_columns([
            ((pl.col(x) - pl.col(f"{x}__r"))**2 + (pl.col(y) - pl.col(f"{y}__r"))**2)
            .sqrt()
            .alias("_dist"),
            (pl.col(side) == pl.col(f"{side}__r")).alias("_same_side")
        ])
    )

    # nearest teammate/opponent per (game,play,frame,pid)
    grp = [game, play, frame, pid]
    nn = (
        joined.group_by(grp)
              .agg([
                  pl.col("_dist").filter(pl.col("_same_side")).min().alias("distance_to_nearest_teammate"),
                  pl.col("_dist").filter(~pl.col("_same_side")).min().alias("distance_to_nearest_opponent"),
                  # coverage density: sum 1/(1+dist) over opponents
                  (1.0 / (1.0 + pl.col("_dist"))).filter(~pl.col("_same_side")).sum().alias("coverage_density"),
              ])
    )

    out = df.join(nn, on=grp, how="left")

    # counts within radii
    for r in radii:
        counts = (
            joined.filter(~pl.col("_same_side"))
                  .with_columns((pl.col("_dist") < r).cast(pl.Int32).alias("_opp_in_r"))
                  .group_by(grp)
                  .agg(pl.col("_opp_in_r").sum().alias(f"opponents_within_{int(r)}yds"))
        )
        out = out.join(counts, on=grp, how="left")

        team_counts = (
            joined.filter(pl.col("_same_side"))
                  .with_columns((pl.col("_dist") < r).cast(pl.Int32).alias("_tm_in_r"))
                  .group_by(grp)
                  .agg(pl.col("_tm_in_r").sum().alias(f"teammates_within_{int(r)}yds"))
        )
        out = out.join(team_counts, on=grp, how="left")

    return out


# ---------------------------------------------------------------------
# Temporal features (rolling) within each player’s timeline
#    Adds: speed_rolling_mean_k, speed_rolling_std_k, accel_rolling_mean_k, accel_rolling_std_k,
#          angular_velocity (wrapped), delta_x, delta_y, cumulative_distance
# Needs: game_id, play_id, nfl_id, frame_id, s, a, dir, x, y
# ---------------------------------------------------------------------
def add_temporal_features(
    df: pl.DataFrame,
    window_sizes=(3, 5),
    game="game_id",
    play="play_id",
    pid="nfl_id",
    frame="frame_id",
    s="s",
    a="a",
    dirc="dir",
    x="x",
    y="y",
) -> pl.DataFrame:
    # Ensure proper time order per player within play
    out = df.sort([game, play, pid, frame])

    # rolling stats for speed/accel
    group_keys = [game, play, pid]
    for w in window_sizes:
        out = out.with_columns([
            pl.col(s).rolling_mean(window_size=w).over(group_keys).alias(f"speed_rolling_mean_{w}"),
            pl.col(s).rolling_std(window_size=w).over(group_keys).alias(f"speed_rolling_std_{w}"),
            pl.col(a).rolling_mean(window_size=w).over(group_keys).alias(f"accel_rolling_mean_{w}"),
            pl.col(a).rolling_std(window_size=w).over(group_keys).alias(f"accel_rolling_std_{w}"),
        ])

    # angular velocity: wrapped diff in degrees to (-180,180]
    # wrap(x) = ((x + 180) % 360) - 180
    out = out.with_columns(
        (((pl.col(dirc).diff().over(group_keys)) + 180.0) % 360.0 - 180.0).alias("angular_velocity")
    )

    # deltas & cumulative distance
    out = out.with_columns([
        pl.col(x).diff().over(group_keys).alias("delta_x"),
        pl.col(y).diff().over(group_keys).alias("delta_y"),
    ])
    out = out.with_columns(
        ((pl.col("delta_x")**2 + pl.col("delta_y")**2).sqrt())
        .cum_sum()
        .over(group_keys)
        .alias("cumulative_distance")
    )

    return out
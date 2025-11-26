#%% Convert CSV into parquet files
# src/data/preprocess.py
from __future__ import annotations
import re
from pathlib import Path
import pathlib
import argparse
import polars as pl


#%% Configure Patterns
INPUT_PATTERN  = "input_2023_w*.csv"
OUTPUT_PATTERN = "output_2023_w*.csv"
TEST_INPUT     = "test_input.csv"    
TEST_OUTPUT    = "test.csv"          

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUT_ROOT = Path("data/parquet")
DEFAULT_OUT_INPUT_DIR = DEFAULT_OUT_ROOT / "train_input"
DEFAULT_OUT_OUTPUT_DIR= DEFAULT_OUT_ROOT / "train_output"

#%% Schema for input (features)
INPUT_SCHEMA = {
    "game_id": pl.Int64,
    "play_id": pl.Int64,
    "player_to_predict": pl.Boolean,
    "nfl_id": pl.Int64,
    "frame_id": pl.Int32,
    "play_direction": pl.Utf8,
    "absolute_yardline_number": pl.Int32,
    "player_name": pl.Utf8,
    "player_height": pl.Utf8,      # keep as string "6-1", format to inches in later step
    "player_weight": pl.Int32,
    "player_birth_date": pl.Date,  # parse to date
    "player_position": pl.Utf8,
    "player_side": pl.Utf8,
    "player_role": pl.Utf8,
    "x": pl.Float64,
    "y": pl.Float64,
    "s": pl.Float64,
    "a": pl.Float64,
    "dir": pl.Float64,
    "o": pl.Float64,
    "num_frames_output": pl.Int32,
    "ball_land_x": pl.Float64,
    "ball_land_y": pl.Float64,
}

#%% Schema for output (labels)
OUTPUT_SCHEMA = {
    "game_id": pl.Int64,
    "play_id": pl.Int64,
    "nfl_id": pl.Int64,
    "frame_id": pl.Int32,
    "x": pl.Float64,
    "y": pl.Float64,
}

#%% Read in funcitons
WEEK_RE = re.compile(r"_w(\d{2})", re.IGNORECASE)

def extract_week(path: Path) -> int:
    m = WEEK_RE.search(path.stem)
    if not m:
        raise ValueError(f"Cannot extract week from filename: {path.name}")
    return int(m.group(1))

def read_csv_with_schema(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    # Polars date parsing: specify types + try_parse_dates=True to coerce ISO-like dates
    return pl.read_csv(
        path,
        schema=schema,
        try_parse_dates=True,
        ignore_errors=False,
    )

#%% Write parquet
def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)

def process_train_inputs(raw_dir: Path = DEFAULT_RAW_DIR) -> list[Path]:
    files = sorted(raw_dir.glob(INPUT_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files matching {INPUT_PATTERN} in {raw_dir.resolve()}")

    written = []
    all_frames: list[pl.DataFrame] = []

    for f in files:
        week = extract_week(f)
        df = read_csv_with_schema(f, INPUT_SCHEMA).with_columns(week=pl.lit(week, dtype=pl.Int32))
        if df.height == 0:
            raise ValueError(f"Empty input file: {f}")
        out_path = DEFAULT_OUT_INPUT_DIR / f"input_w{week:02d}.parquet"
        write_parquet(df, out_path)
        written.append(out_path)
        all_frames.append(df)

    # Also write a combined dataset (handy for DuckDB/fast scans)
    combined = pl.concat(all_frames, how="vertical_relaxed")
    write_parquet(combined, DEFAULT_OUT_ROOT / "train_input.parquet")
    return written

def process_train_outputs(raw_dir: Path = DEFAULT_RAW_DIR) -> list[Path]:
    files = sorted(raw_dir.glob(OUTPUT_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files matching {OUTPUT_PATTERN} in {raw_dir.resolve()}")

    written = []
    all_frames: list[pl.DataFrame] = []

    for f in files:
        week = extract_week(f)
        df = read_csv_with_schema(f, OUTPUT_SCHEMA).with_columns(week=pl.lit(week, dtype=pl.Int32))
        if df.height == 0:
            raise ValueError(f"Empty output file: {f}")
        out_path = DEFAULT_OUT_OUTPUT_DIR / f"output_w{week:02d}.parquet"
        write_parquet(df, out_path)
        written.append(out_path)
        all_frames.append(df)

    combined = pl.concat(all_frames, how="vertical_relaxed")
    write_parquet(combined, DEFAULT_OUT_ROOT / "train_output.parquet")
    return written

def process_test(raw_dir: Path = DEFAULT_RAW_DIR, write_labels: bool = True) -> list[Path]:
    written: list[Path] = []
    test_input_path = raw_dir / TEST_INPUT
    if test_input_path.exists():
        df_in = read_csv_with_schema(test_input_path, INPUT_SCHEMA).with_columns(
            week=pl.lit(-1, dtype=pl.Int32)  # mark test as week -1
        )
        p = DEFAULT_OUT_ROOT / "test_input.parquet"
        write_parquet(df_in, p)
        written.append(p)
    else:
        print(f"NOTE: {TEST_INPUT} not found; skipping test_input parquet.")

    if write_labels:
        test_out_path = raw_dir / TEST_OUTPUT
        if test_out_path.exists():
            df_out = read_csv_with_schema(test_out_path, OUTPUT_SCHEMA).with_columns(
                week=pl.lit(-1, dtype=pl.Int32)
            )
            p2 = DEFAULT_OUT_ROOT / "test_output.parquet"
            write_parquet(df_out, p2)
            written.append(p2)
        else:
            print(f"NOTE: {TEST_OUTPUT} not found; skipping test_output parquet.")
    return written

def sanity_checks():
    # Make sure you have matching weeks across input/output
    in_weeks  = set(pl.read_parquet(DEFAULT_OUT_ROOT / "train_input.parquet")["week"].unique().to_list())
    out_weeks = set(pl.read_parquet(DEFAULT_OUT_ROOT / "train_output.parquet")["week"].unique().to_list())
    if in_weeks != out_weeks:
        missing_in  = out_weeks - in_weeks
        missing_out = in_weeks  - out_weeks
        raise AssertionError(f"Week mismatch. Missing_in:{sorted(missing_in)} Missing_out:{sorted(missing_out)}")
    print(f"Sanity checks OK. Weeks: {sorted(in_weeks)}")

#%% Execute parquet conversion


def parquet_transfer(raw_dir: Path = DEFAULT_RAW_DIR, out_root: Path = DEFAULT_OUT_ROOT, include_test_labels: bool = True):
    print(f"Reading from: {raw_dir.resolve()}")
    print(f"Writing to:   {out_root.resolve()}")
    inp = process_train_inputs(raw_dir)
    out = process_train_outputs(raw_dir)
    tst = process_test(raw_dir, write_labels=include_test_labels)
    sanity_checks()

    print(f"\nWrote {len(inp)} weekly input parquet files.")
    print(f"Wrote {len(out)} weekly output parquet files.")
    if tst:
        print(f"Wrote {len(tst)} test parquet files.")



def run_parquet_transfer_if_needed(
    raw_dir: Path = Path("data/raw"),
    out_root: Path = Path("data/parquet"),
    include_test_labels: bool = True
):
    input_dir  = out_root / "train_input"
    output_dir = out_root / "train_output"
    test_in    = out_root / "test_input.parquet"
    test_out   = out_root / "test_output.parquet"

    input_ok  = input_dir.exists()  and any(input_dir.glob("*.parquet"))
    output_ok = output_dir.exists() and any(output_dir.glob("*.parquet"))

    # ✅ Require test_output if include_test_labels=True
    if include_test_labels:
        test_ok = test_in.exists() and test_out.exists()
    else:
        test_ok = test_in.exists()

    if input_ok and output_ok and test_ok:
        print("✅ Parquet files already complete — skipping conversion.")
        return

    print("⚙️ Missing Parquet files — running conversion...")
    parquet_transfer(raw_dir=raw_dir, out_root=out_root, include_test_labels=include_test_labels)


#%% Run if executed as script
if __name__ == "__main__":
    run_parquet_transfer_if_needed()



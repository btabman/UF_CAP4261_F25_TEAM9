import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # adds project root to sys.path

import gradio as gr
import pandas as pd
import numpy as np
import torch

#paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "parquet"
MODEL_DIR = ROOT_DIR / "src" / "models" / "transformer_models"

#lists parquet datasets
def list_datasets():
    return sorted([p.name for p in (DATA_DIR / "train_input").glob("*.parquet")])

#lists trained models
def list_models():
    return sorted([p.name for p in MODEL_DIR.glob("*.pt")])

#calculates RMSE
def compute_rmse(preds, targets):
    N = len(preds)
    return np.sqrt(np.sum((targets[:, 0]-preds[:, 0])**2 + (targets[:, 1]-preds[:, 1])**2) / (2 * N))

#dummy evaluation (no model load)
def predict_and_evaluate(dataset_name, row_indices, model_name):
    if not dataset_name or not model_name or not row_indices:
        return "Please select dataset, model, and row indices.", None

    input_path = DATA_DIR / "train_input" / dataset_name
    output_path = DATA_DIR / "train_output" / dataset_name.replace("input", "output")
    df_in = pd.read_parquet(input_path)
    df_out = pd.read_parquet(output_path)

    try:
        indices = [int(i.strip()) for i in row_indices.split(",") if i.strip().isdigit()]
    except ValueError:
        return "Invalid row indices format.", None
    if len(indices) == 0:
        return "No valid indices found.", None

    feature_cols = ['x', 'y', 's', 'a', 'dir', 'o', 'absolute_yardline_number']
    X = df_in.iloc[indices][feature_cols].values
    y_true = df_out.iloc[indices][['x', 'y']].values.astype(np.float32)

    #placeholder random predictions for UI testing
    preds = np.random.normal(y_true, scale=0.5)

    rmse = compute_rmse(preds, y_true)
    results = pd.DataFrame({
        "x_true": y_true[:, 0],
        "y_true": y_true[:, 1],
        "x_pred": preds[:, 0],
        "y_pred": preds[:, 1]
    })
    return f"RMSE: {rmse:.4f}", results

#used to build tab UI
def build_ui():
    dataset_options = list_datasets()
    model_options = list_models()

    with gr.Blocks(title="Transformer Evaluator") as demo:
        gr.Markdown("## Transformer Model Evaluator")

        with gr.Row():
            dataset_dropdown = gr.Dropdown(dataset_options, label="Dataset (.parquet)")
            model_dropdown = gr.Dropdown(model_options, label="Transformer Model (.pt)")

        row_indices_box = gr.Textbox(label="Row indices (comma-separated)", placeholder="e.g. 0, 2, 10")
        run_btn = gr.Button("Run Prediction and Compute RMSE")

        rmse_out = gr.Textbox(label="RMSE Result")
        result_df = gr.DataFrame(label="True vs Predicted Coordinates")

        run_btn.click(
            predict_and_evaluate,
            inputs=[dataset_dropdown, row_indices_box, model_dropdown],
            outputs=[rmse_out, result_df]
        )
    return demo

#launches app
if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=True)

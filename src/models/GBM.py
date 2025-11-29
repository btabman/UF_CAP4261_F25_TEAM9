import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#####################################################
# STEP 1:  Read in the combined input/output dataset.
#####################################################
def make_prior_features(df, n_priors):
    """
    Add x_priork and y_priork features for k = 1..n_priors per (game, play, player).
    Drops rows that don't have a full n_priors history.
    """
    df = df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"]).copy()
    groups = df.groupby(["game_id", "play_id", "nfl_id"])

    for prior in range(1, n_priors + 1):
        df["x_prior" + str(prior)] = groups["x"].shift(prior)
        df["y_prior" + str(prior)] = groups["y"].shift(prior)

    last_x_column = "x_prior" + str(n_priors)
    last_y_column = "y_prior" + str(n_priors)

    df = df.dropna(subset=[last_x_column, last_y_column])
    df = df.reset_index(drop=True)

    return df


def run_gbm_and_get_rmse(
    csv_path: str = "combined_input_output.csv",
    n_priors: int = 5,
    write_to_file: bool = True,
    rmse_output_file: str = "gbm_global_rmse.txt",
):
    """
    Runs the full LightGBM pipeline and RETURNS the global RMSE (float).

    Parameters
    ----------
    csv_path : str
        Path to combined_input_output.csv.
    n_priors : int
        Number of prior frames to use as features.
    write_to_file : bool
        If True, also write RMSE to rmse_output_file.
    rmse_output_file : str
        File to write RMSE into if write_to_file is True.

    Returns
    -------
    global_rmse : float
        Global 2D RMSE across all test frames (yards).
    """

    # STEP 1: Load CSV
    df = pd.read_csv(csv_path)

    # Rename columns if they came from a merge
    if "x_y" in df.columns and "y_y" in df.columns:
        df = df.rename(columns={"x_y": "x", "y_y": "y"})

    # STEP 2: Make prior features
    df_with_priors = make_prior_features(df, n_priors)

    # STEP 3: Build feature matrix X and targets y_x, y_y
    feature_columns = []

    for k in range(1, n_priors + 1):
        feature_columns.append("x_prior" + str(k))
        feature_columns.append("y_prior" + str(k))

    extra_frame_features = ["s", "a", "dir", "o"]
    for column in extra_frame_features:
        if column in df_with_priors.columns:
            feature_columns.append(column)

    X = df_with_priors[feature_columns]
    y_x = df_with_priors["x"]
    y_y = df_with_priors["y"]

    # STEP 4: Train LightGBM
    X_train, X_test, yx_train, yx_test, yy_train, yy_test, df_train, df_test = (
        train_test_split(
            X,
            y_x,
            y_y,
            df_with_priors,
            test_size=0.2,
            random_state=42,
        )
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

    gbm_x = LGBMRegressor(**params)
    gbm_y = LGBMRegressor(**params)

    gbm_x.fit(X_train, yx_train)
    gbm_y.fit(X_train, yy_train)

    pred_x = gbm_x.predict(X_test)
    pred_y = gbm_y.predict(X_test)

    # STEP 5: Attach predictions
    df_test = df_test.reset_index(drop=True).copy()
    df_test["pred_x_gbm"] = pred_x
    df_test["pred_y_gbm"] = pred_y

    # STEP 6: Compute global RMSE
    squared_error = (
        (df_test["x"] - df_test["pred_x_gbm"]) ** 2
        + (df_test["y"] - df_test["pred_y_gbm"]) ** 2
    )
    global_rmse = float(np.sqrt(squared_error.mean()))

    # Optional: print and write to file
    print(f"Global RMSE: {global_rmse:.3f} yards")

    if write_to_file:
        with open(rmse_output_file, "w") as file:
            file.write(f"Global RMSE: {global_rmse:.6f} yards\n")

    # *** Tkey line you wanted ***
    return global_rmse


# If you still want to run this file directly from the command line:
if __name__ == "__main__":
    rmse = run_gbm_and_get_rmse()
    # Already printed inside the function; you could also do:
    # print("Returned RMSE:", rmse)
import os
import sqlite3
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import numpy as np
import pandas as pd

from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIG
# ============================================================

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "warsaw_flat_price_retraining"

MODEL_NAME = "warsaw_flat_price_model"
CHAMPION_ALIAS = "champion"

PREDICTION_DB = "prediction_logs.db"

TRAINING_DATA_PATHS = [
    "data/raw/warsaw_flat_prices_synthetic.csv",
    "warsaw_flat_prices_synthetic.csv"
]

OUTPUT_DIR = "outputs"
FEEDBACK_DATA_PATH = "outputs/retraining_feedback_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_IMPROVEMENT_R2 = 0.001

FEATURES = [
    "distance_from_center_km",
    "area_m2",
    "floor",
    "district",
    "rent_pln"
]

NUMERIC_FEATURES = [
    "distance_from_center_km",
    "area_m2",
    "floor",
    "rent_pln"
]

CATEGORICAL_FEATURES = [
    "district"
]

TARGET = "price_pln"


# ============================================================
# DATA
# ============================================================

def find_training_data_path():
    for path in TRAINING_DATA_PATHS:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Training data not found. Expected one of: "
        + ", ".join(TRAINING_DATA_PATHS)
    )


def load_base_training_data():
    path = find_training_data_path()
    df = pd.read_csv(path)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns in training data: "
            + ", ".join(missing_columns)
        )

    return df[required_columns]


def load_prediction_logs():
    if not os.path.exists(PREDICTION_DB):
        print("prediction_logs.db not found. Using only base training data.")
        return pd.DataFrame()

    conn = sqlite3.connect(PREDICTION_DB)

    logs = pd.read_sql_query("""
        SELECT
            distance_from_center_km,
            area_m2,
            floor,
            district,
            rent_pln,
            predicted_price_pln
        FROM prediction_logs
    """, conn)

    conn.close()

    return logs


def create_synthetic_feedback_data(logs):
    """
    Educational simulation.

    In real production, actual_price_pln should come from real business data.
    Here we simulate actual labels from prediction logs.
    """

    if logs.empty:
        return pd.DataFrame()

    feedback = logs.copy()

    rng = np.random.default_rng(RANDOM_STATE)

    noise = rng.normal(
        loc=0,
        scale=50000,
        size=len(feedback)
    )

    feedback[TARGET] = feedback["predicted_price_pln"] + noise
    feedback[TARGET] = feedback[TARGET].clip(250000, 3500000)

    return feedback[FEATURES + [TARGET]]


def build_retraining_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_df = load_base_training_data()
    logs = load_prediction_logs()
    feedback_df = create_synthetic_feedback_data(logs)

    if not feedback_df.empty:
        feedback_df.to_csv(FEEDBACK_DATA_PATH, index=False)

    full_df = pd.concat(
        [base_df, feedback_df],
        ignore_index=True
    )

    return full_df, len(base_df), len(feedback_df)


# ============================================================
# MODEL
# ============================================================

def get_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", get_one_hot_encoder(), CATEGORICAL_FEATURES)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def get_candidate_models():
    return {
        "linear_regression_retrain": LinearRegression(),
        "ridge_alpha_1_retrain": Ridge(alpha=1.0),
        "ridge_alpha_10_retrain": Ridge(alpha=10.0),
        "random_forest_depth_8_retrain": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            random_state=RANDOM_STATE
        )
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "r2_test": float(r2),
        "mae_test": float(mae),
        "rmse_test": float(rmse)
    }


# ============================================================
# CHAMPION
# ============================================================

def load_current_champion():
    champion_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"

    try:
        champion = mlflow.pyfunc.load_model(champion_uri)
        return champion, champion_uri

    except Exception as exc:
        print("Could not load current champion model.")
        print(str(exc))
        return None, champion_uri


def register_new_champion(best_run_id):
    model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=registered_model.version
    )

    return registered_model.version


# ============================================================
# MAIN
# ============================================================

def main():
    if mlflow.active_run() is not None:
        mlflow.end_run()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    full_df, base_rows, feedback_rows = build_retraining_dataset()

    print("Base training rows:", base_rows)
    print("Feedback rows:", feedback_rows)
    print("Total rows:", len(full_df))

    X = full_df[FEATURES]
    y = full_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    champion_model, champion_uri = load_current_champion()

    champion_metrics = None

    if champion_model is not None:
        champion_metrics = evaluate_model(
            champion_model,
            X_test,
            y_test
        )

        print("Current champion:")
        print("Champion URI:", champion_uri)
        print("Champion metrics:", champion_metrics)

    best_run_id = None
    best_model_name = None
    best_r2 = -999999
    best_metrics = None

    candidate_models = get_candidate_models()

    for model_name, raw_model in candidate_models.items():
        pipeline = build_pipeline(raw_model)

        run_name = (
            f"{model_name}_"
            f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        with mlflow.start_run(run_name=run_name):
            pipeline.fit(X_train, y_train)

            metrics = evaluate_model(
                pipeline,
                X_test,
                y_test
            )

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("training_type", "retraining")
            mlflow.log_param("base_rows", base_rows)
            mlflow.log_param("feedback_rows", feedback_rows)
            mlflow.log_param("total_rows", len(full_df))
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("experiment_random_state", RANDOM_STATE)

            # WAŻNE:
            # Parametry modelu zapisujemy z prefiksem model__,
            # żeby nie było konfliktu z parametrami eksperymentu.
            for key, value in raw_model.get_params().items():
                mlflow.log_param(f"model__{key}", value)

            mlflow.log_metric("r2_test", metrics["r2_test"])
            mlflow.log_metric("mae_test", metrics["mae_test"])
            mlflow.log_metric("rmse_test", metrics["rmse_test"])

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model"
            )

            current_run_id = mlflow.active_run().info.run_id

            print("-" * 60)
            print("Candidate:", model_name)
            print("Run ID:", current_run_id)
            print("R2 test:", round(metrics["r2_test"], 4))
            print("MAE test:", round(metrics["mae_test"], 2))
            print("RMSE test:", round(metrics["rmse_test"], 2))

            if metrics["r2_test"] > best_r2:
                best_r2 = metrics["r2_test"]
                best_run_id = current_run_id
                best_model_name = model_name
                best_metrics = metrics

    print("=" * 60)
    print("Best retrained model:", best_model_name)
    print("Best retrained metrics:", best_metrics)

    promote_model = False

    if champion_metrics is None:
        promote_model = True
        print("No champion available. New model will be promoted.")

    else:
        champion_r2 = champion_metrics["r2_test"]
        new_r2 = best_metrics["r2_test"]

        print("Champion R2:", round(champion_r2, 4))
        print("New model R2:", round(new_r2, 4))

        if new_r2 > champion_r2 + MIN_IMPROVEMENT_R2:
            promote_model = True

    if promote_model:
        new_version = register_new_champion(best_run_id)
        print("New champion model registered.")
        print("New champion version:", new_version)

    else:
        print("New model was not promoted. Current champion remains active.")

    print("Retraining pipeline finished.")


if __name__ == "__main__":
    main()

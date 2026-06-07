import os
import json
import sqlite3
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TRAINING_DATA_PATHS = [
    "data/raw/warsaw_flat_prices_synthetic.csv",
    "warsaw_flat_prices_synthetic.csv"
]

PREDICTION_DB = "prediction_logs.db"
OUTPUT_DIR = "outputs"
OUTPUT_PATH = "outputs/drift_report.json"

NUMERIC_COLUMNS = [
    "distance_from_center_km",
    "area_m2",
    "floor",
    "rent_pln",
    "predicted_price_pln"
]

CATEGORICAL_COLUMNS = [
    "district"
]

DRIFT_THRESHOLD_PERCENT = 20.0


# ============================================================
# HELPERS
# ============================================================

def find_training_data_path():
    for path in TRAINING_DATA_PATHS:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Training data not found. Expected one of: "
        + ", ".join(TRAINING_DATA_PATHS)
    )


def load_training_data():
    path = find_training_data_path()
    df = pd.read_csv(path)

    # In training data target is called price_pln.
    # In production logs prediction is called predicted_price_pln.
    if "price_pln" in df.columns:
        df["predicted_price_pln"] = df["price_pln"]

    return df


def load_prediction_logs():
    if not os.path.exists(PREDICTION_DB):
        raise FileNotFoundError("prediction_logs.db not found.")

    conn = sqlite3.connect(PREDICTION_DB)

    logs = pd.read_sql_query("""
        SELECT *
        FROM prediction_logs
    """, conn)

    conn.close()

    return logs


def calculate_numeric_drift(reference_df, production_df, column):
    reference_mean = reference_df[column].mean()
    production_mean = production_df[column].mean()

    if reference_mean == 0:
        percent_change = 0
    else:
        percent_change = (
            (production_mean - reference_mean)
            / reference_mean
        ) * 100

    drift_detected = abs(percent_change) > DRIFT_THRESHOLD_PERCENT

    return {
        "column": column,
        "reference_mean": round(float(reference_mean), 4),
        "production_mean": round(float(production_mean), 4),
        "percent_change": round(float(percent_change), 2),
        "drift_detected": bool(drift_detected)
    }


def calculate_categorical_drift(reference_df, production_df, column):
    reference_distribution = (
        reference_df[column]
        .value_counts(normalize=True)
        .round(4)
        .to_dict()
    )

    production_distribution = (
        production_df[column]
        .value_counts(normalize=True)
        .round(4)
        .to_dict()
    )

    reference_categories = set(reference_distribution.keys())
    production_categories = set(production_distribution.keys())

    new_categories = list(production_categories - reference_categories)
    missing_categories = list(reference_categories - production_categories)

    drift_detected = len(new_categories) > 0

    return {
        "column": column,
        "reference_top_categories": dict(
            list(reference_distribution.items())[:5]
        ),
        "production_top_categories": dict(
            list(production_distribution.items())[:5]
        ),
        "new_categories": new_categories,
        "missing_categories": missing_categories,
        "drift_detected": bool(drift_detected)
    }


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reference_df = load_training_data()
    production_df = load_prediction_logs()

    if production_df.empty:
        report = {
            "status": "no_production_data",
            "message": "No prediction logs available yet."
        }

        with open(OUTPUT_PATH, "w") as f:
            json.dump(report, f, indent=4)

        print(json.dumps(report, indent=4))
        return

    numeric_results = []

    for column in NUMERIC_COLUMNS:
        if column in reference_df.columns and column in production_df.columns:
            numeric_results.append(
                calculate_numeric_drift(reference_df, production_df, column)
            )

    categorical_results = []

    for column in CATEGORICAL_COLUMNS:
        if column in reference_df.columns and column in production_df.columns:
            categorical_results.append(
                calculate_categorical_drift(reference_df, production_df, column)
            )

    drift_detected = any(
        item["drift_detected"] for item in numeric_results
    ) or any(
        item["drift_detected"] for item in categorical_results
    )

    report = {
        "status": "ok",
        "drift_detected": bool(drift_detected),
        "production_rows": int(len(production_df)),
        "reference_rows": int(len(reference_df)),
        "numeric_drift": numeric_results,
        "categorical_drift": categorical_results
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()

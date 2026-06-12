import os
import json
import sqlite3
import subprocess
import uuid
import time
from datetime import datetime, timezone

import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)


# ============================================================
# CONFIG
# ============================================================

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_URI = "models:/warsaw_flat_price_model@champion"
PREDICTION_DB = "prediction_logs.db"
DRIFT_REPORT_PATH = "outputs/drift_report.json"
PROJECT_DIR = "/root/MLOps/Projekt_1"


# ============================================================
# PROMETHEUS METRICS
# ============================================================

PREDICTIONS_TOTAL = Counter(
    "flat_price_predictions_total",
    "Total number of flat price predictions"
)

PREDICTION_ERRORS_TOTAL = Counter(
    "flat_price_prediction_errors_total",
    "Total number of prediction errors"
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "flat_price_prediction_latency_seconds",
    "Prediction latency in seconds"
)

LAST_PREDICTED_PRICE_PLN = Gauge(
    "flat_price_last_predicted_price_pln",
    "Last predicted flat price in PLN"
)

LAST_AREA_M2 = Gauge(
    "flat_price_last_area_m2",
    "Last input flat area in square meters"
)

LAST_DISTANCE_FROM_CENTER_KM = Gauge(
    "flat_price_last_distance_from_center_km",
    "Last input distance from Warsaw center in kilometers"
)

LAST_RENT_PLN = Gauge(
    "flat_price_last_rent_pln",
    "Last input rent in PLN"
)

DATA_DRIFT_DETECTED = Gauge(
    "flat_price_data_drift_detected",
    "Data drift detected flag: 1 = drift detected, 0 = no drift"
)

DRIFT_PRODUCTION_ROWS = Gauge(
    "flat_price_drift_production_rows",
    "Number of production rows used in drift report"
)


# ============================================================
# DATABASE INIT
# ============================================================

def init_prediction_db():
    conn = sqlite3.connect(PREDICTION_DB)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            request_id TEXT PRIMARY KEY,
            timestamp TEXT,
            distance_from_center_km REAL,
            area_m2 REAL,
            floor INTEGER,
            district TEXT,
            rent_pln REAL,
            predicted_price_pln REAL,
            model_uri TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_prediction_log(
    request_id,
    distance_from_center_km,
    area_m2,
    floor,
    district,
    rent_pln,
    predicted_price_pln,
    model_uri
):
    conn = sqlite3.connect(PREDICTION_DB)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO prediction_logs (
            request_id,
            timestamp,
            distance_from_center_km,
            area_m2,
            floor,
            district,
            rent_pln,
            predicted_price_pln,
            model_uri
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        request_id,
        datetime.now(timezone.utc).isoformat(),
        distance_from_center_km,
        area_m2,
        floor,
        district,
        rent_pln,
        predicted_price_pln,
        model_uri
    ))

    conn.commit()
    conn.close()


# ============================================================
# DRIFT HELPERS
# ============================================================

def load_drift_report():
    if not os.path.exists(DRIFT_REPORT_PATH):
        return {
            "status": "missing",
            "message": "Drift report not found. Run: python src/monitoring/drift_report.py"
        }

    with open(DRIFT_REPORT_PATH, "r") as f:
        return json.load(f)


def update_drift_prometheus_metrics():
    report = load_drift_report()

    if report.get("status") != "ok":
        DATA_DRIFT_DETECTED.set(0)
        DRIFT_PRODUCTION_ROWS.set(0)
        return report

    drift_flag = 1 if report.get("drift_detected") else 0
    production_rows = report.get("production_rows", 0)

    DATA_DRIFT_DETECTED.set(drift_flag)
    DRIFT_PRODUCTION_ROWS.set(production_rows)

    return report


# ============================================================
# LOAD MODEL FROM MLFLOW
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Warsaw Flat Price Prediction API",
    description="Simple ML API for predicting flat prices in Warsaw",
    version="1.0.0"
)

init_prediction_db()


# ============================================================
# INPUT SCHEMA
# ============================================================

class FlatInput(BaseModel):
    distance_from_center_km: float
    area_m2: float
    floor: int
    district: str
    rent_pln: float


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Warsaw Flat Price Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_uri": MODEL_URI
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(flat: FlatInput):

    start_time = time.time()

    try:
        input_df = pd.DataFrame([{
            "distance_from_center_km": flat.distance_from_center_km,
            "area_m2": flat.area_m2,
            "floor": flat.floor,
            "district": flat.district,
            "rent_pln": flat.rent_pln
        }])

        prediction = model.predict(input_df)
        predicted_price = round(float(prediction[0]), 2)

        request_id = str(uuid.uuid4())

        save_prediction_log(
            request_id=request_id,
            distance_from_center_km=flat.distance_from_center_km,
            area_m2=flat.area_m2,
            floor=flat.floor,
            district=flat.district,
            rent_pln=flat.rent_pln,
            predicted_price_pln=predicted_price,
            model_uri=MODEL_URI
        )

        PREDICTIONS_TOTAL.inc()
        LAST_PREDICTED_PRICE_PLN.set(predicted_price)
        LAST_AREA_M2.set(flat.area_m2)
        LAST_DISTANCE_FROM_CENTER_KM.set(flat.distance_from_center_km)
        LAST_RENT_PLN.set(flat.rent_pln)

        return {
            "request_id": request_id,
            "predicted_price_pln": predicted_price,
            "model_uri": MODEL_URI
        }

    except Exception:
        PREDICTION_ERRORS_TOTAL.inc()
        raise

    finally:
        latency = time.time() - start_time
        PREDICTION_LATENCY_SECONDS.observe(latency)


# ============================================================
# RECENT LOGS ENDPOINT
# ============================================================

@app.get("/logs/recent")
def recent_logs(limit: int = 10):
    conn = sqlite3.connect(PREDICTION_DB)

    logs = pd.read_sql_query(
        """
        SELECT *
        FROM prediction_logs
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        conn,
        params=(limit,)
    )

    conn.close()

    return logs.to_dict(orient="records")


# ============================================================
# MONITORING SUMMARY ENDPOINT
# ============================================================

@app.get("/monitoring/summary")
def monitoring_summary():
    conn = sqlite3.connect(PREDICTION_DB)

    logs = pd.read_sql_query("""
        SELECT *
        FROM prediction_logs
    """, conn)

    conn.close()

    if logs.empty:
        return {
            "message": "No prediction logs yet"
        }

    total_predictions = len(logs)

    avg_predicted_price = logs["predicted_price_pln"].mean()
    avg_area_m2 = logs["area_m2"].mean()
    avg_distance = logs["distance_from_center_km"].mean()
    avg_rent = logs["rent_pln"].mean()

    min_predicted_price = logs["predicted_price_pln"].min()
    max_predicted_price = logs["predicted_price_pln"].max()

    most_common_districts = (
        logs["district"]
        .value_counts()
        .head(5)
        .to_dict()
    )

    return {
        "total_predictions": int(total_predictions),
        "avg_predicted_price_pln": round(float(avg_predicted_price), 2),
        "min_predicted_price_pln": round(float(min_predicted_price), 2),
        "max_predicted_price_pln": round(float(max_predicted_price), 2),
        "avg_area_m2": round(float(avg_area_m2), 2),
        "avg_distance_from_center_km": round(float(avg_distance), 2),
        "avg_rent_pln": round(float(avg_rent), 2),
        "most_common_districts": most_common_districts,
        "model_uri": MODEL_URI
    }


# ============================================================
# DRIFT MONITORING ENDPOINT
# ============================================================

@app.get("/monitoring/drift")
def monitoring_drift():
    report = update_drift_prometheus_metrics()
    return report


# ============================================================
# PROMETHEUS METRICS ENDPOINT
# ============================================================

@app.get("/metrics")
def prometheus_metrics():
    update_drift_prometheus_metrics()

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================
# ADMIN ENDPOINTS FOR AIRFLOW
# ============================================================

@app.post("/admin/run-drift-report")
def admin_run_drift_report():
    result = subprocess.run(
        ["python", "src/monitoring/drift_report.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "stderr": result.stderr,
                "stdout": result.stdout
            }
        )

    return {
        "status": "ok",
        "task": "drift_report",
        "stdout": result.stdout
    }


@app.post("/admin/retrain")
def admin_retrain():
    result = subprocess.run(
        ["python", "src/training/retrain.py"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=900
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "stderr": result.stderr,
                "stdout": result.stdout
            }
        )

    return {
        "status": "ok",
        "task": "retraining",
        "stdout": result.stdout
    }


@app.post("/admin/reload-model")
def admin_reload_model():
    global model

    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)

        return {
            "status": "ok",
            "message": "Champion model reloaded",
            "model_uri": MODEL_URI
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(exc)
            }
        )

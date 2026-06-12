#!/bin/bash

set -e

echo "========================================"
echo "API SMOKE TESTS STARTED"
echo "========================================"

API_URL="http://127.0.0.1:8000"
PROMETHEUS_URL="http://127.0.0.1:9090"
GRAFANA_URL="http://127.0.0.1:3000"
AIRFLOW_URL="http://127.0.0.1:8080"

echo "1. Checking FastAPI /health..."
curl -fsS "$API_URL/health"
echo ""

echo "2. Checking FastAPI /predict..."
PREDICT_RESPONSE=$(curl -fsS -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "distance_from_center_km": 5.0,
    "area_m2": 55,
    "floor": 4,
    "district": "Mokotów",
    "rent_pln": 850
  }')

echo "$PREDICT_RESPONSE"

echo "$PREDICT_RESPONSE" | grep -q "predicted_price_pln"
echo "$PREDICT_RESPONSE" | grep -q "request_id"

echo "3. Checking FastAPI /metrics..."
curl -fsS "$API_URL/metrics" | grep -q "flat_price_predictions_total"

echo "4. Checking monitoring summary..."
curl -fsS "$API_URL/monitoring/summary" | grep -q "total_predictions"

echo "5. Checking drift endpoint..."
curl -fsS "$API_URL/monitoring/drift" | grep -q "status"

echo "6. Checking Prometheus health..."
curl -fsS "$PROMETHEUS_URL/-/healthy"

echo "7. Checking Prometheus query..."
curl -fsS "$PROMETHEUS_URL/api/v1/query?query=flat_price_predictions_total" | grep -q "success"

echo "8. Checking Grafana health..."
curl -fsS "$GRAFANA_URL/api/health" | grep -q "database"

echo "9. Checking Airflow health..."
curl -fsS "$AIRFLOW_URL/health" | grep -q "metadatabase"

echo "========================================"
echo "ALL API SMOKE TESTS PASSED"
echo "========================================"

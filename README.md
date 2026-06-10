
# Projekt MLOps: predykcja cen mieszkań w Warszawie

To jest kompletny projekt MLOps pokazujący przejście od prostego modelu uczenia maszynowego do działającego systemu produkcyjnego na serwerze VPS.

Projekt obejmuje:

1. wygenerowanie syntetycznej bazy danych
2. trenowanie modelu
3. śledzenie eksperymentów w MLflow
4. rejestrację najlepszego modelu jako champion model
5. wystawienie modelu przez FastAPI
6. logowanie predykcji
7. metryki Prometheus
8. dashboard Grafana
9. wykrywanie dryftu danych
10. automatyczne odświeżanie raportu dryftu przez cron
11. uruchamianie systemu przez Docker Compose
12. automatyczny deployment przez GitHub Actions

## Problem biznesowy

Model przewiduje cenę mieszkania w Warszawie na podstawie pięciu zmiennych:

1. odległość od centrum Warszawy
2. metraż
3. piętro
4. dzielnica
5. wysokość czynszu

Zmienna przewidywana:

```text
price_pln
```

## Model

Pierwszy model produkcyjny jest prostym modelem regresyjnym wytrenowanym na danych syntetycznych.

W projekcie porównywane są różne modele, a najlepszy zostaje oznaczony w MLflow jako model produkcyjny.

Przykładowe modele:

```text
Linear Regression
Ridge Regression
Lasso Regression
Random Forest Regressor
```

## Architektura

```text
Dane syntetyczne
      ↓
Jupyter Notebook / skrypt treningowy
      ↓
MLflow Tracking
      ↓
MLflow Model Registry
      ↓
Champion model
      ↓
FastAPI
      ↓
Logi predykcji
      ↓
Prometheus
      ↓
Grafana
      ↓
Raport dryftu
      ↓
Cron
      ↓
GitHub Actions
      ↓
VPS
```

## Główne komponenty

| Komponent | Rola |
|---|---|
| Jupyter Notebook | eksperymenty Data Science |
| MLflow | śledzenie eksperymentów i wersji modeli |
| FastAPI | produkcyjne API do predykcji |
| SQLite | lokalne metadane i logi predykcji |
| Prometheus | zbieranie metryk |
| Grafana | dashboard monitoringu |
| Cron | automatyczne odświeżanie dryftu |
| Docker Compose | uruchamianie środowiska |
| GitHub Actions | automatyczne CI/CD |
| VPS | serwer produkcyjny |

## Endpointy API

```text
GET  /
GET  /health
POST /predict
GET  /logs/recent
GET  /monitoring/summary
GET  /monitoring/drift
GET  /metrics
```

## Przykład predykcji

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "distance_from_center_km": 5.0,
  "area_m2": 55,
  "floor": 4,
  "district": "Mokotów",
  "rent_pln": 850
}'
```

Przykładowa odpowiedź:

```json
{
  "request_id": "example-request-id",
  "predicted_price_pln": 920387.07,
  "model_uri": "models:/warsaw_flat_price_model@champion"
}
```

## Uruchamianie projektu

```bash
docker compose up -d --build
```

Sprawdzenie kontenerów:

```bash
docker compose ps
```

Zatrzymanie projektu:

```bash
docker compose down
```

## Health check

FastAPI:

```bash
curl http://127.0.0.1:8000/health
```

Prometheus:

```bash
curl http://127.0.0.1:9090/-/healthy
```

Grafana:

```bash
curl http://127.0.0.1:3000/api/health
```

## Monitoring

FastAPI wystawia metryki pod adresem:

```text
http://127.0.0.1:8000/metrics
```

Prometheus zbiera te metryki, a Grafana pokazuje je na dashboardzie.

Najważniejsze metryki:

```text
flat_price_predictions_total
flat_price_prediction_errors_total
flat_price_prediction_latency_seconds
flat_price_last_predicted_price_pln
flat_price_data_drift_detected
flat_price_drift_production_rows
```

## Dryft danych

Projekt porównuje dane treningowe z danymi produkcyjnymi zapisanymi w logach predykcji.

Ręczne uruchomienie raportu dryftu:

```bash
python src/monitoring/drift_report.py
```

Automatyczne uruchamianie raportu dryftu:

```bash
crontab -l
```

Cron uruchamia:

```bash
/root/MLOps/Projekt_1/scripts/run_drift_report.sh
```

## CI/CD

Workflow GitHub Actions znajduje się tutaj:

```text
.github/workflows/ci.yml
```

Workflow wykonuje:

1. podstawowe sprawdzenie kodu Python
2. sprawdzenie konfiguracji Docker Compose
3. połączenie z VPS przez SSH
4. deployment przez Docker Compose
5. health check FastAPI, Prometheus i Grafany

## Status projektu

Obecny status:

```text
Produkcyjne MVP zakończone
```

Zrealizowane elementy:

```text
MLflow tracking
Model registry
FastAPI
Logowanie predykcji
Prometheus
Grafana
Detekcja dryftu
Cron
Docker Compose
GitHub Actions CI/CD
```

Możliwe kolejne kroki:

```text
Airflow retraining pipeline
automatyczne porównywanie modeli
automatyczna aktualizacja champion model
migracja z SQLite do PostgreSQL
wdrożenie na Kubernetes k3s
```


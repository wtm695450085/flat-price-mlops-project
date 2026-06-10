# Architektura projektu

## Cel

Ten projekt pokazuje prosty system MLOps wdrożony na serwerze VPS.

Główny cel to przejście od eksperymentu w notebooku do działającego, monitorowanego API modelu ML.

## Architektura systemu

```text
Generowanie danych
      ↓
Trening modelu
      ↓
MLflow
      ↓
Model Registry
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
Detekcja dryftu
      ↓
Cron
      ↓
GitHub Actions
```

## Architektura uruchomieniowa

```text
Docker Compose
│
├── flat-price-api
│   ├── ładuje champion model z MLflow
│   ├── wystawia /predict
│   ├── wystawia /metrics
│   └── zapisuje logi predykcji
│
├── prometheus
│   └── pobiera metryki z flat-price-api
│
└── grafana
    └── pokazuje dashboard z metrykami
```

## Przepływ danych

1. Użytkownik wysyła dane mieszkania do `/predict`.
2. FastAPI waliduje dane wejściowe.
3. FastAPI używa modelu oznaczonego jako champion.
4. Model zwraca przewidywaną cenę.
5. Zapytanie i predykcja są zapisywane w `prediction_logs.db`.
6. Metryki są wystawiane przez `/metrics`.
7. Prometheus zbiera metryki.
8. Grafana pokazuje dashboard.
9. Skrypt dryftu porównuje dane produkcyjne z treningowymi.
10. Cron automatycznie odświeża raport dryftu.

## Rola MLflow

MLflow odpowiada za:

```text
śledzenie eksperymentów
zapisywanie metryk
artefakty modelu
rejestr modeli
wybór modelu champion
```

## Rola Prometheusa

Prometheus zbiera metryki operacyjne:

```text
liczba predykcji
liczba błędów
czas odpowiedzi modelu
ostatnia przewidywana cena
ostatnie wartości wejściowe
flaga dryftu
```

## Rola Grafany

Grafana pokazuje:

```text
liczbę predykcji
czas odpowiedzi
ostatnią przewidywaną cenę
cechy wejściowe
flagi dryftu
liczbę rekordów użytych do raportu dryftu
```

## Rola GitHub Actions

GitHub Actions wykonuje:

```text
sprawdzenie kodu
sprawdzenie Docker Compose
połączenie z VPS przez SSH
deployment przez docker compose
health check po deploymentcie
```

## Obecne ograniczenia

```text
dane są syntetyczne
SQLite jest używany zamiast PostgreSQL
detekcja dryftu jest prosta i regułowa
retraining nie jest jeszcze zautomatyzowany
Kubernetes nie jest jeszcze użyty
```

## Planowane rozszerzenia

```text
Airflow retraining pipeline
PostgreSQL
automatyczne porównywanie modeli
automatyczna promocja champion model
wdrożenie na Kubernetes k3s
```

# Status projektu

## Obecny etap

```text
Produkcyjne MVP zakończone
```

## Zrealizowane kamienie milowe

### 1. Dane i model

```text
wygenerowano syntetyczną bazę danych
wytrenowano model bazowy
wykonano kilka eksperymentów modelowych
wybrano najlepszy model
zarejestrowano model jako champion w MLflow
```

### 2. MLflow

```text
lokalny backend SQLite
śledzenie eksperymentów
zapisywanie metryk
artefakty modelu
model registry
alias champion
```

### 3. FastAPI

```text
endpoint predykcyjny
endpoint health
logowanie predykcji
endpoint monitoring summary
endpoint drift
endpoint Prometheus metrics
```

### 4. Monitoring

```text
metryki predykcji
metryki czasu odpowiedzi
metryki błędów
flaga dryftu danych
integracja z Prometheus
dashboard Grafana
```

### 5. Detekcja dryftu

```text
porównanie danych treningowych z produkcyjnymi
detekcja dryftu numerycznego
detekcja dryftu kategorycznego
raport JSON
automatyczne odświeżanie przez cron
```

### 6. Deployment

```text
Docker Compose
kontener FastAPI
kontener Prometheus
kontener Grafana
lokalne mapowanie portów
deployment na VPS
```

### 7. CI/CD

```text
repozytorium GitHub
GitHub Actions CI
GitHub Actions deployment na VPS
deployment przez SSH
health check po deploymentcie
```

## Sprawdzenie produkcji

Użyj:

```bash
make ps
make health
make metrics
make prometheus-health
make grafana-health
```

## Główne osiągnięcie

Projekt pokazuje pełną ścieżkę MLOps:

```text
notebook
→ MLflow
→ model registry
→ FastAPI
→ Docker Compose
→ Prometheus
→ Grafana
→ dryft
→ cron
→ GitHub Actions deployment
```

## Następny rekomendowany etap

```text
Airflow retraining pipeline
```

Cel kolejnego etapu:

```text
nowe dane
→ retraining
→ zapis do MLflow
→ porównanie modeli
→ aktualizacja champion model
→ redeployment
```

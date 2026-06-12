.RECIPEPREFIX := >

.PHONY: up down build ps logs api-logs health metrics drift retrain test-predict prometheus-health grafana-health git-status

up:
>docker compose up -d

build:
>docker compose up -d --build

down:
>docker compose down

ps:
>docker compose ps

logs:
>docker compose logs -f

api-logs:
>docker compose logs -f flat-price-api

health:
>curl http://127.0.0.1:8000/health

metrics:
>curl http://127.0.0.1:8000/metrics | grep flat_price

drift:
>python src/monitoring/drift_report.py

retrain:
>python src/training/retrain.py

test-predict:
>curl -X POST "http://127.0.0.1:8000/predict" \
>  -H "Content-Type: application/json" \
>  -d '{"distance_from_center_km":5.0,"area_m2":55,"floor":4,"district":"Mokotów","rent_pln":850}'

prometheus-health:
>curl http://127.0.0.1:9090/-/healthy

grafana-health:
>curl http://127.0.0.1:3000/api/health

git-status:
>git status

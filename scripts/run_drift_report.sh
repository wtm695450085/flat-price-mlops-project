#!/bin/bash

cd /root/MLOps/Projekt_1

/root/miniconda3/bin/python src/monitoring/drift_report.py >> logs/drift_cron.log 2>&1

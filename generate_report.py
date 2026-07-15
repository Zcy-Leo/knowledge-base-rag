import json
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiment_results")

summary = json.load(open(os.path.join(OUTPUT_DIR, "experiment_summary_v3.json")))
all_metrics = []
with open(os.path.join(OUTPUT_DIR, "experiment_metrics_v3.csv"), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    all_metrics = list(reader)

for m in all_metrics:
    for k in ['ndcg@5', 'ndcg@10', 'mrr', 'recall@5', 'recall@10', 'precision@5', 'precision@10', 'f1@5', 'f1@10', 'latency']:
        if k in m:
            m[k] = float(m[k])

from experiment_runner_v3 import generate_docx_report
generate_docx_report(summary, all_metrics)
import json
from pathlib import Path
from problem_generation_pipeline_ray import *

# stats_path = Path("/mnt/weka/home/anzexie/code_agent/code_questions/generate_code_problems/generated_problems/datasets_stats.jsonl")

# a = find_workload()

stats_path = DATASET_SAVE_DIR / "datasets_stats.jsonl"

total_problems = 0

with open(stats_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        stats = json.loads(line)
        total_problems += (stats['total_samples'] - stats['num_invalid'])

print(f"Total valid problems: {total_problems}")   # 971k 
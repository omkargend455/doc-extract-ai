"""Generate comprehensive evaluation comparing baseline vs improved pipeline on 349 development records."""

import os
import time
import json
import pandas as pd
from src.baseline_regex import RegexBaselineExtractor
from src.evaluate import evaluate_predictions

task1_dir = 'C:/Users/Omkar/Downloads/sroie/0325updated.task1train(626p)'
task2_dir = 'C:/Users/Omkar/Downloads/sroie/0325updated.task2train(626p)'

baseline_details_path = 'results/dev_baseline_details.csv'
baseline_summary_path = 'results/dev_baseline_summary.csv'

base_df = pd.read_csv(baseline_details_path)
base_summary = pd.read_csv(baseline_summary_path)

print(f"Loaded {len(base_df)} baseline development records.")

extractor = RegexBaselineExtractor()
improved_preds = []
ground_truths = []

t0 = time.time()
for doc_id in base_df['id']:
    t1_path = os.path.join(task1_dir, f"{doc_id}.txt")
    raw_ocr = open(t1_path, encoding="utf-8", errors="ignore").read()
    pred = extractor.extract(raw_ocr, doc_type="receipt")
    improved_preds.append(pred)

    t2_path = os.path.join(task2_dir, f"{doc_id}.txt")
    with open(t2_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    ground_truths.append(gt)

elapsed_sec = time.time() - t0
avg_ms_per_doc = (elapsed_sec / len(base_df)) * 1000.0

print(f"Evaluated 349 records in {elapsed_sec:.2f}s ({avg_ms_per_doc:.2f} ms/doc).")

eval_res = evaluate_predictions(improved_preds, ground_truths)
improved_summary = eval_res["summary_df"]
improved_details = eval_res["details_df"]

# Save outputs
os.makedirs("results", exist_ok=True)
improved_summary_path = "results/dev_improved_summary.csv"
improved_details_path = "results/dev_improved_details.csv"

improved_summary.to_csv(improved_summary_path, index=False)
improved_details.to_csv(improved_details_path, index=False)

print("\n" + "=" * 80)
print(" SROIE DEVELOPMENT SET BENCHMARK COMPARISON (N = 349)")
print("=" * 80)

merged = pd.merge(
    base_summary,
    improved_summary,
    on="Field",
    suffixes=(" (Baseline)", " (Improved)")
)

merged["Exact Match Diff (pp)"] = merged["Exact Match (%) (Improved)"] - merged["Exact Match (%) (Baseline)"]
merged["Norm Match Diff (pp)"] = merged["Normalized Match (%) (Improved)"] - merged["Normalized Match (%) (Baseline)"]
merged["RapidFuzz Diff"] = merged["Avg RapidFuzz Score (Improved)"] - merged["Avg RapidFuzz Score (Baseline)"]

display_cols = [
    "Field",
    "Exact Match (%) (Baseline)", "Exact Match (%) (Improved)", "Exact Match Diff (pp)",
    "Normalized Match (%) (Baseline)", "Normalized Match (%) (Improved)", "Norm Match Diff (pp)",
    "Avg RapidFuzz Score (Baseline)", "Avg RapidFuzz Score (Improved)", "RapidFuzz Diff"
]

print(merged[display_cols].to_string(index=False))

print("\n" + "=" * 80)
print(" MISSING PREDICTION AUDIT (N = 349)")
print("=" * 80)

fields = ["company", "date", "address", "total"]
for f in fields:
    base_missing = (base_df[f"{f}_pred"].isna()) | (base_df[f"{f}_pred"].astype(str).str.strip().isin(["", "nan"]))
    new_missing = sum(1 for p in improved_preds if p.get(f) is None or str(p.get(f)).strip() == "")
    print(f"Field: {f.upper():<8} | Baseline Missing: {base_missing.sum():>3} ({base_missing.sum()/len(base_df)*100:>5.2f}%) | Improved Missing: {new_missing:>3} ({new_missing/len(base_df)*100:>5.2f}%) | Diff: {new_missing - base_missing.sum():>+3}")

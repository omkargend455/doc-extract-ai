import os
import json
import pandas as pd
from src.baseline_regex import RegexBaselineExtractor
from src.evaluate import match_field

task1_dir = 'C:/Users/Omkar/Downloads/sroie/0325updated.task1train(626p)'
task2_dir = 'C:/Users/Omkar/Downloads/sroie/0325updated.task2train(626p)'

df = pd.read_csv('results/dev_baseline_details.csv')
sample_ids = df['id'].head(10).tolist()

extractor = RegexBaselineExtractor()

print("Comparing on first 10 dev records using Task 1 OCR text:")
for doc_id in sample_ids:
    t1_path = os.path.join(task1_dir, f'{doc_id}.txt')
    lines = []
    with open(t1_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',', 8)
            if len(parts) >= 9:
                lines.append(parts[8])
            elif parts:
                lines.append(parts[-1])
    raw_ocr = '\n'.join(lines)
    preds = extractor.extract(raw_ocr, doc_type='receipt')

    with open(os.path.join(task2_dir, f'{doc_id}.txt'), 'r', encoding='utf-8') as f:
        gt = json.load(f)

    orig_row = df[df['id'] == doc_id].iloc[0]
    print(f"\nID: {doc_id}")
    print(f"  Company: New='{preds['company']}' | Orig='{orig_row['company_pred']}' | GT='{gt.get('company')}'")
    print(f"  Date:    New='{preds['date']}' | Orig='{orig_row['date_pred']}' | GT='{gt.get('date')}'")
    print(f"  Total:   New='{preds['total']}' | Orig='{orig_row['total_pred']}' | GT='{gt.get('total')}'")

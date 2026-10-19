import os
import re
import json
import pandas as pd
from src.evaluate import match_field

AMOUNT_PATTERN = re.compile(r'(?:RM|\$|USD)?\s*([0-9]{1,4}(?:,[0-9]{3})*\.[0-9]{2})\b', re.IGNORECASE)

TOTAL_KEYWORDS = [
    r'\bgrand\s*total\b',
    r'\btotal\s*rounded\b',
    r'\brounded\s*total\b',
    r'\bround\s*total\b',
    r'\bnett\s*total\b',
    r'\bnet\s*total\b',
    r'\btotal\s*amount\b',
    r'\btotal\s*due\b',
    r'\btotal\s*payable\b',
    r'\btotal\s*sales\b',
    r'\btotal\s*incl\.?\s*gst\b',
    r'\btotal\s*\(incl\.?\s*gst\b',
    r'\btotal\s*rm\b',
    r'\bjumlah\b',
    r'\btotal\b',
]

EXCLUDE_TOTAL_LINE = [
    'subtotal', 'sub-total', 'sub total', 'change', 'total item', 'total items', 'total qty',
    'included in total', 'tax included', 'cash', 'tender', 'tendered', 'balance'
]

def improved_predict_total(lines):
    candidates = []
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        if any(neg in line_lower for neg in EXCLUDE_TOTAL_LINE):
            continue
        for kw in TOTAL_KEYWORDS:
            if re.search(kw, line_lower):
                # Check current line
                amts = AMOUNT_PATTERN.findall(line)
                if amts:
                    candidates.append((idx, kw, float(amts[-1].replace(',', ''))))
                else:
                    # Check next 1-2 lines
                    for offset in [1, 2]:
                        if idx + offset < len(lines):
                            nxt = lines[idx + offset]
                            nxt_lower = nxt.lower()
                            if any(neg in nxt_lower for neg in ['cash', 'tender', 'change', 'subtotal', 'tax invoice']):
                                break
                            nxt_amts = AMOUNT_PATTERN.findall(nxt)
                            if nxt_amts:
                                candidates.append((idx, kw, float(nxt_amts[-1].replace(',', ''))))
                                break
                break

    if candidates:
        priority_kws = [
            r'\bgrand\s*total\b', r'\btotal\s*rounded\b', r'\brounded\s*total\b',
            r'\bnet\s*total\b', r'\bnett\s*total\b', r'\btotal\s*amount\b',
            r'\btotal\s*due\b', r'\btotal\s*incl'
        ]
        for p_kw in priority_kws:
            for idx, kw, val in reversed(candidates):
                if re.search(p_kw, kw):
                    return f"{val:.2f}"
        return f"{candidates[-1][2]:.2f}"

    return None

task1_dir = 'C:/Users/Omkar/Downloads/sroie/0325updated.task1train(626p)'
task2_dir = 'C:/Users/Omkar/Downloads/sroie/0325updated.task2train(626p)'
df = pd.read_csv('results/dev_baseline_details.csv')

exact_cnt = 0
norm_cnt = 0
for doc_id in df['id']:
    t1_path = os.path.join(task1_dir, f'{doc_id}.txt')
    lines = []
    with open(t1_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',', 8)
            if len(parts) >= 9:
                lines.append(parts[8])
            elif parts:
                lines.append(parts[-1])
    pred = improved_predict_total(lines)
    with open(os.path.join(task2_dir, f'{doc_id}.txt'), 'r', encoding='utf-8') as f:
        gt = json.load(f)
    exact, norm_m, score = match_field(pred, gt.get('total'), 'total')
    if exact:
        exact_cnt += 1
    if norm_m:
        norm_cnt += 1

orig_exact = df['total_exact'].sum()
orig_norm = df['total_norm'].sum()
print(f"Original Baseline Total (Colab OCR): Exact {orig_exact}/349 ({orig_exact/349*100:.2f}%), Norm {orig_norm}/349 ({orig_norm/349*100:.2f}%)")
print(f"Improved Total (Task 1 OCR):        Exact {exact_cnt}/349 ({exact_cnt/349*100:.2f}%), Norm {norm_cnt}/349 ({norm_cnt/349*100:.2f}%)")

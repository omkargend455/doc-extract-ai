import pandas as pd
import numpy as np

df = pd.read_csv('results/dev_baseline_details.csv')

print("=" * 60)
print("AUDIT: DATE FAILURES")
print("=" * 60)
missing_date = df[df['date_pred'].isna() | (df['date_pred'].astype(str).str.strip() == '') | (df['date_pred'].astype(str).str.strip() == 'nan')]
print(f"Missing date count: {len(missing_date)} / {len(df)} ({len(missing_date)/len(df)*100:.1f}%)")
print("Sample missing dates in GT:")
for _, r in missing_date.head(6).iterrows():
    print(f"  ID: {r['id']} -> GT: '{r['date_gt']}'")

date_exact_fail_norm_pass = df[(~df['date_exact']) & df['date_norm']]
print(f"\nDate exact fail but norm pass (formatting differences): {len(date_exact_fail_norm_pass)}")
for _, r in date_exact_fail_norm_pass.head(6).iterrows():
    print(f"  ID: {r['id']} -> Pred: '{r['date_pred']}' vs GT: '{r['date_gt']}'")

date_norm_fail = df[(~df['date_norm']) & (~df['date_pred'].isna()) & (df['date_pred'].astype(str).str.strip() != 'nan')]
print(f"\nDate wrong prediction (norm fail with pred present): {len(date_norm_fail)}")
for _, r in date_norm_fail.head(8).iterrows():
    print(f"  ID: {r['id']} -> Pred: '{r['date_pred']}' vs GT: '{r['date_gt']}' (Fuzz: {r['date_score']:.1f})")


print("\n" + "=" * 60)
print("AUDIT: TOTAL FAILURES")
print("=" * 60)
missing_total = df[df['total_pred'].isna() | (df['total_pred'].astype(str).str.strip() == '') | (df['total_pred'].astype(str).str.strip() == 'nan')]
print(f"Missing total count: {len(missing_total)} / {len(df)}")
for _, r in missing_total.head(5).iterrows():
    print(f"  ID: {r['id']} -> GT: '{r['total_gt']}'")

total_exact_fail_norm_pass = df[(~df['total_exact']) & df['total_norm']]
print(f"\nTotal exact fail but norm pass (e.g., currency symbol RM): {len(total_exact_fail_norm_pass)}")
for _, r in total_exact_fail_norm_pass.head(6).iterrows():
    print(f"  ID: {r['id']} -> Pred: '{r['total_pred']}' vs GT: '{r['total_gt']}'")

total_norm_fail = df[(~df['total_norm']) & (~df['total_pred'].isna()) & (df['total_pred'].astype(str).str.strip() != 'nan')]
print(f"\nTotal wrong prediction (norm fail with pred present): {len(total_norm_fail)}")
for _, r in total_norm_fail.head(10).iterrows():
    print(f"  ID: {r['id']} -> Pred: '{r['total_pred']}' vs GT: '{r['total_gt']}' (Fuzz: {r['total_score']:.1f})")


print("\n" + "=" * 60)
print("AUDIT: COMPANY FAILURES")
print("=" * 60)
comp_exact_fail_norm_pass = df[(~df['company_exact']) & df['company_norm']]
print(f"Company exact fail but norm pass (case, legal suffix, fuzz>=85): {len(comp_exact_fail_norm_pass)}")
for _, r in comp_exact_fail_norm_pass.head(6).iterrows():
    print(f"  ID: {r['id']} -> Pred: '{r['company_pred']}' vs GT: '{r['company_gt']}'")

comp_norm_fail = df[(~df['company_norm']) & (~df['company_pred'].isna()) & (df['company_pred'].astype(str).str.strip() != 'nan')]
print(f"\nCompany wrong prediction (norm fail): {len(comp_norm_fail)}")
for _, r in comp_norm_fail.head(8).iterrows():
    print(f"  ID: {r['id']} -> Pred: '{r['company_pred']}' vs GT: '{r['company_gt']}' (Fuzz: {r['company_score']:.1f})")


print("\n" + "=" * 60)
print("AUDIT: ADDRESS FAILURES")
print("=" * 60)
print(f"Address exact match: {df['address_exact'].sum()} / {len(df)}")
print(f"Address norm match: {df['address_norm'].sum()} / {len(df)}")
addr_norm_fail = df[~df['address_norm']]
print(f"Address norm fail: {len(addr_norm_fail)}")
for _, r in addr_norm_fail.head(6).iterrows():
    print(f"  ID: {r['id']} -> Pred: '{str(r['address_pred'])[:60]}...' vs GT: '{str(r['address_gt'])[:60]}...' (Fuzz: {r['address_score']:.1f})")

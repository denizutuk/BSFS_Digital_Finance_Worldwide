
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "findex_microdata_2025_labelled_update112425.dta")
OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# Figures given in the project brief 
BRIEF_N_MOBILE = 81_924
BRIEF_N_ACTIVITY = 102_954

# 1. Load 

df = pd.read_stata(DATA_FILE, convert_categoricals=False)
with pd.io.stata.StataReader(DATA_FILE) as reader:
    var_labels = reader.variable_labels()

N = len(df)
ECON = sorted(df["economycode"].unique())
E = len(ECON)
by_econ = df.groupby("economycode")
print(f"Loaded {N:,} respondents, {E} economies, {df.shape[1]} variables")


def economies_with_data(col):
    """Economies where `col` has at least one non-missing answer (i.e. question was asked)."""
    ok = by_econ[col].apply(lambda x: x.notna().any())
    return sorted(ok[ok].index), sorted(ok[~ok].index)


def fmt_list(codes, limit=25):
    return "-" if not codes else (", ".join(codes) if len(codes) <= limit else f"{len(codes)} economies (see Data_Gaps)")


# 2. Key variables: (group, name, coding text, DK/Refused codes if the variable keeps them)

YN = "1 = Yes, 2 = No, 3 = Don't know, 4 = Refused"
BIN = "1 = Yes, 0 = No / Don't know / Refused (derived binary)"
RECV = "1 = Into an account, 2 = In cash only, 3 = Some other way, 4 = Did not receive, 5 = DK/Refused"
KEY_VARS = [
    ("Account ownership", "account", BIN, None),
    ("Account ownership", "account_fin", BIN, None),
    ("Account ownership", "account_mob", BIN, None),
    ("Account ownership", "dig_account", BIN, None),
    ("Account ownership", "fin2", YN, [3, 4]),
    ("Digital payments", "anydigpayment", BIN, None),
    ("Digital payments", "merchantpay_dig", BIN, None),
    ("Digital payments", "receive_wages", RECV, [5]),
    ("Digital payments", "receive_transfers", RECV, [5]),
    ("Digital payments", "receive_pensions", RECV, [5]),
    ("Digital payments", "receive_agriculture", RECV, [5]),
    ("Digital payments", "pay_utilities",
     "1 = From an account, 2 = In cash only, 3 = Some other way, 4 = Did not, 5 = DK/Refused", [5]),
    ("Digital payments", "domestic_remittances",
     "1 = Using an account, 2 = Not using an account, 3 = Did not make or receive, "
     "4 = labelled DK/Refused (see Data_Gaps: suspicious)", [4]),
    ("Saving", "saved", BIN, None),
    ("Saving", "fin17a", YN, [3, 4]),
    ("Borrowing", "borrowed", BIN, None),
    ("Borrowing", "fin22a", YN, [3, 4]),
    ("Borrowing", "fin22b", YN, [3, 4]),
    ("Demographics", "female", "1 = Female, 2 = Male (NOT 0/1)", None),
    ("Demographics", "age", "Continuous, years", None),
    ("Demographics", "educ", "1 = Primary or less, 2 = Secondary, 3 = Tertiary or more", None),
    ("Demographics", "inc_q", "1 = poorest 20% ... 5 = richest 20% (within-economy household income quintile)", None),
    ("Demographics", "emp_in", "1 = In the workforce, 2 = Out of the workforce", None),
    ("Demographics", "urbanicity", "1 = Rural, 2 = Urban (variable label says 'lives in rural area' but code 1 = Rural)", None),
    ("Demographics", "internet_use", BIN, None),
    ("Region / design", "regionwb", "Text, World Bank region ('High income' is its own category)", None),
    ("Region / design", "economycode", "Text, economy code", None),
    ("Region / design", "pop_adult", "Continuous, population 15+ in 2023 (WDI), one value per economy", None),
    ("Region / design", "wgt", "Continuous survey weight, mean 1 per economy", None),
]
missing_cols = [v for _, v, _, _ in KEY_VARS if v not in df.columns]
assert not missing_cols, f"Columns not found in data: {missing_cols}"


# 3. Audit table

rows = []
for group, var, coding, dk_codes in KEY_VARS:
    s = df[var]
    n_ok = int(s.notna().sum())
    with_data, without_data = economies_with_data(var)
    if dk_codes:
        k = int(s.isin(dk_codes).sum())
        dk_text = f"{k:,} ({k / n_ok:.2%} of non-missing)"
    elif "derived binary" in coding:
        dk_text = "Not recoverable: DK/Refused already folded into 0"
    else:
        dk_text = "n/a"
    rows.append({
        "Group": group, "Variable": var, "Label": var_labels.get(var, ""), "Coding": coding,
        "N non-missing": n_ok, "% of all respondents": n_ok / N,
        "Economies with data": f"all {E}" if len(with_data) == E else f"{len(with_data)}/{E}",
        "DK/Refused count": dk_text,
        "Economies with NO data": fmt_list(without_data),
    })
audit = pd.DataFrame(rows)


# 4. Mobile money vs activity variables

mob = df["account_mob"].notna()
act = df["anydigpayment"].notna()
act_same = all(df[c].notna().equals(act) for c in ["saved", "borrowed", "dig_account"])
mob_econ = set(economies_with_data("account_mob")[0])
act_econ = set(economies_with_data("anydigpayment")[0])
share_mob = by_econ["account_mob"].apply(lambda x: x.notna().mean())
partial = share_mob[(share_mob > 0) & (share_mob < 1)]

check = pd.DataFrame([
    ["Brief: mobile money N", f"{BRIEF_N_MOBILE:,}"],
    ["account_mob non-missing (data)", f"{mob.sum():,}"],
    ["Matches brief?", "YES" if mob.sum() == BRIEF_N_MOBILE else "NO"],
    ["Brief: activity variables N", f"{BRIEF_N_ACTIVITY:,}"],
    ["anydigpayment non-missing (data)",
     f"{act.sum():,}" + (" (identical in saved, borrowed, dig_account)" if act_same else " (differs across activity variables!)")],
    ["Matches brief?", "YES" if act.sum() == BRIEF_N_ACTIVITY else "NO"],
    ["Respondents with BOTH account_mob and activity vars", f"{(mob & act).sum():,}"],
    ["Economies with BOTH", f"{len(mob_econ & act_econ)}"],
    ["Economies with account_mob only (no activity vars)",
     f"{', '.join(sorted(mob_econ - act_econ))} ({len(mob_econ - act_econ)})"],
    ["Economies with activity vars only (no account_mob)",
     f"{len(act_econ - mob_econ)}: {', '.join(sorted(act_econ - mob_econ))}"],
    ["Economies with neither", f"{E - len(mob_econ | act_econ)}"],
    ["Economies where account_mob asked", f"{len(mob_econ)} of {E}"],
    ["Economies where activity vars asked", f"{len(act_econ)} of {E}"],
    ["account_mob asked partially within an economy?",
     "No: all-or-nothing in every economy" if partial.empty else f"YES, in {', '.join(partial.index)}"],
], columns=["Check", "Result"])


# 5. Coverage by economy (non-missing count; 0 = question not asked there)

coverage = by_econ[[v for _, v, _, _ in KEY_VARS]].apply(lambda g: g.notna().sum())


# 6. Survey weights

def kish_deff(w):
    """Design effect from unequal weights: n * sum(w^2) / sum(w)^2."""
    return len(w) * (w ** 2).sum() / w.sum() ** 2

weights = pd.DataFrame({
    "n": by_econ.size(),
    "wgt_sum": by_econ["wgt"].sum(),
    "wgt_min": by_econ["wgt"].min(),
    "wgt_max": by_econ["wgt"].max(),
    "pop_adult": by_econ["pop_adult"].first(),
})
weights["wgt_sum/n"] = weights["wgt_sum"] / weights["n"]
weights["max/min"] = weights["wgt_max"] / weights["wgt_min"]
weights["Kish deff"] = by_econ["wgt"].apply(kish_deff)
weights = weights.reset_index()

ratio_top = weights.nlargest(5, "max/min")
ratio_txt = ", ".join(f"{r.economycode} {r['max/min']:.1f}" for _, r in ratio_top.iterrows())
n_1000 = int((weights["n"] == 1000).sum())
small = weights.nsmallest(3, "n")
large = weights.nlargest(2, "n")
sums_to_n = bool(((weights["wgt_sum/n"] - 1).abs() < 1e-6).all())
wgt_ok = df["wgt"].notna().all() and (df["wgt"] > 0).all()

weights_note = pd.DataFrame([
    ["Weight variable",
     f"wgt (label in data: '{var_labels.get('wgt')}'). One per respondent; "
     f"{'never missing and always > 0' if wgt_ok else 'HAS missing or non-positive values'} "
     f"(min {df['wgt'].min():.3f}, max {df['wgt'].max():.2f})."],
    ["What it represents (from the data)",
     ("Post-stratification weight WITHIN each economy: it sums to the sample size n of that economy "
      "(wgt_sum/n = 1 in all economies), so the mean weight is 1. It makes the sample representative of the adult (15+) "
      "population of that economy and carries NO information about economy size."
      if sums_to_n else "Weights do NOT sum to n in every economy; check weights_by_economy.csv.")],
    ["Source documentation",
     "The codebook/technical notes were not available when this was run, so the description above is inferred from the data. "
     "Paste the exact sentence from the technical notes."],
    ["Within an economy", "Use wgt. Weighted share = sum(wgt*x) / sum(wgt) over respondents with non-missing x."],
    ["Across economies (regions, global, pooled)",
     "Use wgt * pop_adult / n of the economy (each economy then sums to its adult population). "
     "Raw wgt pools economies as if each had about 1,000 adults, so large-sample economies count more and small-sample ones less."],
    ["Pitfall: restricted samples",
     f"account_mob exists in {len(mob_econ)} economies and the activity variables in {len(act_econ)}. Pooled or regional averages for them "
     f"cover only those economies; do not compare with averages over all {E}."],
    ["Pitfall: extreme weights",
     f"Largest max/min weight ratio: {ratio_txt} (median economy {weights['max/min'].median():.1f}). "
     "Estimates there are noisier (see Kish deff in Weights_by_Economy); check sensitivity."],
    ["Pitfall: sample sizes",
     f"{n_1000} economies have exactly 1,000 interviews; smallest: "
     + ", ".join(f"{r.economycode} {r.n}" for _, r in small.iterrows()) + "; largest: "
     + ", ".join(f"{r.economycode} {r.n:,}" for _, r in large.iterrows())
     + f". Total weight = {df['wgt'].sum():,.0f}; total adult population in file = {weights['pop_adult'].sum():,.0f}."],
], columns=["Topic", "Note"])

# 7. Data gaps and manual fixes

gaps = []
def gap(issue, detail):
    gaps.append([issue, detail])

years = sorted(df["year"].dropna().unique())
gap("Survey year", f"'year' = {', '.join(str(int(y)) for y in years)} in the data (fieldwork year); the product is named 'Findex 2025' after the release year.")
no_mob = sorted(set(ECON) - mob_econ)
gap("Mobile money coverage", f"account_mob asked in {len(mob_econ)}/{E} economies; not asked in {len(no_mob)}: {', '.join(no_mob)}.")
no_act = sorted(set(ECON) - act_econ)
gap("Activity variables coverage",
    f"dig_account, anydigpayment, saved, borrowed, merchantpay_dig, fin17a, fin22a missing in the same {len(no_act)} economies: {', '.join(no_act)}.")
recv_ok, recv_no = economies_with_data("receive_wages")
gap("Wage/transfer/pension receipt coverage",
    f"receive_wages/transfers/pensions/agriculture and pay_utilities asked in {len(recv_ok)} economies only (missing: {len(recv_no)}).")
gap("Binary variables hide DK/Refused",
    "account, account_fin, account_mob, dig_account, anydigpayment, saved, borrowed, merchantpay_dig, internet_use are derived "
    "binaries: 'No/DK/Ref' = 0. DK/Refused cannot be separated; use the raw fin* questions if needed.")
gap("Mixed coding conventions",
    "Raw questions: 1=Yes 2=No 3=DK 4=Refused. Derived: 1/0. female: 1=Female 2=Male. urbanicity: 1=Rural 2=Urban. "
    "emp_in: 1=in workforce 2=out. con* variables use 8/9 or 98/99 for DK/Refused. Recode before pooling.")
rw = df["receive_wages"].value_counts()
gap("receive_* and pay_utilities", f"5-category, not binary. Digital receipt = code 1 (into an account). Code 5 = DK/Refused "
    f"({int((df[['receive_wages','receive_transfers','receive_pensions','receive_agriculture','pay_utilities']] == 5).sum().min()):,} to "
    f"{int((df[['receive_wages','receive_transfers','receive_pensions','receive_agriculture','pay_utilities']] == 5).sum().max()):,} cases).")

for col in ["inc_q", "emp_in", "educ", "urbanicity"]:
    miss = df[df[col].isna()]
    if miss.empty:
        continue
    per = miss["economycode"].value_counts()
    full = [e for e in per.index if per[e] == (df["economycode"] == e).sum()]
    top = ", ".join(f"{e} ({k:,})" for e, k in per.head(4).items())
    txt = f"{len(miss):,} rows missing; most in {top}."
    if full:
        txt += f" Entirely missing in: {', '.join(full)}."
    gap(f"{col} missing", txt)

rem = df["domestic_remittances"].dropna()
share4 = (rem == 4).mean()
gap("domestic_remittances code 4",
    f"Value label says 4 = DK/ref, yet {int((rem == 4).sum()):,} of {len(rem):,} non-missing answers ({share4:.1%}) have code 4, "
    "versus well under 1% DK/Refused elsewhere. The label is probably wrong. Check the codebook before using this variable.")

income_cols = [c for c in df.columns if "income" in c.lower() or "incgroup" in c.lower()]
gap("Income group variable",
    f"Found: {', '.join(income_cols)}." if income_cols else
    "No economy income-group variable in the file (only regionwb, where 'High income' is a category). "
    "Merge the World Bank income classification via economycode if needed.")
gap("Duplicates", f"{int(df.duplicated().sum())} fully duplicated rows, {int(df['wpid_random'].duplicated().sum())} duplicated wpid_random.")
logic = {
    "account=1 but account_fin=0 and account_mob=0/missing": ((df.account == 1) & (df.account_fin == 0) & (df.account_mob.fillna(0) == 0)).sum(),
    "account=0 but account_fin=1": ((df.account == 0) & (df.account_fin == 1)).sum(),
    "account_mob=1 but account=0": ((df.account_mob == 1) & (df.account == 0)).sum(),
    "dig_account=1 but account=0": ((df.dig_account == 1) & (df.account == 0)).sum(),
}
gap("Logic checks", "; ".join(f"{k}: {int(v)} rows" for k, v in logic.items()) + ".")
gap("Unusual samples",
    f"{E - n_1000} economies have n != 1,000. Smallest: " + ", ".join(f"{r.economycode} {r.n}" for _, r in small.iterrows())
    + "; largest: " + ", ".join(f"{r.economycode} {r.n:,}" for _, r in large.iterrows()) + ".")
gaps = pd.DataFrame(gaps, columns=["Issue", "Detail"])



# 8. Write results

audit.to_csv(os.path.join(OUT_DIR, "audit_table.csv"), index=False)
check.to_csv(os.path.join(OUT_DIR, "mobile_money_check.csv"), index=False)
weights_note.to_csv(os.path.join(OUT_DIR, "weights_note.csv"), index=False)
weights.to_csv(os.path.join(OUT_DIR, "weights_by_economy.csv"), index=False)
coverage.to_csv(os.path.join(OUT_DIR, "coverage_by_economy.csv"))
gaps.to_csv(os.path.join(OUT_DIR, "data_gaps.csv"), index=False)

print(f"\nDone. CSV files written to {OUT_DIR}")
print(check.to_string(index=False))

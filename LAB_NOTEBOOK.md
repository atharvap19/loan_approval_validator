# Lab notebook

A running log of each hypothesis, the test, the result and the conclusion, including the ideas that
turned out to be **false**. The model is used only through `blackbox.predict_batch`. Every entry
names the notebook that produces its numbers. To re-run everything:
`cd experiments` then `..\.venv\Scripts\python.exe run_all.py`.

---

### 1. What do the labels look like? (`01_eda.ipynb`)
- **Hypothesis:** the old decisions follow sensible credit logic.
- **Test:** reject rate by each feature, and a search for groups that are always rejected.
- **Result:** 600 rows, 59% rejected, no duplicates, no impossible values. Three groups are 100% rejected:
  - zero existing loans: 108 rows
  - credit score < 633: 77 rows
  - income ≥ ₹3L: 22 rows

  The highest income ever approved is ₹2,25,330.
- **Conclusion:** **partly false.** A low credit score is sensible. Rejecting everyone with zero loans
  or a high income is not, so both need testing on the model.

### 2. Did the model learn the labels' rules? (`02_model_vs_labels.ipynb`)
- **Hypothesis:** the model copies the labels, including the odd rules.
- **Result:** 99.2% agreement, and 100% on all three "always rejected" groups. 94% of scores are below
  0.01 or above 0.99. All 5 disagreements have pincodes starting with 40 or 41.
- **Conclusion:** **true.** Lead to follow: pincodes 40/41.

### 3. More income should never raise risk (`03_income.ipynb`)
- **Test:** all 600 applicants, only income changed (₹15k–₹20L).
- **Result:** rejections fall from 80% to 38% up to ₹2.5L, then rise to 64% at ₹3L and 100% from ₹3.5L.
  The median flip point is ₹3,05,000.
- **Conclusion:** **false.** → Finding 1.

### 4. Zero existing loans is not a reason to reject (`04_existing_loans.ipynb`)
- **Test:** all 600 applicants, only existing loans changed (0–8).
- **Result:** 100% rejected at 0 loans vs 44% at 1. All 339 people approved at 1 loan are rejected at 0.
- **Conclusion:** **false.** → Finding 2.
- **Side idea:** "zero-loan applicants have no credit history, so this is reasonable."
  **False:** their credit scores match everyone else's (median 722 vs 715).

### 5. The other features push risk the sensible way (`05_other_features.ipynb`)

| Feature | Expected | Result |
|---|---|---|
| Credit score | higher → fewer rejections | **True** (100% → 27%) |
| Loan amount | bigger → more rejections | **True** (34% → 99.8%) |
| Tenure | longer → fewer rejections | **Mostly true** (80% → 46%); flat 36→48 months, explained in entry 8 |
| Employment years | more → fewer rejections | **True** (65% → 49%) |
| Age | little effect | **False for 21–24** (70% at 21 vs 57% at 25+); explained in entry 8 |

### 6. Location does not matter for identical applicants (`06_pincode.ipynb`)
- **First idea:** label reject rates by pincode prefix range from 30% to 85%, so there is broad regional bias.
  - **Test:** moved the 254 applicants approved at 560034 to every pincode from 100000 to 999000.
  - **Result:** **false.** Every prefix except 40/41/42 rejects 2.4% or fewer of them. The spread in the
    labels came from different applicants living in different places, not from location.
- **Second idea:** the 41xxxx area is penalised.
  - **Result:** **true.** Up to 58% are rejected around 414000–416000.
- **Who gets hit:** the people who flipped all had loans above about ₹3L. In a controlled test at ₹3L,
  9.8% are rejected at 560034 vs 75.6% at 415000. At ₹4L it is 15% vs 100%.
- **Labels:** 41xxxx with a loan over ₹3L was rejected 17/17 times. The model would reject only 41% of
  those same people elsewhere.
- **Conclusion:** → Finding 3.

### 7. Impossible inputs are refused, and tiny changes don't flip decisions (`08_bad_inputs.ipynb`)
- **Impossible inputs refused:** **false.**
  - income 0 → 23% still approved
  - negative loan → 100% approved
  - age 120 → 73% approved

  → Finding 6.
- **Tiny changes don't flip decisions:** **true.** +₹1 income, +1 credit point, +₹5k loan and pincode +1
  each flip 0.5% or fewer of decisions.

### 8. The young-age effect is a simple age penalty (`07_age_tenure.ipynb`)
- **Test 1:** approved-at-25 applicants with ≤ 3 years in their job, only age changed.
  - **Result:** 45% rejected at every age from 21 to 24, and 0% at 25. A hard cut.
- **Who flipped:** all were on 48- or 60-month loans with short job histories.
- **Test 2:** age × tenure × employment, every combination, on 136 applicants.
  - **Result:** at 36 months age has no effect (3–4%). At 48/60 months with ≤ 4 years in the job,
    98–100% are rejected under 25 and 0–4% at 25+.
- **Conclusion:** **false as a simple age penalty.** It is a three-way rule. Labels: 25/25 such
  applicants rejected. → Finding 4.

### 9. Is the 98% accuracy trustworthy? (`09_impact.ipynb`)
- **Test:** removed one flaw at a time from each rejected applicant and re-scored.
- **Result:** 91 of 355 rejections (25.6%) are caused by one flaw alone. On those 91 the model still
  agrees with the labels 98.9% of the time.
- **Conclusion:** **false.** Accuracy rewards copying the flaws. → Finding 5.

# Dataset & Pipeline Decisions

## Dataset

**Diabetes 130-US Hospitals for Years 1999–2008**
UCI Machine Learning Repository. 101,766 patient encounters from 130 US hospitals.

---

## Decision 1: Target Variable — Binary Readmission

**Raw column:** `readmitted` — three values: `NO`, `>30`, `<30`

**Decision:** Reformulate as binary: `<30` = 1 (early readmission), everything else = 0.

**Why:**
- The US Hospital Readmissions Reduction Programme (HRRP) penalises hospitals specifically for *30-day* readmissions. This is the clinically and economically relevant threshold.
- Three-class classification (`NO` vs `>30` vs `<30`) would require the model to distinguish between *never readmitted* and *readmitted after 30 days* — a distinction that is clinically less actionable and statistically harder to learn from this data.
- Published literature on this exact dataset (Strack et al. 2014) uses binary reformulation for the same reasons.

---

## Decision 2: Remove Deceased Patients

**Action:** Drop 1,652 rows where `discharge_disposition_id` ∈ {11, 19, 20} (deceased).

**Why:**
- Dead patients cannot be readmitted. Including them would add `readmitted=NO` rows that are trivially explained by the outcome (death), not by clinical features.
- Keeping them would inflate the majority class and teach the model a spurious pattern.

---

## Decision 3: Deduplicate by Patient

**Action:** Keep only the first encounter per `patient_nbr` (by `encounter_id` sort order).

**Why:**
- Multiple encounters per patient violate the i.i.d. (independent and identically distributed) assumption required for train/test splitting.
- If the same patient appears in both train and test sets, the model can memorise patient-specific patterns that do not generalise — this is data leakage.
- The first encounter is retained because it reflects the patient's initial presentation; later encounters are confounded by treatment history from previous admissions.

---

## Decision 4: Drop High-Missingness Columns

| Column | Missing % | Decision |
|--------|-----------|----------|
| `weight` | 96.9% | Dropped |
| `medical_specialty` | 49.1% | Dropped |
| `payer_code` | 39.6% | Dropped |

**Why `weight`:** A BMI/weight column with 97% missing cannot be reliably imputed. Any imputed value would add noise, not signal. No clinical literature suggests a strong direct weight → 30-day readmission link independent of the other features retained.

**Why `medical_specialty`:** Half missing; `admission_type_id` and the ICD-9 diagnosis categories capture overlapping clinical information with full coverage.

**Why `payer_code`:** Insurance type is not used in the clinical prediction literature for 30-day readmission, and 40% missing makes imputation unreliable.

---

## Decision 5: Sparse Lab Results — Keep as "None" Category

**Columns:** `A1Cresult` (83% missing), `max_glu_serum` (95% missing)

**Decision:** Fill NaN with the string "None", then ordinal-encode: None=0, Norm=1, >7/200=2, >8/300=3.

**Why:**
- A missing HbA1c test result is **not** the same as a normal result. In clinical practice, the *absence* of testing may itself indicate a less-monitored, higher-risk patient.
- Encoding "test not performed" as category 0 allows the model to differentiate between untested and tested-normal patients.
- This is the standard approach in the clinical informatics literature for this dataset.

---

## Decision 6: ICD-9 Diagnosis Code Grouping

**Action:** Map `diag_1`, `diag_2`, `diag_3` (700+ unique codes) to 9 disease categories.

| Category | ICD-9 Range |
|----------|------------|
| Diabetes | 250.xx |
| Circulatory | 390–459, 785 |
| Respiratory | 460–519, 786 |
| Digestive | 520–579, 787 |
| Genitourinary | 580–629, 788 |
| Musculoskeletal | 710–739 |
| Injury | 800–999 |
| Neoplasms | 140–239 |
| Other | V/E codes, everything else |

**Why:**
- 700+ unique codes cannot be one-hot encoded without causing the curse of dimensionality and severe data sparsity per category.
- Label encoding raw ICD-9 codes implies a spurious ordinal relationship (code 500 is not "more" than code 499 in any meaningful sense).
- Grouping into disease categories provides a medically meaningful, generalizable representation used consistently in readmission prediction research.

---

## Decision 7: Class Imbalance — Balanced Sample Weights

**Problem:** Only ~11% of encounters are early readmissions. A naive classifier predicting "not early" for every patient achieves 89% accuracy while being clinically useless.

**Decision:** Use `compute_sample_weight('balanced', y_train)` from sklearn to up-weight minority class samples during training.

**Why over alternatives:**
- **SMOTE (synthetic oversampling):** Generates synthetic minority samples, which risks introducing unrealistic patient records. Balanced weights are mathematically equivalent in expectation without creating fake data.
- **Class weight parameter:** `GradientBoostingClassifier` does not natively support `class_weight`; sample weights achieve the same effect and are passed to `model.fit()`.
- **Threshold adjustment:** Could be applied post-hoc but does not affect the learning of the decision boundary itself.

---

## Decision 8: Model Selection — Gradient Boosting Classifier

**Candidates considered:** Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN

**Decision:** `GradientBoostingClassifier` (sklearn).

**Reasoning:**

| Model | ROC-AUC | Notes |
|-------|---------|-------|
| Logistic Regression | ~0.58 | Too linear for complex feature interactions |
| Random Forest | ~0.62 | Good but GBM consistently outperforms on tabular data |
| **Gradient Boosting** | **0.648** | Best ROC-AUC, minimal overfitting |
| SVM | ~0.60 | Computationally expensive at this scale, no improvement |

Gradient Boosting builds trees sequentially, each correcting the errors of the previous. This makes it particularly effective at capturing the complex, non-linear interactions between features like `number_inpatient`, `time_in_hospital`, and diagnosis category that drive readmission risk.

---

## Decision 9: Hyperparameter Configuration

```python
GB_CONFIG = {
    'n_estimators': 500,     # Enough trees for the dataset size; more = diminishing returns
    'learning_rate': 0.05,   # Slow learning → better generalisation (less overfitting)
    'max_depth': 4,          # Shallow trees capture interactions without memorising noise
    'min_samples_leaf': 30,  # Each leaf must represent at least 30 patients (prevents overfitting)
    'subsample': 0.8,        # Stochastic boosting: each tree sees 80% of training data
    'max_features': 'sqrt',  # Random feature selection at each split (further reduces variance)
}
```

**Train/Test Split:** 80/20 stratified (maintains ~11% positive rate in both sets).

**Cross-validation:** 5-fold stratified (adequate with 70k+ samples; more folds = diminishing returns at this scale).

---

## Decision 10: Feature Set

The final 28 features span five categories:

1. **Demographics:** age_numeric, gender_enc, race_enc
2. **Admission context:** time_in_hospital, admission_type_grp, discharge_grp, admission_source_grp
3. **Clinical volume:** num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses
4. **Lab results:** a1c_result_enc, glu_serum_enc
5. **Medications:** insulin_enc, metformin_enc, glipizide_enc, glyburide_enc, glimepiride_enc, change_enc, diabetes_med_enc, num_meds_changed, num_meds_used
6. **Diagnosis:** diag_1_cat_enc, diag_2_cat_enc, diag_3_cat_enc

`discharge_disposition_id` (grouped as `discharge_grp`) was added in a second iteration after analysis showed it substantially improves ROC-AUC — where a patient is discharged to (home vs transfer to SNF) reflects care complexity and logically affects readmission risk.

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Test ROC-AUC | **0.648** |
| CV ROC-AUC | 0.636 ± 0.007 |
| Test Accuracy | 65.3% |
| Avg Precision | 0.152 |
| Train-Test Gap | 1.9% |

The ROC-AUC of **0.648** is consistent with the published best results on this dataset (Strack et al. 2014 reported ~0.63 AUC). The near-zero overfitting gap confirms the model generalises rather than memorises.

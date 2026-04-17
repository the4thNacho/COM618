# K-Means Clustering Analysis

## Setup

K-Means was applied to 70,439 cleaned patient encounters using 14 numeric clinical features:

```
age_numeric, time_in_hospital, num_lab_procedures, num_medications,
number_outpatient, number_emergency, number_inpatient, number_diagnoses,
total_prior_visits, num_meds_changed, num_meds_used,
a1c_result_enc, glu_serum_enc, insulin_enc
```

All features were standardised (zero mean, unit variance) before clustering.
The readmission label was **not** used during clustering — rates per cluster are computed after the fact to assess clinical alignment.

---

## Choosing k

| k | Silhouette | Inertia | Readmit rate spread |
|---|-----------|---------|---------------------|
| 2 | 0.227 | 861,888 | 1.7 pp |
| **3** | **0.226** | 779,576 | **6.3 pp** |
| 4 | 0.112 | 723,677 | 8.1 pp |
| **5** | **0.222** | 680,975 | **6.7 pp** |
| 6 | 0.134 | 632,063 | 8.2 pp |
| 7 | 0.141 | 598,117 | 13.4 pp |
| 8 | 0.124 | 571,188 | 15.1 pp |

**k=3 is recommended.** It has the highest silhouette score alongside k=2, but produces nearly 4× the readmission rate spread (6.3 pp vs 1.7 pp), giving three clinically interpretable patient groups. k=4 shows a pronounced silhouette dip (0.112) — the algorithm is forcing a split that the data does not naturally support. k=5 is a reasonable second choice.

---

## Cluster Profiles at k=3

Clusters are ranked by early readmission rate (highest to lowest).

### Cluster 1 — "Revolving Door" (6.3% of patients, **14.5% readmission**)

**1.6× the dataset baseline rate of 8.9%.**

| Feature | Cluster mean | Dataset mean | Direction |
|---------|-------------|-------------|-----------|
| total_prior_visits | **4.73** | 0.56 | ↑↑ 8× |
| number_inpatient | **1.20** | 0.18 | ↑↑ |
| number_outpatient | **2.75** | 0.28 | ↑↑ |
| number_emergency | **0.78** | 0.10 | ↑↑ |
| has_prior_inpatient | **49.5%** | ~14% | ↑↑ |
| multimorbid (≥7 diag) | **78.2%** | ~65% | ↑ |
| time_in_hospital | 4.28 | 4.28 | ~ |
| num_meds_changed | 0.23 | 0.26 | ~ |

- **Age distribution:** spread evenly across all age groups (no strong skew)
- **Primary diagnoses:** Circulatory 26%, Other 22%, Respiratory 14%, Digestive 9%
- **Top discharge destinations:** Home (id=1: 59%), AMA/Other (id=3: 15%), Home+services (id=6: 17%)
- **Insulin:** roughly even split across No/Steady/Up/Down — not the defining feature

**Interpretation:** These are chronically ill patients already cycling through the healthcare system. Admission complexity (time in hospital, medications) is *average* — the elevated readmission risk comes almost entirely from prior utilisation history. This group most directly validates the model's top feature: `number_inpatient`. Clinical intervention for this group should focus on discharge planning and post-discharge follow-up, not on the current admission itself.

---

### Cluster 2 — "Intensive Treatment" (23.4% of patients, **9.7% readmission**)

Slightly above baseline but markedly different in clinical profile from Cluster 1.

| Feature | Cluster mean | Dataset mean | Direction |
|---------|-------------|-------------|-----------|
| num_meds_changed | **1.00** | 0.26 | ↑↑ 4× |
| num_meds_used | **1.74** | 1.19 | ↑↑ |
| num_medications | **20.1** | 15.7 | ↑ |
| time_in_hospital | **5.44** | 4.28 | ↑ |
| poor_glycaemic_control | **21.6%** | ~5% | ↑↑ |
| number_inpatient | 0.12 | 0.18 | ↓ |
| total_prior_visits | 0.36 | 0.56 | ↓ |

- **Age distribution:** younger skew — 40–55 age group is 30% (vs 26% in other clusters)
- **Primary diagnoses:** Circulatory 30%, Diabetes 15%, Respiratory 14%
- **Insulin:** 43% Down, 39% Up — the vast majority have active dose changes; only 5% are on no insulin
- **African American representation:** 19.5% (vs 18% overall) — marginal elevation

**Interpretation:** Patients undergoing active glycaemic management — insulin dose being titrated, longer stays, more medications overall, and a quarter have poor glycaemic control (elevated A1C despite insulin). Despite the clinical complexity, readmission is *lower* than the Revolving Door cluster because the current admission is addressing an acute management issue rather than representing a pattern of chronic decompensation. Likely includes recently-diagnosed or newly decompensated Type 2 diabetic patients.

---

### Cluster 0 — "Standard Care" (70.3% of patients, **8.2% readmission**)

Below-baseline readmission. The majority group.

| Feature | Cluster mean | Dataset mean | Direction |
|---------|-------------|-------------|-----------|
| num_meds_changed | **0.02** | 0.26 | ↓↓ |
| total_prior_visits | **0.26** | 0.56 | ↓↓ |
| number_inpatient | **0.10** | 0.18 | ↓↓ |
| num_medications | **14.1** | 15.7 | ↓ |
| has_prior_inpatient | **8.9%** | ~14% | ↓ |

- **Insulin:** 64% on no insulin, 36% on steady dose — almost no active titration (0.02 medications changed)
- **Primary diagnoses:** Circulatory 31%, Other 17%, Respiratory 14% — nearly identical distribution to Cluster 1 (diagnosis is not what separates the clusters)
- **Multimorbid:** 62% — slightly lower than other clusters

**Interpretation:** Routine diabetic admissions. Stable medication regimen, minimal prior utilisation, short/average stays. The 8.2% readmission rate is effectively the irreducible baseline for this population — patients who are readmitted from this group likely do so for reasons not captured in the available features.

---

## High-k Behaviour (k=7–8): Extreme High-Risk Cohort

At k=7, a small cluster with **20.1% readmission** emerges. At k=8, this rises to **21.4%**. This cohort combines:
- High prior utilisation (Cluster 1 characteristics)
- Active insulin titration and poor glycaemic control (Cluster 2 characteristics)

At k=3, this double-signal group is absorbed into Cluster 1. Its isolation at higher k suggests it is a genuine extreme-risk subgroup (~1–3% of patients) with readmission rates more than twice the dataset average. This is worth investigating further if clinical resources allow targeted intervention.

---

## Key Findings

1. **Prior utilisation dominates readmission risk.** The clustering algorithm, given only clinical features with no readmission label, independently recovers a high-risk group defined almost entirely by prior inpatient/outpatient visits. This corroborates the model's top feature importances.

2. **Active medication management is a distinct axis.** Heavy insulin titration and medication complexity define a separate patient group that is *not* primarily high-risk — suggesting that active management during admission is protective relative to chronic under-management.

3. **Diagnosis category does not separate clusters.** Circulatory disease appears at 26–31% across all three clusters. The clustering algorithm largely ignores diagnostic category as a discriminating variable, even though ICD-9 grouping is a useful model feature in the supervised setting.

4. **k=4 is unstable.** The silhouette score drops sharply from 0.226 to 0.112 at k=4, indicating the algorithm is artificially splitting a natural group. k=3 or k=5 should be used in preference.

5. **A tiny extreme-risk cohort exists.** Patients with both high prior utilisation and active medication management issues have 20–21% readmission rates at higher k. At k=3 this group is contained within the Revolving Door cluster but not cleanly isolated.

---

## Limitations

- Silhouette scores (0.11–0.23) are low-to-moderate throughout, reflecting genuine overlap in clinical profiles. Patient phenotypes in diabetes are a continuum, not discrete categories.
- K-Means assumes spherical, equally-sized clusters. The actual groups are unequal (6%, 23%, 70%) which violates this assumption and may cause the high-k instability.
- These clusters are data-driven, not clinically validated. The "Revolving Door" and "Intensive Treatment" labels are interpretations, not diagnoses.

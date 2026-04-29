# K-Means Clustering Analysis — Diabetes 130-US Hospitals

**Dataset:** 70,439 patients (post-cleaning, one encounter per patient)  
**Features:** 11 numeric clinical features, log-transformed count variables, StandardScaler normalisation  
**Features excluded:** `number_outpatient`, `number_emergency`, `glu_serum_enc` — near-zero variance (92–98% zeros), add noise without signal  

---

## Silhouette Scores by k

| k   | Silhouette | Notes                                                                   |
| --- | ---------- | ----------------------------------------------------------------------- |
| 2   | 0.223      | Broadest split — medication intensity                                   |
| 3   | **0.228**  | **Best score — three clinically distinct archetypes**                   |
| 4   | 0.145      | Sharp drop — stable group subdivides but doesn't produce new archetypes |
| 5   | 0.150      | Slight recovery — A1C-tested subgroup isolates, but marginal gain       |

k=3 is the optimal choice: highest cohesion and the most interpretable clinical structure. The drop at k=4 indicates the algorithm is splitting a naturally homogeneous group rather than finding a new one.

---

## k=2 — Medication Intensity Split

The most basic division in the data is between patients with actively adjusted medication regimens and those on stable or minimal regimens.

| Rank | n | % | Readmit | Key features |
|------|---|---|---------|-------------|
| #1 High medication activity | 16,595 | 24% | **10.2%** | Meds changed +1.67σ, Insulin +1.45σ, avg 19 meds |
| #2 Stable/minimal | 53,844 | 76% | 8.6% | Meds changed −0.51σ, Insulin −0.45σ, avg 15 meds |

**Insight:** Even at the coarsest level, insulin use and medication adjustment are the primary axes of variation. The gap in readmission rate (10.2% vs 8.6%) is modest but consistent — actively managed patients are more complex, not necessarily worse cared for.

---

## k=3 — Three Clinical Archetypes (Recommended)

At k=3 the data splits into three genuinely distinct patient types that have clear clinical meaning.

| Rank                                  | n      | %   | Readmit   | Age | Stay | Meds | Inpatient hx   | Defining features                           |
| ------------------------------------- | ------ | --- | --------- | --- | ---- | ---- | -------------- | ------------------------------------------- |
| #1 Frequent flyers (often readmitted) | 8,753  | 12% | **14.7%** | 68  | 4.7d | 16   | **1.33 prior** | Prior inpatient +1.91σ, Prior visits +1.57σ |
| #2 Active insulin mgmt                | 15,244 | 22% | 9.4%      | 63  | 5.2d | 19   | 0.05           | Meds changed +1.67σ, Insulin +1.48σ         |
| #3 Stable routine                     | 46,442 | 66% | 7.7%      | 66  | 3.9d | 14   | 0.00           | All features at or below average            |

### Archetype 1 — Frequent Flyers (12%, 14.7% readmit)
Defined almost entirely by **prior hospitalisation history** — 1.33 average prior inpatient visits versus 0.17 overall. These patients have already demonstrated a pattern of repeat admission. Age 68, slightly older than average. Despite this being the highest-risk group, their medication profile is not extreme (16 meds, low insulin) — the risk signal here is behavioural/structural, not pharmacological. Prior inpatient visits carry a z-score of +1.91, the strongest single-feature driver across all k levels.

### Archetype 2 — Active Insulin Management (22%, 9.4% readmit)
Defined by high insulin ordinal encoding (+1.48σ) and high medication change activity (+1.67σ). These are patients whose diabetes management is actively being titrated — more medications, more adjustments. Age 63, younger than the frequent flyer group. The elevated readmission rate likely reflects disease complexity and instability rather than poor care.

### Archetype 3 — Stable Routine (66%, 7.7% readmit)
The majority cohort. Below-average on nearly every feature: fewer meds (14), shorter stays (3.9d), virtually no prior inpatient history. Low readmission rate consistent with lower clinical complexity. This group is heterogeneous — at higher k it subdivides further.

---

## k=4 — Brief Admissions Split Off

The silhouette score drops sharply to 0.145, but one new clinically meaningful group does appear: **brief/low-complexity admissions**.

| Rank | n | % | Readmit | Age | Stay | Meds | Labs | Diagnoses |
|------|---|---|---------|-----|------|------|------|-----------|
| #1 Frequent flyers | 7,874 | 11% | **15.2%** | 68 | 4.8d | 16 | 47 | 7.8 |
| #2 Active insulin mgmt | 14,366 | 20% | 9.5% | 63 | 5.2d | 20 | 48 | 7.6 |
| #3 Older/complex stable | 27,495 | 39% | 9.1% | **70** | 5.1d | 18 | 48 | **8.0** |
| #4 Brief/low-complexity | 20,704 | 29% | **6.0%** | 60 | **2.4d** | 10 | 31 | 5.7 |

**Insight:** The k=3 stable routine group splits into two sub-groups:
- **Older/complex stable** (age 70): longer stays, more diagnoses, but no prior hospital history — multimorbid but first-time admitters
- **Brief/low-complexity** (age 60): very short admissions (2.4 days), few medications, fewest diagnoses — likely straightforward glycaemic management or minor complications

The brief admissions group (6.0% readmit) is the lowest-risk group found at any k level. The frequent flyer and insulin management clusters remain essentially identical to k=3, confirming they represent real, stable structure.

---

## k=5 — A1C-Tested Subgroup Isolates

| Rank | n | % | Readmit | A1C enc | Labs | Key driver |
|------|---|---|---------|---------|------|-----------|
| #1 Frequent flyers | 7,448 | 11% | **15.5%** | 0.19 | 47 | Prior inpatient +2.21σ |
| #2 Active insulin mgmt | 11,884 | 17% | 9.7% | 0.19 | 45 | Meds changed +1.72σ, Insulin +1.60σ |
| #3 Older/complex stable | 24,502 | 35% | 9.2% | 0.10 | 47 | Age +0.34σ, Diagnoses +0.40σ |
| #4 **A1C-tested** | 7,371 | 10% | 7.7% | **2.78** | **55** | A1C +2.57σ, Labs +0.58σ |
| #5 Brief/low-complexity | 19,234 | 27% | 6.1% | 0.08 | 30 | Low complexity across all features |

**Insight:** The most interesting new finding at k=5 is the **A1C-tested subgroup** (10% of patients). These patients have A1C encoding of 2.78 (scale 0–3) — meaning they actually received A1C testing, a marker of more thorough diabetic workup. They also have the highest lab procedure count (55 vs 41 overall). Despite being more complex on paper, their readmission rate is **only 7.7%** — lower than the overall rate. This is consistent with the hypothesis that A1C monitoring is protective: it identifies patients who receive more complete care. Age 58, notably the youngest cluster at k=5.

---

## Cross-k Stability Analysis

The most important finding from comparing across k levels is which clusters are **stable** (genuine structure) versus which only appear at higher k (splitting of homogeneous groups).

| Archetype | First appears | Stable across k? | Readmit range |
|-----------|--------------|-----------------|---------------|
| Frequent flyers | k=3 | Yes — n~11%, readmit~15% at k=3,4,5 | 14.7–15.5% |
| Active insulin management | k=2 | Yes — persists at all k | 9.4–10.2% |
| Brief/low-complexity | k=4 | Yes — persists at k=4,5 | 6.0–6.1% |
| A1C-tested subgroup | k=5 | Only at k=5 | 7.7% |
| Stable routine (generic) | k=3 | Dissolves — splits into sub-types at k=4 | — |

---

## Key Findings

1. **Prior hospitalisation is the strongest unsupervised signal.** The frequent flyer cluster (z=+1.91 to +2.21 for prior inpatient) has consistently ~14.7–15.5% readmission and appears at k=3,4,5. This aligns with the supervised model's feature importance — prior admission history is the single most predictive variable.

2. **Medication adjustment pattern separates as its own archetype.** The insulin management cluster (meds changed +1.67–1.72σ, insulin +1.45–1.60σ) consistently captures ~20–24% of patients. Active regimen changes may reflect disease instability rather than being a risk factor per se.

3. **k=3 is the natural structure of this data.** The sharp silhouette drop from k=3 (0.228) to k=4 (0.145) indicates the algorithm is forced to split a naturally homogeneous group. The three archetypes at k=3 — frequent flyers, active management, stable — each have clear clinical meaning and map directly to actionable interventions.

4. **A1C testing may be protective.** The A1C-tested subgroup (k=5) has the most intensive workup (A1C +2.57σ, labs +0.58σ) but below-average readmission (7.7%). Clinical data often shows this pattern: more thorough investigation correlates with better outcomes, not worse.

5. **Brief admissions are the genuinely low-risk group.** At k=4+ a group of ~27–29% of patients is identified with very short stays (2.4d), few medications and diagnoses, and 6.0–6.1% readmission — roughly half the rate of the highest-risk group. This group warrants minimal follow-up resources compared to frequent flyers.

6. **Age gradient is consistent.** Across all k levels, frequent flyers are oldest (~68), active insulin management is mid-range (~63), and brief/low-complexity are youngest (~58–61). Risk increases with age in the unsupervised structure, matching the supervised analysis.

---

## Limitations

- Silhouette scores (0.145–0.228) are low throughout, reflecting that clinical data doesn't naturally partition into sharply separated groups — patients exist on continua rather than in discrete types.
- Three sparse features excluded from clustering (`number_outpatient`, `number_emergency`, `glu_serum_enc`) may contain signal that cannot be recovered without better data collection.
- Clustering is run on the cleaned single-encounter-per-patient dataset. Patients with multiple encounters may have had their most complex visit excluded.

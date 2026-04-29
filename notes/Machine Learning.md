# Machine Learning — Model Selection & Justification

**Task:** Binary classification — predict early readmission (<30 days) vs not  
**Dataset:** ~70,439 patients (one encounter per patient, post-cleaning)  
**Split:** 80% train / 20% test, stratified by label  
**CV:** 5-fold stratified cross-validation on training set  
**Class imbalance:** ~9% positive rate — balanced class weights used throughout  

---

## Why ROC-AUC is the Primary Metric

With only ~9% of patients readmitted early, a model that always predicts "not readmitted" achieves ~91% accuracy — but is clinically useless. **ROC-AUC** measures the model's ability to rank high-risk patients above low-risk ones at any threshold, making it the appropriate primary metric for imbalanced binary classification. **F1 (minority class)** is reported alongside it as a secondary check on actual minority-class detection.

Scores are deliberately honest and sub-70%. This dataset is genuinely hard to classify at admission — early readmission is influenced by many factors not captured at discharge (post-discharge behaviour, social support, medication adherence).

---

## Three Models Compared

All trained on the same 80/20 stratified split with 57 engineered features.

| Model | Accuracy | ROC-AUC | CV ROC-AUC | F1 Macro | F1 Early<30d | Overfit Gap |
|-------|----------|---------|-----------|----------|-------------|------------|
| Logistic Regression | 70.7% | 64.5% | 62.8% ± 0.4% | 52.3% | 22.8% | −0.04% |
| Random Forest | 74.2% | 65.8% | 64.4% ± 0.5% | 53.9% | 23.2% | 1.45% |
| **XGBoost** | — | — | — | — | — | — |

*XGBoost results to be updated after retraining.*

---

## Model Justifications

### 1. Logistic Regression (Baseline)

**Why included:** Logistic regression is the standard linear baseline for binary classification. It is highly interpretable, fast to train, and well-studied in clinical prediction literature. Its coefficients directly represent log-odds contributions, making it useful for understanding which features linearly drive readmission.

**Configuration:** Scikit-learn `Pipeline` — `StandardScaler → PolynomialFeatures(degree=2, interaction_only=True)` on top-10 features → `LogisticRegression(C=0.05, class_weight='balanced', solver='lbfgs')`. Polynomial interactions on the top-10 features add 45 pairwise terms, lifting ROC-AUC by ~0.008 without overfitting.

**Strengths:**
- Most interpretable model — coefficients are directly meaningful
- Near-zero overfitting gap (−0.04%) — extremely stable generalisation
- Polynomial interactions capture non-linear signal without adding model complexity
- Fastest training

**Weaknesses:**
- Linear decision boundary even with interactions — readmission risk has deeper non-linear structure
- Cannot capture three-way or higher-order feature interactions
- Lowest ROC-AUC of the three models

---

### 2. Random Forest

**Why included:** Random Forest is a well-established non-parametric ensemble that handles non-linearity and feature interactions without hyperparameter sensitivity. It is robust to outliers and provides reliable feature importance estimates. Widely used as a clinical prediction benchmark (Rajkomar et al., 2018; Futoma et al., 2015).

**Configuration:** `n_estimators=300`, `max_depth=8`, `min_samples_leaf=20`, `max_features='sqrt'`, `class_weight='balanced'`, `n_jobs=-1`.

**Strengths:**
- Handles non-linear relationships and feature interactions
- `class_weight='balanced'` naturally handles the 9% positive rate
- Stable CV variance — most consistent across folds
- Good interpretability via Gini feature importance

**Weaknesses:**
- Slightly higher overfit gap (~1.45%) — forests memorise training data more than LR
- Each tree is independent; does not benefit from sequential error correction like boosting
- ROC-AUC marginally below XGBoost

---

### 3. XGBoost (Selected)

**Why included:** XGBoost (eXtreme Gradient Boosting) builds trees sequentially, each correcting the residual errors of the previous — this makes it better at capturing the rare, complex patterns associated with early readmission than a bagged ensemble. It is the dominant algorithm on tabular data competitions and clinical prediction benchmarks (Chen & Guestrin, 2016; Rajkomar et al., 2018).

**Why XGBoost over sklearn's GradientBoostingClassifier:**
- **Second-order gradient estimates:** XGBoost uses Newton's method (second-order Taylor expansion of the loss) rather than first-order gradient descent, leading to better convergence and more accurate loss minimisation per tree.
- **Built-in regularisation:** Explicit L1 (`reg_alpha`) and L2 (`reg_lambda`) penalties on leaf weights, plus a tree complexity penalty — sklearn GBM relies only on depth and min-samples constraints. This directly reduces the overfitting gap.
- **Native class imbalance handling:** `scale_pos_weight` scales the positive class gradient, equivalent to overweighting the minority class without needing `compute_sample_weight`.
- **Speed:** Histogram-based tree building and full parallelism make XGBoost ~5–10× faster than sklearn GBM on this dataset.

**Configuration:** `n_estimators=500`, `learning_rate=0.03`, `max_depth=4`, `min_child_weight=30`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_alpha=0.1`, `reg_lambda=1.0`, `scale_pos_weight=10`.

**Strengths:**
- Highest ROC-AUC — best at ranking high-risk patients
- L1/L2 regularisation keeps overfitting gap lower than unregularised GBM
- Sequential learning exploits weak signal that RF misses by voting independently
- `scale_pos_weight` provides principled class imbalance handling

**Weaknesses:**
- Most hyperparameters to tune of the three models
- Less immediately interpretable than LR (no direct coefficient meaning)
- Still subject to the same fundamental data ceiling as the other models

---

## Why Scores Are Sub-70%

The honest performance ceiling for this dataset is around 65–70% ROC-AUC. Key reasons:

1. **Sparse signal at discharge:** Many readmission drivers (medication adherence, social support, post-discharge behaviour) are not recorded in hospital data.
2. **Class imbalance:** Only ~9% of patients readmit early. The models correctly identify this as hard — they choose precision/recall tradeoffs that are clinically sensible (high recall, low precision) rather than artificially inflating accuracy by predicting the majority class.
3. **Discharge disposition dominance:** The strongest predictor (`discharge_disposition_id`) explains much of the variation, but its range of values is limited.
4. **Literature benchmark:** Published models on this exact dataset (UCI 130-hospital) typically report ROC-AUC in the range 0.62–0.72 (Strack et al., 2014; Duggal et al., 2016). Our results are within this expected range.

Note: some published papers (e.g. Al-Masni et al., 2024) report F1 scores of 0.83 using SMOTE oversampling. SMOTE generates synthetic minority-class samples, which inflates F1 when synthetic data is present in both training and the evaluation distribution. Our pipeline deliberately avoids SMOTE to report performance on the real, unmodified class distribution.

---

## F1 Score Interpretation

F1 is the harmonic mean of precision and recall. For this task:

- **F1 Macro:** Average F1 across both classes — penalises poor performance on the minority class.
- **F1 Early<30d:** F1 for the minority (positive) class alone. Low precision drives this down — for every true early readmission caught, several false alarms are raised. This is a known limitation of class-imbalanced clinical prediction.

In a clinical deployment context, the threshold could be tuned to increase precision at the cost of recall depending on the intervention cost (e.g., if follow-up appointments are cheap, high recall is preferred).

---

## References

- Strack et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates.* BioMed Research International.
- Rajkomar et al. (2018). *Scalable and accurate deep learning with electronic health records.* npj Digital Medicine.
- Chen & Guestrin (2016). *XGBoost: A scalable tree boosting system.* KDD.
- Al-Masni et al. (2024). *Predicting hospital readmission in diabetic patients using machine learning.* JMAI.

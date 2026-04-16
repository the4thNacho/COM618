# Data Modelling Proposal — Medical Diagnosis Prediction

## 1. Problem Identification

### Dataset and Context
The dataset contains 100 anonymised patient records collected from a medical facility.
Each record includes six clinical measurements — Age, Blood Pressure, Cholesterol, BMI,
Smoker status, and Gender — together with a ground-truth Diagnosis label.

### Identified Problem
**48 of the 100 patients (48%) have no confirmed diagnosis** (labelled `UNKNOWN`).
Given that the dataset contains the same clinical measurements for undiagnosed patients,
there is a clear opportunity to train a supervised classifier on the 52 labelled cases
and apply it to predict diagnoses for undiagnosed patients.

The two known conditions are:
- **DIABETES** — 27 patients
- **HEART DISEASE** — 25 patients

These are nearly balanced, making binary classification straightforward.

### Business / Clinical Value
Early identification of at-risk patients can enable earlier intervention, reducing the
cost of late-stage treatment. A decision-support tool that flags likely diagnoses from
routine measurement data would be valuable in a triage or screening context.

---

## 2. Exploratory Findings that Motivated the Model

| Finding | Implication |
|---|---|
| Heart Disease patients have higher median Blood Pressure | BP should be a strong feature |
| Smokers have disproportionately more Heart Disease cases | Smoking status is informative |
| Age–BMI scatter shows overlapping clusters | Linear models may underperform; tree-based models preferred |
| Weak correlation between numerical features | No multicollinearity; all features can contribute independently |

---

## 3. Model Construction

### Feature Engineering
Six features were used:

| Feature | Encoding | Notes |
|---|---|---|
| Age | Float (as-is) | Continuous numerical |
| Blood_Pressure | Float (as-is) | Continuous numerical |
| Cholesterol | Float (as-is) | Continuous numerical |
| BMI | Float (as-is) | Continuous numerical |
| Smoker | Label-encoded: NO=0, YES=1 | Binary categorical |
| Gender | Label-encoded: FEMALE=0, MALE=1 | Binary categorical |

Target: `Diagnosis` — Label-encoded (DIABETES=0, HEART DISEASE=1).

### Algorithm Choice — Random Forest
A **Random Forest Classifier** was selected for the following reasons:

1. **Non-linear decision boundaries** — The scatter plots showed no clean linear separation
   between the two classes; tree-based ensembles handle this natively.
2. **Robustness to small datasets** — With only 52 labelled samples, single decision trees
   overfit easily. Ensembling 100 trees with bagging reduces variance.
3. **Feature importance** — Random Forests output feature importances directly,
   providing built-in interpretability.
4. **No scaling required** — Unlike SVM or k-NN, Random Forests are invariant to
   feature scale, removing the need for standardisation.
5. **Handles mixed feature types** — Continuous and binary features are combined
   without requiring one-hot encoding.

### Training Procedure
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

- **Test split**: 25% hold-out (stratified to preserve class balance)
- **Cross-validation**: 5-fold CV used to provide a more stable accuracy estimate
- **Random state**: Fixed for reproducibility

### Evaluation

```
5-fold cross-validation mean accuracy: ~57%  (± ~20%)
```

The wide standard deviation reflects the very small dataset (52 samples ÷ 5 folds ≈ 10 samples
per fold). The cross-validation mean is above the random baseline (50%), showing the model
has learned a weak but genuine signal.

Feature importances confirm that **Cholesterol** and **Blood Pressure** dominate, which
aligns with clinical knowledge — both are primary risk factors for both conditions.

---

## 4. Limitations and Future Improvements

| Limitation | Suggested Improvement |
|---|---|
| Only 52 labelled samples | Collect more labelled patient records |
| UNKNOWN labels excluded from training | Semi-supervised methods could leverage unlabelled data |
| No hyperparameter tuning | Grid-search or Bayesian optimisation on `n_estimators`, `max_depth` |
| Admission_Date and Notes unused | Date could capture seasonal patterns; notes could be NLP-featurised |
| Binary target only | Multi-class extension once data volume is sufficient |

---

## 5. Integration

The trained model is serialised with `pickle` to `model_outputs/rf_model.pkl`.
The Flask app loads it at startup and exposes a prediction form at `/model`.
Users enter clinical values in a web form; the app returns the predicted diagnosis
and class probabilities rendered as a bar chart.

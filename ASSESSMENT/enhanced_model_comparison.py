"""
Enhanced model comparison for small medical datasets.

Adds additional models specifically chosen for small dataset performance:
- K-Nearest Neighbors (optimal for small data)
- Linear SVM (fewer parameters than RBF)
- Ridge/Lasso Regression (strong regularization) 
- Extra Trees (more robust than Random Forest)
- Gradient Boosting (with heavy regularization)
"""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def enhanced_model_comparison(X, y, cv_folds=5):
    """
    Enhanced model comparison optimized for small datasets.
    
    Focus on models that work well with limited training data:
    - Fewer parameters to reduce overfitting
    - Built-in regularization
    - Good performance with 50-100 samples
    """
    
    # Models optimized for small datasets
    small_data_models = {
        # Distance-based (no training phase)
        'KNN (k=3)': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=3, weights='distance'))
        ]),
        
        'KNN (k=5)': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
        ]),
        
        # Linear models (fewer parameters)
        'Linear SVM (C=0.1)': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', C=0.1, probability=True, random_state=42))
        ]),
        
        'Linear SVM (C=1.0)': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', C=1.0, probability=True, random_state=42))
        ]),
        
        # Regularized linear models
        'Lasso Logistic (C=0.1)': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=1000, random_state=42))
        ]),
        
        'Ridge Logistic (C=0.1)': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42))
        ]),
        
        'Ridge Classifier': Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeClassifier(alpha=1.0, random_state=42))
        ]),
        
        # Tree-based with heavy regularization
        'Extra Trees (constrained)': ExtraTreesClassifier(
            n_estimators=50, max_depth=2, min_samples_leaf=5, 
            min_samples_split=10, random_state=42
        ),
        
        'Gradient Boosting (regularized)': GradientBoostingClassifier(
            n_estimators=50, max_depth=2, min_samples_leaf=5,
            learning_rate=0.1, subsample=0.8, random_state=42
        ),
        
        # Original models for comparison
        'SVM (RBF) - Original': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42))
        ]),
        
        'Random Forest - Original': RandomForestClassifier(
            n_estimators=100, random_state=42
        )
    }
    
    results = {}
    print("Testing models optimized for small datasets...")
    print("=" * 60)
    
    for name, model in small_data_models.items():
        try:
            # Cross-validation scores
            cv_acc = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            cv_roc = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')
            
            results[name] = {
                'cv_acc_mean': round(float(cv_acc.mean()) * 100, 1),
                'cv_acc_std':  round(float(cv_acc.std())  * 100, 1),
                'cv_roc_mean': round(float(cv_roc.mean()) * 100, 1),
                'cv_roc_std':  round(float(cv_roc.std())  * 100, 1),
                'model_type': 'small_data_optimized'
            }
            
            print(f"{name:25} | Acc: {results[name]['cv_acc_mean']:5.1f}% ± {results[name]['cv_acc_std']:4.1f} | "
                  f"ROC: {results[name]['cv_roc_mean']:5.1f}% ± {results[name]['cv_roc_std']:4.1f}")
            
        except Exception as e:
            print(f"{name:25} | ERROR: {str(e)}")
            
    print("=" * 60)
    
    # Find best performing models
    if results:
        best_acc_model = max(results, key=lambda x: results[x]['cv_acc_mean'])
        best_roc_model = max(results, key=lambda x: results[x]['cv_roc_mean'])
        
        print(f"Best Accuracy: {best_acc_model} ({results[best_acc_model]['cv_acc_mean']:.1f}%)")
        print(f"Best ROC-AUC:  {best_roc_model} ({results[best_roc_model]['cv_roc_mean']:.1f}%)")
        
    return results


def small_dataset_recommendations():
    """
    Print recommendations for improving model performance on small datasets.
    """
    print("\n" + "🎯 RECOMMENDATIONS FOR SMALL DATASETS (52 samples)")
    print("=" * 65)
    
    recommendations = [
        ("✅ Use K-Nearest Neighbors", "No training phase, works well with 50-100 samples"),
        ("✅ Try Linear SVM", "Fewer parameters than RBF kernel, less overfitting"),
        ("✅ Strong Regularization", "Use L1/L2 penalties to prevent overfitting"),
        ("✅ Cross-Validation", "Use LOO or stratified k-fold for reliable estimates"),
        ("✅ Feature Selection", "Reduce dimensionality to avoid curse of dimensionality"),
        ("⚠️  Avoid Deep Models", "Neural networks, deep trees need 1000+ samples"),
        ("⚠️  Careful with Ensembles", "Random Forest can overfit with <100 samples"),
        ("❌ No Boosting (XGBoost)", "Typically needs 500+ samples to work well")
    ]
    
    for title, desc in recommendations:
        print(f"{title:25} | {desc}")
    
    print("\n" + "💡 ADVANCED TECHNIQUES FOR TINY DATASETS")
    print("=" * 50)
    
    advanced = [
        ("Bayesian Approaches", "Include uncertainty in predictions"),
        ("Leave-One-Out CV", "Maximum use of training data for evaluation"),
        ("Stratified Sampling", "Ensure balanced class representation"),
        ("Feature Engineering", "Create domain-specific features"),
        ("Ensemble of Simple Models", "Combine 3-5 different simple models"),
        ("Bootstrap Aggregating", "Reduce variance with resampling")
    ]
    
    for technique, desc in advanced:
        print(f"• {technique:20} | {desc}")


if __name__ == "__main__":
    # Example usage
    from predictor import _load_and_prepare
    
    # Load your data
    X, y, _ = _load_and_prepare('realworld_medical_dirty_cleaned_1.csv')
    
    print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Run enhanced comparison
    cv_folds = min(5, len(y) // 2)  # Adjust for tiny datasets
    results = enhanced_model_comparison(X, y, cv_folds)
    
    # Print recommendations
    small_dataset_recommendations()
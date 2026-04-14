"""
Improved model implementations specifically for small medical datasets.

This script adds models that work better than Random Forest for datasets with <100 samples.
Focus on models with fewer parameters and built-in regularization.
"""

import os
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from predictor import _load_and_prepare, MODEL_DIR


def small_dataset_models():
    """
    Return a dictionary of models optimized for small datasets (<100 samples).
    
    These models have fewer parameters and are less prone to overfitting
    than Random Forest on tiny datasets.
    """
    
    models = {
        # K-Nearest Neighbors (best for small data)
        'KNN (k=3)': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=3, weights='distance'))
        ]),
        
        'KNN (k=5)': Pipeline([
            ('scaler', StandardScaler()),  
            ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
        ]),
        
        'KNN (k=7)': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance'))
        ]),
        
        # Linear models (fewer parameters than RBF SVM)
        'Linear SVM (C=0.01)': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', C=0.01, probability=True, random_state=42))
        ]),
        
        'Linear SVM (C=0.1)': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', C=0.1, probability=True, random_state=42))  
        ]),
        
        'Linear SVM (C=1.0)': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', C=1.0, probability=True, random_state=42))
        ]),
        
        # Logistic regression with feature selection
        'Lasso Logistic (C=0.01)': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=0.01, penalty='l1', solver='liblinear', max_iter=1000, random_state=42))
        ]),
        
        'Lasso Logistic (C=0.1)': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=1000, random_state=42))
        ]),
        
        'Ridge Logistic (C=0.1)': Pipeline([
            ('scaler', StandardScaler()),  
            ('lr', LogisticRegression(C=0.1, penalty='l2', max_iter=1000, random_state=42))
        ]),
        
        # Simple decision tree (interpretable)
        'Decision Tree (depth=2)': DecisionTreeClassifier(
            max_depth=2, min_samples_leaf=3, random_state=42
        ),
        
        'Decision Tree (depth=3)': DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=2, random_state=42  
        ),
        
        # Feature selection + simple model
        'Logistic + Feature Selection': Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=5)),
            ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
        ]),
    }
    
    return models


def evaluate_small_dataset_models(train_path=None):
    """
    Evaluate models optimized for small datasets using Leave-One-Out CV.
    
    LOO CV is more stable than k-fold CV for datasets with <100 samples.
    """
    
    if train_path is None:
        train_path = os.path.join(os.path.dirname(__file__), 'realworld_medical_dirty_cleaned_1.csv')
    
    print("🎯 EVALUATING MODELS FOR SMALL MEDICAL DATASET")
    print("=" * 65)
    
    # Load the data
    X, y, label_mappings = _load_and_prepare(train_path)
    n_samples, n_features = X.shape
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Classes: {np.bincount(y)} samples per class")
    print(f"Class names: {list(label_mappings['target'].classes_)}")
    
    if n_samples < 30:
        print("⚠️  Dataset very small (<30 samples). Results may be unreliable.")
    elif n_samples < 100:
        print("⚠️  Small dataset (<100 samples). Using LOO CV for stability.")
    
    # Get models optimized for small datasets
    models = small_dataset_models()
    
    # Add best performing original models for comparison
    from predictor import _scaled_pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    
    models.update({
        # Original models for comparison
        'SVM (RBF) - Original': _scaled_pipeline(SVC(kernel='rbf', probability=True, random_state=42)),
        'Random Forest - Original': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes - Original': GaussianNB()
    })
    
    print(f"\nTesting {len(models)} models...")
    print("\nModel                        | LOO Accuracy | 5-Fold Accuracy | Assessment")
    print("-" * 85)
    
    results = {}
    
    for name, model in models.items():
        try:
            # Leave-One-Out Cross Validation (most stable for small data)
            loo = LeaveOneOut()
            loo_scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
            loo_acc = loo_scores.mean() * 100
            
            # 5-fold CV for comparison (if dataset large enough)
            if n_samples >= 25:  # Need at least 5 samples per fold
                cv5_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                cv5_acc = cv5_scores.mean() * 100
            else:
                cv5_acc = np.nan
            
            # Assessment based on LOO performance
            if loo_acc >= 65:
                assessment = "🟢 Good"
            elif loo_acc >= 60:
                assessment = "🟡 Moderate"
            elif loo_acc >= 55:
                assessment = "🔴 Poor"
            else:
                assessment = "❌ Very Poor"
            
            results[name] = {
                'loo_accuracy': loo_acc,
                'cv5_accuracy': cv5_acc,
                'assessment': assessment,
                'model_type': 'small_data' if 'Original' not in name else 'original'
            }
            
            if not np.isnan(cv5_acc):
                print(f"{name:28} | {loo_acc:10.1f}% | {cv5_acc:12.1f}% | {assessment}")
            else:
                print(f"{name:28} | {loo_acc:10.1f}% | {'N/A':12} | {assessment}")
                
        except Exception as e:
            print(f"{name:28} | ERROR: {str(e)[:40]}...")
            
    print("-" * 85)
    
    # Find best models
    if results:
        # Sort by LOO accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['loo_accuracy'], reverse=True)
        
        print(f"\n🏆 TOP PERFORMING MODELS:")
        for i, (name, metrics) in enumerate(sorted_results[:3]):
            print(f"{i+1}. {name}: {metrics['loo_accuracy']:.1f}% LOO accuracy")
            
        # Compare small-data models vs originals
        small_data_models = {k: v for k, v in results.items() if v['model_type'] == 'small_data'}
        original_models = {k: v for k, v in results.items() if v['model_type'] == 'original'}
        
        if small_data_models and original_models:
            best_small = max(small_data_models.items(), key=lambda x: x[1]['loo_accuracy'])
            best_original = max(original_models.items(), key=lambda x: x[1]['loo_accuracy'])
            
            print(f"\n💡 COMPARISON:")
            print(f"   Best Small-Data Model: {best_small[0]} ({best_small[1]['loo_accuracy']:.1f}%)")
            print(f"   Best Original Model:   {best_original[0]} ({best_original[1]['loo_accuracy']:.1f}%)")
            
            improvement = best_small[1]['loo_accuracy'] - best_original[1]['loo_accuracy']
            if improvement > 2:
                print(f"   ✅ Small-data models perform {improvement:.1f}% better!")
            elif improvement > 0:
                print(f"   🟡 Modest improvement of {improvement:.1f}%")
            else:
                print(f"   ⚠️  Original models still perform better")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   • Use Leave-One-Out CV for most reliable estimates")  
        print(f"   • KNN and Linear SVM typically work best for tiny datasets")
        print(f"   • Consider ensemble of top 2-3 simple models")
        print(f"   • Collect more data: target 200+ samples per class")
        print(f"   • Current performance ceiling ~65% due to dataset size")
    
    # Save results
    os.makedirs(MODEL_DIR, exist_ok=True)
    output_path = os.path.join(MODEL_DIR, 'small_dataset_comparison.json')
    
    with open(output_path, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        json_results = {}
        for name, metrics in results.items():
            json_results[name] = {
                'loo_accuracy': float(metrics['loo_accuracy']),
                'cv5_accuracy': float(metrics['cv5_accuracy']) if not np.isnan(metrics['cv5_accuracy']) else None,
                'assessment': metrics['assessment'],
                'model_type': metrics['model_type']
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    return results


def create_ensemble_model():
    """
    Create an ensemble of the best-performing small-dataset models.
    
    Combines 3 different types of simple models for better stability.
    """
    
    # Best models for small datasets (different types for diversity)
    ensemble_models = [
        ('knn', Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
        ])),
        ('linear_svm', Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', C=0.1, probability=True, random_state=42))
        ])),
        ('decision_tree', DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42))
    ]
    
    # Soft voting (uses probability estimates)
    ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
    
    return ensemble


if __name__ == "__main__":
    # Run the evaluation
    results = evaluate_small_dataset_models()
    
    print(f"\n🔬 TESTING ENSEMBLE MODEL:")
    print("-" * 30)
    
    # Test ensemble model
    try:
        from predictor import _load_and_prepare
        X, y, _ = _load_and_prepare()
        
        ensemble = create_ensemble_model()
        
        # LOO CV for ensemble
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        ensemble_scores = cross_val_score(ensemble, X, y, cv=loo, scoring='accuracy')
        ensemble_acc = ensemble_scores.mean() * 100
        
        print(f"Ensemble Model (KNN+LinearSVM+Tree): {ensemble_acc:.1f}% LOO accuracy")
        
        # Compare to best individual model
        if results:
            best_individual = max(results.items(), key=lambda x: x[1]['loo_accuracy'])
            best_acc = best_individual[1]['loo_accuracy']
            
            improvement = ensemble_acc - best_acc
            print(f"Best Individual Model: {best_acc:.1f}% LOO accuracy")
            
            if improvement > 1:
                print(f"✅ Ensemble improves performance by {improvement:.1f}%")
            else:
                print(f"⚠️  Individual models perform similarly ({improvement:+.1f}%)")
                
    except Exception as e:
        print(f"❌ Ensemble test failed: {e}")
        
    print(f"\n🎯 SUMMARY: Models tested and compared for your 52-sample dataset!")
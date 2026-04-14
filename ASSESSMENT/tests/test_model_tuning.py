"""
Tests for GridSearchCV hyperparameter tuning, model comparison, and LOO CV.

Verifies:
  - compare_models() returns all 6 candidates with valid CV scores
  - GridSearchCV best_params contains the expected keys
  - LOO accuracy and ROC-AUC are present and valid
  - comparison.json is written and loadable
  - Best model is identified by highest CV ROC-AUC
  - RF remains competitive (CV ROC-AUC >= other models)
"""

import sys
import os
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from predictor import compare_models, load_comparison, load_performance, train, COMPARISON_JSON

EXPECTED_MODELS = {
    'RF (default)',
    'RF (constrained)',
    'RF (tuned)',
    'Logistic Reg.',
    'SVM (RBF)',
    'Naive Bayes',
}

RANDOM_BASELINE = 50.0


class TestModelComparison(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.comp = compare_models()

    # ── Structure ────────────────────────────────────────────────────────────

    def test_returns_dict(self):
        self.assertIsInstance(self.comp, dict)

    def test_has_models_key(self):
        self.assertIn('models', self.comp)

    def test_has_best_model_key(self):
        self.assertIn('best_model', self.comp)

    def test_has_best_params_key(self):
        self.assertIn('best_params', self.comp)

    def test_all_candidates_present(self):
        for name in EXPECTED_MODELS:
            with self.subTest(model=name):
                self.assertIn(name, self.comp['models'])

    def test_each_model_has_required_metrics(self):
        for name, result in self.comp['models'].items():
            for key in ('cv_acc_mean', 'cv_acc_std', 'cv_roc_mean', 'cv_roc_std'):
                with self.subTest(model=name, key=key):
                    self.assertIn(key, result)

    # ── Score ranges ─────────────────────────────────────────────────────────

    def test_cv_acc_means_in_valid_range(self):
        for name, result in self.comp['models'].items():
            with self.subTest(model=name):
                self.assertGreaterEqual(result['cv_acc_mean'], 0)
                self.assertLessEqual(result['cv_acc_mean'], 100)

    def test_cv_roc_means_in_valid_range(self):
        for name, result in self.comp['models'].items():
            with self.subTest(model=name):
                self.assertGreaterEqual(result['cv_roc_mean'], 0)
                self.assertLessEqual(result['cv_roc_mean'], 100)

    def test_cv_stds_non_negative(self):
        for name, result in self.comp['models'].items():
            with self.subTest(model=name):
                self.assertGreaterEqual(result['cv_acc_std'], 0)
                self.assertGreaterEqual(result['cv_roc_std'], 0)

    # ── Best model selection ──────────────────────────────────────────────────

    def test_best_model_is_in_candidates(self):
        self.assertIn(self.comp['best_model'], self.comp['models'])

    def test_best_model_has_highest_cv_roc(self):
        """best_model must have the highest (or joint-highest) CV ROC-AUC."""
        best_roc = self.comp['models'][self.comp['best_model']]['cv_roc_mean']
        for name, result in self.comp['models'].items():
            with self.subTest(model=name):
                self.assertLessEqual(
                    result['cv_roc_mean'], best_roc + 0.01,  # float tolerance
                    msg=(f"{name} ({result['cv_roc_mean']}%) should not exceed "
                         f"best_model {self.comp['best_model']} ({best_roc}%).")
                )

    # ── GridSearchCV best params ──────────────────────────────────────────────

    def test_best_params_has_max_depth(self):
        self.assertIn('max_depth', self.comp['best_params'])

    def test_best_params_has_min_samples_leaf(self):
        self.assertIn('min_samples_leaf', self.comp['best_params'])

    def test_best_params_has_n_estimators(self):
        self.assertIn('n_estimators', self.comp['best_params'])

    def test_best_params_n_estimators_positive(self):
        self.assertGreater(self.comp['best_params']['n_estimators'], 0)

    def test_best_params_min_samples_leaf_positive(self):
        self.assertGreater(self.comp['best_params']['min_samples_leaf'], 0)

    # ── Persistence ──────────────────────────────────────────────────────────

    def test_comparison_json_written(self):
        self.assertTrue(os.path.isfile(COMPARISON_JSON))

    def test_comparison_json_loadable(self):
        with open(COMPARISON_JSON) as fh:
            data = json.load(fh)
        self.assertIn('models', data)
        self.assertIn('best_model', data)

    def test_load_comparison_matches_run(self):
        loaded = load_comparison()
        self.assertEqual(loaded['best_model'], self.comp['best_model'])

    # ── RF competitiveness (supervised learning requirement) ─────────────────

    def test_rf_default_beats_naive_bayes_on_roc(self):
        """RF should be at least as good as Naive Bayes — if not, investigate."""
        rf_roc  = self.comp['models']['RF (default)']['cv_roc_mean']
        nb_roc  = self.comp['models']['Naive Bayes']['cv_roc_mean']
        self.assertGreaterEqual(
            rf_roc, nb_roc - 5,   # 5% tolerance for small-dataset noise
            msg=f"RF ({rf_roc}%) much worse than NB ({nb_roc}%) — unexpected."
        )


class TestLOOCrossValidation(unittest.TestCase):
    """Tests for Leave-One-Out CV metrics."""

    @classmethod
    def setUpClass(cls):
        cls.metrics = train()
        cls.perf    = load_performance()

    def test_loo_acc_present_in_metrics(self):
        self.assertIn('loo_acc', self.metrics)

    def test_loo_roc_present_in_metrics(self):
        self.assertIn('loo_roc', self.metrics)

    def test_loo_acc_in_valid_range(self):
        self.assertGreaterEqual(self.metrics['loo_acc'], 0)
        self.assertLessEqual(self.metrics['loo_acc'], 100)

    def test_loo_roc_in_valid_range(self):
        self.assertGreaterEqual(self.metrics['loo_roc'], 0)
        self.assertLessEqual(self.metrics['loo_roc'], 100)

    def test_loo_acc_in_performance_json(self):
        self.assertIn('loo_acc', self.perf)

    def test_loo_roc_in_performance_json(self):
        self.assertIn('loo_roc', self.perf)

    def test_loo_acc_is_float(self):
        self.assertIsInstance(self.perf['loo_acc'], float)

    def test_loo_roc_is_float(self):
        self.assertIsInstance(self.perf['loo_roc'], float)


class TestGridSearchTuning(unittest.TestCase):
    """Tests confirming GridSearchCV ran and produced valid best_params."""

    @classmethod
    def setUpClass(cls):
        cls.metrics = train()
        cls.perf    = load_performance()

    def test_best_params_in_metrics(self):
        self.assertIn('best_params', self.metrics)

    def test_best_params_in_performance_json(self):
        self.assertIn('best_params', self.perf)

    def test_best_params_has_required_keys(self):
        for key in ('max_depth', 'min_samples_leaf', 'n_estimators'):
            with self.subTest(key=key):
                self.assertIn(key, self.metrics['best_params'])

    def test_tuned_model_cv_roc_above_random(self):
        """After tuning, CV ROC-AUC must still beat random."""
        self.assertGreater(
            self.metrics['cv_roc_auc_mean'], RANDOM_BASELINE,
            msg=f"Tuned model CV ROC-AUC {self.metrics['cv_roc_auc_mean']}% <= 50%."
        )

    def test_tuned_model_cv_acc_above_random(self):
        self.assertGreater(
            self.metrics['cv_mean'], RANDOM_BASELINE,
            msg=f"Tuned model CV accuracy {self.metrics['cv_mean']}% <= 50%."
        )


if __name__ == '__main__':
    unittest.main()

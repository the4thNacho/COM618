"""
Tests for model performance and overfitting detection.

These tests assert:
  1. The model beats a random baseline on cross-validation (not just memorising).
  2. Overfitting IS detected (train acc > test acc) — expected on 52 samples.
  3. The overfitting gap does not exceed a "severe" threshold (we allow up to 75%
     given the tiny dataset, but flag anything above 20% as a warning in the app).
  4. ROC-AUC (CV) is above the random baseline of 0.50.
  5. Confidence scores are valid floats in [50, 100].
  6. Per-class metrics are structurally correct and non-negative.
  7. Performance JSON contains all expected keys.
"""

import sys
import os
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from predictor import train, load_performance, PERFORMANCE_JSON

# Thresholds ────────────────────────────────────────────────────────────────
# The dataset has only 52 labelled samples, so we use generous bounds.
# The goal is to detect *catastrophic* failures and document expected behaviour,
# not to chase arbitrary accuracy numbers on an inherently small dataset.

RANDOM_BASELINE        = 50.0   # % — binary random classifier
SEVERE_OVERFIT_LIMIT   = 75.0   # % — gap above this implies something is wrong
CV_ACC_MIN             = RANDOM_BASELINE   # CV accuracy must beat random
CV_ROC_MIN             = RANDOM_BASELINE   # CV ROC-AUC must beat random
CONFIDENCE_MIN         = 50.0   # % — max-prob of binary classifier ≥ 50
CONFIDENCE_MAX         = 100.0


class TestModelPerformanceMetrics(unittest.TestCase):
    """Verify the metrics dict returned by train() is complete and sane."""

    @classmethod
    def setUpClass(cls):
        cls.metrics = train()
        cls.perf    = load_performance()

    # ── Completeness ─────────────────────────────────────────────────────────

    def test_metrics_has_all_keys(self):
        required = [
            'accuracy', 'train_accuracy', 'overfitting_gap',
            'roc_auc', 'cv_roc_auc_mean', 'cv_roc_auc_std',
            'cv_mean', 'cv_std', 'cv_scores', 'cv_roc_scores',
            'mean_confidence', 'confidence_scores',
            'per_class', 'report', 'classes',
        ]
        for key in required:
            with self.subTest(key=key):
                self.assertIn(key, self.metrics)

    def test_performance_json_exists(self):
        self.assertTrue(os.path.isfile(PERFORMANCE_JSON))

    def test_performance_json_loadable(self):
        with open(PERFORMANCE_JSON) as fh:
            data = json.load(fh)
        self.assertIsInstance(data, dict)

    def test_performance_json_has_all_keys(self):
        required = [
            'y_test', 'y_pred', 'y_proba', 'fpr', 'tpr',
            'class_names', 'train_acc', 'test_acc', 'roc_auc',
            'cv_acc', 'cv_roc', 'confidence_scores', 'mean_confidence',
            'per_class', 'overfitting_gap',
        ]
        for key in required:
            with self.subTest(key=key):
                self.assertIn(key, self.perf)

    # ── Accuracy ranges ──────────────────────────────────────────────────────

    def test_test_accuracy_in_valid_range(self):
        self.assertGreaterEqual(self.metrics['accuracy'], 0)
        self.assertLessEqual(self.metrics['accuracy'], 100)

    def test_train_accuracy_in_valid_range(self):
        self.assertGreaterEqual(self.metrics['train_accuracy'], 0)
        self.assertLessEqual(self.metrics['train_accuracy'], 100)

    def test_train_accuracy_not_lower_than_test(self):
        """A well-fitted model should score at least as well on its own training data."""
        self.assertGreaterEqual(
            self.metrics['train_accuracy'],
            self.metrics['accuracy'],
            msg="Train accuracy should be >= test accuracy."
        )

    # ── Overfitting detection ─────────────────────────────────────────────────

    def test_overfitting_gap_is_detected(self):
        """
        With 52 samples and 100 trees, Random Forest memorises training data.
        We assert the gap IS positive (expected behaviour, not a bug).
        The performance page surfaces this as a warning to the user.
        """
        gap = self.metrics['overfitting_gap']
        self.assertGreater(
            gap, 0,
            msg=f"Expected overfitting gap > 0 on this small dataset, got {gap}%."
        )

    def test_overfitting_gap_not_catastrophic(self):
        """Gap should be explainable by dataset size, not a code bug."""
        gap = self.metrics['overfitting_gap']
        self.assertLessEqual(
            gap, SEVERE_OVERFIT_LIMIT,
            msg=f"Overfitting gap {gap}% exceeds severe threshold {SEVERE_OVERFIT_LIMIT}%."
        )

    # ── Cross-validation beats random baseline ───────────────────────────────

    def test_cv_accuracy_beats_random(self):
        """CV accuracy must be above random (50%) — i.e. the model has learned something."""
        self.assertGreater(
            self.metrics['cv_mean'], RANDOM_BASELINE,
            msg=(f"CV accuracy {self.metrics['cv_mean']}% does not beat random baseline "
                 f"({RANDOM_BASELINE}%).")
        )

    def test_cv_roc_auc_beats_random(self):
        """CV ROC-AUC must be above 0.50 — better than random ranking."""
        self.assertGreater(
            self.metrics['cv_roc_auc_mean'], RANDOM_BASELINE,
            msg=(f"CV ROC-AUC {self.metrics['cv_roc_auc_mean']}% does not beat random "
                 f"baseline ({RANDOM_BASELINE}%).")
        )

    def test_cv_scores_correct_count(self):
        self.assertEqual(len(self.metrics['cv_scores']), 5)

    def test_cv_roc_scores_correct_count(self):
        self.assertEqual(len(self.metrics['cv_roc_scores']), 5)

    def test_cv_scores_all_in_valid_range(self):
        for i, s in enumerate(self.metrics['cv_scores']):
            with self.subTest(fold=i + 1):
                self.assertGreaterEqual(s, 0)
                self.assertLessEqual(s, 100)

    # ── ROC-AUC ──────────────────────────────────────────────────────────────

    def test_roc_auc_in_valid_range(self):
        self.assertGreaterEqual(self.metrics['roc_auc'], 0)
        self.assertLessEqual(self.metrics['roc_auc'], 100)

    def test_fpr_tpr_same_length(self):
        self.assertEqual(len(self.perf['fpr']), len(self.perf['tpr']))

    def test_fpr_starts_at_zero(self):
        self.assertAlmostEqual(self.perf['fpr'][0], 0.0, places=5)

    def test_tpr_ends_at_one(self):
        self.assertAlmostEqual(self.perf['tpr'][-1], 1.0, places=5)

    # ── Confidence scores ────────────────────────────────────────────────────

    def test_confidence_scores_all_valid(self):
        for i, c in enumerate(self.metrics['confidence_scores']):
            with self.subTest(sample=i):
                self.assertGreaterEqual(c, CONFIDENCE_MIN,
                    msg=f"Confidence {c}% < {CONFIDENCE_MIN}% (impossible for max-prob)")
                self.assertLessEqual(c, CONFIDENCE_MAX)

    def test_mean_confidence_in_valid_range(self):
        self.assertGreaterEqual(self.metrics['mean_confidence'], CONFIDENCE_MIN)
        self.assertLessEqual(self.metrics['mean_confidence'], CONFIDENCE_MAX)

    def test_confidence_count_matches_test_set(self):
        """One confidence score per test sample."""
        n_test = len(self.perf['y_test'])
        self.assertEqual(len(self.metrics['confidence_scores']), n_test)

    # ── Per-class metrics ────────────────────────────────────────────────────

    def test_per_class_has_both_classes(self):
        for cls in ['DIABETES', 'HEART DISEASE']:
            with self.subTest(cls=cls):
                self.assertIn(cls, self.metrics['per_class'])

    def test_per_class_metrics_all_non_negative(self):
        for cls, vals in self.metrics['per_class'].items():
            for metric, val in vals.items():
                with self.subTest(cls=cls, metric=metric):
                    self.assertGreaterEqual(
                        val, 0,
                        msg=f"{cls} {metric} = {val} is negative."
                    )

    def test_per_class_metrics_at_most_100(self):
        for cls, vals in self.metrics['per_class'].items():
            for metric, val in vals.items():
                with self.subTest(cls=cls, metric=metric):
                    self.assertLessEqual(val, 100)

    def test_per_class_keys_present(self):
        for cls, vals in self.metrics['per_class'].items():
            for key in ('precision', 'recall', 'f1'):
                with self.subTest(cls=cls, key=key):
                    self.assertIn(key, vals)

    # ── Probabilities sum to 1 (sanity check on y_proba) ────────────────────

    def test_y_proba_rows_sum_to_one(self):
        for i, row in enumerate(self.perf['y_proba']):
            with self.subTest(sample=i):
                self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_y_proba_all_non_negative(self):
        for i, row in enumerate(self.perf['y_proba']):
            for j, p in enumerate(row):
                with self.subTest(sample=i, class_idx=j):
                    self.assertGreaterEqual(p, 0.0)

    # ── Classes ──────────────────────────────────────────────────────────────

    def test_classes_correct(self):
        self.assertEqual(sorted(self.metrics['classes']),
                         ['DIABETES', 'HEART DISEASE'])


class TestOverfittingDiagnosis(unittest.TestCase):
    """
    Dedicated tests that document and verify the overfitting behaviour.

    These tests exist to ensure the *overfitting detection machinery*
    works correctly — not to fail CI if the model overfits (which is
    expected with 52 samples).
    """

    @classmethod
    def setUpClass(cls):
        cls.perf = load_performance()

    def test_train_acc_is_100_percent(self):
        """
        Random Forest with default depth perfectly memorises 39 training samples.
        This is the primary overfitting indicator.
        """
        train_acc = self.perf['train_acc'] * 100
        self.assertAlmostEqual(
            train_acc, 100.0, delta=1.0,
            msg=f"Expected near-100% train accuracy on 39 samples, got {train_acc:.1f}%."
        )

    def test_overfitting_gap_exceeds_warning_threshold(self):
        """
        The gap must exceed the 20% warning threshold so the app banner fires.
        This test would need updating if more data is added and overfitting reduces.
        """
        gap = self.perf['overfitting_gap']
        self.assertGreater(
            gap, 20.0,
            msg=(f"Gap {gap}% did not exceed 20% warning threshold. "
                 "If this fails after adding more training data, "
                 "update or remove this test.")
        )

    def test_cv_acc_higher_than_single_split_test_acc(self):
        """
        5-fold CV uses more data for testing, so it gives a better (higher) estimate
        than the single 25% hold-out split.
        """
        cv_mean   = sum(self.perf['cv_acc'])  / len(self.perf['cv_acc'])  * 100
        test_acc  = self.perf['test_acc'] * 100
        # CV should generally be >= single-split test acc on small datasets
        # (not guaranteed but expected here given the split sizes)
        self.assertGreaterEqual(
            cv_mean, test_acc - 5,   # allow 5% tolerance
            msg=(f"CV mean ({cv_mean:.1f}%) much lower than single-split test "
                 f"({test_acc:.1f}%) — unexpected.")
        )


if __name__ == '__main__':
    unittest.main()

"""
Unit tests for the predictor module.
Tests cover model training, prediction output shape/type,
and boundary/edge cases for the predict() function.
"""

import sys
import os
import unittest

# Add ASSESSMENT directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from predictor import train, predict, load_model, FEATURE_COLS


class TestPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Train the model once for the entire test class."""
        cls.metrics = train()

    # ------------------------------------------------------------------ #
    # Training metrics                                                    #
    # ------------------------------------------------------------------ #

    def test_train_returns_dict(self):
        self.assertIsInstance(self.metrics, dict)

    def test_metrics_has_accuracy(self):
        self.assertIn('accuracy', self.metrics)

    def test_metrics_accuracy_is_numeric(self):
        self.assertIsInstance(self.metrics['accuracy'], (int, float))

    def test_metrics_accuracy_in_valid_range(self):
        self.assertGreaterEqual(self.metrics['accuracy'], 0)
        self.assertLessEqual(self.metrics['accuracy'], 100)

    def test_metrics_cv_mean_present(self):
        self.assertIn('cv_mean', self.metrics)

    def test_metrics_has_classes(self):
        self.assertIn('classes', self.metrics)
        self.assertEqual(sorted(self.metrics['classes']), ['DIABETES', 'HEART DISEASE'])

    def test_metrics_has_report(self):
        self.assertIn('report', self.metrics)
        self.assertIsInstance(self.metrics['report'], str)

    # ------------------------------------------------------------------ #
    # Model loading                                                       #
    # ------------------------------------------------------------------ #

    def test_load_model_returns_two_objects(self):
        model, encoders = load_model()
        self.assertIsNotNone(model)
        self.assertIsNotNone(encoders)

    def test_encoders_have_expected_keys(self):
        _, encoders = load_model()
        for key in ('smoker', 'gender', 'target'):
            self.assertIn(key, encoders)

    # ------------------------------------------------------------------ #
    # Prediction output structure                                         #
    # ------------------------------------------------------------------ #

    def test_predict_returns_dict(self):
        result = predict(55, 140, 220, 30.5, 'YES', 'MALE')
        self.assertIsInstance(result, dict)

    def test_predict_diagnosis_is_valid_class(self):
        result = predict(55, 140, 220, 30.5, 'YES', 'MALE')
        self.assertIn(result['diagnosis'], ['DIABETES', 'HEART DISEASE'])

    def test_predict_probabilities_sum_to_100(self):
        result = predict(55, 140, 220, 30.5, 'YES', 'MALE')
        total = sum(result['probabilities'].values())
        self.assertAlmostEqual(total, 100.0, delta=0.5)

    def test_predict_probabilities_all_non_negative(self):
        result = predict(40, 120, 200, 24.0, 'NO', 'FEMALE')
        for p in result['probabilities'].values():
            self.assertGreaterEqual(p, 0)

    def test_predict_classes_list_returned(self):
        result = predict(55, 140, 220, 30.5, 'YES', 'MALE')
        self.assertIn('classes', result)
        self.assertIsInstance(result['classes'], list)

    # ------------------------------------------------------------------ #
    # Gender / smoker case-insensitivity                                  #
    # ------------------------------------------------------------------ #

    def test_predict_lowercase_smoker_accepted(self):
        result = predict(50, 130, 210, 28.0, 'no', 'MALE')
        self.assertIn(result['diagnosis'], ['DIABETES', 'HEART DISEASE'])

    def test_predict_lowercase_gender_accepted(self):
        result = predict(50, 130, 210, 28.0, 'NO', 'female')
        self.assertIn(result['diagnosis'], ['DIABETES', 'HEART DISEASE'])

    # ------------------------------------------------------------------ #
    # Edge-value inputs                                                   #
    # ------------------------------------------------------------------ #

    def test_predict_young_non_smoker(self):
        result = predict(25, 120, 180, 22.5, 'NO', 'MALE')
        self.assertIn(result['diagnosis'], ['DIABETES', 'HEART DISEASE'])

    def test_predict_elderly_high_risk(self):
        result = predict(65, 160, 300, 40.0, 'YES', 'MALE')
        self.assertIn(result['diagnosis'], ['DIABETES', 'HEART DISEASE'])


if __name__ == '__main__':
    unittest.main()

"""
Integration tests for the Flask application routes.
These tests use Flask's test client to verify each route returns
a successful HTTP response and contains expected content.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app import app


class TestFlaskRoutes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    # ------------------------------------------------------------------ #
    # GET routes                                                          #
    # ------------------------------------------------------------------ #

    def test_index_returns_200(self):
        r = self.client.get('/')
        self.assertEqual(r.status_code, 200)

    def test_index_contains_title(self):
        r = self.client.get('/')
        self.assertIn(b'Medical Data Analysis', r.data)

    def test_index_shows_patient_count(self):
        r = self.client.get('/')
        # Dataset has 100 rows
        self.assertIn(b'100', r.data)

    def test_cleaning_returns_200(self):
        r = self.client.get('/cleaning')
        self.assertEqual(r.status_code, 200)

    def test_cleaning_contains_strategy_table(self):
        r = self.client.get('/cleaning')
        self.assertIn(b'Median Imputation', r.data)

    def test_exploration_returns_200(self):
        r = self.client.get('/exploration')
        self.assertEqual(r.status_code, 200)

    def test_exploration_contains_insight(self):
        r = self.client.get('/exploration')
        self.assertIn(b'Insight', r.data)

    def test_model_get_returns_200(self):
        r = self.client.get('/model')
        self.assertEqual(r.status_code, 200)

    def test_model_page_contains_form(self):
        r = self.client.get('/model')
        self.assertIn(b'Predict Diagnosis', r.data)

    # ------------------------------------------------------------------ #
    # POST to /model                                                      #
    # ------------------------------------------------------------------ #

    def test_model_post_returns_200(self):
        r = self.client.post('/model', data={
            'age': '55', 'bp': '140', 'chol': '220',
            'bmi': '30.5', 'smoker': 'YES', 'gender': 'MALE'
        })
        self.assertEqual(r.status_code, 200)

    def test_model_post_shows_result(self):
        r = self.client.post('/model', data={
            'age': '55', 'bp': '140', 'chol': '220',
            'bmi': '30.5', 'smoker': 'YES', 'gender': 'MALE'
        })
        # Result box should be in the response
        self.assertIn(b'Predicted Diagnosis', r.data)

    def test_model_post_result_is_valid_diagnosis(self):
        r = self.client.post('/model', data={
            'age': '40', 'bp': '120', 'chol': '200',
            'bmi': '24.0', 'smoker': 'NO', 'gender': 'FEMALE'
        })
        data = r.data.decode()
        self.assertTrue('DIABETES' in data or 'HEART DISEASE' in data)

    # ------------------------------------------------------------------ #
    # Image serving                                                       #
    # ------------------------------------------------------------------ #

    def test_image_missing_data_comparison(self):
        r = self.client.get('/image/missing_data_comparison.png')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content_type, 'image/png')

    def test_image_summary_dashboard(self):
        r = self.client.get('/image/summary_dashboard.png')
        self.assertEqual(r.status_code, 200)

    def test_image_nonexistent_returns_404(self):
        r = self.client.get('/image/does_not_exist.png')
        self.assertEqual(r.status_code, 404)


if __name__ == '__main__':
    unittest.main()

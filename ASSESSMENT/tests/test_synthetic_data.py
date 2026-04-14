"""
Unit tests for synthetic data generation and integration with predictor.
Tests cover synthetic data creation, check-before-create logic, and 
integration with the training pipeline.
"""

import sys
import os
import unittest
import tempfile
import shutil
import json

# Add ASSESSMENT directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import synthesise
from predictor import train, load_performance, _load_and_prepare


class TestSyntheticData(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp()
        self.original_synthetic_path = synthesise.SYNTHETIC_PATH
        self.original_stats_path = synthesise.STATS_PATH
        
        # Point synthetic outputs to our test directory
        synthesise.SYNTHETIC_PATH = os.path.join(self.test_dir, 'synthetic_test.csv')
        synthesise.STATS_PATH = os.path.join(self.test_dir, 'stats_test.json')

    def tearDown(self):
        """Clean up test directory and restore original paths."""
        synthesise.SYNTHETIC_PATH = self.original_synthetic_path
        synthesise.STATS_PATH = self.original_stats_path
        shutil.rmtree(self.test_dir)

    # ------------------------------------------------------------------ #
    # Synthetic data generation                                           #
    # ------------------------------------------------------------------ #

    def test_generate_creates_csv_file(self):
        """Test that generate() creates a CSV file."""
        csv_path = synthesise.generate(n_samples=50)
        self.assertTrue(os.path.exists(csv_path))
        self.assertEqual(csv_path, synthesise.SYNTHETIC_PATH)

    def test_generate_creates_stats_file(self):
        """Test that generate() creates a stats audit file."""
        synthesise.generate(n_samples=50)
        self.assertTrue(os.path.exists(synthesise.STATS_PATH))

    def test_generated_csv_has_correct_columns(self):
        """Test that generated CSV has all expected columns."""
        synthesise.generate(n_samples=50)
        
        import pandas as pd
        df = pd.read_csv(synthesise.SYNTHETIC_PATH)
        
        expected_cols = ['Patient_ID', 'Age', 'Gender', 'Blood_Pressure', 'Cholesterol', 
                        'BMI', 'Smoker', 'Diagnosis', 'Admission_Date', 'Notes']
        self.assertEqual(sorted(df.columns), sorted(expected_cols))

    def test_generated_csv_has_requested_sample_count(self):
        """Test that generated CSV has approximately the requested number of samples."""
        n_samples = 100
        synthesise.generate(n_samples=n_samples)
        
        import pandas as pd
        df = pd.read_csv(synthesise.SYNTHETIC_PATH)
        
        # Allow small variance due to class balancing
        self.assertGreaterEqual(len(df), n_samples - 10)
        self.assertLessEqual(len(df), n_samples + 10)

    def test_generated_data_has_valid_diagnoses(self):
        """Test that generated data only contains valid diagnosis classes."""
        synthesise.generate(n_samples=50)
        
        import pandas as pd
        df = pd.read_csv(synthesise.SYNTHETIC_PATH)
        
        valid_diagnoses = {'DIABETES', 'HEART DISEASE'}
        actual_diagnoses = set(df['Diagnosis'].unique())
        self.assertTrue(actual_diagnoses.issubset(valid_diagnoses))

    # ------------------------------------------------------------------ #
    # Check-before-create logic                                           #
    # ------------------------------------------------------------------ #

    def test_exists_returns_false_when_no_file(self):
        """Test that exists() returns False when no synthetic file exists."""
        self.assertFalse(synthesise.exists())

    def test_exists_returns_true_after_generation(self):
        """Test that exists() returns True after file is generated."""
        synthesise.generate(n_samples=50)
        self.assertTrue(synthesise.exists())

    def test_generate_skips_creation_when_file_exists(self):
        """Test that generate() doesn't recreate file when it exists."""
        # Generate initial file
        first_path = synthesise.generate(n_samples=50)
        first_mtime = os.path.getmtime(first_path)
        
        # Small delay to ensure different timestamp if file was recreated
        import time
        time.sleep(0.01)
        
        # Call generate again
        second_path = synthesise.generate(n_samples=50)
        second_mtime = os.path.getmtime(second_path)
        
        self.assertEqual(first_path, second_path)
        self.assertEqual(first_mtime, second_mtime)  # File was not modified

    def test_generate_force_recreates_file(self):
        """Test that generate(force=True) recreates file even when it exists."""
        # Generate initial file
        first_path = synthesise.generate(n_samples=50)
        first_mtime = os.path.getmtime(first_path)
        
        # Small delay to ensure different timestamp
        import time
        time.sleep(0.01)
        
        # Force regeneration
        second_path = synthesise.generate(n_samples=50, force=True)
        second_mtime = os.path.getmtime(second_path)
        
        self.assertEqual(first_path, second_path)
        self.assertNotEqual(first_mtime, second_mtime)  # File was modified

    # ------------------------------------------------------------------ #
    # Integration with predictor                                          #
    # ------------------------------------------------------------------ #

    def test_load_and_prepare_works_with_synthetic_data(self):
        """Test that _load_and_prepare can load synthetic data."""
        # Generate synthetic data
        syn_path = synthesise.generate(n_samples=100)
        
        # Load synthetic data
        X, y, encoders = _load_and_prepare(syn_path)
        
        self.assertGreater(len(X), 0)
        self.assertGreater(len(y), 0)
        self.assertEqual(len(X), len(y))
        self.assertIn('smoker', encoders)
        self.assertIn('gender', encoders) 
        self.assertIn('target', encoders)

    def test_training_with_synthetic_data_includes_training_mode(self):
        """Test that training with synthetic data records training_mode."""
        # Generate synthetic data
        synthesise.generate(n_samples=100)
        
        # Train model (should use synthetic data)
        metrics = train()
        
        # Check training mode is recorded
        self.assertIn('training_mode', metrics)
        self.assertEqual(metrics['training_mode'], 'synthetic')

    def test_training_mode_in_performance_json(self):
        """Test that performance.json includes training_mode after training."""
        # Generate synthetic data
        synthesise.generate(n_samples=100)
        
        # Train model
        train()
        
        # Load performance data
        perf = load_performance()
        
        self.assertIn('training_mode', perf)
        self.assertIn(perf['training_mode'], ['synthetic', 'real_only'])

    def test_synthetic_training_uses_all_real_data_for_test(self):
        """Test that synthetic training uses all real data as test set."""
        # Generate synthetic data
        synthesise.generate(n_samples=100)
        
        # Train model 
        train()
        
        # Load performance data
        perf = load_performance()
        
        if perf['training_mode'] == 'synthetic':
            # Should use all 52 real samples for testing
            self.assertEqual(perf['n_test'], 52)
            # Should use synthetic samples for training
            self.assertGreater(perf['n_train'], 52)


if __name__ == '__main__':
    unittest.main()
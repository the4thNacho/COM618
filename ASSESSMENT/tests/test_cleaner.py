"""
Unit tests for the MedicalDataCleaner class.
Tests cover loading, duplicate detection/removal, imputation strategies,
and the cleaning summary.
"""

import sys
import os
import unittest
import tempfile

import pandas as pd
import numpy as np

# Add the scripts directory to sys.path so MedicalDataCleaner can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from medical_data_cleaner import MedicalDataCleaner


def _make_csv(rows: list[dict], tmp_dir: str) -> str:
    """Write rows to a temporary CSV and return its path."""
    df = pd.DataFrame(rows)
    path = os.path.join(tmp_dir, 'test_data.csv')
    df.to_csv(path, index=False)
    return path


SAMPLE_ROWS = [
    {'Patient_ID': 'P01', 'Age': 55.0,  'Gender': 'MALE',   'Blood_Pressure': 130.0, 'Cholesterol': 200.0, 'BMI': 27.0, 'Smoker': 'YES',  'Diagnosis': 'DIABETES',      'Admission_Date': '2023-01-01', 'Notes': np.nan},
    {'Patient_ID': 'P02', 'Age': np.nan,'Gender': 'FEMALE', 'Blood_Pressure': np.nan,'Cholesterol': np.nan,'BMI': np.nan,'Smoker': np.nan, 'Diagnosis': np.nan,          'Admission_Date': '2023-02-01', 'Notes': np.nan},
    {'Patient_ID': 'P03', 'Age': 45.0,  'Gender': np.nan,   'Blood_Pressure': 140.0, 'Cholesterol': 250.0, 'BMI': 30.0, 'Smoker': 'NO',   'Diagnosis': 'HEART DISEASE', 'Admission_Date': '2023-03-01', 'Notes': 'stable'},
    {'Patient_ID': 'P04', 'Age': 65.0,  'Gender': 'male',   'Blood_Pressure': 150.0, 'Cholesterol': 220.0, 'BMI': 35.0, 'Smoker': 'Y',    'Diagnosis': 'diabetes',      'Admission_Date': '2023-04-01', 'Notes': ''},
    # Duplicate of P04
    {'Patient_ID': 'P04', 'Age': 65.0,  'Gender': 'male',   'Blood_Pressure': 150.0, 'Cholesterol': 220.0, 'BMI': 35.0, 'Smoker': 'Y',    'Diagnosis': 'diabetes',      'Admission_Date': '2023-04-01', 'Notes': ''},
]


class TestMedicalDataCleaner(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.csv_path = _make_csv(SAMPLE_ROWS, self.tmp)
        self.cleaner = MedicalDataCleaner(self.csv_path)
        self.cleaner.load_data()

    # ------------------------------------------------------------------ #
    # Loading                                                             #
    # ------------------------------------------------------------------ #

    def test_load_data_returns_dataframe(self):
        self.assertIsInstance(self.cleaner.df, pd.DataFrame)

    def test_load_data_row_count(self):
        self.assertEqual(len(self.cleaner.df), len(SAMPLE_ROWS))

    def test_load_data_columns(self):
        expected = ['Patient_ID', 'Age', 'Gender', 'Blood_Pressure',
                    'Cholesterol', 'BMI', 'Smoker', 'Diagnosis',
                    'Admission_Date', 'Notes']
        self.assertEqual(list(self.cleaner.df.columns), expected)

    # ------------------------------------------------------------------ #
    # Duplicate detection & removal                                       #
    # ------------------------------------------------------------------ #

    def test_check_duplicates_finds_duplicates(self):
        dupes = self.cleaner.check_duplicates()
        # P04 appears twice → 2 rows in the duplicates report
        self.assertEqual(len(dupes), 2)

    def test_remove_duplicates_reduces_row_count(self):
        before = len(self.cleaner.df)
        self.cleaner.remove_duplicates()
        self.assertEqual(len(self.cleaner.df), before - 1)

    def test_remove_duplicates_no_duplicates_remain(self):
        self.cleaner.remove_duplicates()
        self.assertEqual(self.cleaner.df.duplicated().sum(), 0)

    # ------------------------------------------------------------------ #
    # Missing-data analysis                                               #
    # ------------------------------------------------------------------ #

    def test_analyze_missing_data_returns_dataframe(self):
        stats = self.cleaner.analyze_missing_data()
        self.assertIsInstance(stats, pd.DataFrame)
        self.assertIn('Missing_Count', stats.columns)
        self.assertIn('Missing_Percentage', stats.columns)

    def test_analyze_missing_data_detects_age_missing(self):
        stats = self.cleaner.analyze_missing_data()
        age_row = stats[stats['Column'] == 'Age']
        self.assertEqual(int(age_row['Missing_Count'].iloc[0]), 1)

    # ------------------------------------------------------------------ #
    # Cleaning — numerical imputation                                     #
    # ------------------------------------------------------------------ #

    def test_clean_age_no_nulls_after_cleaning(self):
        self.cleaner.clean_data()
        self.assertEqual(self.cleaner.df_cleaned['Age'].isna().sum(), 0)

    def test_clean_age_uses_median(self):
        self.cleaner.clean_data()
        # The imputed value should equal the pre-clean median
        median_age = self.cleaner.df['Age'].median()
        imputed_val = self.cleaner.df_cleaned.loc[
            self.cleaner.df['Age'].isna(), 'Age'
        ].iloc[0]
        self.assertAlmostEqual(imputed_val, median_age)

    def test_clean_blood_pressure_no_nulls(self):
        self.cleaner.clean_data()
        self.assertEqual(self.cleaner.df_cleaned['Blood_Pressure'].isna().sum(), 0)

    def test_clean_cholesterol_no_nulls(self):
        self.cleaner.clean_data()
        self.assertEqual(self.cleaner.df_cleaned['Cholesterol'].isna().sum(), 0)

    def test_clean_bmi_no_nulls(self):
        self.cleaner.clean_data()
        self.assertEqual(self.cleaner.df_cleaned['BMI'].isna().sum(), 0)

    # ------------------------------------------------------------------ #
    # Cleaning — categorical imputation & normalisation                  #
    # ------------------------------------------------------------------ #

    def test_clean_gender_uppercase(self):
        self.cleaner.clean_data()
        for val in self.cleaner.df_cleaned['Gender'].dropna():
            self.assertEqual(val, val.upper())

    def test_clean_gender_no_nulls(self):
        self.cleaner.clean_data()
        self.assertEqual(self.cleaner.df_cleaned['Gender'].isna().sum(), 0)

    def test_clean_smoker_standardised(self):
        self.cleaner.clean_data()
        valid = {'YES', 'NO'}
        for val in self.cleaner.df_cleaned['Smoker']:
            self.assertIn(val, valid)

    def test_clean_diagnosis_unknown_for_missing(self):
        self.cleaner.clean_data()
        self.assertNotIn(np.nan, self.cleaner.df_cleaned['Diagnosis'].values)
        # P02 had no diagnosis — should now be UNKNOWN
        p02_diag = self.cleaner.df_cleaned.loc[
            self.cleaner.df['Patient_ID'] == 'P02', 'Diagnosis'
        ].iloc[0]
        self.assertEqual(p02_diag, 'UNKNOWN')

    def test_clean_notes_na_for_empty_and_missing(self):
        self.cleaner.clean_data()
        for val in self.cleaner.df_cleaned['Notes']:
            self.assertNotEqual(val, '')
            self.assertFalse(pd.isna(val))

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #

    def test_get_summary_keys_present(self):
        self.cleaner.clean_data()
        summary = self.cleaner.get_summary()
        for key in ['original_shape', 'cleaned_shape', 'original_missing', 'cleaned_missing']:
            self.assertIn(key, summary)

    def test_get_summary_cleaned_missing_is_zero(self):
        self.cleaner.clean_data()
        summary = self.cleaner.get_summary()
        self.assertEqual(summary['cleaned_missing'], 0)

    # ------------------------------------------------------------------ #
    # Save                                                                #
    # ------------------------------------------------------------------ #

    def test_save_creates_file(self):
        self.cleaner.clean_data()
        out_path = os.path.join(self.tmp, 'output.csv')
        self.cleaner.save_cleaned_data(out_path)
        self.assertTrue(os.path.isfile(out_path))

    def test_save_file_is_valid_csv(self):
        self.cleaner.clean_data()
        out_path = os.path.join(self.tmp, 'output.csv')
        self.cleaner.save_cleaned_data(out_path)
        loaded = pd.read_csv(out_path)
        self.assertEqual(len(loaded), len(self.cleaner.df_cleaned))


if __name__ == '__main__':
    unittest.main()

import pandas as pd
import numpy as np
import os
import glob
from typing import Optional


# A class to clean medical datasets with missing values using appropriate
# imputation strategies based on the data type and distribution.
class MedicalDataCleaner:
    
    # Initialize the cleaner with a CSV file path.
    # Args:
    #     filepath: Path to the CSV file to clean
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.df_cleaned = None
    
    # Load the CSV file into a pandas DataFrame.
    # Returns:
    #     Loaded DataFrame
    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded dataset with {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df
    
    # Analyze and report missing data in the dataset.
    # Returns:
    #     DataFrame with missing data statistics
    def analyze_missing_data(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        missing_stats = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': [self.df[col].isna().sum() for col in self.df.columns],
            'Missing_Percentage': [self.df[col].isna().sum() / len(self.df) * 100 
                                  for col in self.df.columns]
        })
        
        print("\nMissing Data Analysis:")
        print(missing_stats[missing_stats['Missing_Count'] > 0])
        return missing_stats
    
    # Check for and report duplicate rows in the dataset.
    # Returns:
    #     DataFrame containing the duplicate rows
    def check_duplicates(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        duplicates = self.df[self.df.duplicated(keep=False)]
        num_duplicates = self.df.duplicated().sum()
        
        print(f"\nDuplicate Analysis:")
        
        if num_duplicates > 0:
            print(f"Total duplicate rows found: {num_duplicates}")
            print(f"Duplicate rows (showing all occurrences):")
            print(duplicates.sort_values(by=list(self.df.columns)))
        else:
            print("No duplicate rows found.")
        
        return duplicates
    
    # Remove duplicate rows from the dataset.
    # Args:
    #     subset: Optional list of column names to consider for identifying duplicates.
    #             If None, all columns are used.
    #     keep: Which duplicates to keep ('first', 'last', or False to remove all).
    #           Default is 'first'.
    # Returns:
    #     DataFrame with duplicates removed
    def remove_duplicates(self, subset=None, keep='first') -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        duplicates_before = self.df.duplicated(subset=subset).sum()
        
        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        
        print(f"\nRemoved {duplicates_before} duplicate rows")
        print(f"Dataset now has {len(self.df)} rows")
        
        return self.df
    
    # Clean the dataset by applying appropriate imputation strategies
    # for each column based on its characteristics.
    # Returns:
    #     Cleaned DataFrame
    def clean_data(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create a copy to avoid modifying original data
        self.df_cleaned = self.df.copy()
        
        # Replace string 'nan' values with actual NaN
        self.df_cleaned = self.df_cleaned.replace('nan', np.nan)
        self.df_cleaned = self.df_cleaned.replace('', np.nan)
        
        print("\nApplying cleaning strategies...")
        
        # Clean each column with appropriate strategy
        self._clean_age()
        self._clean_gender()
        self._clean_blood_pressure()
        self._clean_cholesterol()
        self._clean_bmi()
        self._clean_smoker()
        self._clean_diagnosis()
        self._clean_notes()
        
        print("\nData cleaning completed!")
        return self.df_cleaned
    
    # Clean Age column using Median Imputation.
    # Justification: Age is a numerical variable that may have outliers.
    # Median is robust to outliers and represents a typical patient age
    # better than mean in medical datasets where age distribution may be skewed.
    def _clean_age(self):
        missing_before = self.df_cleaned['Age'].isna().sum()
        median_age = self.df_cleaned['Age'].median()
        self.df_cleaned['Age'] = self.df_cleaned['Age'].fillna(median_age)
        print(f"Age: Filled {missing_before} missing values with median ({median_age})")
    
    # Clean Gender column using Mode Imputation.
    # Justification: Gender is a categorical variable. Using the mode
    # (most frequent value) maintains the existing distribution and is
    # appropriate for categorical data. NaN values (including string 'nan')
    # are replaced with the most common gender value. All text is normalized
    # to uppercase for consistency.
    def _clean_gender(self):
        missing_before = self.df_cleaned['Gender'].isna().sum()
        
        # Standardize gender format to uppercase
        self.df_cleaned['Gender'] = self.df_cleaned['Gender'].str.upper()
        
        # Get mode and fill missing values (NaN values are filled with most common gender)
        mode_gender = self.df_cleaned['Gender'].mode()[0]
        self.df_cleaned['Gender'] = self.df_cleaned['Gender'].fillna(mode_gender)
        print(f"Gender: Filled {missing_before} missing values with mode ({mode_gender})")
    
    # Clean Blood_Pressure column using Mean Imputation.
    # Justification: Blood pressure is a continuous physiological variable
    # that typically follows a normal distribution. Mean imputation is
    # appropriate as it preserves the central tendency and is suitable
    # for normally distributed continuous variables.
    def _clean_blood_pressure(self):
        missing_before = self.df_cleaned['Blood_Pressure'].isna().sum()
        mean_bp = self.df_cleaned['Blood_Pressure'].mean()
        self.df_cleaned['Blood_Pressure'] = self.df_cleaned['Blood_Pressure'].fillna(mean_bp)
        print(f"Blood_Pressure: Filled {missing_before} missing values with mean ({mean_bp:.2f})")
    
    # Clean Cholesterol column using Median Imputation.
    # Justification: Cholesterol levels can have outliers and may be
    # skewed towards higher values in patients with conditions.
    # Median is more robust to outliers than mean and better represents
    # the typical cholesterol level in the dataset.
    def _clean_cholesterol(self):
        missing_before = self.df_cleaned['Cholesterol'].isna().sum()
        median_chol = self.df_cleaned['Cholesterol'].median()
        self.df_cleaned['Cholesterol'] = self.df_cleaned['Cholesterol'].fillna(median_chol)
        print(f"Cholesterol: Filled {missing_before} missing values with median ({median_chol})")
    
    # Clean BMI column using Mean Imputation.
    # Justification: BMI is a calculated continuous variable that typically
    # follows a relatively normal distribution. Mean imputation is appropriate
    # as it maintains the average body mass index across the patient population.
    def _clean_bmi(self):
        missing_before = self.df_cleaned['BMI'].isna().sum()
        mean_bmi = self.df_cleaned['BMI'].mean()
        self.df_cleaned['BMI'] = self.df_cleaned['BMI'].fillna(mean_bmi)
        print(f"BMI: Filled {missing_before} missing values with mean ({mean_bmi:.2f})")
    
    # Clean Smoker column using Mode Imputation.
    # Justification: Smoker is a categorical/binary variable (YES/NO).
    # Mode imputation maintains the most common smoking status in the
    # dataset. We standardize Y/N variants to YES/NO in uppercase.
    def _clean_smoker(self):
        missing_before = self.df_cleaned['Smoker'].isna().sum()
        
        # Standardize smoker values: Y/N to YES/NO (uppercase)
        smoker_map = {'Y': 'YES', 'N': 'NO', 'y': 'YES', 'n': 'NO', 
                      'Yes': 'YES', 'No': 'NO', 'yes': 'YES', 'no': 'NO'}
        self.df_cleaned['Smoker'] = self.df_cleaned['Smoker'].replace(smoker_map)
        
        # Get mode and fill missing values
        mode_smoker = self.df_cleaned['Smoker'].mode()[0]
        self.df_cleaned['Smoker'] = self.df_cleaned['Smoker'].fillna(mode_smoker)
        print(f"Smoker: Filled {missing_before} missing values with mode ({mode_smoker})")
    
    # Clean Diagnosis column using Constant Value Imputation.
    # Justification: Missing diagnosis could mean the patient hasn't been
    # diagnosed with any specific condition yet. Using 'UNKNOWN' as a constant
    # value is more appropriate than using mode, as it explicitly indicates
    # the absence of diagnosis information rather than assuming a specific condition.
    # All text is normalized to uppercase for consistency.
    def _clean_diagnosis(self):
        missing_before = self.df_cleaned['Diagnosis'].isna().sum()
        
        # Normalize existing diagnoses to uppercase
        self.df_cleaned['Diagnosis'] = self.df_cleaned['Diagnosis'].str.upper()
        
        # Fill missing values with UNKNOWN
        self.df_cleaned['Diagnosis'] = self.df_cleaned['Diagnosis'].fillna('UNKNOWN')
        print(f"Diagnosis: Filled {missing_before} missing values with constant value ('UNKNOWN')")
    
    # Clean Notes column using Constant Value Imputation.
    # Justification: Empty notes simply indicate no additional information
    # was recorded. Replacing with 'N/A' (Not Applicable) is the most
    # appropriate strategy as it clearly indicates the absence of notes
    # without implying any medical information. All text is normalized
    # to uppercase for consistency.
    def _clean_notes(self):
        missing_before = self.df_cleaned['Notes'].isna().sum()
        
        # First, replace both NaN and empty strings with 'N/A' before normalization
        self.df_cleaned['Notes'] = self.df_cleaned['Notes'].fillna('N/A')
        self.df_cleaned['Notes'] = self.df_cleaned['Notes'].replace('', 'N/A')
        
        # Then normalize all notes to uppercase (including the N/A values)
        self.df_cleaned['Notes'] = self.df_cleaned['Notes'].str.upper()
        
        print(f"Notes: Filled {missing_before} missing values with 'N/A')")
    
    # Save the cleaned dataset to a CSV file with auto-incrementing number.
    # Args:
    #     output_path: Optional custom output path. If None, uses auto-incrementing naming.
    # Returns:
    #     Path to the saved file
    def save_cleaned_data(self, output_path: Optional[str] = None) -> str:
        if self.df_cleaned is None:
            raise ValueError("No cleaned data available. Call clean_data() first.")
        
        if output_path is None:
            # Generate auto-incrementing filename
            base_path = os.path.dirname(self.filepath)
            base_name = os.path.basename(self.filepath).replace('.csv', '')
            
            # Find existing cleaned files with numbering
            pattern = os.path.join(base_path, f"{base_name}_cleaned_*.csv")
            existing_files = glob.glob(pattern)
            
            # Determine next number
            if not existing_files:
                next_num = 1
            else:
                # Extract numbers from existing files
                numbers = []
                for f in existing_files:
                    try:
                        num = int(f.split('_cleaned_')[1].replace('.csv', ''))
                        numbers.append(num)
                    except (IndexError, ValueError):
                        continue
                next_num = max(numbers) + 1 if numbers else 1
            
            output_path = os.path.join(base_path, f"{base_name}_cleaned_{next_num}.csv")
        
        self.df_cleaned.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
        return output_path
    
    # Get a summary comparison of the original and cleaned datasets.
    # Returns:
    #     Dictionary with summary statistics
    def get_summary(self) -> dict:
        if self.df_cleaned is None:
            raise ValueError("No cleaned data available. Call clean_data() first.")
        
        summary = {
            'original_shape': self.df.shape,
            'cleaned_shape': self.df_cleaned.shape,
            'original_missing': self.df.isna().sum().sum(),
            'cleaned_missing': self.df_cleaned.isna().sum().sum(),
            'missing_by_column_before': self.df.isna().sum().to_dict(),
            'missing_by_column_after': self.df_cleaned.isna().sum().to_dict()
        }
        
        print("\n" + "="*60)
        print("CLEANING SUMMARY")
        print("="*60)
        print(f"Original dataset: {summary['original_shape'][0]} rows × {summary['original_shape'][1]} columns")
        print(f"Total missing values before: {summary['original_missing']}")
        print(f"Total missing values after: {summary['cleaned_missing']}")
        print("="*60)
        
        return summary


# Main function to demonstrate the usage of MedicalDataCleaner class.
def main():
    # Initialize the cleaner
    cleaner = MedicalDataCleaner('/home/daniel/UNIVERSITY/COM618/COM618/ASSESSMENT/realworld_medical_dirty.csv')
    
    # Load the data
    cleaner.load_data()
    
    # Check for duplicates
    duplicates = cleaner.check_duplicates()
    
    # Remove duplicates if any exist
    if len(duplicates) > 0:
        print(f"\n{len(duplicates)} duplicate rows detected. Proceeding with removal...")
        cleaner.remove_duplicates()
    else:
        print("\nNo duplicates to remove.")
    
    # Analyze missing data
    cleaner.analyze_missing_data()
    
    # Clean the data
    cleaner.clean_data()
    
    # Save the cleaned data
    cleaner.save_cleaned_data()
    
    # Get and display summary
    cleaner.get_summary()
    
    # Display first few rows of cleaned data
    print("\nFirst 5 rows of cleaned data:")
    print(cleaner.df_cleaned.head())


if __name__ == "__main__":
    main()

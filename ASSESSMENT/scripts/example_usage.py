"""
Example usage of the MedicalDataCleaner class

This script demonstrates how to use the MedicalDataCleaner to clean
a medical dataset with missing values.
"""

from medical_data_cleaner import MedicalDataCleaner


def example_usage():
    """Demonstrate basic usage of the cleaner."""
    
    # Path to your dirty dataset
    input_file = '/home/daniel/UNIVERSITY/COM618/COM618/ASSESSMENT/realworld_medical_dirty.csv'
    
    # Create cleaner instance
    cleaner = MedicalDataCleaner(input_file)
    
    # Step 1: Load the data
    print("Step 1: Loading data...")
    cleaner.load_data()
    
    # Step 2: Analyze missing data (optional but recommended)
    print("\nStep 2: Analyzing missing data...")
    cleaner.analyze_missing_data()
    
    # Step 3: Clean the data
    print("\nStep 3: Cleaning data...")
    cleaned_df = cleaner.clean_data()
    
    # Step 4: Save the cleaned data
    print("\nStep 4: Saving cleaned data...")
    output_file = cleaner.save_cleaned_data()
    
    # Step 5: View summary
    print("\nStep 5: Viewing summary...")
    summary = cleaner.get_summary()
    
    # Optional: Display sample of cleaned data
    print("\n" + "="*60)
    print("SAMPLE OF CLEANED DATA")
    print("="*60)
    print(cleaned_df.head(10))
    
    return cleaned_df


if __name__ == "__main__":
    example_usage()

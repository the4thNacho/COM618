import os
import glob
from medical_data_cleaner import MedicalDataCleaner
from visualize_cleaning import DataCleaningVisualizer


# Main orchestrator script that cleans data and generates visualizations
def main():
    print("="*60)
    print("MEDICAL DATA CLEANING AND VISUALIZATION PIPELINE")
    print("="*60)
    
    # Define input file path
    input_file = '/home/daniel/UNIVERSITY/COM618/COM618/ASSESSMENT/realworld_medical_dirty.csv'
    
    # Step 1: Clean the data
    print("\n[STEP 1] DATA CLEANING")
    print("-"*60)
    
    cleaner = MedicalDataCleaner(input_file)
    cleaner.load_data()
    
    # Check for duplicates
    duplicates = cleaner.check_duplicates()
    if len(duplicates) > 0:
        print(f"\n{len(duplicates)} duplicate rows detected. Proceeding with removal...")
        cleaner.remove_duplicates()
    else:
        print("\nNo duplicates to remove.")
    
    # Analyze missing data
    cleaner.analyze_missing_data()
    
    # Clean the data
    cleaner.clean_data()
    
    # Save cleaned data (auto-increments filename)
    cleaned_file_path = cleaner.save_cleaned_data()
    
    # Get summary
    cleaner.get_summary()
    
    # Step 2: Find the most recently created cleaned CSV
    print("\n[STEP 2] LOCATING CLEANED FILE")
    print("-"*60)
    
    base_path = os.path.dirname(input_file)
    pattern = os.path.join(base_path, 'realworld_medical_dirty_cleaned_*.csv')
    cleaned_files = glob.glob(pattern)
    
    if not cleaned_files:
        print("Error: No cleaned files found!")
        return
    
    # Get the most recently modified file
    most_recent_cleaned = max(cleaned_files, key=os.path.getmtime)
    print(f"Most recent cleaned file: {most_recent_cleaned}")
    
    # Step 3: Generate visualizations
    print("\n[STEP 3] GENERATING VISUALIZATIONS")
    print("-"*60)
    
    visualizer = DataCleaningVisualizer(input_file, most_recent_cleaned)
    visualizer.load_data()
    visualizer.generate_all_visualizations()
    
    # Step 4: Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nCleaned data saved to: {most_recent_cleaned}")
    print(f"Visualizations saved to: {base_path}/")
    print("\nGenerated visualizations:")
    print("  - missing_data_comparison.png")
    print("  - missing_percentage_comparison.png")
    print("  - numerical_distributions.png")
    print("  - boxplot_comparison.png")
    print("  - categorical_distributions.png")
    print("  - summary_dashboard.png")
    print("="*60)


if __name__ == "__main__":
    main()

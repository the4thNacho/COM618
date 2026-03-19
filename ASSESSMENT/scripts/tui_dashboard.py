#!/usr/bin/env python3
"""
Terminal UI for displaying data cleaning dashboards in Kitty terminal.
Uses Kitty's icat kitten to display PNG images directly in the terminal.
Also provides functionality to run the data cleaning pipeline.
"""

import os
import subprocess
import sys
import glob
from medical_data_cleaner import MedicalDataCleaner
from visualize_cleaning import DataCleaningVisualizer


# Display an image in Kitty terminal using icat kitten
# Args:
#     image_path: Full path to the PNG image file
#     title: Title to display above the image
def display_image(image_path: str, title: str = None):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return False
    
    if title:
        print(f"\n{'=' * 80}")
        print(f"{title:^80}")
        print(f"{'=' * 80}\n")
    
    try:
        # Use Kitty's icat kitten to display the image
        subprocess.run(['kitty', '+kitten', 'icat', image_path], check=True)
        print()  # Add spacing after image
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error displaying image: {e}")
        return False
    except FileNotFoundError:
        print("Error: Kitty terminal or icat kitten not found.")
        print("This script requires Kitty terminal with icat support.")
        return False


# Find the most recent cleaned CSV file in the assessment directory
def find_latest_cleaned_csv(base_dir: str) -> str:
    pattern = os.path.join(base_dir, 'realworld_medical_dirty_cleaned_*.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# Clear the terminal screen
def clear_screen():
    os.system('clear')


# Display the main menu and get user choice
def show_menu():
    print("\n" + "=" * 80)
    print("DATA CLEANING DASHBOARD VIEWER".center(80))
    print("=" * 80)
    print("\n0. Run Data Cleaning Pipeline")
    print("1. View Summary Dashboard")
    print("2. View Missing Data Comparison")
    print("3. View Missing Percentage Comparison")
    print("4. View Numerical Distributions")
    print("5. View Boxplot Comparison")
    print("6. View Categorical Distributions")
    print("7. View All Visualizations")
    print("8. Clear Screen")
    print("9. Exit")
    print("\n" + "=" * 80)
    
    choice = input("\nEnter your choice (0-9): ").strip()
    return choice


# Run the complete data cleaning pipeline
def run_cleaning_pipeline(base_dir: str):
    print("\n" + "=" * 80)
    print("RUNNING DATA CLEANING PIPELINE".center(80))
    print("=" * 80 + "\n")
    
    # Path to the original dirty CSV
    original_csv = os.path.join(base_dir, 'realworld_medical_dirty.csv')
    
    if not os.path.exists(original_csv):
        print(f"Error: Original CSV not found at {original_csv}")
        return False
    
    try:
        # Initialize cleaner
        print(f"Loading data from: {os.path.basename(original_csv)}")
        cleaner = MedicalDataCleaner(original_csv)
        
        # Load and analyze data
        cleaner.load_data()
        print("\nAnalyzing missing data...")
        cleaner.analyze_missing_data()
        
        # Check for duplicates
        print("\nChecking for duplicates...")
        duplicates = cleaner.check_duplicates()
        if len(duplicates) > 0:
            print(f"Found {len(duplicates)} duplicate rows")
            response = input("Remove duplicates? (y/n): ").strip().lower()
            if response == 'y':
                cleaner.remove_duplicates()
        
        # Clean the data
        print("\nCleaning data...")
        cleaner.clean_data()
        
        # Save cleaned data
        print("\nSaving cleaned data...")
        output_path = cleaner.save_cleaned_data()
        print(f"Cleaned data saved to: {os.path.basename(output_path)}")
        
        # Generate visualizations
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS".center(80))
        print("=" * 80 + "\n")
        
        visualizer = DataCleaningVisualizer(original_csv, output_path)
        visualizer.load_data()
        
        print("Creating visualizations...")
        visualizer.visualize_missing_data()
        visualizer.visualize_missing_percentage()
        visualizer.visualize_numerical_distributions()
        visualizer.visualize_boxplots()
        visualizer.visualize_categorical_distributions()
        visualizer.create_summary_dashboard()
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!".center(80))
        print("=" * 80)
        
        input("\nPress Enter to return to menu...")
        return True
        
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to return to menu...")
        return False


# Main TUI loop
def main():
    # Get the base directory (parent of scripts folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    visuals_dir = os.path.join(base_dir, 'cleaning_visuals')
    dashboard_dir = os.path.join(visuals_dir, 'dashboard')
    
    # Show initial status
    clear_screen()
    print("\n" + "=" * 80)
    print("MEDICAL DATA CLEANING DASHBOARD".center(80))
    print("=" * 80)
    
    # Check if visualizations exist
    if not os.path.exists(visuals_dir):
        print("\nNo visualizations found.")
        print("You can run the cleaning pipeline from the menu (Option 0).")
    else:
        # Find latest cleaned CSV
        latest_csv = find_latest_cleaned_csv(base_dir)
        if latest_csv:
            csv_name = os.path.basename(latest_csv)
            print(f"\nMost recent cleaned data: {csv_name}")
    
    input("\nPress Enter to continue to menu...")
    
    while True:
        choice = show_menu()
        
        if choice == '0':
            clear_screen()
            run_cleaning_pipeline(base_dir)
            clear_screen()
        
        elif choice == '1':
            clear_screen()
            image_path = os.path.join(dashboard_dir, 'summary_dashboard.png')
            display_image(image_path, "SUMMARY DASHBOARD")
        
        elif choice == '2':
            clear_screen()
            image_path = os.path.join(visuals_dir, 'missing_data_comparison.png')
            display_image(image_path, "MISSING DATA COMPARISON")
        
        elif choice == '3':
            clear_screen()
            image_path = os.path.join(visuals_dir, 'missing_percentage_comparison.png')
            display_image(image_path, "MISSING PERCENTAGE COMPARISON")
        
        elif choice == '4':
            clear_screen()
            image_path = os.path.join(visuals_dir, 'numerical_distributions.png')
            display_image(image_path, "NUMERICAL DISTRIBUTIONS")
        
        elif choice == '5':
            clear_screen()
            image_path = os.path.join(visuals_dir, 'boxplot_comparison.png')
            display_image(image_path, "BOXPLOT COMPARISON")
        
        elif choice == '6':
            clear_screen()
            image_path = os.path.join(visuals_dir, 'categorical_distributions.png')
            display_image(image_path, "CATEGORICAL DISTRIBUTIONS")
        
        elif choice == '7':
            clear_screen()
            images = [
                (os.path.join(dashboard_dir, 'summary_dashboard.png'), "SUMMARY DASHBOARD"),
                (os.path.join(visuals_dir, 'missing_data_comparison.png'), "MISSING DATA COMPARISON"),
                (os.path.join(visuals_dir, 'missing_percentage_comparison.png'), "MISSING PERCENTAGE COMPARISON"),
                (os.path.join(visuals_dir, 'numerical_distributions.png'), "NUMERICAL DISTRIBUTIONS"),
                (os.path.join(visuals_dir, 'boxplot_comparison.png'), "BOXPLOT COMPARISON"),
                (os.path.join(visuals_dir, 'categorical_distributions.png'), "CATEGORICAL DISTRIBUTIONS")
            ]
            for img_path, title in images:
                display_image(img_path, title)
                input("Press Enter to continue...")
                clear_screen()
        
        elif choice == '8':
            clear_screen()
        
        elif choice == '9':
            print("\nExiting dashboard viewer. Goodbye!")
            break
        
        else:
            print("\nInvalid choice. Please enter a number between 0 and 9.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    # Check if running in Kitty terminal
    if 'KITTY_WINDOW_ID' not in os.environ:
        print("Warning: This script is optimized for Kitty terminal.")
        print("Image display may not work in other terminals.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(0)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)

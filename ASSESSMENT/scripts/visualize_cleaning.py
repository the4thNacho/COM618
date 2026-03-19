import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# A class to create visualizations comparing original and cleaned datasets
class DataCleaningVisualizer:
    
    # Initialize the visualizer with file paths to original and cleaned datasets.
    # Args:
    #     original_path: Path to the original dirty CSV file
    #     cleaned_path: Path to the cleaned CSV file
    def __init__(self, original_path: str, cleaned_path: str):
        self.original_path = original_path
        self.cleaned_path = cleaned_path
        self.df_original = None
        self.df_cleaned = None
    
    # Load both original and cleaned datasets.
    # Note: keep_default_na=False prevents pandas from treating 'N/A' as NaN
    def load_data(self):
        self.df_original = pd.read_csv(self.original_path)
        self.df_cleaned = pd.read_csv(self.cleaned_path, keep_default_na=False, na_values=[''])
        print(f"Loaded original dataset: {len(self.df_original)} rows")
        print(f"Loaded cleaned dataset: {len(self.df_cleaned)} rows")
    
    # Create a comparison visualization of missing data before and after cleaning.
    def visualize_missing_data(self):
        if self.df_original is None or self.df_cleaned is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Calculate missing values
        missing_original = self.df_original.isna().sum()
        missing_cleaned = self.df_cleaned.isna().sum()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original dataset missing values
        columns = missing_original.index
        ax1.barh(columns, missing_original.values, color='#e74c3c')
        ax1.set_xlabel('Number of Missing Values')
        ax1.set_title('Missing Values - Original Dataset')
        ax1.grid(axis='x', alpha=0.3)
        
        # Cleaned dataset missing values
        ax2.barh(columns, missing_cleaned.values, color='#27ae60')
        ax2.set_xlabel('Number of Missing Values')
        ax2.set_title('Missing Values - Cleaned Dataset')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        visuals_dir = os.path.join(os.path.dirname(self.cleaned_path), 'cleaning_visuals')
        os.makedirs(visuals_dir, exist_ok=True)
        output_path = os.path.join(visuals_dir, 'missing_data_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {output_path}")
        plt.close()
    
    # Create a side-by-side comparison of missing data percentages.
    def visualize_missing_percentage(self):
        if self.df_original is None or self.df_cleaned is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Calculate missing percentages
        missing_pct_original = (self.df_original.isna().sum() / len(self.df_original) * 100)
        missing_pct_cleaned = (self.df_cleaned.isna().sum() / len(self.df_cleaned) * 100)
        
        # Create grouped bar chart
        columns = missing_pct_original.index
        x = np.arange(len(columns))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, missing_pct_original.values, width, label='Original', color='#e74c3c')
        bars2 = ax.bar(x + width/2, missing_pct_cleaned.values, width, label='Cleaned', color='#27ae60')
        
        ax.set_xlabel('Columns')
        ax.set_ylabel('Missing Data (%)')
        ax.set_title('Missing Data Percentage Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(columns, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        visuals_dir = os.path.join(os.path.dirname(self.cleaned_path), 'cleaning_visuals')
        os.makedirs(visuals_dir, exist_ok=True)
        output_path = os.path.join(visuals_dir, 'missing_percentage_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    # Create distribution comparisons for numerical columns.
    # Uses side-by-side subplots for clearer comparison.
    def visualize_numerical_distributions(self):
        if self.df_original is None or self.df_cleaned is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Identify numerical columns
        numerical_cols = ['Age', 'Blood_Pressure', 'Cholesterol', 'BMI']
        
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        
        for idx, col in enumerate(numerical_cols):
            # Original data on left
            ax_orig = axes[idx, 0]
            ax_orig.hist(self.df_original[col].dropna(), bins=20, 
                        color='#e74c3c', edgecolor='black', alpha=0.7)
            ax_orig.set_xlabel(col)
            ax_orig.set_ylabel('Frequency')
            ax_orig.set_title(f'{col} - Original Dataset')
            ax_orig.grid(alpha=0.3)
            
            # Cleaned data on right
            ax_clean = axes[idx, 1]
            ax_clean.hist(self.df_cleaned[col].dropna(), bins=20, 
                         color='#27ae60', edgecolor='black', alpha=0.7)
            ax_clean.set_xlabel(col)
            ax_clean.set_ylabel('Frequency')
            ax_clean.set_title(f'{col} - Cleaned Dataset')
            ax_clean.grid(alpha=0.3)
        
        plt.tight_layout()
        visuals_dir = os.path.join(os.path.dirname(self.cleaned_path), 'cleaning_visuals')
        os.makedirs(visuals_dir, exist_ok=True)
        output_path = os.path.join(visuals_dir, 'numerical_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    # Create box plots for numerical columns to show outliers and distributions.
    def visualize_boxplots(self):
        if self.df_original is None or self.df_cleaned is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        numerical_cols = ['Age', 'Blood_Pressure', 'Cholesterol', 'BMI']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols):
            ax = axes[idx]
            
            # Prepare data for box plots
            data_to_plot = [
                self.df_original[col].dropna(),
                self.df_cleaned[col].dropna()
            ]
            
            bp = ax.boxplot(data_to_plot, labels=['Original', 'Cleaned'],
                           patch_artist=True)
            
            # Color the boxes
            bp['boxes'][0].set_facecolor('#e74c3c')
            bp['boxes'][1].set_facecolor('#27ae60')
            
            ax.set_ylabel(col)
            ax.set_title(f'{col} - Box Plot Comparison')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        visuals_dir = os.path.join(os.path.dirname(self.cleaned_path), 'cleaning_visuals')
        os.makedirs(visuals_dir, exist_ok=True)
        output_path = os.path.join(visuals_dir, 'boxplot_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    # Create categorical data distribution comparisons.
    def visualize_categorical_distributions(self):
        if self.df_original is None or self.df_cleaned is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        categorical_cols = ['Gender', 'Smoker', 'Diagnosis']
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        for idx, col in enumerate(categorical_cols):
            ax = axes[idx]
            
            # Get value counts
            original_counts = self.df_original[col].value_counts()
            cleaned_counts = self.df_cleaned[col].value_counts()
            
            # Combine all categories
            all_categories = sorted(set(original_counts.index) | set(cleaned_counts.index))
            
            x = np.arange(len(all_categories))
            width = 0.35
            
            original_values = [original_counts.get(cat, 0) for cat in all_categories]
            cleaned_values = [cleaned_counts.get(cat, 0) for cat in all_categories]
            
            ax.bar(x - width/2, original_values, width, label='Original', 
                  color='#e74c3c', alpha=0.8)
            ax.bar(x + width/2, cleaned_values, width, label='Cleaned', 
                  color='#27ae60', alpha=0.8)
            
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.set_title(f'{col} Distribution')
            ax.set_xticks(x)
            ax.set_xticklabels(all_categories, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        visuals_dir = os.path.join(os.path.dirname(self.cleaned_path), 'cleaning_visuals')
        os.makedirs(visuals_dir, exist_ok=True)
        output_path = os.path.join(visuals_dir, 'categorical_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    # Create a comprehensive summary dashboard.
    def create_summary_dashboard(self):
        if self.df_original is None or self.df_cleaned is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 4, hspace=0.6, wspace=0.35, top=0.92, bottom=0.05, height_ratios=[1, 0.65, 1, 1])
        
        # 1. Total missing values comparison
        ax1 = fig.add_subplot(gs[0, 0])
        total_missing_orig = self.df_original.isna().sum().sum()
        total_missing_clean = self.df_cleaned.isna().sum().sum()
        ax1.bar(['Original', 'Cleaned'], [total_missing_orig, total_missing_clean],
               color=['#e74c3c', '#27ae60'])
        ax1.set_ylabel('Total Missing Values')
        ax1.set_title('Total Missing Data')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Data completeness percentage
        ax2 = fig.add_subplot(gs[0, 1])
        completeness_orig = ((1 - self.df_original.isna().sum().sum() / 
                             (len(self.df_original) * len(self.df_original.columns))) * 100)
        completeness_clean = ((1 - self.df_cleaned.isna().sum().sum() / 
                              (len(self.df_cleaned) * len(self.df_cleaned.columns))) * 100)
        ax2.bar(['Original', 'Cleaned'], [completeness_orig, completeness_clean],
               color=['#e74c3c', '#27ae60'])
        ax2.set_ylabel('Completeness (%)')
        ax2.set_title('Data Completeness')
        ax2.set_ylim([0, 100])
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Row count comparison
        ax3 = fig.add_subplot(gs[0, 2:])
        ax3.bar(['Original', 'Cleaned'], [len(self.df_original), len(self.df_cleaned)],
               color=['#e74c3c', '#27ae60'])
        ax3.set_ylabel('Number of Rows')
        ax3.set_title('Dataset Size')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Missing values by column
        ax4 = fig.add_subplot(gs[1, :])
        missing_orig = self.df_original.isna().sum()
        missing_clean = self.df_cleaned.isna().sum()
        x = np.arange(len(missing_orig))
        width = 0.35
        ax4.bar(x - width/2, missing_orig.values, width, label='Original', color='#e74c3c')
        ax4.bar(x + width/2, missing_clean.values, width, label='Cleaned', color='#27ae60')
        ax4.set_ylabel('Missing Values')
        ax4.set_title('Missing Values by Column')
        ax4.set_xticks(x)
        ax4.set_xticklabels(missing_orig.index, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Age distribution - side by side
        ax5_orig = fig.add_subplot(gs[2, 0])
        ax5_orig.hist(self.df_original['Age'].dropna(), bins=15, 
                     color='#e74c3c', edgecolor='black', alpha=0.7)
        ax5_orig.set_xlabel('Age')
        ax5_orig.set_ylabel('Frequency')
        ax5_orig.set_title('Age - Original')
        ax5_orig.grid(alpha=0.3)
        
        ax5_clean = fig.add_subplot(gs[2, 1])
        ax5_clean.hist(self.df_cleaned['Age'].dropna(), bins=15, 
                      color='#27ae60', edgecolor='black', alpha=0.7)
        ax5_clean.set_xlabel('Age')
        ax5_clean.set_ylabel('Frequency')
        ax5_clean.set_title('Age - Cleaned')
        ax5_clean.grid(alpha=0.3)
        
        # 6. BMI distribution - side by side
        ax6_orig = fig.add_subplot(gs[2, 2])
        ax6_orig.hist(self.df_original['BMI'].dropna(), bins=15, 
                     color='#e74c3c', edgecolor='black', alpha=0.7)
        ax6_orig.set_xlabel('BMI')
        ax6_orig.set_ylabel('Frequency')
        ax6_orig.set_title('BMI - Original')
        ax6_orig.grid(alpha=0.3)
        
        ax6_clean = fig.add_subplot(gs[2, 3])
        ax6_clean.hist(self.df_cleaned['BMI'].dropna(), bins=15, 
                      color='#27ae60', edgecolor='black', alpha=0.7)
        ax6_clean.set_xlabel('BMI')
        ax6_clean.set_ylabel('Frequency')
        ax6_clean.set_title('BMI - Cleaned')
        ax6_clean.grid(alpha=0.3)
        
        # 7. Gender distribution - side by side
        ax7_orig = fig.add_subplot(gs[3, :2])
        gender_orig = self.df_original['Gender'].value_counts()
        all_genders = sorted(set(gender_orig.index) | (set(self.df_cleaned['Gender'].value_counts().index) if 'Gender' in self.df_cleaned else set()))
        x_pos = np.arange(len(all_genders))
        orig_vals = [gender_orig.get(g, 0) for g in all_genders]
        ax7_orig.bar(x_pos, orig_vals, color='#e74c3c')
        ax7_orig.set_xlabel('Gender')
        ax7_orig.set_ylabel('Count')
        ax7_orig.set_title('Gender Distribution - Original')
        ax7_orig.set_xticks(x_pos)
        ax7_orig.set_xticklabels(all_genders, rotation=45, ha='right')
        ax7_orig.grid(axis='y', alpha=0.3)
        
        ax7_clean = fig.add_subplot(gs[3, 2:])
        gender_clean = self.df_cleaned['Gender'].value_counts()
        clean_vals = [gender_clean.get(g, 0) for g in all_genders]
        ax7_clean.bar(x_pos, clean_vals, color='#27ae60')
        ax7_clean.set_xlabel('Gender')
        ax7_clean.set_ylabel('Count')
        ax7_clean.set_title('Gender Distribution - Cleaned')
        ax7_clean.set_xticks(x_pos)
        ax7_clean.set_xticklabels(all_genders, rotation=45, ha='right')
        ax7_clean.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Data Cleaning Summary Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        dashboard_dir = os.path.join(os.path.dirname(self.cleaned_path), 'cleaning_visuals', 'dashboard')
        os.makedirs(dashboard_dir, exist_ok=True)
        output_path = os.path.join(dashboard_dir, 'summary_dashboard.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    # Generate all visualizations at once.
    def generate_all_visualizations(self):
        print("\nGenerating visualizations...")
        print("="*60)
        
        self.visualize_missing_data()
        self.visualize_missing_percentage()
        self.visualize_numerical_distributions()
        self.visualize_boxplots()
        self.visualize_categorical_distributions()
        self.create_summary_dashboard()
        
        print("="*60)
        print("All visualizations generated successfully!")


# Main function to demonstrate usage
def main():
    # Define file paths
    original_file = '/home/daniel/UNIVERSITY/COM618/COM618/ASSESSMENT/realworld_medical_dirty.csv'
    cleaned_file = '/home/daniel/UNIVERSITY/COM618/COM618/ASSESSMENT/realworld_medical_dirty_cleaned_1.csv'
    
    # Create visualizer
    visualizer = DataCleaningVisualizer(original_file, cleaned_file)
    
    # Load data
    visualizer.load_data()
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()

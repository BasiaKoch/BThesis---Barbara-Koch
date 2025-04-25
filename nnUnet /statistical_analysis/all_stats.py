import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from pathlib import Path

# Set seaborn style for better visualization
sns.set_style("whitegrid")
sns.set_palette("colorblind")

# Define paths
OUTPUT_DIR = "/home/u873859/nnUNet/NNunet/eval_output"
JSON_PATH = "/home/u873859/nnUNet/NNunet/eval_output/fold0_eval.json"

def load_json_data(json_path):
    """Load JSON data with error handling."""
    try:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at {json_path}")
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        raise

def create_dataframes(data):
    """Create DataFrames from nnUNet evaluation JSON."""
    try:
        # Foreground metrics
        foreground_metrics = pd.DataFrame(data['foreground_mean'], index=['Foreground'])

        # Mean class metrics
        mean_metrics = pd.DataFrame(data['mean']).T
        mean_metrics.index.name = 'Class'

        # Per-case metrics
        case_metrics = []
        for case in data['metric_per_case']:
            case_name = os.path.basename(case['prediction_file']).replace('.nii.gz', '')
            for class_id, metrics in case['metrics'].items():
                metrics_df = pd.DataFrame(metrics, index=[f"{case_name}_class{class_id}"])
                metrics_df['Case'] = case_name
                metrics_df['Class'] = class_id
                case_metrics.append(metrics_df)
        case_metrics_df = pd.concat(case_metrics)
        case_metrics_df.reset_index(drop=True, inplace=True)

        return foreground_metrics, mean_metrics, case_metrics_df
    except Exception as e:
        print(f"Error creating DataFrames: {e}")
        raise

def save_metrics(foreground_metrics, mean_metrics, case_metrics_df, output_dir):
    """Save metrics to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        foreground_metrics.to_csv(f"{output_dir}/foreground_metrics.csv")
        mean_metrics.to_csv(f"{output_dir}/mean_class_metrics.csv")
        case_metrics_df.to_csv(f"{output_dir}/per_case_metrics.csv")
    except Exception as e:
        print(f"Error saving metrics: {e}")
        raise

def calculate_summary_stats(df, name, output_dir):
    """Calculate and save summary statistics."""
    try:
        stats = df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        stats.to_csv(f"{output_dir}/{name}_summary_stats.csv")
        return stats
    except Exception as e:
        print(f"Error calculating summary stats for {name}: {e}")
        raise

def plot_metric_distribution(metric, data, title, filename, output_dir):
    """Plot distribution of a metric."""
    plt.figure(figsize=(10, 6))
    if 'Class' in data.columns:
        sns.boxplot(x='Class', y=metric, data=data)
        plt.xlabel('Class (1: Necrotic Core, 2: Edema, 3: Enhancing Tumor)')
    else:
        data[metric].hist(bins=20, edgecolor='black')
    plt.title(title)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_class_comparison(mean_metrics, title, filename, output_dir):
    """Plot class-wise comparison of metrics."""
    plt.figure(figsize=(12, 6))
    x = np.arange(3)
    width = 0.35

    for i, metric in enumerate(['Dice', 'IoU']):
        values = [mean_metrics.loc[c, metric] for c in ['1', '2', '3']]
        plt.bar(x + i * width, values, width, label=metric)

    plt.xticks(x + width / 2, ['Necrotic Core', 'Edema', 'Enhancing Tumor'])
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_case_performance(case_data, metric, filename, output_dir, top_n=20):
    """Plot case-wise performance for top N cases."""
    plt.figure(figsize=(15, 6))
    top_cases = case_data.groupby('Case')[metric].mean().sort_values(ascending=False).index[:top_n]
    case_data = case_data[case_data['Case'].isin(top_cases)].sort_values(by=metric, ascending=False)
    sns.barplot(x='Case', y=metric, hue='Class', data=case_data)
    plt.xticks(rotation=90)
    plt.title(f'Top {top_n} Cases: {metric} Performance')
    plt.legend(title='Class')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300)
    plt.close()

def plot_correlation_matrix(data, output_dir):
    """Plot correlation matrix of metrics."""
    corr_matrix = data[['Dice', 'IoU', 'FN', 'FP', 'TP', 'TN']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Metric Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metric_correlation.png", dpi=300)
    plt.close()

def statistical_tests(case_metrics_df, output_dir):
    """Perform statistical tests on class performance."""
    try:
        classes = ['1', '2', '3']
        class_pairs = [(1, 2), (1, 3), (2, 3)]
        with open(f"{output_dir}/statistical_tests.txt", 'w') as f:
            f.write("Statistical Tests for Class Performance Differences (Dice Score)\n")
            f.write("=" * 60 + "\n\n")
            for pair in class_pairs:
                c1, c2 = str(pair[0]), str(pair[1])
                data1 = case_metrics_df[case_metrics_df['Class'] == c1]['Dice']
                data2 = case_metrics_df[case_metrics_df['Class'] == c2]['Dice']
                t_stat, p_val = stats.ttest_ind(data1, data2)
                u_stat, u_p_val = stats.mannwhitneyu(data1, data2)
                f.write(f"Class {c1} vs Class {c2}:\n")
                f.write(f"T-test: t = {t_stat:.3f}, p = {p_val:.5f}\n")
                f.write(f"Wilcoxon rank-sum: U = {u_stat:.1f}, p = {u_p_val:.5f}\n")
                f.write(f"Mean Dice - Class {c1}: {data1.mean():.3f}, Class {c2}: {data2.mean():.3f}\n")
                f.write("-" * 50 + "\n")
    except Exception as e:
        print(f"Error in statistical tests: {e}")
        raise

def error_analysis(case_metrics_df, output_dir):
    """Perform error analysis."""
    error_metrics = case_metrics_df.copy()
    error_metrics['Error_Rate'] = (error_metrics['FN'] + error_metrics['FP']) / \
                                  (error_metrics['TP'] + error_metrics['FN'] + error_metrics['FP'] + error_metrics['TN'])
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Error_Rate', data=error_metrics)
    plt.title('Error Rate by Class')
    plt.xlabel('Class (1: Necrotic Core, 2: Edema, 3: Enhancing Tumor)')
    plt.ylabel('Error Rate (FP+FN)/(TP+FP+FN+TN)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_rate_by_class.png", dpi=300)
    plt.close()

def volumetric_analysis(case_metrics_df, output_dir):
    """Perform volumetric analysis."""
    vol_metrics = case_metrics_df.copy()
    vol_metrics['Volume_Diff'] = (vol_metrics['n_pred'] - vol_metrics['n_ref']) / vol_metrics['n_ref']
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Volume_Diff', data=vol_metrics)
    plt.title('Volume Difference (Predicted vs Reference) by Class')
    plt.xlabel('Class (1: Necrotic Core, 2: Edema, 3: Enhancing Tumor)')
    plt.ylabel('(Predicted - Reference)/Reference')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/volume_difference.png", dpi=300)
    plt.close()

def plot_dice_vs_volume(case_metrics_df, output_dir):
    """Plot Dice vs Reference Volume."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='n_ref', y='Dice', hue='Class', size='Class', data=case_metrics_df)
    plt.xscale('log')
    plt.xlabel('Reference Volume (log scale)')
    plt.ylabel('Dice Score')
    plt.title('Dice Score vs Reference Volume by Class')
    plt.legend(title='Class')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dice_vs_volume.png", dpi=300)
    plt.close()

def df_to_latex(df, caption, label, filename, output_dir):
    """Generate LaTeX table."""
    try:
        latex = df.to_latex(float_format="%.3f", caption=caption, label=label, index=True)
        with open(f"{output_dir}/{filename}.tex", 'w') as f:
            f.write(latex)
    except Exception as e:
        print(f"Error generating LaTeX table for {filename}: {e}")
        raise

def main():
    """Main function to run the analysis."""
    try:
        # Load data
        data = load_json_data(JSON_PATH)

        # Create DataFrames
        foreground_metrics, mean_metrics, case_metrics_df = create_dataframes(data)

        # Save raw metrics
        save_metrics(foreground_metrics, mean_metrics, case_metrics_df, OUTPUT_DIR)

        # Calculate summary statistics
        foreground_stats = calculate_summary_stats(foreground_metrics, "foreground", OUTPUT_DIR)
        mean_stats = calculate_summary_stats(mean_metrics, "class_mean", OUTPUT_DIR)
        case_stats = calculate_summary_stats(case_metrics_df.drop(['Case', 'Class'], axis=1), "per_case", OUTPUT_DIR)

        # Generate visualizations
        plot_metric_distribution('Dice', case_metrics_df, 'Distribution of Dice Scores Across All Cases', 'dice_distribution', OUTPUT_DIR)
        plot_metric_distribution('IoU', case_metrics_df, 'Distribution of IoU Scores Across All Cases', 'iou_distribution', OUTPUT_DIR)
        plot_class_comparison(mean_metrics, 'Class-wise Performance Comparison', 'class_comparison', OUTPUT_DIR)
        plot_case_performance(case_metrics_df, 'Dice', 'top_cases_dice_performance', OUTPUT_DIR)
        plot_correlation_matrix(case_metrics_df, OUTPUT_DIR)
        error_analysis(case_metrics_df, OUTPUT_DIR)
        volumetric_analysis(case_metrics_df, OUTPUT_DIR)
        plot_dice_vs_volume(case_metrics_df, OUTPUT_DIR)

        # Perform statistical tests
        statistical_tests(case_metrics_df, OUTPUT_DIR)

        # Generate LaTeX tables
        df_to_latex(foreground_stats, 'Foreground Segmentation Performance Metrics', 'tab:foreground_stats', 'foreground_stats', OUTPUT_DIR)
        df_to_latex(mean_stats, 'Class-wise Mean Segmentation Performance Metrics', 'tab:class_stats', 'class_stats', OUTPUT_DIR)
        df_to_latex(case_stats, 'Per-case Segmentation Performance Metrics', 'tab:case_stats', 'case_stats', OUTPUT_DIR)

        print(f"Analysis complete. Results saved to {OUTPUT_DIR}")

    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()

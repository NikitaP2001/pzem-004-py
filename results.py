import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib as mpl

# Set up matplotlib for high-resolution figures
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# Define file paths
stat_files = [
    "atlas-gen-bmk.csv",
    "atlas-kv-bmk.csv",
    "belle2-gen-sim-reco-ma-bmk.csv",
    "cms-reco-bmk.csv",
    "lhcb-sim-run3-ma-bm.csv"
]

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Function to load and process data
def load_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        print(f"Warning: File {file_path} not found.")
        return None

# Initialize results dictionary
results = {}

# Process each file
for file_name in stat_files:
    df = load_data(file_name)
    if df is not None:
        # Extract workload name from filename
        workload = os.path.splitext(file_name)[0]
        
        # Calculate statistics
        pzem_mean = df['taskEnergy'].mean()
        rapl_mean = df['hepscoreEnergy'].mean()
        diff_percent = ((pzem_mean - rapl_mean) / rapl_mean) * 100
        ratio = pzem_mean / rapl_mean
        pearson_r, p_value = stats.pearsonr(df['taskEnergy'], df['hepscoreEnergy'])
        
        # Store results
        results[workload] = {
            'pzem_mean': pzem_mean,
            'rapl_mean': rapl_mean,
            'diff_percent': diff_percent,
            'ratio': ratio,
            'pearson_r': pearson_r,
            'p_value': p_value,
            'std_diff': np.std(df['taskEnergy'] - df['hepscoreEnergy']),
            'mean_diff': np.mean(df['taskEnergy'] - df['hepscoreEnergy']),
            'relative_error': np.mean(np.abs(df['taskEnergy'] - df['hepscoreEnergy']) / df['rapl_mean']) * 100 if 'rapl_mean' in df.columns else None,
            'data': df
        }

# Create summary table
if results:
    summary_data = {
        'Workload': [],
        'PZEM (J)': [],
        'RAPL (J)': [],
        'Difference (%)': [],
        'Ratio PZEM/RAPL': []
    }
    
    for workload, data in results.items():
        summary_data['Workload'].append(workload)
        summary_data['PZEM (J)'].append(f"{data['pzem_mean']:.2f}")
        summary_data['RAPL (J)'].append(f"{data['rapl_mean']:.2f}")
        summary_data['Difference (%)'].append(f"{data['diff_percent']:.2f}")
        summary_data['Ratio PZEM/RAPL'].append(f"{data['ratio']:.4f}")
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("results/summary_table.csv", index=False)
    print("Summary Table:")
    print(summary_df)
    
    # Calculate overall statistics
    all_pzem = np.concatenate([data['data']['taskEnergy'].values for data in results.values()])
    all_rapl = np.concatenate([data['data']['hepscoreEnergy'].values for data in results.values()])
    print("data: ", all_pzem, all_rapl)
    overall_pearson_r, overall_p_value = stats.pearsonr(all_pzem, all_rapl)
    print("R, P: ", overall_pearson_r, overall_p_value)
    overall_mean_diff = np.mean(all_pzem - all_rapl)
    overall_std_diff = np.std(all_pzem - all_rapl)
    overall_mean_relative_error = np.mean(np.abs(all_pzem - all_rapl) / all_rapl) * 100
    
    # Statistical analysis table
    stats_data = {
        'Statistical Metric': [
            'Mean difference between PZEM and RAPL',
            'Standard deviation of difference',
            'Pearson correlation coefficient',
            'p-value',
            'Mean relative error'
        ],
        'Value': [
            f"{overall_mean_diff:.2f} J",
            f"{overall_std_diff:.2f} J",
            f"{overall_pearson_r:.4f}",
            f"{overall_p_value:.8f}",
            f"{overall_mean_relative_error:.2f} %"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv("results/statistical_analysis.csv", index=False)
    print("\nStatistical Analysis:")
    print(stats_df)
    
    # Generate visualizations
    
    # 1. Correlation between PZEM and RAPL
    plt.figure(figsize=(10, 8))
    plt.scatter(all_rapl, all_pzem, alpha=0.7)
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_rapl, all_pzem)
    x_line = np.linspace(min(all_rapl), max(all_rapl), 100)
    plt.plot(x_line, slope * x_line + intercept, 'r-', 
             label=f'Regression line: y={slope:.4f}x+{intercept:.2f}')
    
    # Add identity line (y=x)
    plt.plot(x_line, x_line, 'k--', label='Identity line (y=x)')
    
    plt.xlabel('RAPL Energy Measurement (J)')
    plt.ylabel('PZEM Energy Measurement (J)')
    plt.title(f'Correlation between PZEM and RAPL Measurements\nPearson r = {overall_pearson_r:.4f}, p-value = {overall_p_value:.8f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/correlation_plot.png", bbox_inches='tight')
    
    # 2. Bar chart of PZEM/RAPL ratios
    plt.figure(figsize=(12, 8))
    workloads = list(results.keys())
    ratios = [results[w]['ratio'] for w in workloads]
    
    # Calculate 95% confidence intervals
    confidence_intervals = []
    for workload in workloads:
        df = results[workload]['data']
        ratios_per_measurement = df['taskEnergy'] / df['hepscoreEnergy']
        ci = stats.t.interval(0.95, len(ratios_per_measurement)-1, 
                             loc=np.mean(ratios_per_measurement), 
                             scale=stats.sem(ratios_per_measurement))
        confidence_intervals.append((ci[1] - ci[0]) / 2)
    
    plt.bar(workloads, ratios, yerr=confidence_intervals, capsize=5)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Ideal ratio (1.0)')
    
    plt.xlabel('Workload')
    plt.ylabel('PZEM/RAPL Ratio')
    plt.title('PZEM to RAPL Energy Measurement Ratio for Different Workloads')
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/ratio_bar_chart.png", bbox_inches='tight')
    
    # 3. Bland-Altman plot
    plt.figure(figsize=(10, 8))
    mean_values = (all_pzem + all_rapl) / 2
    diff_values = all_pzem - all_rapl
    
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)
    
    plt.scatter(mean_values, diff_values, alpha=0.7)
    plt.axhline(y=mean_diff, color='k', linestyle='-', label=f'Mean difference: {mean_diff:.2f} J')
    plt.axhline(y=mean_diff + 1.96*std_diff, color='r', linestyle='--', 
                label=f'Upper limit of agreement: {mean_diff + 1.96*std_diff:.2f} J')
    plt.axhline(y=mean_diff - 1.96*std_diff, color='r', linestyle='--',
                label=f'Lower limit of agreement: {mean_diff - 1.96*std_diff:.2f} J')
    
    plt.xlabel('Mean of PZEM and RAPL (J)')
    plt.ylabel('Difference (PZEM - RAPL) (J)')
    plt.title('Bland-Altman Plot of PZEM vs RAPL Energy Measurements')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/bland_altman_plot.png", bbox_inches='tight')
    
    # 4. Box plot of differences by workload
    plt.figure(figsize=(12, 8))
    diff_by_workload = []
    labels = []
    
    for workload in workloads:
        df = results[workload]['data']
        diff = df['taskEnergy'] - df['hepscoreEnergy']
        diff_by_workload.append(diff)
        labels.append(workload)
    
    plt.boxplot(diff_by_workload, labels=labels)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Workload')
    plt.ylabel('Difference (PZEM - RAPL) (J)')
    plt.title('Distribution of Differences Between PZEM and RAPL by Workload')
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/difference_boxplot.png", bbox_inches='tight')
    
    # 5. Time series plot for the first workload (as example)
    if len(workloads) > 0:
        example_workload = workloads[0]
        df = results[example_workload]['data']
        
        plt.figure(figsize=(12, 8))
        plt.plot(range(len(df)), df['taskEnergy'], 'b-', label='PZEM')
        plt.plot(range(len(df)), df['hepscoreEnergy'], 'g-', label='RAPL')
        plt.plot(range(len(df)), df['taskEnergy'] - df['hepscoreEnergy'], 'r--', 
                 label='Difference (PZEM - RAPL)')
        
        plt.xlabel('Measurement Index')
        plt.ylabel('Energy (J)')
        plt.title(f'Time Series of Energy Measurements for {example_workload}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/time_series_{example_workload}.png", bbox_inches='tight')
    
    print("\nAnalysis complete! Results saved to 'results' directory.")
else:
    print("No data was processed. Please check your file paths.")



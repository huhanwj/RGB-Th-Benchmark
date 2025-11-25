import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import glob
import re
import os

def create_consistency_plot(df, model_name):
    """Generates the Score Consistency Box Plot with the instructional guide."""
    
    unique_labels = df['ground_truth_label'].unique()
    num_labels = len(unique_labels)
    # Dynamic height
    plot_height = max(10, num_labels * 0.5) + 4 

    # Create figure with 2 subplots (Guide on top, Data on bottom)
    fig, (ax_guide, ax_data) = plt.subplots(2, 1, figsize=(14, plot_height), 
                                            gridspec_kw={'height_ratios': [1.5, num_labels * 0.5]})

    # --- PART 1: The Guide ---
    ax_guide.set_title(f"Score Consistency: {model_name}\n(How to Read This Plot)", fontsize=16, weight='bold', pad=20)
    ax_guide.set_xlim(0, 10)
    ax_guide.set_ylim(0, 4)
    ax_guide.axis('off')

    # Schematic Boxplot
    rect = mpatches.Rectangle((3, 1.5), 4, 1, linewidth=2, edgecolor='black', facecolor='#69b3a2')
    ax_guide.add_patch(rect)
    ax_guide.plot([5, 5], [1.5, 2.5], color='black', linewidth=3) # Median
    ax_guide.plot([1, 3], [2, 2], color='black', linewidth=2) # Left whisker
    ax_guide.plot([7, 9], [2, 2], color='black', linewidth=2) # Right whisker
    ax_guide.plot([1, 1], [1.7, 2.3], color='black', linewidth=2) # Left cap
    ax_guide.plot([9, 9], [1.7, 2.3], color='black', linewidth=2) # Right cap
    ax_guide.plot(9.5, 2, 'o', color='black', markersize=8) # Outlier

    # Annotations
    ax_guide.annotate('Median Score', xy=(5, 2.5), xytext=(5, 3.5), arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
    ax_guide.annotate('Consistency Box', xy=(3, 1.5), xytext=(3, 0.5), arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
    ax_guide.annotate('Range', xy=(1.5, 2), xytext=(1.5, 3), arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
    ax_guide.annotate('Outliers', xy=(9.5, 2), xytext=(9.5, 1), arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

    # --- PART 2: The Data ---
    order = df.groupby('ground_truth_label')['judge_score'].median().sort_values(ascending=False).index
    sns.boxplot(x='judge_score', y='ground_truth_label', data=df, order=order, palette='viridis', ax=ax_data)
    
    ax_data.set_title(f'Score Consistency by Action - {model_name}', fontsize=14)
    ax_data.set_xlabel('Judge Score (0.0 = Fail, 1.0 = Perfect)', fontsize=12)
    ax_data.set_ylabel('Action', fontsize=12)
    ax_data.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    filename = f'score_consistency_{model_name}.png'
    plt.savefig(filename)
    plt.close()
    return filename

# --- Main Execution ---
all_files = glob.glob('benchmark_results_*_judged.csv')
dfs = []

print(f"Found {len(all_files)} files.")

for filename in all_files:
    # 1. Extract Model Name
    # Tries to find text between 'results_' and the date like '_2025...'
    match = re.search(r'benchmark_results_(.*?)_\d{8}', filename)
    if match:
        model_name = match.group(1)
    else:
        # Fallback if filename format is slightly different
        model_name = os.path.basename(filename).replace('benchmark_results_', '').replace('.csv', '')
    
    print(f"Processing model: {model_name}")
    
    # 2. Read Data
    temp_df = pd.read_csv(filename)
    temp_df['model'] = model_name
    dfs.append(temp_df)
    
    # 3. Generate Individual Consistency Plot
    plot_file = create_consistency_plot(temp_df, model_name)
    print(f"  -> Saved {plot_file}")

# --- Comparison Plots ---
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)

    # A. Heatmap
    heatmap_data = combined_df.groupby(['ground_truth_label', 'model'])['judge_score'].mean().unstack()
    heatmap_data['average_difficulty'] = heatmap_data.mean(axis=1)
    # Sort by difficulty so hard tasks are together
    heatmap_data = heatmap_data.sort_values('average_difficulty', ascending=False).drop('average_difficulty', axis=1)
    
    plot_height = max(10, len(heatmap_data) * 0.4)
    plt.figure(figsize=(10, plot_height))
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.2f', linewidths=.5, cbar_kws={'label': 'Avg Score'})
    plt.title('Model Comparison Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('model_comparison_heatmap.png')
    print("  -> Saved model_comparison_heatmap.png")

    # B. Leaderboard
    leaderboard = combined_df.groupby('model')['judge_score'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(x='judge_score', y='model', data=leaderboard, palette='viridis')
    plt.title('Overall Model Leaderboard', fontsize=16)
    plt.xlim(0, 1.05)
    for index, row in leaderboard.iterrows():
        plt.text(row.judge_score + 0.02, index, f"{row.judge_score:.3f}", va='center')
    plt.tight_layout()
    plt.savefig('model_leaderboard.png')
    print("  -> Saved model_leaderboard.png")
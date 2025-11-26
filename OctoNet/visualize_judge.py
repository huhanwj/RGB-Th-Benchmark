import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import glob
import re
import os

# --- 1. Robust Parsing Logic ---
def parse_filename_robust(filename):
    basename = os.path.basename(filename)
    # Regex to capture Model and optional Judge Suffix
    pattern = r"^benchmark_results_(?P<model>.+?)_\d{8}_\d{6}_judged(?P<suffix>.*)\.csv$"
    match = re.match(pattern, basename)
    
    if match:
        model = match.group('model')
        suffix = match.group('suffix')
        if not suffix:
            judge = 'qwen3-max'
        else:
            judge = suffix.lstrip('_')
        return model, judge
    return "Unknown", "Unknown"

# --- 2. Individual Consistency Plot (Landscape + Guide) ---
def create_landscape_consistency_plot(df, evaluated_model, judge_model):
    unique_labels = df['ground_truth_label'].unique()
    num_labels = len(unique_labels)
    
    # Dynamic width: 0.3 inches per label
    plot_width = max(14, num_labels * 0.3)
    
    fig = plt.figure(figsize=(plot_width + 6, 8)) 
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.1)
    
    ax_data = fig.add_subplot(gs[0])
    ax_guide = fig.add_subplot(gs[1])
    
    # Sort data
    order = df.groupby('ground_truth_label')['judge_score'].median().sort_values(ascending=False).index
    
    # Main Boxplot
    sns.boxplot(x='ground_truth_label', y='judge_score', data=df, order=order, palette='viridis', ax=ax_data)
    
    ax_data.set_title(f'Score Consistency: {evaluated_model}\n(Judge: {judge_model})', fontsize=16)
    ax_data.set_ylabel('Judge Score', fontsize=12)
    ax_data.set_xlabel('Action', fontsize=12)
    ax_data.set_ylim(-0.05, 1.05)
    ax_data.tick_params(axis='x', rotation=90) # Rotate labels
    
    # Guide Panel
    ax_guide.set_title("How to Read", fontsize=14, weight='bold', pad=10)
    ax_guide.axis('off'); ax_guide.set_xlim(0, 5); ax_guide.set_ylim(0, 11)
    
    # Draw Schematic
    rect = mpatches.Rectangle((0.5, 3), 2, 4, linewidth=2, edgecolor='black', facecolor='#69b3a2')
    ax_guide.add_patch(rect)
    ax_guide.plot([0.5, 2.5], [5, 5], color='black', linewidth=3)
    ax_guide.plot([1.5, 1.5], [1, 3], color='black', linewidth=2)
    ax_guide.plot([1.5, 1.5], [7, 9], color='black', linewidth=2)
    ax_guide.plot([1.0, 2.0], [1, 1], color='black', linewidth=2)
    ax_guide.plot([1.0, 2.0], [9, 9], color='black', linewidth=2)
    ax_guide.plot(1.5, 9.5, 'o', color='black', markersize=8)
    
    # Annotations
    ap = dict(facecolor='black', shrink=0.05)
    ax_guide.annotate('Outlier', xy=(1.5, 9.5), xytext=(3, 9.5), arrowprops=ap, ha='left', va='center')
    ax_guide.annotate('Range', xy=(1.5, 8), xytext=(3, 8), arrowprops=ap, ha='left', va='center')
    ax_guide.annotate('Consistency', xy=(2.5, 4), xytext=(3, 4), arrowprops=ap, ha='left', va='center')
    ax_guide.annotate('Median', xy=(2.5, 5), xytext=(3, 6), arrowprops=ap, ha='left', va='center')

    plt.tight_layout()
    
    # Safe filename
    safe_eval = evaluated_model.replace(' ', '_').replace('/', '-')
    safe_judge = judge_model.replace(' ', '_').replace('/', '-')
    filename = f'score_consistency_{safe_eval}_by_{safe_judge}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    return filename

# --- 3. Main Execution Loop ---
all_files = glob.glob('benchmark_results_*.csv')
dfs = []

print(f"Found {len(all_files)} files.")

for filename in all_files:
    evaluated_model, judge_model = parse_filename_robust(filename)
    if evaluated_model == "Unknown":
        print(f"Skipping unknown file: {filename}")
        continue
        
    print(f"Processing: {evaluated_model} (Judge: {judge_model})")
    
    temp_df = pd.read_csv(filename)
    temp_df['evaluated_model'] = evaluated_model
    temp_df['judge_model'] = judge_model
    temp_df['display_name'] = f"{evaluated_model}\n(Judge: {judge_model})"
    dfs.append(temp_df)
    
    # GENERATE CONSISTENCY PLOT
    plot_file = create_landscape_consistency_plot(temp_df, evaluated_model, judge_model)
    print(f"  -> Saved {plot_file}")

# --- 4. Comparison Plots (Heatmap + Leaderboard) ---
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)

    # --- A. LANDSCAPE HEATMAP (FIXED) ---
    heatmap_data = combined_df.groupby(['display_name', 'ground_truth_label'])['judge_score'].mean().unstack()
    
    # Sort Actions by Difficulty
    action_difficulty = heatmap_data.mean(axis=0).sort_values(ascending=False)
    heatmap_data = heatmap_data[action_difficulty.index]
    
    num_actions = len(heatmap_data.columns)
    num_models = len(heatmap_data.index)
    
    # Dimensions: 0.6 inches per column for spacing
    plot_width = max(20, num_actions * 0.6) 
    plot_height = max(6, num_models * 1.5) 
    
    plt.figure(figsize=(plot_width, plot_height))
    
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.2f', 
                linewidths=.5, 
                annot_kws={"size": 10},
                cbar_kws={'label': 'Avg Score', 'shrink': 0.8}) 
    
    plt.title('Model Comparison Landscape', fontsize=18, pad=20)
    plt.xlabel('Action (Sorted by Difficulty)', fontsize=14, labelpad=10)
    plt.ylabel('Configuration', fontsize=14)
    plt.xticks(rotation=90, fontsize=10) # Rotated labels
    plt.yticks(rotation=0, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('model_comparison_heatmap_landscape_fixed.png', dpi=300)
    print("  -> Saved model_comparison_heatmap_landscape_fixed.png")

    # --- B. LEADERBOARD ---
    leaderboard = combined_df.groupby('display_name')['judge_score'].mean().sort_values(ascending=False).reset_index()
    
    lb_height = max(6, len(leaderboard) * 0.8)
    plt.figure(figsize=(10, lb_height))
    
    sns.barplot(x='judge_score', y='display_name', data=leaderboard, palette='viridis')
    plt.title('Overall Leaderboard', fontsize=16)
    plt.xlabel('Global Average Score', fontsize=12)
    plt.xlim(0, 1.05)
    
    for index, row in leaderboard.iterrows():
        plt.text(row.judge_score + 0.01, index, f"{row.judge_score:.3f}", va='center')
        
    plt.tight_layout()
    plt.savefig('model_leaderboard.png')
    print("  -> Saved model_leaderboard.png")
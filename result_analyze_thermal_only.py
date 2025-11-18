#!/usr/bin/env python3

"""
================================================================================
RGB-Th-Bench Benchmark Analysis Script
*** THERMAL-ONLY COMPARISON MODE ***
================================================================================
... (script header) ...
MODIFICATION: Now also saves summary tables and skill matrices to CSV files.
--------------------------------------------------------------------------------
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration: Skill Mappings ---
SKILL_ABBREVIATIONS = {
    'Scene Understanding': 'Scene',
    'Detailed Object Presence': 'ObjPr',
    'Instance Attributes': 'Attr',
    'Instance Location w.r.t. Image': 'LocIm',
    'Instance Spatial Relation': 'SpRel',
    'Instance Counting': 'Count',
    'Instance Interaction': 'Interact',
    'Temperature to Color Mapping Understanding': 'Temp2Col',
    'RGB-Thermal-Heatmap Alignment': 'HeatAlign',
    'Instance Thermal Attribute': 'ThermAttr',
    'Relative Thermal Attribute': 'RelTherm',
    'Warmest Areas Detection': 'Warmest',
    'Coldest Areas Detection': 'Coldest',
    'Risk and Anomaly Detection': 'Anomaly'
}

RGB_TXT_SKILLS = [
    'Scene Understanding', 'Detailed Object Presence', 'Instance Attributes',
    'Instance Location w.r.t. Image', 'Instance Spatial Relation',
    'Instance Counting', 'Instance Interaction'
]

RGB_TH_TXT_SKILLS = [
    'Temperature to Color Mapping Understanding', 'RGB-Thermal-Heatmap Alignment',
    'Instance Thermal Attribute', 'Relative Thermal Attribute',
    'Warmest Areas Detection', 'Coldest Areas Detection',
    'Risk and Anomaly Detection'
]


def load_all_results() -> pd.DataFrame:
    """Finds and loads ONLY thermal_only benchmark_results_*.json files."""
    all_results = []
    
    # This glob pattern now ONLY finds files ending in _thermal_only.json
    json_files = glob.glob('benchmark_results_*_thermal_only.json')
    
    if not json_files:
        print("Error: No 'benchmark_results_*_thermal_only.json' files found.")
        return pd.DataFrame()

    print(f"Found {len(json_files)} thermal-only result files: {json_files}")
    
    for f in json_files:
        with open(f, 'r') as file:
            try:
                data = json.load(file)
                all_results.extend(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {f}. Skipping.")
    
    if not all_results:
        print("Error: No valid results were loaded.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    
    if 'data_type_original' in df.columns:
        df['data_type'] = df['data_type_original']
    else:
        print("Warning: 'data_type_original' column not found. Skill categorization may be incorrect.")
        
    print(f"Loaded {len(df)} total results from {df['model'].nunique()} models.")
    return df


def calculate_metrics(df: pd.DataFrame):
    """Calculates and prints QAcc and SAcc metrics."""
    
    if df.empty:
        return None, None

    # --- 1. QAcc (Question-level Accuracy) Calculations ---
    qacc_overall = df.groupby('model')['is_correct'].mean() * 100
    
    df_rgb_txt = df[df['skill'].isin(RGB_TXT_SKILLS)]
    qacc_rgb_txt = df_rgb_txt.groupby('model')['is_correct'].mean() * 100
    
    df_rgb_th = df[df['skill'].isin(RGB_TH_TXT_SKILLS)]
    qacc_rgb_th = df_rgb_th.groupby('model')['is_correct'].mean() * 100

    # --- 2. SAcc (Skill-level Accuracy) Calculations ---
    sacc_passes = df.groupby(['model', 'file', 'skill'])['is_correct'].all()
    sacc_overall = sacc_passes.groupby('model').mean() * 100
    
    sacc_df = sacc_passes.reset_index()
    sacc_df['skill_group'] = sacc_df['skill'].apply(
        lambda s: 'RGB-Txt' if s in RGB_TXT_SKILLS else 'RGB-Th-Txt'
    )
    
    sacc_rgb_txt = sacc_df[
        sacc_df['skill_group'] == 'RGB-Txt'
    ].groupby('model')['is_correct'].mean() * 100
    
    sacc_rgb_th = sacc_df[
        sacc_df['skill_group'] == 'RGB-Th-Txt'
    ].groupby('model')['is_correct'].mean() * 100

    # --- 3. Format & Print Summary Table ---
    summary_df = pd.DataFrame({
        'QAcc (Overall)': qacc_overall,
        'SAcc (Overall)': sacc_overall,
        'QAcc (RGB-Txt)': qacc_rgb_txt,
        'SAcc (RGB-Txt)': sacc_rgb_txt,
        'QAcc (RGB-Th-Txt)': qacc_rgb_th,
        'SAcc (RGB-Th-Txt)': sacc_rgb_th,
    }).sort_values('SAcc (Overall)', ascending=False)
    
    print("\n--- Model Performance Summary (Thermal-Only) ---")
    print(summary_df.to_string(float_format="%.2f%%"))
    print("--------------------------------------------------\n")
    
    # --- START OF MODIFICATION ---
    # Save the summary dataframe to CSV
    summary_filename = "benchmark_summary_thermal_only.csv"
    summary_df.to_csv(summary_filename, float_format="%.4f")
    print(f"Saved summary table to {summary_filename}")
    # --- END OF MODIFICATION ---
    
    # --- 4. Prepare Matrices for Heatmaps ---
    qacc_matrix = df.groupby(['model', 'skill'])['is_correct'].mean().unstack()
    sacc_matrix = sacc_passes.groupby(['model', 'skill']).mean().unstack()
    
    return qacc_matrix * 100, sacc_matrix * 100


def plot_heatmaps(qacc_matrix, sacc_matrix):
    """Generates and saves QAcc and SAcc heatmaps."""
    
    if qacc_matrix is None or sacc_matrix is None:
        print("Skipping plotting due to missing data.")
        return

    # Sort skills by the paper's order
    skill_order = RGB_TXT_SKILLS + RGB_TH_TXT_SKILLS
    skill_abbr_order = [SKILL_ABBREVIATIONS[s] for s in skill_order]

    # Apply abbreviations and re-order columns
    qacc_matrix_plt = qacc_matrix.rename(columns=SKILL_ABBREVIATIONS)[skill_abbr_order]
    sacc_matrix_plt = sacc_matrix.rename(columns=SKILL_ABBREVIATIONS)[skill_abbr_order]
    
    # Sort models (rows) by SAcc performance
    model_order = sacc_matrix_plt.mean(axis=1).sort_values(ascending=False).index
    qacc_matrix_plt = qacc_matrix_plt.reindex(model_order)
    sacc_matrix_plt = sacc_matrix_plt.reindex(model_order)

    # --- START OF MODIFICATION ---
    # Save the heatmap matrices to CSV
    qacc_matrix_filename = "qacc_matrix_thermal_only.csv"
    sacc_matrix_filename = "sacc_matrix_thermal_only.csv"
    qacc_matrix_plt.to_csv(qacc_matrix_filename, float_format="%.4f")
    print(f"Saved QAcc matrix data to {qacc_matrix_filename}")
    sacc_matrix_plt.to_csv(sacc_matrix_filename, float_format="%.4f")
    print(f"Saved SAcc matrix data to {sacc_matrix_filename}")
    # --- END OF MODIFICATION ---

    # --- Plot QAcc Heatmap (like Figure 4) ---
    print("Generating QAcc heatmap...")
    fig_height = max(5, 0.5 * len(model_order))
    plt.figure(figsize=(15, fig_height))
    sns.heatmap(
        qacc_matrix_plt,
        annot=True,
        fmt=".1f",
        linewidths=.5,
        cmap="YlGnBu", 
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'QAcc (%)'}
    )
    plt.title('QAcc (Question-level Accuracy) by Model and Skill [Thermal-Only]', fontsize=16)
    plt.ylabel('Model (Run)')
    plt.xlabel('Skill Dimension')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    qacc_filename = 'qacc_heatmap_thermal_only.png'
    plt.savefig(qacc_filename, dpi=300)
    print(f"Saved QAcc heatmap to {qacc_filename}")
    plt.close()

    # --- Plot SAcc Heatmap (like Figure 5) ---
    print("Generating SAcc heatmap...")
    plt.figure(figsize=(15, fig_height))
    sns.heatmap(
        sacc_matrix_plt,
        annot=True,
        fmt=".1f",
        linewidths=.5,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'SAcc (%)'}
    )
    plt.title('SAcc (Skill-level Accuracy) by Model and Skill [Thermal-Only]', fontsize=16)
    plt.ylabel('Model (Run)')
    plt.xlabel('Skill Dimension')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    sacc_filename = 'sacc_heatmap_thermal_only.png'
    plt.savefig(sacc_filename, dpi=300)
    print(f"Saved SAcc heatmap to {sacc_filename}")
    plt.close()


def main():
    """Main execution function."""
    df = load_all_results()
    if not df.empty:
        qacc_matrix, sacc_matrix = calculate_metrics(df)
        plot_heatmaps(qacc_matrix, sacc_matrix)
    print("Analysis complete.")

if __name__ == "__main__":
    main()
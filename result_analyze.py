#!/usr/bin/env python3

"""
================================================================================
RGB-Th-Bench Benchmark Analysis Script
================================================================================

This script loads all 'benchmark_results_*.json' files from the current
directory to calculate and visualize performance metrics, mirroring the
analysis in the RGB-Th-Bench paper (arXiv:2503.19654v3).

It calculates:
1.  QAcc (Question-level Accuracy): Overall, RGB-Txt, and RGB-Th-Txt.
2.  SAcc (Skill-level Accuracy): Overall, RGB-Txt, and RGB-Th-Txt.
3.  Heatmaps: Generates QAcc and SAcc heatmaps, saved as PNG files,
    comparing all models across all 14 skill dimensions.

--------------------------------------------------------------------------------
SETUP:
--------------------------------------------------------------------------------
1.  Make sure you have run the benchmark and have one or more
    `benchmark_results_{provider}.json` files in this directory.

2.  Install necessary libraries:
    pip install pandas matplotlib seaborn
--------------------------------------------------------------------------------
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration: Skill Mappings ---

# This mapping uses the abbreviations from the paper [cite: 139, 141]
# to make the heatmap columns readable.
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

# Define which skills belong to which data type, as per the paper [cite: 137, 141]
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
    """Finds and loads all benchmark_results_*.json files."""
    all_results = []
    json_files = glob.glob('benchmark_results_*.json')
    if not json_files:
        print("Error: No 'benchmark_results_*.json' files found.")
        return pd.DataFrame()

    print(f"Found {len(json_files)} result files: {json_files}")
    
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
    print(f"Loaded {len(df)} total results from {df['model'].nunique()} models.")
    return df


def calculate_metrics(df: pd.DataFrame):
    """Calculates and prints QAcc and SAcc metrics."""
    
    if df.empty:
        return None, None

    # --- 1. QAcc (Question-level Accuracy) Calculations ---
    
    # QAcc Overall
    qacc_overall = df.groupby('model')['is_correct'].mean() * 100
    
    # QAcc for RGB-Txt
    df_rgb_txt = df[df['skill'].isin(RGB_TXT_SKILLS)]
    qacc_rgb_txt = df_rgb_txt.groupby('model')['is_correct'].mean() * 100
    
    # QAcc for RGB-Th-Txt (This is the "Racc" you asked for)
    df_rgb_th = df[df['skill'].isin(RGB_TH_TXT_SKILLS)]
    qacc_rgb_th = df_rgb_th.groupby('model')['is_correct'].mean() * 100

    # --- 2. SAcc (Skill-level Accuracy) Calculations ---
    
    # As defined in the paper: a "Pass" (True) if ALL questions
    # for a (model, file, skill) group are correct, else "Fail" (False).
    sacc_passes = df.groupby(['model', 'file', 'skill'])['is_correct'].all()
    
    # SAcc Overall
    sacc_overall = sacc_passes.groupby('model').mean() * 100
    
    # Merge skill type back in to split SAcc
    sacc_df = sacc_passes.reset_index()
    sacc_df['data_type'] = sacc_df['skill'].apply(
        lambda s: 'RGB-Txt' if s in RGB_TXT_SKILLS else 'RGB-Th-Txt'
    )
    
    # SAcc for RGB-Txt
    sacc_rgb_txt = sacc_df[
        sacc_df['data_type'] == 'RGB-Txt'
    ].groupby('model')['is_correct'].mean() * 100
    
    # SAcc for RGB-Th-Txt
    sacc_rgb_th = sacc_df[
        sacc_df['data_type'] == 'RGB-Th-Txt'
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
    
    print("\n--- Model Performance Summary ---")
    print(summary_df.to_string(float_format="%.2f%%"))
    print("---------------------------------\n")
    
    # --- 4. Prepare Matrices for Heatmaps ---
    
    # QAcc Matrix (Model x Skill)
    qacc_matrix = df.groupby(['model', 'skill'])['is_correct'].mean().unstack()
    
    # SAcc Matrix (Model x Skill)
    sacc_matrix = sacc_passes.groupby(['model', 'skill']).mean().unstack()
    
    return qacc_matrix * 100, sacc_matrix * 100


def plot_heatmaps(qacc_matrix, sacc_matrix):
    """Generates and saves QAcc and SAcc heatmaps."""
    
    if qacc_matrix is None or sacc_matrix is None:
        print("Skipping plotting due to missing data.")
        return

    # Sort skills by the paper's order (RGB-Txt then RGB-Th-Txt)
    skill_order = RGB_TXT_SKILLS + RGB_TH_TXT_SKILLS
    skill_abbr_order = [SKILL_ABBREVIATIONS[s] for s in skill_order]

    # Apply abbreviations and re-order columns
    qacc_matrix_plt = qacc_matrix.rename(columns=SKILL_ABBREVIATIONS)[skill_abbr_order]
    sacc_matrix_plt = sacc_matrix.rename(columns=SKILL_ABBREVIATIONS)[skill_abbr_order]
    
    # Sort models (rows) by SAcc performance
    model_order = sacc_matrix_plt.mean(axis=1).sort_values(ascending=False).index
    qacc_matrix_plt = qacc_matrix_plt.reindex(model_order)
    sacc_matrix_plt = sacc_matrix_plt.reindex(model_order)

    # --- Plot QAcc Heatmap (like Figure 4) ---
    print("Generating QAcc heatmap...")
    plt.figure(figsize=(15, max(5, 0.5 * len(model_order))))
    sns.heatmap(
        qacc_matrix_plt,
        annot=True,
        fmt=".1f",
        linewidths=.5,
        cmap="YlGnBu", # A good blue-green sequential colormap
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'QAcc (%)'}
    )
    plt.title('QAcc (Question-level Accuracy) by Model and Skill', fontsize=16)
    plt.ylabel('Model')
    plt.xlabel('Skill Dimension')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    qacc_filename = 'qacc_heatmap.png'
    plt.savefig(qacc_filename)
    print(f"Saved QAcc heatmap to {qacc_filename}")
    plt.close()

    # --- Plot SAcc Heatmap (like Figure 5) ---
    print("Generating SAcc heatmap...")
    plt.figure(figsize=(15, max(5, 0.5 * len(model_order))))
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
    plt.title('SAcc (Skill-level Accuracy) by Model and Skill', fontsize=16)
    plt.ylabel('Model')
    plt.xlabel('Skill Dimension')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    sacc_filename = 'sacc_heatmap.png'
    plt.savefig(sacc_filename)
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
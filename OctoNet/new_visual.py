import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os
import numpy as np

# ==========================================
# 1. 全局配置
# ==========================================
# 输出文件命名
OUT_LEADERBOARD = "Figure_1_Leaderboard.png"
OUT_HEATMAP = "Figure_2_Heatmap.png"
OUT_BOXPLOT = "Figure3_Stability.png"
OUT_STATS = "Table_1_Detailed_Statistics.csv"

# ==========================================
# 2. 数据加载与集成核心逻辑
# ==========================================
def parse_filename(filename):
    basename = os.path.basename(filename)
    # 匹配模式: benchmark_results_{MODEL_NAME}_{TIMESTAMP}_judged_{JUDGE_NAME}.csv
    try:
        if "_judged_" in basename:
            parts = basename.split("_judged_")
            left_part = parts[0]
            judge_part = parts[1].replace(".csv", "")
            
            # 提取模型名 (去掉前缀和时间戳)
            model_name = left_part.replace("benchmark_results_", "")
            model_name = re.sub(r'_\d{8}_\d{6}$', '', model_name)
            
            return model_name, judge_part
    except:
        pass
    return "Unknown", "Unknown"

def load_and_ensemble_data():
    all_files = glob.glob('*_judged_*.csv')
    print(f"Found {len(all_files)} judged files.")
    
    # 1. 按“被测模型”分组
    model_groups = {}
    for f in all_files:
        model, judge = parse_filename(f)
        if model == "Unknown": continue
        
        try:
            df = pd.read_csv(f)
            # 确保有分数列
            if 'judge_score' in df.columns:
                df['judge_score'] = pd.to_numeric(df['judge_score'], errors='coerce').fillna(0.0)
                if model not in model_groups: model_groups[model] = []
                model_groups[model].append({'judge': judge, 'df': df})
        except Exception as e:
            print(f"  [Error] Failed to read {f}: {e}")

    # 2. 计算集成平均分 (Ensemble)
    final_dfs = []
    
    for model_name, items in model_groups.items():
        print(f"Processing '{model_name}' (Judged by {len(items)} models)...")
        
        if len(items) == 0: continue
        
        # 提取共同的基础列 (Video Path & GT)
        base_df = items[0]['df'][['video_path', 'ground_truth_label']].copy()
        
        # 收集所有裁判的分数
        score_cols = []
        for i, item in enumerate(items):
            j_name = item['judge']
            # 为了防止重名，加后缀
            col_name = f"score_{j_name}_{i}" 
            temp_df = item['df'][['video_path', 'judge_score']].rename(columns={'judge_score': col_name})
            
            # Merge 确保对齐
            base_df = pd.merge(base_df, temp_df, on='video_path', how='inner')
            score_cols.append(col_name)
        
        # 计算行平均分 (Ensemble Score)
        base_df['judge_score'] = base_df[score_cols].mean(axis=1)
        # 计算行标准差 (Disagreement)
        base_df['judge_std'] = base_df[score_cols].std(axis=1).fillna(0.0)
        
        # 标记来源
        judge_display = "Ensemble" if len(items) > 1 else items[0]['judge']
        base_df['evaluated_model'] = model_name
        base_df['judge_info'] = judge_display
        base_df['display_name'] = f"{model_name}\n({judge_display})"
        
        final_dfs.append(base_df)

    if not final_dfs: return None
    return pd.concat(final_dfs, ignore_index=True)

# ==========================================
# 3. 绘图函数集
# ==========================================

def plot_leaderboard(df):
    """画总体排行榜 (Bar Chart)"""
    print("  -> Drawing Leaderboard...")
    leaderboard = df.groupby('display_name')['judge_score'].mean().sort_values(ascending=False).reset_index()
    
    plt.figure(figsize=(10, max(5, len(leaderboard)*0.8)))
    # 金色高亮 Ensemble 结果
    colors = ['#FFD700' if 'Ensemble' in x else '#4682B4' for x in leaderboard['display_name']]
    
    ax = sns.barplot(x='judge_score', y='display_name', data=leaderboard, palette=colors)
    plt.title('Overall VLM Performance Leaderboard (0.0 - 1.0)', fontsize=16, weight='bold')
    plt.xlabel('Average Accuracy (Ensemble Mean)', fontsize=12)
    plt.ylabel('')
    plt.xlim(0, 1.05)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # 标数值
    for i, v in enumerate(leaderboard['judge_score']):
        ax.text(v + 0.01, i, f"{v:.1%}", va='center', fontsize=10, weight='bold')
        
    plt.tight_layout()
    plt.savefig(OUT_LEADERBOARD, dpi=300)

def plot_heatmap(df):
    """画分项热力图 (Heatmap)"""
    print("  -> Drawing Category Heatmap...")
    # 聚合数据
    pivot_table = df.groupby(['display_name', 'ground_truth_label'])['judge_score'].mean().unstack()
    
    # 智能排序：
    # 1. 动作按“难度”排序 (平均分越低越难，排左边)
    action_difficulty = pivot_table.mean(axis=0).sort_values(ascending=True)
    pivot_table = pivot_table[action_difficulty.index]
    
    # 2. 模型按“能力”排序 (平均分越高越强，排上面)
    model_power = pivot_table.mean(axis=1).sort_values(ascending=False)
    pivot_table = pivot_table.reindex(model_power.index)
    
    # 动态尺寸
    w = max(12, len(pivot_table.columns) * 0.8)
    h = max(6, len(pivot_table.index) * 1.0)
    
    plt.figure(figsize=(w, h))
    sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', fmt='.0%', 
                vmin=0, vmax=1, linewidths=1, linecolor='white',
                cbar_kws={'label': 'Accuracy'})
    
    plt.title('Performance Heatmap: Models vs Actions (Sorted by Difficulty)', fontsize=16, pad=20)
    plt.xlabel('Action Category (Hardest -> Easiest)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(OUT_HEATMAP, dpi=300)

def plot_stability(df):
    """画稳定性箱线图 (Boxplot)"""
    print("  -> Drawing Stability Boxplot...")
    
    # 只选样本数足够的动作 (>5)
    counts = df.groupby('ground_truth_label')['video_path'].nunique()
    valid_actions = counts[counts >= 5].index
    if len(valid_actions) == 0:
        print("    [Skip] Not enough samples per action for boxplot.")
        return
        
    plot_df = df[df['ground_truth_label'].isin(valid_actions)].copy()
    
    # 排序
    action_order = plot_df.groupby('ground_truth_label')['judge_score'].median().sort_values(ascending=False).index
    
    w = max(14, len(valid_actions) * 1.5)
    plt.figure(figsize=(w, 8))
    
    sns.boxplot(x='ground_truth_label', y='judge_score', hue='display_name',
                data=plot_df, order=action_order, palette="Set2",
                linewidth=1.2, fliersize=3)
    
    plt.title('Stability Analysis: Consistency Across Samples', fontsize=16, weight='bold')
    plt.ylabel('Score Distribution (0=Fail, 1=Perfect)', fontsize=12)
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Models', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_BOXPLOT, dpi=300)

def export_statistics(df):
    """生成详细统计表 (含 Risk Factor)"""
    print(f"  -> Exporting Statistics to {OUT_STATS}...")
    
    stats = df.groupby(['evaluated_model', 'ground_truth_label'])['judge_score'].agg(
        ['count', 'mean', 'std', 'min', 'max']
    ).reset_index()
    
    # Risk Factor: 平均分很高，但最小值很低，说明有致命盲区
    stats['risk_factor'] = stats['mean'] - stats['min']
    
    # Stability Score: 1 - 标准差
    stats['stability_score'] = 1.0 - stats['std'].fillna(0)
    
    # 格式化
    stats = stats.round(3)
    stats.sort_values(['evaluated_model', 'mean'], ascending=[True, False], inplace=True)
    
    stats.to_csv(OUT_STATS, index=False)

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    print("=== VLM Benchmark Master Visualizer ===")
    
    # 1. 加载并集成数据
    full_df = load_and_ensemble_data()
    
    if full_df is not None:
        # 2. 导出统计表
        export_statistics(full_df)
        
        # 3. 绘制三张核心图
        plot_leaderboard(full_df)
        plot_heatmap(full_df)
        plot_stability(full_df)
        
        print("\n=== All Done! Check the generated PNG and CSV files. ===")
    else:
        print("No valid data found to visualize.")
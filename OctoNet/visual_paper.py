import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

# ==========================================
# 1. 设置 ACM 论文绘图风格 (Single Column Optimization)
# ==========================================
def set_acm_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    
    # 保持字号，但微调布局
    base_size = 8
    plt.rcParams['font.size'] = base_size
    plt.rcParams['axes.labelsize'] = base_size
    plt.rcParams['axes.titlesize'] = base_size + 1
    plt.rcParams['xtick.labelsize'] = base_size - 1
    plt.rcParams['ytick.labelsize'] = base_size - 1
    plt.rcParams['legend.fontsize'] = base_size - 2
    plt.rcParams['figure.titlesize'] = base_size + 2
    
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

set_acm_style()

# ==========================================
# 2. 数据处理
# ==========================================
try:
    df = pd.read_csv('Table_1_Detailed_Statistics.csv')
    
    name_map = {
        'unsloth_Qwen3-VL-30B-A3B-Thinking-GGUF_Q5_K_M': 'Qwen3-VL-30B (5-bit)',
        'unsloth_Qwen3-VL-4B-Thinking-GGUF_Q8_0': 'Qwen3-VL-4B (8-bit)',
        'qwen3-vl-plus': 'Qwen3-VL-Plus',
        'grok-4-1-fast-reasoning': 'Grok-4.1-Fast',
        'gpt-5.1': 'GPT-5.1',
        'gemini-2.5-pro': 'Gemini-2.5-Pro'
    }
    df['Model'] = df['evaluated_model'].map(name_map).fillna(df['evaluated_model'])

    def get_category(label):
        label = label.lower()
        if any(x in label for x in ['fall', 'stagger', 'faint']): return 'Safety-Critical'
        if any(x in label for x in ['sit', 'walk', 'bow', 'handshake', 'hug']): return 'Macro-Action'
        if any(x in label for x in ['yawn', 'sign', 'type', 'phone', 'eat', 'drink']): return 'Micro-Action'
        return 'Other'

    df['Category'] = df['ground_truth_label'].apply(get_category)
    df_filtered = df[df['Category'] != 'Other']

    # ==========================================
    # 3. 绘图 - 针对用户反馈调整 (Taller & Cleaner)
    # ==========================================

    # --- Figure 1: Leaderboard (高度微调) ---
    plt.figure(figsize=(3.33, 2.5)) # 稍微拉高一点点
    
    leaderboard = df.groupby('Model')['mean'].mean().sort_values(ascending=False).reset_index()
    
    sns.barplot(data=leaderboard, x='mean', y='Model', palette='viridis', linewidth=0)
    
    plt.title('Overall Accuracy', pad=8)
    plt.xlabel('Mean Accuracy')
    plt.ylabel('') 
    plt.xlim(0, 0.65)
    
    for i, v in enumerate(leaderboard['mean']):
        plt.text(v + 0.01, i, f"{v:.1%}", va='center', fontsize=7)
    
    plt.tight_layout(pad=0.2)
    plt.savefig('Figure_1_Leaderboard_SingleCol_v2.png', dpi=300, bbox_inches='tight')
    print("Saved Figure 1 v2")

    # --- Figure 2: Heatmap (解决Acc遮挡，拉高) ---
    plt.figure(figsize=(3.33, 3.2)) # 拉高高度，给图例和标签更多空间
    
    pivot = df_filtered.groupby(['Model', 'Category'])['mean'].mean().unstack()
    pivot = pivot[['Macro-Action', 'Safety-Critical', 'Micro-Action']] 
    pivot = pivot.reindex(leaderboard['Model']) 

    # cbar_kws: use horizontal orientation or adjust padding if needed. 
    # Here we keep vertical but make sure figure is wide enough or tight_layout handles it.
    ax = sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', 
                vmin=0, vmax=1, 
                cbar_kws={'label': 'Acc.', 'shrink': 0.8}, # Shrink colorbar slightly
                annot_kws={"size": 7})
    
    plt.title('Performance by Category', pad=10)
    plt.ylabel('')
    plt.xlabel('')
    plt.xticks(rotation=15, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    
    # 关键：使用 bbox_inches='tight' 保存，防止边缘被切
    plt.tight_layout(pad=0.3)
    plt.savefig('Figure_2_Category_Heatmap_SingleCol_v2.png', dpi=300, bbox_inches='tight')
    print("Saved Figure 2 v2")

    # --- Figure 3: Scatter Plot (解决Label喧宾夺主，大幅拉高) ---
    plt.figure(figsize=(3.33, 4.2)) # 高度增加到 4.2，给三个区域更多垂直空间分布
    
    key_actions = ['sit', 'walk', 'handshake', 'falldown', 'stagger', 'yawn', 'makeoksign']
    scatter_df = df[df['ground_truth_label'].isin(key_actions)].copy()

    ax = plt.gca()
    sns.scatterplot(
        data=scatter_df, 
        x='mean', 
        y='stability_score', 
        hue='ground_truth_label', 
        style='Model', 
        s=45, # 点再小一点点
        palette='deep',
        ax=ax,
        legend='brief' 
    )

    # Thresholds
    ACC_THRESHOLD = 0.6
    STABILITY_THRESHOLD = 0.7 

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.4, 1.05)

    # Zones - 颜色变浅，文字变小，去粗体，位置微调
    # Failure Zone
    rect_fail = patches.Rectangle((0, 0), ACC_THRESHOLD, STABILITY_THRESHOLD, 
                                  linewidth=0, facecolor='#ffcccc', alpha=0.2, zorder=0) # alpha 0.3 -> 0.2
    ax.add_patch(rect_fail)
    ax.text(ACC_THRESHOLD/2, 0.42, 'FAILURE', 
            color='darkred', fontsize=6, alpha=0.7, ha='center', va='bottom') # 去掉 bold, fontsize 7->6, 加 alpha

    # Illusion Zone
    rect_illusion = patches.Rectangle((ACC_THRESHOLD, 0), 1.0-ACC_THRESHOLD, STABILITY_THRESHOLD, 
                                      linewidth=0, facecolor='#ffe5cc', alpha=0.2, zorder=0)
    ax.add_patch(rect_illusion)
    ax.text((1.0+ACC_THRESHOLD)/2, 0.42, 'ILLUSION', 
            color='#cc6600', fontsize=6, alpha=0.7, ha='center', va='bottom')

    # Ideal Zone
    rect_ideal = patches.Rectangle((ACC_THRESHOLD, STABILITY_THRESHOLD), 1.0-ACC_THRESHOLD, 1.05-STABILITY_THRESHOLD,
                                   linewidth=0, facecolor='#ccffcc', alpha=0.2, zorder=0)
    ax.add_patch(rect_ideal)
    ax.text((1.0+ACC_THRESHOLD)/2, 0.98, 'RELIABLE', 
            color='darkgreen', fontsize=6, alpha=0.7, ha='center', va='top')

    ax.axvline(x=ACC_THRESHOLD, color='grey', linestyle='--', linewidth=0.6, alpha=0.4)
    ax.axhline(y=STABILITY_THRESHOLD, color='grey', linestyle='--', linewidth=0.6, alpha=0.4)

    plt.title('Stability vs. Accuracy', pad=8)
    plt.xlabel('Mean Accuracy', fontsize=8)
    plt.ylabel('Stability ($1-\sigma$)', fontsize=8)
    
    # Legend 放在底部外侧，不挤占图内空间
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', 
               ncol=2, fontsize=6, frameon=False, borderaxespad=0.)
    
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout(pad=0.5) # 给 Legend 留出空间
    plt.savefig('Figure_3_Stability_SingleCol_v2.png', dpi=300, bbox_inches='tight')
    print("Saved Figure 3 v2")

except Exception as e:
    print(f"Error: {e}")
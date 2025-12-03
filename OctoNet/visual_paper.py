import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
# 1. 加载数据
df = pd.read_csv('Table_1_Detailed_Statistics.csv')

# 2. 修正模型名称 (按照你的要求)
name_map = {
    'unsloth_Qwen3-VL-30B-A3B-Thinking-GGUF_Q5_K_M': 'Qwen3-VL-30B (5-bit)',
    'unsloth_Qwen3-VL-4B-Thinking-GGUF_Q8_0': 'Qwen3-VL-4B (Q8)',
    'qwen3-vl-plus': 'Qwen3-VL-Plus (Commercial)',
    'grok-4-1-fast-reasoning': 'Grok-4.1-Fast-Reason',
    'gpt-5.1': 'GPT-5.1',
    'gemini-2.5-pro': 'Gemini-2.5-Pro'
}
df['Model'] = df['evaluated_model'].map(name_map).fillna(df['evaluated_model'])

# 3. 定义动作类别
def get_category(label):
    label = label.lower()
    if any(x in label for x in ['falldown', 'stagger', 'cough']): return 'Safety-Critical'
    if any(x in label for x in ['sit', 'walk', 'bow', 'handshake', 'hug','someone']): return 'Macro-Action'
    if any(x in label for x in ['yawn', 'thumb', 'type', 'phone', 'eat', 'drink','sign']): return 'Micro-Action'
    return 'Other'

df['Category'] = df['ground_truth_label'].apply(get_category)
df_filtered = df[df['Category'] != 'Other']

# --- Figure 1: Overall Leaderboard ---
plt.figure(figsize=(10, 6))
leaderboard = df.groupby('Model')['mean'].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=leaderboard, x='mean', y='Model', palette='viridis')
plt.title('Overall Zero-Shot Accuracy on Thermal OctoNet')
plt.xlabel('Mean Accuracy')
plt.xlim(0, 0.6)
for i, v in enumerate(leaderboard['mean']):
    plt.text(v + 0.005, i, f"{v:.1%}", va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('Figure_1_Leaderboard.png')

# --- Figure 2: Category Gap Heatmap ---
pivot = df_filtered.groupby(['Model', 'Category'])['mean'].mean().unstack()
pivot = pivot[['Macro-Action', 'Safety-Critical', 'Micro-Action']] # Order columns
pivot = pivot.reindex(leaderboard['Model']) # Order rows

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', vmin=0, vmax=1)
plt.title('The Capability Gap: Accuracy by Action Type')
plt.tight_layout()
plt.savefig('Figure_2_Category_Heatmap.png')

# --- Figure 3: Stability Analysis (Scatter) ---
# Select representative actions for clarity
key_actions = ['sit', 'eat', 'handshake', 'falldown', 'stagger', 'yawn', 'thumbup','cough']
scatter_df = df[df['ground_truth_label'].isin(key_actions)].copy()
# --- Re-draw Figure 3 with Explicit Rectangles ---
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter Plot
sns.scatterplot(
    data=scatter_df, 
    x='mean', 
    y='stability_score', 
    hue='ground_truth_label', 
    style='Model', 
    s=150, 
    palette='deep',
    ax=ax
)

# Define Thresholds
ACC_THRESHOLD = 0.6
STABILITY_THRESHOLD = 0.7  # Mean stability is roughly 0.76, so 0.7 is a good cut-off for "Low"

# Zone 1: Critical Failure (Low Acc, Low Stability)
# Rectangle(xy, width, height)
# Assuming plot limits: X (0, 1.0), Y (0, 1.0)
# We set limits explicitly to make sure rectangles fit
ax.set_xlim(0, 1.05)
ax.set_ylim(0.4, 1.05) # Stability usually doesn't go to 0, zoom in a bit

# Draw "Critical Failure Zone" (Bottom Left)
rect_fail = patches.Rectangle((0, 0), ACC_THRESHOLD, STABILITY_THRESHOLD, 
                                linewidth=0, edgecolor='none', facecolor='red', alpha=0.1, zorder=0)
ax.add_patch(rect_fail)
ax.text(ACC_THRESHOLD/2, 0.45, 'CRITICAL FAILURE\n(Blind & Unstable)', 
        color='darkred', fontsize=12, fontweight='bold', ha='center', va='bottom')

# Draw "Safety Illusion Zone" (Bottom Right)
rect_illusion = patches.Rectangle((ACC_THRESHOLD, 0), 1.0-ACC_THRESHOLD, STABILITY_THRESHOLD, 
                                    linewidth=0, edgecolor='none', facecolor='orange', alpha=0.1, zorder=0)
ax.add_patch(rect_illusion)
ax.text((1.0+ACC_THRESHOLD)/2, 0.45, 'SAFETY ILLUSION\n(Lucky & Unstable)', 
        color='darkorange', fontsize=12, fontweight='bold', ha='center', va='bottom')

# Draw "Ideal Zone" (Top Right)
rect_ideal = patches.Rectangle((ACC_THRESHOLD, STABILITY_THRESHOLD), 1.0-ACC_THRESHOLD, 1.05-STABILITY_THRESHOLD,
                                linewidth=0, edgecolor='none', facecolor='green', alpha=0.05, zorder=0)
ax.add_patch(rect_ideal)
ax.text((1.0+ACC_THRESHOLD)/2, 0.95, 'RELIABLE ZONE', 
        color='green', fontsize=12, fontweight='bold', ha='center')

# Add Reference Lines
ax.axvline(x=ACC_THRESHOLD, color='grey', linestyle='--', alpha=0.5)
ax.axhline(y=STABILITY_THRESHOLD, color='grey', linestyle='--', alpha=0.5)

ax.set_title('The Safety Illusion Landscape (Stability vs. Accuracy)', fontsize=16)
ax.set_xlabel('Mean Accuracy (Performance)', fontsize=14)
ax.set_ylabel('Stability Score (Consistency)', fontsize=14)

# Legend
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.savefig('Figure_3_Stability_Overlay_Fixed.png', dpi=300)

print("Figure 3 regenerated with explicit zones.")


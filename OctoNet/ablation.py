import pandas as pd
import glob
import numpy as np

# ================= 配置区域 =================
DYNAMIC_ACTIONS = [
    'falldown', 'stagger', 'jumprope', 'boxing', 
    'kicksomeone', 'punchsomeone', 'jumpingjack', 
    'jog', 'dance','drawzigzag','drawcircleclockwise','drawcirclecounterclockwise'
]

# 请确保这里的文件名是正确的
FILES = {
    "Qwen3-VL-30B": {
        "3fps": "benchmark_results_unsloth_Qwen3-VL-30B-A3B-Thinking-GGUF_Q5_K_M_20251124_162149_judged_*.csv",
        "10fps": "benchmark_results_unsloth_Qwen3-VL-30B-A3B-Thinking-GGUF_Q5_K_M_20251204_160026_judged_*.csv"
    },
    "Qwen3-VL-4B": {
        "3fps": "benchmark_results_unsloth_Qwen3-VL-4B-Thinking-GGUF_Q8_0_20251124_174910_judged_*.csv", 
        "10fps": "benchmark_results_unsloth_Qwen3-VL-4B-Thinking-GGUF_Q8_0_20251204_190151_judged_*.csv" 
    }
}

def calculate_metrics(pattern, fps_label):
    files = glob.glob(pattern)
    if not files: return None
    
    # 1. 读取并合并裁判分数
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df = df.sort_values('video_path').reset_index(drop=True)
            dfs.append(df['judge_score'])
        except: pass
    
    if not dfs: return None
    
    # 2. 计算每个视频的 Ensemble Mean Score
    judge_scores = pd.concat(dfs, axis=1)
    video_scores = judge_scores.mean(axis=1) # 这是每个视频的最终得分
    
    # 3. 构造基础数据表
    base_df = pd.read_csv(files[0])[['video_path', 'ground_truth_label']].copy()
    base_df['score'] = video_scores
    
    # 4. 筛选动态动作
    # 使用模糊匹配 (只要包含关键词就算)
    mask = base_df['ground_truth_label'].apply(lambda x: any(d in str(x).lower() for d in DYNAMIC_ACTIONS))
    dynamic_df = base_df[mask].copy()
    
    if len(dynamic_df) == 0: return None
    
    # 5. [核心修正] 计算稳定性
    # 先按具体动作分组计算 std，再取平均，避免不同动作难度的基准差异干扰
    action_stats = dynamic_df.groupby('ground_truth_label')['score'].agg(['mean', 'std'])
    
    # Stability = 1 - std
    # 如果某个动作只有一个视频，std为NaN，视为稳定(1.0)，或者是0(根据你的定义)
    # 通常填0意味着无波动。这里我们fillna(0)
    action_stats['stability'] = 1.0 - action_stats['std'].fillna(0.0)
    
    # 最终指标：所有动态动作的平均准确率 和 平均稳定性
    final_acc = action_stats['mean'].mean()
    final_stab = action_stats['stability'].mean()
    
    return final_acc, final_stab

# ================= 主程序 =================
results = []

print("Running corrected ablation analysis...")

for model, conditions in FILES.items():
    # 计算 3fps
    res_3 = calculate_metrics(conditions["3fps"], "3fps")
    # 计算 10fps
    res_10 = calculate_metrics(conditions["10fps"], "10fps")
    
    if res_3 and res_10:
        acc_3, stab_3 = res_3
        acc_10, stab_10 = res_10
        
        results.append({
            "Model": model,
            "Acc (3fps)": f"{acc_3:.1%}",
            "Acc (10fps)": f"{acc_10:.1%}",
            "Δ Acc": f"+{acc_10 - acc_3:.1%}",
            "Stab (3fps)": f"{stab_3:.3f}",
            "Stab (10fps)": f"{stab_10:.3f}",
            "Δ Stab": f"{stab_10 - stab_3:+.3f}"
        })

print(pd.DataFrame(results))
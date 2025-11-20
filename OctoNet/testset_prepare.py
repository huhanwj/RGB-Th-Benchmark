import os
import json
import cv2
import sys
from datetime import datetime

# --- 1. 配置 ---

# 输入：你原始的 'data' 文件夹
INPUT_BASE_PATH = r"D:\OctoNet-upload\node_3\seekThermal"

# 输出：一个新的、干净的、处理后的数据集文件夹
OUTPUT_BASE_PATH = r"D:\OctoNet\processed_thermal_dataset"
OUTPUT_VIDEO_DIR = os.path.join(OUTPUT_BASE_PATH, "videos")
OUTPUT_JSON_FILE = os.path.join(OUTPUT_BASE_PATH, "video_qa_set.json")

# 视频设置
FPS = 30  # 你希望合成的视频的帧率

# --- 2. [复用] 时间戳解析函数 ---

def extract_time_from_filename(filename):
    """
    从 'thermal_YYYY-MM-DD HH_MM_SS.ffffff.png' 格式的文件名中提取datetime对象
    """
    try:
        timestamp_str = filename[len("thermal_"):-len(".png")]
    except IndexError:
        return None 

    if ' ' in timestamp_str:
        timestamp_str_fmt = timestamp_str.replace('_', ':', 2)
    else:
        timestamp_str_fmt = timestamp_str.replace('_', ' ', 1)
        timestamp_str_fmt = timestamp_str_fmt.replace('_', ':', 2)

    if '.' not in timestamp_str_fmt:
        timestamp_str_fmt += ".000000"
        
    try:
        return datetime.strptime(timestamp_str_fmt, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        # print(f"警告: 解析时间戳失败: '{timestamp_str_fmt}'")
        return None

# --- 3. [复用] 视频创建辅助函数 ---

def create_video_clip(image_paths, output_path, fps):
    if not image_paths:
        return False
    
    try:
        first_frame = cv2.imread(image_paths[0])
        if first_frame is None:
            return False
        height, width, _ = first_frame.shape
        frame_size = (width, height)
    except Exception:
        return False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    if not out.isOpened():
        return False

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None: continue
        
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, frame_size)
        out.write(img)
    
    out.release()
    return True

# --- 4. [修改] 主处理函数 ---

def process_data_into_dataset():
    print(f"正在创建输出目录: {OUTPUT_BASE_PATH}")
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    
    qa_pairs = []
    video_counter = 1
    
    print(f"开始遍历输入目录: {INPUT_BASE_PATH}")
    
    for root, dirs, files in os.walk(INPUT_BASE_PATH):
        
        # 1. 检查是否包含PNG
        png_files = [f for f in files if f.endswith(".png")]
        if not png_files:
            continue
            
        print(f"\n--- 正在处理会话: {root} ---")
        
        # 2. [修正点] 智能提取活动标签
        folder_name = os.path.basename(root)
        parts = folder_name.split("_")
        
        activity_label = "unknown" # 默认值
        
        # 逻辑：查找 "activity" 关键词，然后取它后面的那个词
        if "activity" in parts:
            try:
                index = parts.index("activity")
                # 确保 "activity" 不是最后一个词
                if index + 1 < len(parts):
                    activity_label = parts[index + 1]
            except ValueError:
                pass
        
        # 如果没找到 activity 标签，我们可以选择跳过或者标记为 unknown
        # 这里我们打印一个警告
        if activity_label == "unknown":
            print(f"  警告: 无法在文件夹名中找到 'activity' 标签。文件夹名: {folder_name}")
            # 如果你希望严格跳过没标签的数据，取消下面这行的注释：
            # continue 
        else:
            print(f"  成功识别活动: {activity_label}")

        # 3. 排序帧
        files_with_times = []
        for file_name in png_files:
            parsed_time = extract_time_from_filename(file_name)
            if parsed_time:
                files_with_times.append((parsed_time, os.path.join(root, file_name)))
            
        if not files_with_times:
            continue
            
        files_with_times.sort(key=lambda x: x[0])
        sorted_file_paths = [path for dt, path in files_with_times]

        # 4. 创建视频
        # 文件名带上标签方便人工检查
        video_filename = f"session_{video_counter:04d}_{activity_label}.mp4"
        absolute_video_path = os.path.join(OUTPUT_VIDEO_DIR, video_filename)
        
        print(f"  正在创建视频 ({len(sorted_file_paths)} 帧)...")
        success = create_video_clip(sorted_file_paths, absolute_video_path, FPS)
        
        if not success:
            print(f"  创建视频失败，跳过。")
            continue

        # 5. 生成QA对
        relative_video_path = os.path.join("videos", video_filename)
        
        qa_pair = {
            "question": "根据这段热成像视频，描述人物的动作。",
            "video_path": relative_video_path,
            "action": activity_label
        }
        qa_pairs.append(qa_pair)
        video_counter += 1

    # 6. 保存QA JSON
    if not qa_pairs:
        print("\n警告：未生成任何数据。")
        return

    print(f"\n--- 处理完成 ---")
    print(f"总共生成: {len(qa_pairs)} 个视频 QA 对")
    
    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
        
    print(f"JSON保存至: {OUTPUT_JSON_FILE}")

if __name__ == "__main__":
    if 'cv2' not in sys.modules:
        print("请先安装 opencv-python")
        sys.exit()
    process_data_into_dataset()
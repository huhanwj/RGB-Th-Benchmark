import json
import os
import time
import base64
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod

# ==========================================
# 0. 配置与 API Key
# ==========================================

# 输入输出文件配置
# 指向 testset_prepare_v2.py 生成的 JSON
INPUT_JSON = r'F:\OctoNet\node_3\seekThermal_v2\video_qa_set.json'
OUTPUT_CSV = 'benchmark_results_raw_gemini.csv'

# API KEYS
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-your-qwen-key")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-openai-key")

# Qwen Base URL
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ==========================================
# 1. 图像处理工具 (Image Utilities)
# ==========================================

def load_sampled_frames(video_relative_path, base_dir):
    """
    根据 video_path (e.g. 'videos/session_001.mp4')
    加载对应的采样帧 (e.g. 'sampled_frames/session_001/*.jpg')
    返回 Base64 编码的图像列表
    """
    # 1. 获取视频文件名（无扩展名）
    video_filename = os.path.basename(video_relative_path)
    video_name_no_ext = os.path.splitext(video_filename)[0]
    
    # 2. 构造采样帧目录路径
    # 假设目录结构:
    # root/
    #   videos/
    #   sampled_frames/
    #     session_001/
    #       frame_00.jpg
    #       ...
    frames_dir = os.path.join(base_dir, "sampled_frames", video_name_no_ext)
    
    if not os.path.exists(frames_dir):
        print(f"Warning: Frames dir not found: {frames_dir}")
        return []
    
    # 3. 读取所有jpg并按文件名排序
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    encoded_frames = []
    for f in frame_files:
        file_path = os.path.join(frames_dir, f)
        with open(file_path, "rb") as img_file:
            b64_str = base64.b64encode(img_file.read()).decode('utf-8')
            encoded_frames.append(b64_str)
            
    return encoded_frames

# ==========================================
# 2. 模型封装 (Model Wrappers)
# ==========================================

class BaseVLM(ABC):
    @abstractmethod
    def generate_response(self, video_path, prompt, base_dir):
        pass

# --- Model A: Qwen (Qwen3-VL-Plus/Max) ---
class QwenVLM(BaseVLM):
    def __init__(self, model_name="qwen3-vl-plus", api_key=None, base_url=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        print(f"Initialized Qwen VLM: {model_name}")

    def generate_response(self, video_path, prompt, base_dir):
        frames = load_sampled_frames(video_path, base_dir)
        if not frames: return "ERROR: No frames found"

        # 构造多图输入
        content = []
        for b64_img in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })
        content.append({"type": "text", "text": prompt})

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            # temperature=0.01, # Removed as requested
        )
        return completion.choices[0].message.content

# --- Model B: Gemini (Gemini-1.5/3.0) ---
class GeminiVLM(BaseVLM):
    def __init__(self, model_name="gemini-1.5-pro", api_key=None):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print(f"Initialized Gemini VLM (Multi-Image): {model_name}")

    def generate_response(self, video_path, prompt, base_dir):
        from google.genai import types
        
        frames = load_sampled_frames(video_path, base_dir)
        if not frames: return "ERROR: No frames found"
        
        # 构造内容: 多个 Image Blob + 文本
        parts = []
        for b64_img in frames:
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        data=base64.b64decode(b64_img), 
                        mime_type='image/jpeg'
                    )
                )
            )
        parts.append(types.Part(text=prompt))

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Content(parts=parts)],
            # config=types.GenerateContentConfig(temperature=0.0) # Removed
        )
        return response.text

# ==========================================
# 3. 主程序 (Main Execution)
# ==========================================

def main():
    # 1. 加载数据
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found.")
        return

    # 获取数据集根目录 (假设 JSON 在根目录下)
    DATASET_ROOT = os.path.dirname(INPUT_JSON)

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # 修复路径分隔符
    for item in full_data:
        item['video_path'] = item['video_path'].replace('\\', '/')

    print(f"Loaded {len(full_data)} videos from {INPUT_JSON}")
    print(f"Dataset Root: {DATASET_ROOT}")

    # ==========================================
    # 2. 模型选择
    # ==========================================
    
    # 选项 A: Qwen
    # runner = QwenVLM(model_name="qwen3-vl-plus", api_key=DASHSCOPE_API_KEY, base_url=QWEN_BASE_URL)
    
    # 选项 B: Gemini
    runner = GeminiVLM(model_name="gemini-2.5-pro", api_key=GEMINI_API_KEY)
    
    # ==========================================
    
    # 3. 开始测试
    results = []
    
    prompt = (
        "Watch this thermal video sequence carefully. Describe the single main human action shown. "
        "Use a short phrase (e.g., 'walking', 'falling down', 'sitting', 'yawning'). "
        "Do not describe the background or temperature colors."
    )

    print(f"Starting inference using {runner.model_name}...")
    
    for item in tqdm(full_data, desc="Processing"):
        video_path = item['video_path']
        gt_action = item['action']
        
        try:
            response_text = ""
            for attempt in range(3):
                try:
                    # 传入 DATASET_ROOT 以便找到 sampled_frames
                    response_text = runner.generate_response(video_path, prompt, DATASET_ROOT)
                    break
                except Exception as e:
                    print(f"Retry {attempt+1} for {video_path}: {e}")
                    time.sleep(2)
            
            if not response_text:
                response_text = "ERROR_API_FAIL"

            results.append({
                "video_path": video_path,
                "ground_truth_label": gt_action,
                "vlm_raw_response": response_text
            })
            
            time.sleep(0.5)

        except Exception as e:
            print(f"Critical error on {video_path}: {e}")
            results.append({
                "video_path": video_path,
                "ground_truth_label": gt_action,
                "vlm_raw_response": f"CRITICAL_ERROR: {str(e)}"
            })

    # 4. 保存结果
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Results saved to: {os.path.abspath(OUTPUT_CSV)}")

if __name__ == "__main__":
    main()
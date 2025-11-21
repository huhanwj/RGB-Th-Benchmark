import json
import os
import time
import base64
import math
import cv2
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod

# ==========================================
# 0. 配置与 API Key (请在此处填入你的Key)
# ==========================================

# 输入输出文件配置
INPUT_JSON = 'video_qa_set.json'
OUTPUT_CSV = 'benchmark_results_raw_gemini.csv'

# API KEYS
# 建议使用环境变量，或者直接替换下方的字符串
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-your-qwen-key")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-openai-key")

# Qwen Base URL (阿里云百炼)
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ==========================================
# 1. 视频处理工具 (Video Utilities)
# ==========================================

def encode_video_base64(video_path):
    """
    [Qwen专用] 读取整个视频文件并转为 Base64 字符串
    """
    if not os.path.exists(video_path):
        return None
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

def extract_frames_1fps(video_path):
    """
    [GPT-4o专用] 按 1fps (每秒一帧) 抽取关键帧并转为 Base64
    """
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    
    # 获取视频原始帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0 or fps <= 0:
        cap.release()
        return None

    # 计算步长：保证每秒抽1帧
    step = int(math.ceil(fps))
    
    encoded_frames = []
    for frame_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # 可选：缩放以节省 Token (320x240其实可以不缩放)
            # frame = cv2.resize(frame, (320, 240)) 
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            encoded_frames.append(base64_image)
            
    cap.release()
    
    # 兜底：如果视频极短，至少取第一帧
    if not encoded_frames and total_frames > 0:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            encoded_frames.append(base64.b64encode(buffer).decode('utf-8'))
        cap.release()

    return encoded_frames

# ==========================================
# 2. 模型封装 (Model Wrappers)
# ==========================================

class BaseVLM(ABC):
    @abstractmethod
    def generate_response(self, video_path, prompt):
        pass

# --- Model A: Qwen (Qwen3-VL-Plus/Max) ---
class QwenVLM(BaseVLM):
    def __init__(self, model_name="qwen3-vl-plus", api_key=None, base_url=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        print(f"Initialized Qwen VLM: {model_name}")

    def generate_response(self, video_path, prompt):
        b64_video = encode_video_base64(video_path)
        if not b64_video: return "ERROR: Video file missing"

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{b64_video}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.01,
        )
        return completion.choices[0].message.content

# --- Model B: Gemini (Gemini-1.5/3.0 - New SDK) ---
class GeminiVLM(BaseVLM):
    def __init__(self, model_name="gemini-1.5-pro", api_key=None):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print(f"Initialized Gemini VLM (Inline): {model_name}")

    def generate_response(self, video_path, prompt):
        from google.genai import types
        
        if not os.path.exists(video_path): return "ERROR: Video file missing"
        
        # 读取二进制数据
        with open(video_path, 'rb') as f:
            video_bytes = f.read()

        # 使用最新的 inline data 方式 (v1.0+ SDK style)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                data=video_bytes, 
                                mime_type='video/mp4'
                            )
                        ),
                        types.Part(text=prompt)
                    ]
                )
            ]
        )
        return response.text

# --- Model C: OpenAI (GPT-4o / GPT-5) ---
class GPTVLM(BaseVLM):
    def __init__(self, model_name="gpt-5.1", api_key=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        print(f"Initialized GPT VLM (1fps mode): {model_name}")

    def generate_response(self, video_path, prompt):
        frames = extract_frames_1fps(video_path)
        if not frames: return "ERROR: Extraction failed"

        content_payload = [{"type": "text", "text": prompt}]
        for f in frames:
            content_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{f}",
                    "detail": "low" # 强制低细节模式节省 Token
                }
            })

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content_payload}],
            temperature=0.0
        )
        return response.choices[0].message.content

# ==========================================
# 3. 主程序 (Main Execution)
# ==========================================

def main():
    # 1. 加载数据
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found.")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # 修复路径分隔符 (Windows "\" -> Linux "/")
    for item in full_data:
        item['video_path'] = item['video_path'].replace('\\', '/')

    print(f"Loaded {len(full_data)} videos from {INPUT_JSON}")

    # ==========================================
    # 2. 模型选择 (请在此处 取消注释 你要测试的模型)
    # ==========================================
    
    # 选项 A: Qwen (阿里云)
    runner = QwenVLM(model_name="qwen3-vl-plus", api_key=DASHSCOPE_API_KEY, base_url=QWEN_BASE_URL)
    
    # 选项 B: Gemini (Google) - 使用最新的 gemini-2.0-flash 或 1.5-pro
    # 注意：如果你的账号有 gemini-3-pro-preview 权限，直接改 model_name 即可
    # runner = GeminiVLM(model_name="gemini-3-pro-preview", api_key=GEMINI_API_KEY)
    
    # 选项 C: GPT (OpenAI)
    # runner = GPTVLM(model_name="gpt-4o", api_key=OPENAI_API_KEY)

    # ==========================================
    
    # 3. 开始测试
    results = []
    
    # Prompt: 专注于动作描述，简短有力
    prompt = (
        "Watch this thermal video carefully. Describe the single main human action shown in the video. "
        "Use a short phrase (e.g., 'walking', 'falling down', 'sitting', 'yawning'). "
        "Do not describe the background or temperature colors."
    )

    print(f"Starting inference using {runner.model_name}...")
    
    # 可以在这里用 full_data[:10] 先测前10个
    for item in tqdm(full_data, desc="Processing"):
        video_path = item['video_path']
        gt_action = item['action']
        
        try:
            # 简单的重试机制
            response_text = ""
            for attempt in range(3):
                try:
                    response_text = runner.generate_response(video_path, prompt)
                    break
                except Exception as e:
                    # 打印错误但继续重试
                    print(f"Retry {attempt+1} for {video_path}: {e}")
                    time.sleep(2)
            
            if not response_text:
                response_text = "ERROR_API_FAIL"

            results.append({
                "video_path": video_path,
                "ground_truth_label": gt_action,
                "vlm_raw_response": response_text
            })
            
            # 速率限制保护
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
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
INPUT_JSON = r'F:\OctoNet\node_3\seekThermal_v2\high\video_qa_set.json'
OUTPUT_CSV = 'benchmark_results_unified_highfps.csv'

# API KEYS
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

# Local Llama.cpp Configuration
LOCAL_LLAMA_URL = "http://192.168.77.42:8000/v1"
LOCAL_LLAMA_API_KEY = "no-key-required"

# ==========================================
# 1. 核心 Prompt (Core Prompt)
# ==========================================
# 不要用具体的分析步骤，直接给出最终的action来避免模型陷入分析步骤的陷阱
CORE_PROMPT = """
### Role
You are an expert in Intelligent Surveillance and Human Activity Recognition (HAR) using Thermal Infrared Cameras.

### Visual Context
- The provided video frames are from a low-resolution thermal camera (brighter pixels = hotter objects).
- The background is static and cold (dark).
- Your goal is to identify the specific human action despite the lack of texture and facial features.

### Final Output Format
Action: [The final action label]
"""

# ==========================================
# 2. 图像处理工具 (Image Utilities)
# ==========================================

def load_sampled_frames(video_relative_path, base_dir):
    """
    根据 video_path (e.g. 'videos/session_001.mp4')
    加载对应的采样帧 (e.g. 'sampled_frames/session_001/*.jpg')
    返回 Base64 编码的图像列表
    """
    video_filename = os.path.basename(video_relative_path)
    video_name_no_ext = os.path.splitext(video_filename)[0]
    
    frames_dir = os.path.join(base_dir, "sampled_frames", video_name_no_ext)
    
    if not os.path.exists(frames_dir):
        # 尝试 fallback 到 videos 同级目录
        # 假设结构可能是 root/videos 和 root/sampled_frames
        # base_dir 应该是 dataset root
        pass

    if not os.path.exists(frames_dir):
        print(f"Warning: Frames dir not found: {frames_dir}")
        return []
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    encoded_frames = []
    for f in frame_files:
        file_path = os.path.join(frames_dir, f)
        with open(file_path, "rb") as img_file:
            b64_str = base64.b64encode(img_file.read()).decode('utf-8')
            encoded_frames.append(b64_str)
            
    return encoded_frames

# ==========================================
# 3. 模型封装 (Model Wrappers)
# ==========================================

class BaseVLM(ABC):
    @abstractmethod
    def generate_response(self, video_path, prompt, base_dir):
        pass

# --- Model A: Qwen (Qwen3-VL-Plus/Max) ---
class QwenVLM(BaseVLM):
    def __init__(self, model_name="qwen3-vl-plus", api_key=None):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_name = model_name
        print(f"Initialized Qwen VLM: {model_name}")

    def generate_response(self, video_path, prompt, base_dir):
        frames = load_sampled_frames(video_path, base_dir)
        if not frames: return "ERROR: No frames found"

        content = []
        for b64_img in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })
        content.append({"type": "text", "text": prompt})

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            extra_body={'enable_thinking': True},
        )
        return completion.choices[0].message.content

# --- Model B: Gemini (Gemini-1.5/2.5/3.0) ---
class GeminiVLM(BaseVLM):
    def __init__(self, model_name="gemini-2.5-pro", api_key=None):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print(f"Initialized Gemini VLM: {model_name}")

    def generate_response(self, video_path, prompt, base_dir):
        from google.genai import types
        
        frames = load_sampled_frames(video_path, base_dir)
        if not frames: return "ERROR: No frames found"
        
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
        )
        return response.text

# --- Model C: OpenAI (GPT-4o / GPT-5) ---
class GPTVLM(BaseVLM):
    def __init__(self, model_name="gpt-4o", api_key=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        print(f"Initialized GPT VLM: {model_name}")

    def generate_response(self, video_path, prompt, base_dir):
        frames = load_sampled_frames(video_path, base_dir)
        if not frames: return "ERROR: No frames found"

        # Construct the input content list
        input_content = [{"type": "input_text", "text": prompt}]
        for f in frames:
            input_content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{f}"
            })

        response = self.client.responses.create(
            model=self.model_name,
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "user",
                    "content": input_content
                }
            ]
        )
        
        # TODO: Verify the correct field to access the text content.
        # For now, returning the response object as a string or trying to access 'output_text' if it exists.
        # Based on typical new API patterns, it might be response.output_text or similar.
        # If the user example just prints it, maybe str(response) is enough for debugging.
        try:
            return response.output_text
        except AttributeError:
            return str(response)

# --- Model D: Grok (xAI) ---
class GrokVLM(BaseVLM):
    def __init__(self, model_name="grok-vision-beta", api_key=None):
        try:
            from xai_sdk import Client
        except ImportError:
            print("Error: xai_sdk not installed.")
            raise
        self.client = Client(api_key=api_key)
        self.model_name = model_name
        print(f"Initialized Grok VLM: {model_name}")

    def generate_response(self, video_path, prompt, base_dir):
        from xai_sdk.chat import user, image
        
        frames = load_sampled_frames(video_path, base_dir)
        if not frames: return "ERROR: No frames found"

        content_parts = []
        content_parts.append(prompt)
        
        for f in frames:
            data_uri = f"data:image/jpeg;base64,{f}"
            content_parts.append(image(image_url=data_uri, detail="high"))
            
        chat = self.client.chat.create(model=self.model_name)
        chat.append(user(*content_parts))
        
        response = chat.sample()
        return response.content

# --- Model E: Local Llama.cpp (OpenAI Compatible) ---
class LlamaCppVLM(BaseVLM):
    def __init__(self, model_name="local-model", base_url=None, api_key=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        print(f"Initialized Local Llama.cpp VLM: {model_name} at {base_url}")

    def generate_response(self, video_path, prompt, base_dir):
        frames = load_sampled_frames(video_path, base_dir)
        if not frames: return "ERROR: No frames found"

        content_payload = [{"type": "text", "text": prompt}]
        for f in frames:
            content_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{f}",
                    # Local models might ignore 'detail', but we send it anyway
                }
            })

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content_payload}],
        )
        return response.choices[0].message.content

# ==========================================
# 4. 主程序 (Main Execution)
# ==========================================

def main():
    # 1. 加载数据
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found.")
        return

    DATASET_ROOT = os.path.dirname(INPUT_JSON)

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # 修复路径
    for item in full_data:
        item['video_path'] = item['video_path'].replace('\\', '/')

    print(f"Loaded {len(full_data)} videos from {INPUT_JSON}")
    print(f"Dataset Root: {DATASET_ROOT}")

    # ==========================================
    # 2. 模型选择 (请取消注释你要运行的模型)
    # ==========================================
    
    # --- Qwen ---
    # runner = QwenVLM(model_name="qwen3-vl-plus", api_key=DASHSCOPE_API_KEY)
    
    # --- Gemini ---
    # runner = GeminiVLM(model_name="gemini-3-pro-preview", api_key=GEMINI_API_KEY)
    
    # --- GPT ---
    # runner = GPTVLM(model_name="gpt-5.1", api_key=OPENAI_API_KEY)
    
    # --- Grok ---
    # runner = GrokVLM(model_name="grok-4-1-fast-reasoning", api_key=XAI_API_KEY)
    
    # --- Local Llama.cpp ---
    # 假设你本地启动了 llama-server，端口 8000
    runner = LlamaCppVLM(
        model_name="unsloth/Qwen3-VL-4B-Thinking-GGUF:Q8_0", # 这里的名字通常不重要，取决于server加载的模型
        base_url=LOCAL_LLAMA_URL, 
        api_key=LOCAL_LLAMA_API_KEY
    )

    # ==========================================
    
    # 3. 开始测试
    results = []
    
    print(f"Starting inference using {runner.model_name}...")
    print(f"Prompt Length: {len(CORE_PROMPT)} chars")
    
    for item in tqdm(full_data, desc="Processing"):
        video_path = item['video_path']
        gt_action = item['action']
        
        try:
            response_text = ""
            for attempt in range(3):
                try:
                    response_text = runner.generate_response(video_path, CORE_PROMPT, DATASET_ROOT)
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
    # 动态生成文件名以避免覆盖
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_safe_name = runner.model_name.replace("/", "_").replace(":", "_")
    final_csv_name = f"benchmark_results_{model_safe_name}_{timestamp}.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(final_csv_name, index=False)
    print(f"\nDone! Results saved to: {os.path.abspath(final_csv_name)}")

if __name__ == "__main__":
    main()

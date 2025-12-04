import pandas as pd
import os
import argparse
from openai import OpenAI
from zai import ZhipuAiClient
from tqdm import tqdm
import time
import glob

# Configuration
LOCAL_LLAMA_URL = "http://192.168.77.38:8000/v1"
LOCAL_LLAMA_API_KEY = "no-key-required"
JUDGE_MODEL_LOCAL = "local-model"

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
JUDGE_MODEL_DASHSCOPE = "qwen3-max"

# ZAI_BASE_URL = "https://api.zhipuai.com/v1"
ZAI_API_KEY = os.getenv("ZAI_API_KEY", "")
JUDGE_MODEL_ZAI = "glm-4.6"

def get_judge_response(client, ground_truth, prediction, model_name):
    prompt = f"""
### Task
You are an impartial judge evaluating the performance of a Vision-Language Model (VLM) on a Human Activity Recognition (HAR) task.
Your goal is to determine if the VLM's predicted action matches the Ground Truth action. 

### Context & Constraints
1. **Input Nature**: The VLM sees a short, low-resolution thermal video clip.
2. **Label Nature**: Ground Truth labels are high-level categories (e.g., "sit", "gym").
3. **Temporal Partiality**: A "sit" label might correspond to a clip of a person "standing up" or "sitting down". These are CORRECT.
4. **Abstract Labels**: Labels like "gym", "freestyle", or "exercise" encompass many movements. Any specific exercise or active movement is CORRECT. However, generic static descriptions like "standing" for these dynamic labels should be penalized unless the video clearly shows inactivity. Here freestyle do not mean freestyle swimming or dancing, it just tells the test person to do whatever action he/she wants.

### Input Data
- **Ground Truth Action**: "{ground_truth}"
- **VLM Predicted Action**: "{prediction}"

### Evaluation Criteria
1. **High Score (0.8 - 1.0)**:
   - Exact match or semantic equivalent (e.g., "walk" == "ambulate").
   - Logical temporal part of the action (e.g., GT="sit", Pred="standing up" -> Correct).
   - Specific valid action for abstract labels (e.g., GT="gym", Pred="lifting weights" -> Correct).

2. **Medium Score (0.4 - 0.7)**:
   - Related but less specific (e.g., GT="boxing", Pred="exercising").
   - Plausible but ambiguous interaction (e.g., GT="handshake", Pred="interacting").

3. **Low Score (0.0 - 0.3)**:
   - Completely different action (e.g., GT="run", Pred="sleep").
   - Generic static label for clearly dynamic GT (e.g., GT="gym", Pred="standing" -> Low Score).
   - Hallucinations or error messages.

Final words: When you grade, take the similarity of the action into account, not all rely on the exactness of the action.
### Output Format
Return a JSON object with two fields:
- "score": A float between 0.0 and 1.0.
- "reasoning": A brief explanation of your decision.

Example Output:
{{
    "score": 1.0,
    "reasoning": "The prediction 'standing up' is a valid temporal part of the ground truth action 'sit'."
}}
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            thinking={
                "type": "disabled"  # 启用深度思考模式
            },
            # extra_body={"enable_thinking":False},
            # temperature=0.05,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return f'{{"score": 0.0, "reasoning": "Error calling judge: {str(e)}"}}'

def process_file(input_path, client, model_name):
    print(f"\nProcessing: {input_path}")
    
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_judged_{model_name}{ext}"
    
    if os.path.exists(output_path):
        print(f"Skipping {input_path} (Output already exists: {output_path})")
        return

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return
    
    # Check for required columns
    required_columns = ['ground_truth_label', 'vlm_raw_response']
    for col in required_columns:
        if col not in df.columns:
            print(f"Skipping {input_path}: Missing required column '{col}'")
            return

    judge_scores = []
    judge_reasonings = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Judging {os.path.basename(input_path)}"):
        gt = row['ground_truth_label']
        pred = row['vlm_raw_response']
        
        # Simple pre-check to save time on obvious matches
        if str(gt).lower().strip() == str(pred).lower().strip():
             judge_scores.append(1.0)
             judge_reasonings.append("Exact string match (auto-judged).")
             continue

        json_response = get_judge_response(client, gt, pred, model_name)
        
        try:
            import json
            result = json.loads(json_response)
            judge_scores.append(result.get("score", 0.0))
            judge_reasonings.append(result.get("reasoning", "No reasoning provided."))
        except json.JSONDecodeError:
            # print(f"Warning: Failed to parse JSON for row {index}. Raw response: {json_response}")
            judge_scores.append(0.0)
            judge_reasonings.append(f"JSON Parse Error. Raw: {json_response}")

    df['judge_score'] = judge_scores
    df['judge_reasoning'] = judge_reasonings
    
    # Calculate average score
    avg_score = df['judge_score'].mean()
    print(f"Average Score for {os.path.basename(input_path)}: {avg_score:.4f}")

    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM benchmark results using a local LLM judge.")
    parser.add_argument("--input_path", help="Path to a specific CSV file or directory. Defaults to current directory.", default=".")
    parser.add_argument("--backend", choices=["local", "dashscope", "zai"], default="local", help="Choose the judge backend: 'local' (llama.cpp) or 'dashscope' (Aliyun Qwen).")
    parser.add_argument("--model", help="Specific model name to use. Defaults to 'local-model' for local or 'qwen-plus' for dashscope.", default=None)
    
    args = parser.parse_args()
    
    target_path = args.input_path
    files_to_process = []

    if os.path.isfile(target_path):
        files_to_process.append(target_path)
    elif os.path.isdir(target_path):
        print(f"Scanning directory: {os.path.abspath(target_path)}")
        # Find all csvs
        all_csvs = glob.glob(os.path.join(target_path, "benchmark_results_unsloth_Qwen3-VL-4B-Thinking-GGUF_Q8_0_20251204_190151.csv"))
        for f in all_csvs:
            # Filter logic:
            # 1. Must be a csv (glob handled this)
            # 2. Should not be a judged file (ends with _judged.csv)
            # 3. Should probably look like a result file (optional, but safer)
            if "_judged" not in f:
                files_to_process.append(f)
    else:
        print(f"Error: Path not found: {target_path}")
        return

    if not files_to_process:
        print("No suitable CSV files found to process.")
        return

    print(f"Found {len(files_to_process)} files to process.")

    # Initialize OpenAI client
    client = None
    model_name = args.model

    if args.backend == "local":
        print(f"Using Local Backend: {LOCAL_LLAMA_URL}")
        try:
            client = OpenAI(base_url=LOCAL_LLAMA_URL, api_key=LOCAL_LLAMA_API_KEY)
            client.models.list() # Test connection
            if not model_name:
                model_name = JUDGE_MODEL_LOCAL
            print(f"Connected to Local LLM. Model: {model_name}")
        except Exception as e:
            print(f"Error connecting to Local LLM: {e}")
            return
            
    elif args.backend == "dashscope":
        print(f"Using DashScope Backend")
        if not DASHSCOPE_API_KEY:
            print("Error: DASHSCOPE_API_KEY environment variable not set.")
            return
        try:
            client = OpenAI(base_url=DASHSCOPE_BASE_URL, api_key=DASHSCOPE_API_KEY)
            if not model_name:
                model_name = JUDGE_MODEL_DASHSCOPE
            print(f"Connected to DashScope. Model: {model_name}")
        except Exception as e:
            print(f"Error connecting to DashScope: {e}")
            return
    elif args.backend == "zai":
        print(f"Using Zai Backend")
        if not ZAI_API_KEY:
            print("Error: ZAI_API_KEY environment variable not set.")
            return
        try:
            client = ZhipuAiClient(api_key=ZAI_API_KEY)
            if not model_name:
                model_name = JUDGE_MODEL_ZAI
            print(f"Connected to Zai. Model: {model_name}")
        except Exception as e:
            print(f"Error connecting to Zai: {e}")
            return
    else:
        print(f"Error: Invalid backend: {args.backend}")
        return
    for file_path in files_to_process:
        process_file(file_path, client, model_name)
        
    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()

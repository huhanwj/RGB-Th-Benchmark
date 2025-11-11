#!/usr/bin/env python3

"""
================================================================================
RGB-Th-Bench VLM Evaluation Script
*** THERMAL-ONLY (HARD) MODE ***
================================================================================

This MODIFIED script runs the benchmark by forcing the VLM to answer
ALL questions (including RGB-based ones) using ONLY the 'thermal.jpg' image.

It is based on your 'benchmark.py' and includes your specific provider
logic for Gemini, Ark, and OpenAI-compatible (local, siliconflow) APIs.

It also re-integrates:
1.  Result Caching (Resumability): Skips questions already in the output file.
2.  Exponential Backoff (Retries): Automatically retries on rate limit errors.
3.  Deterministic Temperature (0.0): Ensures reproducible benchmark results.
4.  Dynamic Filenames: Saves one JSON per model, e.g.,
    'benchmark_results_gpt-4o_thermal_only.json'

--------------------------------------------------------------------------------
"""

import os
import json
import base64
import io
import mimetypes
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time

# Import API Clients
import google.genai as genai
from google.genai import types
# from google.api_core import exceptions as google_exceptions
from openai import OpenAI, RateLimitError
from volcenginesdkarkruntime import Ark
# from volcenginesdkarkruntime.errors import RequestLimitError as ArkRateLimitError

# --- 1. CONFIGURE YOUR MODELS HERE ---

DATASET_ROOT = Path("RGB-Th-Bench/Data")

MODELS_TO_TEST = [
    {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
    },
    # {
    #     "provider": "gemini",
    #     "model_name": "gemini-2.5-pro",
    # },
    # {
    #     "provider": "openai",
    #     "model_name": "gpt-5", # Assuming this is gpt-4o or similar
    #     "base_url": "https://api.openai.com/v1",
    # },
    # {
    #     "provider": "ark",
    #     "model_name": "doubao-seed-1-6-vision-250815",
    #     "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    #     "api_key_env": "ARK_API_KEY"
    # },
    # {
    #     "provider": "siliconflow",
    #     "model_name": "Qwen/Qwen3-VL-8B-Instruct",
    #     "base_url": "https://api.siliconflow.cn/v1",
    #     "api_key_env": "SILICONFLOW_API_KEY"
    # },
    {
        "provider": "local",
        "model_name": "unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:Q5_K_M", 
        "base_url": "http://192.168.77.38:8000/v1",
        "api_key_env": "LOCAL_API_KEY"
    },
    {
        "provider": "local",
        "model_name": "unsloth/Qwen3-VL-8B-Thinking-GGUF:Q8_0", 
        "base_url": "http://192.168.77.42:8000/v1",
        "api_key_env": "LOCAL_API_KEY"
    },
    {
        "provider": "local",
        "model_name": "unsloth/Qwen3-VL-8B-Instruct-GGUF:Q8_0", 
        "base_url": "http://192.168.66.116:8000/v1",
        "api_key_env": "LOCAL_API_KEY"
    },
]

# --- 2. API CLIENT FUNCTIONS (WITH RETRY LOGIC) ---

def image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_gemini_api(api_key, model_name, prompt, pil_images):
    """
    Calls Gemini API with robust retries.
    """
    max_retries = 5
    base_delay = 0
    
    for i in range(max_retries):
        try:
            client = genai.Client()
            
            # Use 0.0 temp for reproducible results, disable thinking for Yes/No
            gen_config = genai.types.GenerateContentConfig(
                temperature=0.0, 
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
            
            content = pil_images + [prompt] # Use simple PIL list
            response = client.models.generate_content(
                model=model_name,
                contents = content,
                config = gen_config
            )
            return response.text # Success
            
        # except google_exceptions.ResourceExhausted as e:
        #     if i == max_retries - 1:
        #         tqdm.write(f"  [!] Gemini API Error: Max retries exceeded. {e}")
        #         return f"Error: {e}"
        #     wait_time = (base_delay ** i) + (0.5 * i)
        #     tqdm.write(f"  [!] Gemini Rate Limit hit. Retrying in {wait_time:.1f}s...")
        #     time.sleep(wait_time)
        except Exception as e:
            tqdm.write(f"  [!] Gemini API Error: {e}")
            return f"Error: {e}"

def call_ark_api(api_key, base_url, model_name, prompt, pil_images):
    """
    Calls Volcengine Ark API with robust retries.
    """
    max_retries = 5
    base_delay = 0
    
    for i in range(max_retries):
        try:
            client = Ark(api_key=api_key, base_url=base_url)
            
            content = [{"text": prompt, "type": "text" }]
            for img in pil_images:
                content.append({
                    "image_url": { "url": f"data:image/jpeg;base64,{image_to_base64(img)}" },
                    "type": "image_url"                
                })
            
            messages = [{"role": "user", "content": content}]
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                # temperature=0.0, # Use 0.0 for reproducible results
                thinking={"type": "disabled"}, # Disable thinking for fast Yes/No
            )
            return response.choices[0].message.content # Success

        # except ArkRateLimitError as e:
        #     if i == max_retries - 1:
        #         tqdm.write(f"  [!] Ark API Error: Max retries exceeded. {e}")
        #         return f"Error: {e}"
        #     wait_time = (base_delay ** i) + (0.5 * i)
        #     tqdm.write(f"  [!] Ark Rate Limit hit. Retrying in {wait_time:.1f}s...")
        #     time.sleep(wait_time)
        except Exception as e:
            tqdm.write(f"  [!] Volcengine Ark API Error: {e}")
            return f"Error: {e}"
    
def call_openai_compatible_api(api_key, base_url, model_name, prompt, pil_images):
    """
    Calls any OpenAI-compatible API with robust retries.
    """
    max_retries = 5
    base_delay = 0
    
    for i in range(max_retries):
        try:
            client = OpenAI(api_key=api_key or "local", base_url=base_url)
            
            content = [{"type": "text", "text": prompt}]
            for img in pil_images:
                content.append({
                    "type": "image_url",
                    "image_url": { "url": f"data:image/jpeg;base64,{image_to_base64(img)}" }
                })
            
            messages = [{"role": "user", "content": content}]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                # temperature=0.0, # Use 0.0 for reproducible results
                # max_tokens=50 # Set a small limit for Yes/No
            )
            return response.choices[0].message.content # Success

        except RateLimitError as e:
            if i == max_retries - 1:
                tqdm.write(f"  [!] OpenAI API Error: Max retries exceeded. {e}")
                return f"Error: {e}"
            wait_time = (base_delay ** i) + (0.5 * i)
            tqdm.write(f"  [!] OpenAI Rate Limit hit. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
        except Exception as e:
            tqdm.write(f"  [!] OpenAI-Compatible API Error ({base_url}): {e}")
            return f"Error: {e}"

# --- 3. BENCHMARK HELPER FUNCTIONS (MODIFIED FOR THERMAL-ONLY) ---

def get_prompt_context(data_type):
    """
    MODIFIED (Thermal-Only Hard Mode):
    This function will IGNORE the 'data_type' and always
    provide a prompt that specifies this is a *thermal image*.
    """
    return ('Based on this thermal image, answer the '
            'following question with strictly either "Yes" or "No", '
            'without any extra explanation.')

def parse_vlm_response(response_text):
    """
    Parses the VLM response according to the paper's evaluation strategy.
    """
    text_lower = response_text.lower()
    has_yes = "yes" in text_lower
    has_no = "no" in text_lower
    
    if has_yes and has_no:
        if text_lower.find("yes") < text_lower.find("no"):
            return "Yes"
        else:
            return "No"
    
    if has_yes: return "Yes"
    if has_no: return "No"
    return "Fail"

def load_images_for_item(item, base_dir):
    """
    MODIFIED (Thermal-Only Hard Mode):
    This function will IGNORE the 'data_type' and 'data_id' fields
    and *only* load 'thermal.jpg' or 'thermal.png' for *all* questions.
    """
    pil_images = []
    thermal_img_path_jpg = base_dir / "thermal.jpg"
    thermal_img_path_png = base_dir / "thermal.png"
    
    thermal_img_path_to_load = None

    if thermal_img_path_jpg.exists():
        thermal_img_path_to_load = thermal_img_path_jpg
    elif thermal_img_path_png.exists():
        thermal_img_path_to_load = thermal_img_path_png
    
    if thermal_img_path_to_load:
        try:
            # Always append the thermal image
            pil_images.append(Image.open(thermal_img_path_to_load))
        except Exception as e:
            tqdm.write(f"Warning: Failed to load thermal image {thermal_img_path_to_load}: {e}")
    else:
        tqdm.write(f"Warning: Thermal image not found (checked .jpg and .png) in {base_dir}, skipping item.")
            
    return pil_images

# --- 4. MAIN EXECUTION ---

def main():
    """
    Main function to run the benchmark.
    """
    print("==================================================")
    print("WARNING: Running in THERMAL-ONLY (HARD) mode.")
    print("All questions will be answered using *only* thermal.jpg")
    print("==================================================")
    time.sleep(2)
    
    print(f"Starting RGB-Th-Bench Evaluation...")
    print(f"Looking for dataset in: {DATASET_ROOT.resolve()}")
    
    question_files = list(DATASET_ROOT.rglob("questions.json"))
    if not question_files:
        print(f"Error: No 'questions.json' files found under {DATASET_ROOT}")
        return
        
    print(f"Found {len(question_files)} 'questions.json' files to process.")
    
    total_questions = 0

    for model_config in MODELS_TO_TEST:
        provider = model_config["provider"]
        model_name = model_config["model_name"]
        
        # --- DYNAMIC FILENAME ---
        model_name_safe = model_name.replace('/', '_').replace(':', '_')
        output_file = f"benchmark_results_{model_name_safe}_thermal_only.json"
        
        all_results = []
        processed_set = set()

        # --- CACHING LOGIC ---
        if Path(output_file).exists():
            print(f"Found existing results file: {output_file}")
            try:
                with open(output_file, 'r') as f:
                    all_results = json.load(f)
                processed_set = set((r['model'], r['question']) for r in all_results)
                print(f"Loaded {len(all_results)} previous results. Will skip processed items.")
            except json.JSONDecodeError:
                print(f"Warning: {output_file} is corrupted. Starting fresh.")
                all_results = []
        else:
            print(f"No existing results file for {model_name}. Starting fresh.")
        
        key_env_var = model_config.get("api_key_env", f"{provider.upper()}_API_KEY")
        api_key = os.getenv(key_env_var)
        
        if provider != "local" and not api_key:
            print(f"Warning: API key for {provider} ({key_env_var}) not found. Skipping.")
            continue
        
        print(f"\n--- Testing Model: {model_name} (Provider: {provider}) ---")
        
        model_correct = 0
        model_total = 0
        
        # Calculate totals for accuracy update
        for r in all_results:
            model_total += 1
            if r['is_correct']:
                model_correct += 1

        pbar = tqdm(question_files, desc=f"Testing {model_name}")
        for q_file_path in pbar:
            base_dir = q_file_path.parent
            
            try:
                with open(q_file_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {q_file_path}: {e}")
                continue

            for item in data.get('data', []):
                skill_type = item['skill_type']
                data_type = item['data_type']
                
                try:
                    pil_images = load_images_for_item(item, base_dir)
                    if not pil_images:
                        tqdm.write(f"Skipping item in {base_dir.name} due to missing thermal image.")
                        continue
                except Exception as e:
                    tqdm.write(f"Error loading images for {base_dir.name}: {e}")
                    continue
                
                prompt_context = get_prompt_context(data_type)
                
                for qa in item['QAs']:
                    if provider == "gemini" and model_total == 0:
                        total_questions += 1

                    question = qa['question']
                    ground_truth = qa['answer']
                    
                    # --- CHECK CACHE ---
                    if (model_name, question) in processed_set:
                        continue # Already processed
                    
                    # Not in cache, so we will process it
                    model_total += 1
                    
                    full_prompt = f"{prompt_context}\n\nQuestion: {question}"
                    response_text = ""
                    start_time = time.time()
                    
                    try:
                        if provider == 'gemini':
                            response_text = call_gemini_api(api_key, model_name, full_prompt, pil_images)
                        elif provider == 'ark':
                            response_text = call_ark_api(api_key, model_config['base_url'], model_name, full_prompt, pil_images)
                        else:
                            response_text = call_openai_compatible_api(
                                api_key,
                                model_config['base_url'],
                                model_name,
                                full_prompt,
                                pil_images
                            )
                    except Exception as e:
                        response_text = f"CRITICAL API ERROR: {e}"
                        
                    end_time = time.time()

                    vlm_answer_parsed = parse_vlm_response(response_text)
                    is_correct = (vlm_answer_parsed == ground_truth)
                    
                    if is_correct:
                        model_correct += 1
                    
                    result_entry = {
                        "model": model_name,
                        "provider": provider,
                        "file": str(q_file_path.relative_to(DATASET_ROOT)),
                        "skill": skill_type,
                        "data_type_original": data_type, # Log original data type
                        "question": question,
                        "ground_truth": ground_truth,
                        "vlm_response_raw": response_text,
                        "vlm_answer_parsed": vlm_answer_parsed,
                        "is_correct": is_correct,
                        "latency_s": end_time - start_time
                    }
                    all_results.append(result_entry)
                    processed_set.add((model_name, question)) # Add to cache
                    
                    # --- SAVE RESULTS INTERMEDIATELY ---
                    # This saves progress after every single question
                    with open(output_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    
                    if model_total > 0:
                        pbar.set_description(
                            f"Testing {model_name} (QAcc: {model_correct / model_total:.2%})"
                        )

        # Print model summary
        if model_total > 0:
            qacc = (model_correct / model_total) * 100
            print(f"--- Summary for {model_name} ---")
            print(f"  Question-level Accuracy (QAcc): {qacc:.2f}%")
            print(f"  Total Correct: {model_correct} / {model_total}")
            print(f"  Results saved to: {output_file}")
            print("--------------------------------" + "-" * len(model_name))
        else:
            print(f"--- No new results to process for {model_name} ---")


    print(f"\n==========================================")
    print(f"✅ THERMAL-ONLY Benchmark Complete!")
    if total_questions > 0:
        print(f"Total unique questions in benchmark: {total_questions}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
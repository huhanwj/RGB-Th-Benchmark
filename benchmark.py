#!/usr/bin/env python3

"""
================================================================================
RGB-Th-Bench VLM Evaluation Script
================================================================================

This script runs the RGB-Th-Bench benchmark (arXiv:2503.19654) 
across multiple VLM API providers.

--------------------------------------------------------------------------------
SETUP:
--------------------------------------------------------------------------------
1.  Install necessary libraries:
    pip install google-generativeai openai volcengine-python-sdk pillow tqdm

2.  Set Environment Variables:
    You MUST set API keys for the providers you want to test.

    # For Gemini
    export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    
    # For OpenAI
    export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    
    # For Volcengine Ark
    export ARK_API_KEY="YOUR_VOLCENGINE_API_KEY"
    
    # For SiliconFlow (or other OpenAI-compatible)
    export SILICONFLOW_API_KEY="YOUR_SILICONFLOW_API_KEY"

3.  Configure Models:
    Edit the `MODELS_TO_TEST` list below to match the model names and 
    base URLs for each provider.

4.  Dataset Structure:
    This script assumes your data is in the following structure,
    starting from the script's location or a defined path:
    
    RGB-Th-Bench/
    └── Data/
        ├── Source 1/
        │   ├── Basement door/
        │   │   ├── questions.json
        │   │   ├── rgb.jpg
        │   │   └── thermal.jpg
        │   └── ... (other scenes)
        ├── Source 2/
        │   └── ...
        └── ...

5.  Run the script:
    python run_benchmark.py

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
import google.generativeai as genai
from openai import OpenAI, APIError, APITimeoutError
from volcenginesdkarkruntime import Ark
# --- 1. CONFIGURE YOUR MODELS HERE ---

# Define the root directory for the dataset
# Assumes "RGB-Th-Bench" is in the same directory as the script.
DATASET_ROOT = Path("RGB-Th-Bench/Data")

# Define the models you want to test
MODELS_TO_TEST = [
    {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        # "api_key_env": "GEMINI_API_KEY" # Automatically inferred
    },
    {
        "provider": "gemini",
        "model_name": "gemini-2.5-pro",
        # "api_key_env": "GEMINI_API_KEY" # Automatically inferred
    },
    {
        "provider": "openai",
        "model_name": "gpt-5",
        "base_url": "https://api.openai.com/v1",
        # "api_key_env": "OPENAI_API_KEY" # Automatically inferred
    },
    {
        "provider": "volcengine",
        "model_name": "doubao-seed-1-6-vision-250815",  # <-- Doubao-Seed-1.6-Vision
        # "base_url": "https://api.volcengine.com/v1/chat/completions", #
        # "api_key_env": "VOLCENGINE_API_KEY" # Automatically inferred
    },
    {
        "provider": "siliconflow",
        "model_name": "Qwen/Qwen3-VL-235B-A22B-Thinking", # <-- Qwen3-VL-235B-A22B-Thinking
        "base_url": "https://api.siliconflow.com/v1", # <-- TODO: VERIFY URL
        "api_key_env": "SILICONFLOW_API_KEY"
    },
    {
        "provider": "siliconflow",
        "model_name": "Qwen/Qwen3-VL-30B-A3B-Thinking", # <-- Qwen-3-VL-30B-A3B-Thinking
        "base_url": "https://api.siliconflow.com/v1", # <-- TODO: VERIFY URL
        "api_key_env": "SILICONFLOW_API_KEY"
    },
    {
        "provider": "siliconflow",
        "model_name": "Qwen/Qwen3-VL-8B-Thinking", # <-- Qwen3-VL-8B-Thinking
        "base_url": "https://api.siliconflow.com/v1", # <-- TODO: VERIFY URL
        "api_key_env": "SILICONFLOW_API_KEY"
    },
    {
        "provider": "siliconflow",
        "model_name": "zai-org/GLM-4.5V", # GLM-4.5V
        "base_url": "https://api.siliconflow.com/v1", # <-- TODO: VERIFY URL
        "api_key_env": "SILICONFLOW_API_KEY"
    },
    # Add more models as needed...
]

# --- 2. API CLIENT FUNCTIONS ---

def image_to_base64(pil_image):
    """Converts a PIL Image to a base64 string for OpenAI API."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_gemini_api(api_key, model_name, prompt, pil_images):
    """
    Calls the Gemini API with a prompt and a list of PIL Images.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Build the content list [prompt, image1, image2, ...]
        content = [prompt] + pil_images
        
        response = model.generate_content(
            content,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=50 # "Yes" or "No" should be small
            )
        )
        return response.text
    except Exception as e:
        print(f"  [!] Gemini API Error: {e}")
        return f"Error: {e}"
def call_ark_api(api_key, model_name, prompt, pil_images):
    """
    Calls the Volcengine Ark API with a prompt and a list of PIL Images.
    """
    try:
        client = Ark(api_key=os.environ.get("ARK_API_KEY"))
        
        # Build the message content
        content = [{"type": "text", "text": prompt}]
        for img in pil_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_base64(img)}"
                }
            })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=50
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [!] Volcengine Ark API Error: {e}")
        return f"Error: {e}"
    
def call_openai_compatible_api(api_key, base_url, model_name, prompt, pil_images):
    """
    Calls any OpenAI-compatible API (OpenAI, Volcengine, SiliconFlow).
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Build the message content
        content = [{"type": "text", "text": prompt}]
        for img in pil_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_base64(img)}"
                }
            })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=50
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [!] OpenAI-Compatible API Error ({base_url}): {e}")
        return f"Error: {e}"

# --- 3. BENCHMARK HELPER FUNCTIONS ---

def get_prompt_context(data_type):
    """
    Returns the exact prompt context specified in the RGB-Th-Bench paper,
    Section 3.3: Instruction Design.
    """
    if data_type == "Single RGB Image":
        return ('Based on this image, answer the '
                'following question with strictly either "Yes" or "No", '
                'without any extra explanation')
    elif data_type == "RGB-Thermal Pair":
        return ('Based on these two images, '
                'and the fact that the second image is the thermal image '
                'taken from the same scene as the first image, answer the '
                'following question with strictly either "Yes" or "No", '
                'without any extra explanation.')
    else:
        print(f"Warning: Unknown data_type '{data_type}'")
        return ""

def parse_vlm_response(response_text):
    """
    Parses the VLM response according to the paper's evaluation strategy.
    Section 3.3: "If the model's response does not include Yes or No 
    (case insensitive), it is marked as a ... 'Fail'."
    """
    text_lower = response_text.lower()
    
    # Check for "yes" or "no"
    has_yes = "yes" in text_lower
    has_no = "no" in text_lower
    
    # Handle ambiguous cases (e.g., "Yes, but also no...")
    if has_yes and has_no:
        # Prioritize the first occurrence
        if text_lower.find("yes") < text_lower.find("no"):
            return "Yes"
        else:
            return "No"
    
    if has_yes:
        return "Yes"
    
    if has_no:
        return "No"
        
    # If neither "Yes" nor "No" is found, it's a "Fail"
    return "Fail"

def load_images_for_item(item, base_dir):
    """
    Loads PIL Images based on the questions.json item.
    """
    pil_images = []
    if item['data_type'] == "Single RGB Image":
        # data_id is a single string, e.g., "rgb.jpg"
        # Note: The provided 'Basement door/rgb.jpg' seems like an error in the
        # sample json. We'll correct for it by just taking the filename.
        img_name = Path(item['data_id']).name
        img_path = base_dir / img_name
        if img_path.exists():
            pil_images.append(Image.open(img_path))
        else:
            print(f"Warning: Image not found {img_path}")
            
    elif item['data_type'] == "RGB-Thermal Pair":
        # data_id is a list of strings, e.g., ["rgb.jpg", "thermal.jpg"]
        rgb_name = Path(item['data_id'][0]).name
        thermal_name = Path(item['data_id'][1]).name
        
        rgb_path = base_dir / rgb_name
        thermal_path = base_dir / thermal_name
        
        if rgb_path.exists():
            pil_images.append(Image.open(rgb_path))
        else:
            print(f"Warning: Image not found {rgb_path}")
            
        if thermal_path.exists():
            pil_images.append(Image.open(thermal_path))
        else:
            print(f"Warning: Image not found {thermal_path}")
    
    return pil_images

# --- 4. MAIN EXECUTION ---

def main():
    """
    Main function to run the benchmark.
    """
    print(f"Starting RGB-Th-Bench Evaluation...")
    print(f"Looking for dataset in: {DATASET_ROOT.resolve()}")
    
    # Find all questions.json files
    question_files = list(DATASET_ROOT.rglob("questions.json"))
    if not question_files:
        print(f"Error: No 'questions.json' files found under {DATASET_ROOT}")
        print("Please check your DATASET_ROOT path and file structure.")
        return
        
    print(f"Found {len(question_files)} 'questions.json' files to process.")
    
    all_results = []
    total_questions = 0 # We'll count them as we go

    for model_config in MODELS_TO_TEST:
        provider = model_config["provider"]
        model_name = model_config["model_name"]
        
        # Get API Key
        key_env_var = model_config.get("api_key_env", f"{provider.upper()}_API_KEY")
        api_key = os.getenv(key_env_var)
        
        if not api_key:
            print(f"Warning: API key for {provider} ({key_env_var}) not found. Skipping.")
            continue
            
        print(f"\n--- Testing Model: {model_name} (Provider: {provider}) ---")
        
        model_correct = 0
        model_total = 0

        # Use tqdm for a global progress bar for this model
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
                
                # Load images once per item
                try:
                    pil_images = load_images_for_item(item, base_dir)
                    if not pil_images:
                        tqdm.write(f"Skipping item in {base_dir.name} due to missing images.")
                        continue
                except Exception as e:
                    tqdm.write(f"Error loading images for {base_dir.name}: {e}")
                    continue
                
                # Get the prompt context
                prompt_context = get_prompt_context(data_type)
                
                for qa in item['QAs']:
                    if provider == "gemini" and model_total == 0:
                        total_questions += 1 # Only count total questions on the first model run

                    model_total += 1
                    question = qa['question']
                    ground_truth = qa['answer']
                    
                    # Construct the full prompt
                    full_prompt = f"{prompt_context}\n\nQuestion: {question}"

                    # Call the appropriate API
                    response_text = ""
                    start_time = time.time()
                    
                    try:
                        if provider == 'gemini':
                            response_text = call_gemini_api(api_key, model_name, full_prompt, pil_images)
                        elif provider == 'volcengine':
                            # For Volcengine Ark
                            response_text = call_ark_api(api_key, model_name, full_prompt, pil_images)
                        else:
                            # For OpenAI, SiliconFlow
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

                    # Parse the response
                    vlm_answer_parsed = parse_vlm_response(response_text)
                    is_correct = (vlm_answer_parsed == ground_truth)
                    
                    if is_correct:
                        model_correct += 1
                    
                    # Store results
                    result_entry = {
                        "model": model_name,
                        "provider": provider,
                        "file": str(q_file_path.relative_to(DATASET_ROOT)),
                        "skill": skill_type,
                        "data_type": data_type,
                        "question": question,
                        "ground_truth": ground_truth,
                        "vlm_response_raw": response_text,
                        "vlm_answer_parsed": vlm_answer_parsed,
                        "is_correct": is_correct,
                        "latency_s": end_time - start_time
                    }
                    all_results.append(result_entry)

                    # Update progress bar description with running accuracy
                    pbar.set_description(
                        f"Testing {model_name} (QAcc: {model_correct / model_total:.2%})"
                    )

        # Print model summary
        if model_total > 0:
            qacc = (model_correct / model_total) * 100
            print(f"--- Summary for {model_name} ---")
            print(f"  Question-level Accuracy (QAcc): {qacc:.2f}%")
            print(f"  Total Correct: {model_correct} / {model_total}")
            print("--------------------------------" + "-" * len(model_name))

    # Save all results to a JSON file
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n==========================================")
    print(f"✅ Benchmark Complete!")
    print(f"Total questions processed (per model): {total_questions}")
    print(f"Detailed results saved to: {output_file}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
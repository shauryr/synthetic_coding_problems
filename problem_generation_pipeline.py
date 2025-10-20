import json, os, random, time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

from prompt_templates import *
# from vllm import LLM, EngineArgs, SamplingParams
# from vllm.utils import FlexibleArgumentParser

MODEL_NAMEs = ["Qwen/Qwen3-32B", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
MODEL_NAME = MODEL_NAMEs[1]
DATASET_SAVE_DIR = Path("/mnt/weka/home/anzexie/code_agent/code_questions/generate_code_problems/test_gp")


openai_api_key = "EMPTY"
openai_api_base = "http://fs-mbz-gpu-945:8000/v1"

temp_seed = """learn_model( tf_idfSVM, tf_idfNB, target)  def get_clean_review(raw_review): letters_only = re.sub( "[^a-zA-Z]", " ", raw_review)
"""

data_path = Path("/mnt/sharefs/users/omkar.pangarkar/opencoder/original/chunk_00001.jsonl")

def load_data(data_path):
    n = 0
    
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            n += 1
            if n > 10:
                break
            data.append(json.loads(line))
    
    code_seeds = []    
    for item in tqdm(data, desc="Extracting code seeds"):
        seed = extract_seed_code(15, 30, item['text'])
        item['seed'] = seed
        code_seeds.append(item)
    
    return code_seeds

def extract_seed_code(min_lines, max_lines, document) -> str:
    lines = document.splitlines(keepends=True)
    start_index = random.choice(range(len(lines)))
    n_lines_to_consider = random.randint(min_lines, max_lines)
    code = "".join(lines[start_index : start_index + n_lines_to_consider])
    return code

def parse_problem_solution(response_text: str) -> tuple[str, str] | None:
    lines = response_text.splitlines(keepends=True)
    problem_start_index: int | None = None
    solution_start_index: int | None = None
    for idx, line in enumerate(lines):
        if "[problem description]" in line.lower() and problem_start_index is None:
            problem_start_index = idx
        if "[solution]" in line.lower() and solution_start_index is None:
            solution_start_index = idx
    if problem_start_index is None or solution_start_index is None:
        return None
    if problem_start_index >= solution_start_index:
        return None
    problem = "".join(lines[problem_start_index + 1 : solution_start_index]).strip()
    solution = "".join(lines[solution_start_index + 1 :]).strip()
    return problem, solution

def query_vllm(sys_prompt, user_prompt, model_name):
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    
    chat_response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": sys_prompt},
        {
         "role": "user", 
         "content": user_prompt
         },
    ],
    max_tokens=8192,
    temperature=0,
    top_p=0.95,
    extra_body={
        "top_k": 20,
    },
    )

    return chat_response.choices[0].message.content

def save_to_jsonl(dataset, stats, file_name):
    full_path = DATASET_SAVE_DIR / file_name
    with open(full_path, 'w') as f:
        for item in tqdm(dataset, desc=f"Saving {file_name.stem} to JSONL"):
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            
    stats_filename = DATASET_SAVE_DIR / "datasets_stats.jsonl"
    
    datasets_stats = []
    already_exists = False
    with open(stats_filename, 'r') as f:
        for line in f:
            temp_stats = json.loads(line.strip())
            if temp_stats['file_name'] == file_name.name:                               
                already_exists = True
                datasets_stats.append(stats)  # update with new stats
            else:
                datasets_stats.append(json.loads(line.strip()))
    if not already_exists:
        datasets_stats.append(stats)
    with open(stats_filename, 'w') as f:
        for item in datasets_stats:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            

def main():
    seeds = load_data(data_path)
    
    num_invalid = 0
    data_dict = []
    time_start = time.time()
    for seed in tqdm(seeds[:10], desc="Generating problems"):
        sys_prompt = MAGIC_CODER_TEMPLATE_SYS
        user_prompt = MAGIC_CODER_TEMPLATE_USER.format(code=seed)
        output = parse_problem_solution(query_vllm(sys_prompt, user_prompt, MODEL_NAME))
        if output is None:
            print("Failed to parse problem and solution from response.")
            num_invalid += 1
            continue
        else:
            problem, solution = output
            
        
        seed['problem'] = problem
        seed['solution'] = solution
        
        data_dict.append({
            "seed": seed['seed'],
            "problem": problem,
            "solution": solution
        })
    time_end = time.time()
    
    save_file_name = Path(f"problems_{Path(MODEL_NAME).stem}.jsonl")
    invalid_ratio = num_invalid / len(seeds)
    print(f"Invalid ratio: {invalid_ratio}")
    stats = {
        "file_name": str(save_file_name),
        "total_samples": len(seeds),
        "num_invalid": num_invalid,
        "valid_ratio": 1 - invalid_ratio,
        "generating_model": MODEL_NAME,
        "time_used": time_end - time_start
    }
    
    save_to_jsonl(data_dict, stats, save_file_name)
    
    

if __name__ == "__main__":
    main()
    
    

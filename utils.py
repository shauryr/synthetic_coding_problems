import json, os, random, time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

from prompt_templates import *

def load_data(data_path):
    n = 0
    
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            # n += 1
            # if n > 10:
            #     break
            data.append(json.loads(line))
    
    code_seeds = []    
    for item in tqdm(data, desc="Extracting code seeds"):
        try:
            seed = extract_seed_code(15, 30, item['text'])
        except:
            print(item['text'])
            exit("load_data")
        
        if len(seed) > 4096:
            seed = seed[:4096]
        
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

def save_to_jsonl(dataset, file_name, dataset_save_dir):
    full_path = dataset_save_dir / file_name
    with open(full_path, 'w') as f:
        for item in tqdm(dataset, desc=f"Saving {file_name.stem} to JSONL"):
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def save_stats(stats, dataset_save_dir):
    file_name = Path(stats['file_name'])
    stats_filename = dataset_save_dir / "datasets_stats.jsonl"
    
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
            
def prepare_prompts(raw_data, tokenizer):
    # sys_prompt = CLAUDE_TEMPLATE_SYS
    # user_prompt = CLAUDE_TEMPLATE_USER
    sys_prompt = MAGIC_CODER_TEMPLATE_SYS
    user_empty_prompt = MAGIC_CODER_TEMPLATE_USER
    
    prompts = []
    for sample in raw_data:
        user_prompt = user_empty_prompt.format(code=sample['seed'])
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
            # reasoning_effort="high"
            )
        
        # assert len(tokenizer.encode(formatted)) < 4096, f"prompt: {user_prompt}, len: {len(user_prompt)}"
        prompts.append(formatted)
    
    # assert len(prompts) == 100000
    return prompts
    
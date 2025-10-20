import json, os, random, time, ray, torch
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser

from prompt_templates import *
from utils import load_data, parse_problem_solution, save_to_jsonl, prepare_prompts, save_stats




MODEL_NAMEs = ["Qwen/Qwen3-32B", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
MODEL_NAME = MODEL_NAMEs[1]
DATASET_SAVE_DIR = Path("/mnt/weka/home/anzexie/code_agent/code_questions/generate_code_problems/generated_problems")
# DATASET_SAVE_DIR = Path("/mnt/weka/home/anzexie/code_agent/code_questions/generate_code_problems/test_gp")

# data_path = Path("/mnt/sharefs/users/omkar.pangarkar/opencoder/original/chunk_00001.jsonl")
ORIG_DATA_DIR = Path("/mnt/sharefs/users/omkar.pangarkar/opencoder/original/")

@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, params=None):
        self.llm = LLM(model=MODEL_NAME)
        self.sampling_params = SamplingParams(
            temperature=0.7,    # recommended for non-thinking mode
            top_p=0.8,         # nucleus sampling
            top_k=20,           # top-k sampling
            max_tokens=2048,  # maximum tokens to generate
            n=1,
            stop=["<|endoftext|>"]
        ) if params is None else params
        self.tokenizer = self.llm.get_tokenizer()
        
    def generate_code_problems(self, files):
        files = files[:5] # fs-mbz-gpu-126
        
        all_stats = []
        
        for file in files:
            all_stats.append(self.generate_problem_for_file(file))
        
        return all_stats
    
    def generate_problem_for_file(self, file_path):
        raw_data = load_data(file_path)
        
        num_invalid = 0
        output_dict = []
        
        time_start = time.time()
        prompts = prepare_prompts(raw_data, self.tokenizer)
        llm_outputs = self.generate(prompts)
        for i, output in tqdm(enumerate(llm_outputs), desc="Processing LLM outputs", total=len(llm_outputs)):
            temp = parse_problem_solution(output.outputs[0].text)
            if temp is None:
                num_invalid += 1
                continue
            else:
                problem, solution = temp
            
            
            raw_data[i]['problem'] = problem
            raw_data[i]['solution'] = solution
            
            # raw_data[i]['all_outputs'] = output.outputs[0].text
            
            # output_dict.append({
            #     "seed": raw_data[i]['seed'],
            #     "problem": problem,
            #     "solution": solution
            # })
            output_dict.append(raw_data[i])
        time_end = time.time()
        
        save_file_name = Path(f"problems_{Path(file_path).stem}.jsonl")
        invalid_ratio = num_invalid / len(raw_data)
        print(f"Invalid ratio: {invalid_ratio}")
        stats = {
            "file_name": str(save_file_name),
            "total_samples": len(raw_data),
            "num_invalid": num_invalid,
            "valid_ratio": f"{(1 - invalid_ratio):1%}",
            "generating_model": MODEL_NAME,
            "time_used": time_end - time_start
        }
        
        save_to_jsonl(output_dict, save_file_name, DATASET_SAVE_DIR)

        return stats
    
    def generate(self, prompts):
        return self.llm.generate(prompts, self.sampling_params)

def split_workload(total_workloads, num_workers):
    chunk_size = len(total_workloads) // num_workers
    print(f"Total workloads: {len(total_workloads)}, Chunk size: {chunk_size}, Num workers: {num_workers}")
    return  [total_workloads[i:i + chunk_size] for i in range(0, len(total_workloads), chunk_size)]

def find_workload(num_gpus):
    total_workloads = list(ORIG_DATA_DIR.glob("*.jsonl"))
        
    completed_workloads = list(DATASET_SAVE_DIR.glob("problems_chunk_*.jsonl"))
    compeleted_orginal_files = [ORIG_DATA_DIR / f"chunk_{file.stem.split('_')[-1]}.jsonl" for file in completed_workloads]
    
    to_do_workloads = []
    for file in total_workloads:
        if file in compeleted_orginal_files:
            continue
        else:
            to_do_workloads.append(file)
            
    excess = len(to_do_workloads) % num_gpus
    if excess != 0:
        to_do_workloads = to_do_workloads[:-excess]
    
    return to_do_workloads
    
def get_stats():
    stats_path = DATASET_SAVE_DIR / "datasets_stats.jsonl"

    total_problems = 0

    with open(stats_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            stats = json.loads(line)
            total_problems += (stats['total_samples'] - stats['num_invalid'])

    print(f"Total valid problems: {total_problems}")   # 971k problems

def main():
    ray.init(address='10.24.0.143:6379')
    # ray.init()
    num_gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f"Detected {num_gpus} GPUs in the Ray cluster.")
    
    workers = [ModelWorker.remote() for i in range(num_gpus)]
    
    total_workloads = find_workload(num_gpus)
    
    file_chunks = split_workload(total_workloads, len(workers))
    
    time_s = time.time()
    futures = [workers[i].generate_code_problems.remote(chunk) for i, chunk in enumerate(file_chunks)]
    all_worker_stats = ray.get(futures)
    time_e = time.time()
    print(f"Total time taken: {time_e - time_s} seconds")
    
    for all_stats in all_worker_stats:
        for stats in all_stats:
            save_stats(stats, DATASET_SAVE_DIR)
    
    get_stats()
    
    
    
    
    

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Generate solutions for coding problems using vLLM with gpt-oss-120b model.
This script processes JSONL files in a distributed manner across Slurm array jobs.
"""

import os

# Set HuggingFace cache directory BEFORE importing transformers/vllm
os.environ["HF_HOME"] = "/mnt/sharefs/users/shaurya.rohatgi/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = (
    "/mnt/sharefs/users/shaurya.rohatgi/.cache/huggingface"
)
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Most reliable offline mode
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
import argparse
import signal
import sys
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle preemption signals gracefully."""
    global shutdown_requested
    print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True


def setup_signal_handlers():
    """Set up handlers for SLURM preemption signals."""
    signal.signal(
        signal.SIGTERM, signal_handler
    )  # SLURM sends SIGTERM before preemption
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C for testing


def load_checkpoint(checkpoint_file: Path) -> Set[int]:
    """Load set of already processed line indices from checkpoint."""
    processed = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            for line in f:
                try:
                    processed.add(int(line.strip()))
                except ValueError:
                    continue
    return processed


def save_to_checkpoint(checkpoint_file: Path, line_idx: int):
    """Append a processed line index to checkpoint file."""
    with open(checkpoint_file, "a") as f:
        f.write(f"{line_idx}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate solutions for coding problems"
    )
    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="Task ID from Slurm array job (0-based index)",
    )
    parser.add_argument(
        "--total-tasks",
        type=int,
        required=True,
        help="Total number of tasks in the array",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/weka/shrd/k2m/anze.xie/synthetic_code_problems_opencoder",
        help="Directory containing input JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./solutions_output",
        help="Directory to save output JSONL files",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Batch size for vLLM inference"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism",
    )
    return parser.parse_args()


def get_input_files(input_dir: str) -> List[Path]:
    """Get all JSONL files from input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all .jsonl and .json files
    jsonl_files = sorted(
        list(input_path.glob("*.jsonl")) + list(input_path.glob("*.json"))
    )

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {input_dir}")

    return jsonl_files


def stream_problems(files: List[Path], task_id: int, total_tasks: int):
    """
    Stream problems from JSONL files, yielding only those assigned to this task.
    Uses modulo sharding to distribute work across tasks.
    """
    global_line_idx = 0

    for file_path in files:
        with open(file_path, "r") as f:
            for line in f:
                # Shard data: each task processes lines where (global_idx % total_tasks == task_id)
                if global_line_idx % total_tasks == task_id:
                    try:
                        data = json.loads(line.strip())
                        if "problem" in data:
                            yield data, global_line_idx
                    except json.JSONDecodeError as e:
                        print(
                            f"Warning: Failed to parse line {global_line_idx} in {file_path}: {e}"
                        )

                global_line_idx += 1


def parse_vllm_output(output_text: str) -> tuple[str, str]:
    """
    Parse vLLM output to extract think tokens and solution.

    Think tokens: everything between 'assistant' and 'assistantfinal' tokens
    Solution: everything after 'assistantfinal' token

    Returns: (think_tokens, solution)
    """
    # Look for the assistant thinking delimiter
    if "assistantfinal" in output_text:
        parts = output_text.split("assistantfinal", 1)
        think_tokens = parts[0].strip()
        solution = parts[1].strip() if len(parts) > 1 else ""
    else:
        # If no assistantfinal token, treat entire output as solution
        think_tokens = ""
        solution = output_text.strip()

    # Remove "analysis" token from the beginning of think_tokens if present
    if think_tokens.startswith("analysis"):
        think_tokens = think_tokens[len("analysis") :].strip()

    return think_tokens, solution


def process_batch(
    batch: List[tuple[Dict[str, Any], int]],
    llm: LLM,
    tokenizer: Any,
    sampling_params: SamplingParams,
) -> List[Dict[str, Any]]:
    """Process a batch of problems and return formatted results."""

    # Format prompts
    formatted_prompts = []
    for item, _ in tqdm(batch, desc="Formatting prompts", leave=False):
        problem = item["problem"]

        messages = [
            {"role": "user", "content": problem},
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=None,
            reasoning_effort="high",
        )

        formatted_prompts.append(formatted_prompt)

    # Generate with vLLM
    print(f"Calling vLLM for batch of {len(batch)} problems...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Parse outputs and create result objects
    results = []
    for (item, line_idx), output in zip(batch, outputs):
        output_text = output.outputs[0].text
        think_tokens, solution = parse_vllm_output(output_text)

        # Skip results where think tokens are empty
        if not think_tokens:
            continue

        # Count tokens for think and answer
        token_count_think = len(tokenizer.encode(think_tokens))
        token_count_answer = len(tokenizer.encode(solution))

        result = {
            "conversation": [
                {"role": "user", "content": item["problem"]},
                {"role": "assistant", "think": think_tokens, "content": solution},
            ],
            "token_count_think": token_count_think,
            "token_count_answer": token_count_answer,
        }
        results.append(result)

    return results


def main():
    global shutdown_requested

    args = parse_args()

    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()

    print(f"Starting task {args.task_id}/{args.total_tasks}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint and output files for this task
    output_file = output_dir / f"solutions_task_{args.task_id:04d}.jsonl"
    checkpoint_file = output_dir / f".checkpoint_task_{args.task_id:04d}.txt"
    temp_output_file = output_dir / f".temp_solutions_task_{args.task_id:04d}.jsonl"

    # Load checkpoint to resume from where we left off
    processed_lines = load_checkpoint(checkpoint_file)
    if processed_lines:
        print(
            f"Resuming from checkpoint: {len(processed_lines)} problems already processed"
        )

    # Initialize vLLM model
    print("Initializing vLLM model...")
    # Use direct path to cached model snapshot to avoid network requests
    model_path = "/mnt/sharefs/users/shaurya.rohatgi/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        max_tokens=16384,
        temperature=1.0,  # Use temperature 0 for deterministic extraction
        top_p=1.0,
    )

    print("Model initialized successfully!")

    # Get input files
    input_files = get_input_files(args.input_dir)
    print(f"Found {len(input_files)} input files")

    # Process problems in batches
    batch = []
    batch_line_indices = []
    processed_count = 0

    with open(temp_output_file, "a") as out_f:
        for problem_data, line_idx in tqdm(
            stream_problems(input_files, args.task_id, args.total_tasks),
            desc=f"Processing problems (Task {args.task_id})",
        ):
            # Check for shutdown signal
            if shutdown_requested:
                print("\nShutdown requested. Saving progress and exiting...")
                break

            # Skip already processed problems
            if line_idx in processed_lines:
                continue

            batch.append((problem_data, line_idx))
            batch_line_indices.append(line_idx)

            # Process batch when it reaches batch_size
            if len(batch) >= args.batch_size:
                results = process_batch(batch, llm, tokenizer, sampling_params)

                # Write results to output file and update checkpoint
                for result in results:
                    out_f.write(json.dumps(result) + "\n")

                # Update checkpoint for all processed items in this batch
                for idx in batch_line_indices:
                    save_to_checkpoint(checkpoint_file, idx)
                    processed_lines.add(idx)

                processed_count += len(batch)
                batch = []
                batch_line_indices = []

                # Flush to ensure progress is saved
                out_f.flush()

                # Check for shutdown after each batch
                if shutdown_requested:
                    print("\nShutdown requested. Saving progress and exiting...")
                    break

        # Process remaining items in batch
        if batch and not shutdown_requested:
            results = process_batch(batch, llm, tokenizer, sampling_params)

            for result in results:
                out_f.write(json.dumps(result) + "\n")

            # Update checkpoint for remaining items
            for idx in batch_line_indices:
                save_to_checkpoint(checkpoint_file, idx)

            processed_count += len(batch)

    # If job completed successfully (not shutdown), consolidate temp file to final output
    if not shutdown_requested:
        print("Job completed successfully. Consolidating results...")
        if temp_output_file.exists():
            temp_output_file.rename(output_file)
            checkpoint_file.unlink(missing_ok=True)  # Remove checkpoint file
        print(f"Task {args.task_id} completed! Processed {processed_count} problems.")
        print(f"Output saved to: {output_file}")
    else:
        print(
            f"Job preempted. Progress saved. Processed {processed_count} problems so far."
        )
        print(f"Temp output: {temp_output_file}")
        print(f"Checkpoint: {checkpoint_file}")
        sys.exit(143)  # Exit code for SIGTERM (128 + 15)


if __name__ == "__main__":
    main()

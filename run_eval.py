from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from mind2web2.llm_client.base_client import LLMClient
from mind2web2.eval_runner import evaluate_task, merge_all_results
from mind2web2.utils.path_config import PathConfig

# --------------------------------------------------------------------------- #
# CLI                                                                        #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Mind2Web2 task evaluation.")
    
    # Task specification
    p.add_argument("--task_id", help="Task folder name (if not provided, evaluates all tasks)")
    p.add_argument("--agent_name", required=True, help="Agent name for evaluation")
    
    # Required path
    p.add_argument("--answer_folder", type=Path, required=True,
                   help="Directory containing answer files (required)")
    
    # Optional path overrides
    p.add_argument("--eval_scripts_root", type=Path, 
                   help="Override evaluation scripts directory")
    p.add_argument("--eval_results_root", type=Path, 
                   help="Override output directory for results/logs")
    p.add_argument("--cache_root", type=Path, 
                   help="Override cache directory")
    
    # LLM configuration
    p.add_argument("--llm_provider", choices=["openai", "azure_openai"], 
                   default="openai", help="LLM provider to use")
    
    # Runtime options
    p.add_argument("--concurrency", type=int, default=5,
                   help="Maximum number of concurrent network requests")
    p.add_argument("--dump_cache", action="store_true", default=True,
                   help="Persist cache to disk at the end (default: True)")
    p.add_argument("--self_debug", action="store_true",
                   help="Add *_debug suffix to logs / result files")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing results")
    return p


# --------------------------------------------------------------------------- #
# Helpers                                                                    #
# --------------------------------------------------------------------------- #


async def evaluate_single_task(
    task_id: str,
    agent_name: str,
    client: LLMClient,
    paths: PathConfig,
    args: argparse.Namespace
) -> List[Dict[str, Any]]:
    """Evaluate a single task."""
    # Resolve evaluation script
    script_path = paths.default_script_for(task_id)
    if not script_path.exists():
        logging.error(f"Evaluation script not found: {script_path}")
        return []
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.concurrency)
    
    # Invoke evaluation
    return await evaluate_task(
        client=client,
        task_id=task_id,
        agent_name=agent_name,
        answer_dir=paths.answers_root,
        cache_dir=paths.cache_root,
        output_dir=paths.eval_results_root,
        script_path=script_path,
        dump_cache=args.dump_cache,
        is_self_debug=args.self_debug,
        overwrite=args.overwrite,
        max_concurrent_answers=args.concurrency,
        semaphore=semaphore,
    )


async def evaluate_all_tasks(
    agent_name: str,
    client: LLMClient,
    paths: PathConfig,
    args: argparse.Namespace
) -> Dict[str, List[Dict[str, Any]]]:
    """Evaluate all tasks in the eval_scripts directory."""
    results = {}
    
    # Find all task scripts
    eval_scripts = list(paths.eval_scripts_root.glob("*.py"))
    if not eval_scripts:
        logging.warning(f"No evaluation scripts found in {paths.eval_scripts_root}")
        return results
    
    logging.info(f"Found {len(eval_scripts)} evaluation scripts")
    
    # Create a global semaphore to control overall concurrency
    global_semaphore = asyncio.Semaphore(args.concurrency)
    
    # Create tasks for all evaluations
    async_tasks = []
    for script_path in eval_scripts:
        task_id = script_path.stem
        async_tasks.append((task_id, evaluate_single_task(
            task_id=task_id,
            agent_name=agent_name,
            client=client,
            paths=paths,
            args=args
        )))
    
    # Run evaluations with progress tracking
    for task_id, task_coro in async_tasks:
        logging.info(f"Evaluating task: {task_id}")
        try:
            task_results = await task_coro
            results[task_id] = task_results
            if task_results:
                logging.info(f"✅ Task {task_id}: {len(task_results)} results")
            else:
                logging.warning(f"⚠️ Task {task_id}: No results")
        except Exception as e:
            logging.error(f"❌ Failed to evaluate task {task_id}: {e}")
            results[task_id] = []
    
    return results


async def run_evaluation(args: argparse.Namespace, paths: PathConfig):
    """Main evaluation runner."""
    # Build async client
    client = LLMClient(provider=args.llm_provider, is_async=True)
    
    if args.task_id:
        # Evaluate single task
        logging.info(f"Evaluating single task: {args.task_id}")
        results = await evaluate_single_task(
            task_id=args.task_id,
            agent_name=args.agent_name,
            client=client,
            paths=paths,
            args=args
        )
        return {args.task_id: results}
    else:
        # Evaluate all tasks
        logging.info("Evaluating all tasks")
        return await evaluate_all_tasks(
            agent_name=args.agent_name,
            client=client,
            paths=paths,
            args=args
        )


# --------------------------------------------------------------------------- #
# Entrypoint                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Initialize paths
    project_root = Path(__file__).resolve().parent
    paths = PathConfig(project_root)
    
    # Parse arguments
    args = build_parser().parse_args()
    
    # Apply path overrides
    paths.apply_overrides(
        answers_root=args.answer_folder,  # Required argument
        eval_scripts_root=args.eval_scripts_root,
        eval_results_root=args.eval_results_root,
        cache_root=args.cache_root,
    )
    
    # Validate answer folder structure
    agent_dir = paths.answers_root / args.agent_name
    if not agent_dir.exists():
        logging.error(f"Agent directory not found: {agent_dir}")
        logging.error(f"Expected structure: {paths.answers_root}/<agent_name>/<task_id>/answer_*.md")
        return
    
    logging.info("=" * 60)
    logging.info("Mind2Web2 Evaluation Runner")
    logging.info("=" * 60)
    logging.info(f"Agent: {args.agent_name}")
    logging.info(f"Answer folder: {paths.answers_root}")
    logging.info(f"Eval scripts root: {paths.eval_scripts_root}")
    logging.info(f"Eval results root: {paths.eval_results_root}")
    logging.info(f"Cache root: {paths.cache_root}")
    logging.info(f"LLM Provider: {args.llm_provider}")
    logging.info(f"Concurrency: {args.concurrency}")
    logging.info("=" * 60)
    
    # Run async evaluation
    results = asyncio.run(run_evaluation(args, paths))
    
    # Log summary
    logging.info("=" * 60)
    logging.info("Evaluation Summary")
    logging.info("=" * 60)
    
    if args.task_id:
        task_results = results.get(args.task_id, [])
        logging.info(f"Task {args.task_id}: {len(task_results)} results")
        for res in task_results:
            score = res.get('final_score', 'N/A')
            answer = res.get('answer_name', 'unknown')
            logging.info(f"  - {answer}: score={score}")
    else:
        total_results = sum(len(r) for r in results.values())
        logging.info(f"Evaluated {len(results)} tasks with {total_results} total results")
        for task_id, task_results in sorted(results.items()):
            if task_results:
                avg_score = sum(r.get('final_score', 0) for r in task_results) / len(task_results)
                logging.info(f"  - {task_id}: {len(task_results)} results, avg_score={avg_score:.2f}")
            else:
                logging.info(f"  - {task_id}: No results")
    
    # Merge all results if evaluating all tasks
    if not args.task_id and results:
        logging.info("=" * 60)
        logging.info("Merging all results...")
        merge_all_results(paths.eval_results_root)
        logging.info("✅ Results merged successfully")
    
    logging.info("=" * 60)
    logging.info("🎉 Evaluation completed!")


if __name__ == "__main__":
    main()
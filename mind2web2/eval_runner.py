from __future__ import annotations

import asyncio
import json
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

from tqdm import tqdm

from .utils.cache import CacheClass
from .utils.logging_setup import create_logger, cleanup_logger
from .utils.load_eval_script import load_eval_script


# --------------------------------------------------------------------------- #
# Path helpers                                                                #
# --------------------------------------------------------------------------- #


def _all_answer_paths(task_root: Path) -> List[Path]:
    """Return a *sorted* list of every answer file under <answer_dir>/<task_id>/<agent>/<answer_file>."""

    priority = {"human": 0, "hf_open_deep_research": -1, "openai_deep_research": 1}

    agent_dirs = sorted(
        (d for d in task_root.iterdir() if d.is_dir()),
        key=lambda p: (priority.get(p.name, 2), p.name),
    )

    paths: List[Path] = []
    for agent_dir in agent_dirs:
        paths.extend(sorted(p for p in agent_dir.iterdir() if p.is_file()))
    return paths


import re


def _answer_base(answer_name: str) -> str:
    """Strip the trailing extension, e.g. 'answer_3.md' → 'answer_3'."""
    return answer_name.rsplit(".", 1)[0]


def _extract_ts_from_name(fname: str) -> str | None:
    """
    Extract a 14-digit timestamp (YYYYMMDDHHMMSS) from the filename, or return None if not found.
    """
    m = re.search(r"(\d{8})[_]?(\d{6})", fname)
    if not m:
        return None
    return "".join(m.groups())


def _latest_json(result_dir: Path) -> Path | None:
    """Return newest *.json file in <result_dir> (by timestamp in filename)."""
    if not result_dir.exists():
        return None
    json_files = [p for p in result_dir.iterdir() if p.suffix == ".json"]
    if not json_files:
        return None

    def _ts_key(fp: Path):
        ts_str = _extract_ts_from_name(fp.name)
        return datetime.strptime(ts_str, "%Y%m%d%H%M%S") if ts_str else datetime.min

    return max(json_files, key=_ts_key)


# --------------------------------------------------------------------------- #
# Single‑answer evaluation                                                    #
# --------------------------------------------------------------------------- #


async def _eval_one_answer(
        eval_fn,
        client,
        task_id: str,
        answer_path: Path,
        cache: CacheClass,
        semaphore: asyncio.Semaphore,
        output_dir: Path,
        is_self_debug: bool = False,
):
    """Evaluate a single answer file and write its result JSON / logs."""

    agent_name = answer_path.parent.name
    answer_name = answer_path.name
    answer_base = _answer_base(answer_name)

    # ---------- Create isolated logging ----------
    log_dir = output_dir / task_id / agent_name / answer_base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Use a more specific logger name to ensure uniqueness
    log_tag = f"{task_id}_{agent_name}_{answer_name}"
    if is_self_debug:
        log_tag += "_debug"

    # Important: Disable console output in concurrent environments to avoid log confusion
    logger, timestamp = create_logger(
        log_tag,
        str(log_dir),
        enable_console=False  # Disable console output during concurrency, only output to file
    )

    # Add structured log for task start
    logger.info(
        f"🚀 Starting evaluation for {agent_name}/{answer_name}",
        extra={
            "task_id": task_id,
            "agent_name": agent_name,
            "answer_name": answer_name,
            "answer_base": answer_base,
            "is_debug": is_self_debug,
            "operation": "eval_start"
        }
    )

    # ---------- Read answer ----------
    try:
        answer_text = answer_path.read_text(encoding="utf-8")
        logger.debug(
            f"Answer loaded: {len(answer_text)} characters",
            extra={"answer_length": len(answer_text)}
        )
    except Exception as e:
        logger.error(f"Failed to read answer file: {e}")
        return e

    result = None
    try:
        # Log waiting outside the semaphore
        logger.debug("Waiting for evaluation semaphore")

        async with semaphore:  # Control concurrency inside eval_fn
            logger.info("🔄 Starting evaluation function")

            result: Dict = await eval_fn(
                client=client,
                answer=answer_text,
                agent_name=agent_name,
                answer_name=answer_name,
                cache=cache,
                semaphore=semaphore,
                logger=logger,
                model="o4-mini",
            )

            logger.info(
                f"✅ Evaluation completed with score: {result.get('final_score', 'unknown')}",
                extra={
                    "final_score": result.get("final_score"),
                    "operation": "eval_complete"
                }
            )

    except Exception as exc:
        logger.exception(
            "❌ Evaluation raised an exception",
            extra={
                "error_type": type(exc).__name__,
                "operation": "eval_error"
            }
        )
        return exc
    finally:
        # Clean up logger resources
        try:
            cleanup_logger(logger)
        except Exception:
            pass  # Cleanup failure should not affect main flow

    # ---------- Save result ----------
    try:
        if result is not None:
            _save_result_json(result, output_dir / task_id, timestamp, is_self_debug)
    except Exception as e:
        print(f"Failed to save result for {agent_name}/{answer_name}: {e}")
        return e

    return result


def _save_result_json(result: Dict, task_out_dir: Path, ts: str, is_debug: bool):
    """Write per‑answer result JSON to disk."""

    agent = result["agent_name"]
    answer = result["answer_name"]
    answer_base = _answer_base(answer)

    save_dir = task_out_dir / agent / answer_base / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{ts}_{answer}{'_debug' if is_debug else ''}.json"
    with (save_dir / fname).open("w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=4)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


async def evaluate_task(
        client,
        task_id: str,
        answer_dir: Union[str, Path],
        cache_dir: Union[str, Path],
        output_dir: Union[str, Path],
        script_path: Union[str, Path],
        dump_cache: bool = True,
        is_self_debug: bool = False,
        overwrite: bool = False,
        max_concurrent_answers: int = 6,  # How many answer files can be under evaluation at the same time
        semaphore: int = 10,  # The maximum parallelism **inside a single answer evaluation** (passed as
        # an `asyncio.Semaphore` to `eval_fn`).
):
    """Evaluate *all* answers of a task with two‑level concurrency control.

    Parameters
    ----------
    max_concurrent_answers : int, default 6
        How many answer files can be under evaluation at the same time.
    semaphore : int, default 10
        The maximum parallelism **inside a single answer evaluation** (passed as
        an `asyncio.Semaphore` to `eval_fn`).

    Other parameters keep their original meanings.
    """

    # ------------------------------------------------------------------
    # 0. Setup paths & ensure dirs exist
    # ------------------------------------------------------------------
    answer_root = Path(answer_dir) / task_id
    output_root = Path(output_dir)
    cache_root = Path(cache_dir)

    answer_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Create main task logger (for overall progress tracking)
    # ------------------------------------------------------------------
    main_log_dir = output_root / task_id / "main_logs"
    main_log_dir.mkdir(parents=True, exist_ok=True)
    main_logger, main_timestamp = create_logger(
        f"main_{task_id}",
        str(main_log_dir),
        enable_console=True  # Main logger can output to console
    )

    try:
        main_logger.info(
            f"🎯 Starting task evaluation: {task_id}",
            extra={
                "task_id": task_id,
                "max_concurrent_answers": max_concurrent_answers,
                "inner_semaphore": semaphore,
                "operation": "task_start"
            }
        )

        # ------------------------------------------------------------------
        # 2. Load eval script & cache
        # ------------------------------------------------------------------
        main_logger.info("📜 Loading evaluation script")
        eval_fn = load_eval_script(script_path)

        cache_path = cache_root / f"{task_id}.pkl"
        cache = CacheClass(cache_path=str(cache_path))
        cache.merge(str(cache_root / "gt_urls.pkl"))
        main_logger.info(f"💾 Cache loaded from {cache_path}")

        # ------------------------------------------------------------------
        # 3. Collect answer files
        # ------------------------------------------------------------------
        answer_paths = _all_answer_paths(answer_root)
        main_logger.info(
            f"📁 Found {len(answer_paths)} answer files to evaluate",
            extra={
                "answer_count": len(answer_paths),
                "answer_paths": [f"{p.parent.name}/{p.name}" for p in answer_paths]
            }
        )
        print("-->> Answer Root:", answer_root)
        print("-->> Answers to Eval:", answer_paths)

        ok_results: List[Dict] = []

        # ------------------------------------------------------------------
        # 4. Concurrency primitives
        # ------------------------------------------------------------------
        outer_semaphore = asyncio.Semaphore(max_concurrent_answers)

        # ------------------------------------------------------------------
        # 5. Define per‑answer coroutine
        # ------------------------------------------------------------------
        async def _process_answer(ans_path: Path):
            async with outer_semaphore:  # Control concurrency inside answer
                agent_name = ans_path.parent.name
                answer_name = ans_path.name
                answer_base = _answer_base(answer_name)

                main_logger.info(
                    f"👉 Processing {agent_name}/{answer_name}",
                    extra={
                        "agent_name": agent_name,
                        "answer_name": answer_name,
                        "operation": "answer_start"
                    }
                )
                print(f"👉 Starting {agent_name} {answer_name}")

                # 5‑A. Copy original md to workspace (if not copied yet)
                dst_md = output_root / task_id / agent_name / answer_name
                if not dst_md.exists():
                    dst_md.parent.mkdir(parents=True, exist_ok=True)
                    dst_md.write_bytes(ans_path.read_bytes())
                    main_logger.debug(f"📋 Copied answer file to {dst_md}")

                # 5‑B. Result reuse check
                result_dir = dst_md.parent / answer_base / "results"
                latest = _latest_json(result_dir)
                if latest and not overwrite:
                    main_logger.info(
                        f"⚠️ Using existing result for {agent_name}/{answer_name}",
                        extra={
                            "agent_name": agent_name,
                            "answer_name": answer_name,
                            "existing_result": str(latest),
                            "operation": "reuse_result"
                        }
                    )
                    print(f"⚠️ Existing result -- {agent_name} {answer_name}")
                    try:
                        result = json.loads(latest.read_text(encoding="utf-8"))
                        main_logger.debug(
                            f"✅ Loaded existing result with score: {result.get('final_score')}",
                            extra={
                                "agent_name": agent_name,
                                "answer_name": answer_name,
                                "final_score": result.get('final_score'),
                                "operation": "existing_result_loaded"
                            }
                        )
                        return result
                    except Exception as exc:
                        main_logger.error(
                            f"❌ Failed to load existing result: {exc}",
                            extra={
                                "agent_name": agent_name,
                                "answer_name": answer_name,
                                "error": str(exc),
                                "operation": "existing_result_error"
                            }
                        )
                        traceback.print_exception(type(exc), exc, exc.__traceback__)

                # 5‑C. Real evaluation
                inner_semaphore = asyncio.Semaphore(semaphore)
                try:
                    res = await _eval_one_answer(
                        eval_fn,
                        client,
                        task_id,
                        ans_path,
                        cache,
                        inner_semaphore,
                        output_root,
                        is_self_debug,
                    )

                    if isinstance(res, dict):
                        main_logger.info(
                            f"✅ Successfully evaluated {agent_name}/{answer_name}",
                            extra={
                                "agent_name": agent_name,
                                "answer_name": answer_name,
                                "final_score": res.get('final_score'),
                                "operation": "answer_complete"
                            }
                        )
                    else:
                        main_logger.error(
                            f"❌ Evaluation failed for {agent_name}/{answer_name}: {res}",
                            extra={
                                "agent_name": agent_name,
                                "answer_name": answer_name,
                                "error": str(res),
                                "operation": "answer_error"
                            }
                        )

                    return res
                except Exception as exc:
                    main_logger.exception(
                        f"💥 Unexpected error evaluating {agent_name}/{answer_name}",
                        extra={
                            "agent_name": agent_name,
                            "answer_name": answer_name,
                            "error_type": type(exc).__name__,
                            "operation": "answer_exception"
                        }
                    )
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
                    return exc

        # ------------------------------------------------------------------
        # 6. Kick off evaluations
        # ------------------------------------------------------------------
        main_logger.info(f"🚀 Starting concurrent evaluation of {len(answer_paths)} answers")
        tasks = [asyncio.create_task(_process_answer(p)) for p in answer_paths]

        completed_count = 0
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"[{task_id}] Evaluating"):
            res = await coro
            completed_count += 1

            if isinstance(res, dict):
                ok_results.append(res)
                main_logger.debug(
                    f"✅ [{completed_count}/{len(tasks)}] Completed evaluation for {res.get('agent_name')}/{res.get('answer_name')}",
                    extra={
                        "completed_count": completed_count,
                        "total_count": len(tasks),
                        "agent_name": res.get('agent_name'),
                        "answer_name": res.get('answer_name'),
                        "final_score": res.get('final_score'),
                        "operation": "progress_update"
                    }
                )
            else:
                main_logger.error(
                    f"❌ [{completed_count}/{len(tasks)}] Evaluation failed with error: {res}",
                    extra={
                        "completed_count": completed_count,
                        "total_count": len(tasks),
                        "error": str(res),
                        "operation": "progress_error"
                    }
                )

        # ------------------------------------------------------------------
        # 7. Persist cache & merge summary
        # ------------------------------------------------------------------
        if dump_cache:
            cache.dump(str(cache_path))
            main_logger.info("💾 Cache dumped successfully")

        _merge_and_save_summary(output_root / task_id, ok_results)
        main_logger.info("📊 Summary saved successfully")

        main_logger.info(
            f"🎉 Task evaluation completed: {len(ok_results)}/{len(answer_paths)} successful results",
            extra={
                "task_id": task_id,
                "successful_count": len(ok_results),
                "total_count": len(answer_paths),
                "success_rate": len(ok_results) / len(answer_paths) if answer_paths else 0,
                "operation": "task_complete"
            }
        )

        return ok_results

    except Exception as e:
        main_logger.exception(
            f"💥 Task evaluation failed: {e}",
            extra={
                "task_id": task_id,
                "error_type": type(e).__name__,
                "operation": "task_error"
            }
        )
        raise
    finally:
        # Clean up main logger
        try:
            cleanup_logger(main_logger)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Summary helper                                                              #
# --------------------------------------------------------------------------- #


def _merge_and_save_summary(task_out_dir: Path, all_results: List[Dict]):
    """Aggregate per‑answer results into task‑level summary."""

    merged: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for res in all_results:
        agent = res["agent_name"]
        answer = res["answer_name"]
        merged[agent][answer] = {
            "score": float(res["final_score"]),
            "status": "success" if res["final_score"] > 0 else "failed",
            "success": res["final_score"] == 1,
        }

    summary = {
        agent: [
            {"answer_name": ans, **info}
            for ans, info in sorted(answers.items())
        ]
        for agent, answers in sorted(merged.items())
    }

    with (task_out_dir / "result.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=4)
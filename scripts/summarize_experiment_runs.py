from __future__ import annotations

"""
Summarize EPM experiment runs from `epm/runs`.

Data sources:
- final dish feedback: `recipe_feedback_*.json`
- step outcomes and per-step timing: `memory/long_horizon_history.txt`
- planner API/model/timing stats: `memory/planner_request_metrics.jsonl`

Typical usage:
    python epm/scripts/summarize_experiment_runs.py --group-by method
    python epm/scripts/summarize_experiment_runs.py --group-by method_model
    python epm/scripts/summarize_experiment_runs.py --group-by model
    python epm/scripts/summarize_experiment_runs.py --group-by method --method reflexion
    python epm/scripts/summarize_experiment_runs.py --group-by method_model --model gpt-5-mini

Optional filters:
- `--method reflexion`
- `--model gpt-5-mini`
- `--provider openai_compatible`

Outputs:
- per-run JSON summary
- grouped JSON summary
- per-run CSV summary

Default output directory:
- `epm/runs/analysis`
"""

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from _benchmark_defaults import current_benchmark_size


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(_read_text(path))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = _read_text(path)
    if not raw:
        return rows
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_run_name(run_name: str) -> tuple[str, int | None]:
    parts = [p for p in str(run_name or "").strip().split("-") if p]
    if len(parts) >= 3 and parts[-1].isdigit():
        return parts[1], int(parts[-1])
    if len(parts) >= 2 and parts[-1].isdigit():
        return parts[-2], int(parts[-1])
    return "unknown", None


def _latest_feedback_file(run_root: Path) -> Path | None:
    matches = sorted(run_root.glob("recipe_feedback_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _extract_feedback_summary(feedback_obj: dict[str, Any]) -> tuple[str, list[str]]:
    if not isinstance(feedback_obj, dict):
        return "", []
    complaints = feedback_obj.get("complaints")
    if not isinstance(complaints, dict):
        return "", []
    lines: list[str] = []
    missing_items: list[str] = []
    for section_name in ("flavors", "technique", "temperature"):
        items = complaints.get(section_name)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            product = str(item.get("LocalizedProductName") or "").strip()
            message = str(item.get("Message") or "").strip()
            reason = str(item.get("Reason") or "").strip()
            text = " / ".join([x for x in (product, message or reason) if x])
            if text:
                lines.append(text)
            if message.lower() == "not enough" and product:
                missing_items.append(product)
    unwanted = complaints.get("unwantedProducts")
    if isinstance(unwanted, list):
        for item in unwanted:
            if not isinstance(item, dict):
                continue
            product = str(item.get("LocalizedProductName") or "").strip()
            if product:
                lines.append(f"unwanted / {product}")
    return " | ".join(lines[:12]), missing_items


def _normalize_error_text(text: Any) -> str:
    s = str(text or "").strip().strip("'").strip('"')
    if not s:
        return ""
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    return s


def _classify_error(error_text: str) -> str:
    s = str(error_text or "").strip().lower()
    if not s:
        return ""
    if "unauthorized" in s or "未提供令牌" in s or "token" in s or "api_key" in s:
        return "auth"
    if "timeout" in s or "timed out" in s:
        return "timeout"
    if "planner_exception" in s or "invalid_plan" in s or "planner_" in s:
        return "planner"
    if "navigation_failed" in s or "astar_path_not_found" in s or "precise_adjustment_failed" in s:
        return "navigation"
    if "pick_up_" in s or "not_holding_after_click" in s or "already_holding_item" in s:
        return "pick_up"
    if "put_down_" in s or "still_holding_item" in s or "hands_empty" in s:
        return "put_down"
    if "enter_" in s and "_mode_failed" in s:
        return "mode_switch"
    if "exit_" in s and "_mode_failed" in s:
        return "mode_switch"
    if "pour" in s or "liquid_bottle_not_found" in s:
        return "pour"
    if "sprinkle" in s:
        return "sprinkle"
    if "cut" in s or "chop" in s:
        return "cut"
    if "repair" in s:
        return "repair"
    if "camera_info" in s or "percept" in s or "query_scene" in s:
        return "perception"
    if "submit" in s or "checkout" in s:
        return "submit"
    if "gui_" in s or "computer" in s:
        return "gui"
    return "other"


def _counter_to_json(counter: Counter[str], *, top_k: int = 20) -> str:
    items = counter.most_common(top_k)
    return json.dumps([{ "name": k, "count": v } for k, v in items], ensure_ascii=False)


def _summarize_run(run_root: Path) -> dict[str, Any]:
    method, dish_id = _parse_run_name(run_root.name)
    history_path = run_root / "memory" / "long_horizon_history.txt"
    metrics_path = run_root / "memory" / "planner_request_metrics.jsonl"
    feedback_path = _latest_feedback_file(run_root)

    history_rows = _read_jsonl(history_path)
    metric_rows = _read_jsonl(metrics_path)
    feedback_obj = _read_json(feedback_path) if feedback_path else {}

    provider_counter: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()
    planner_api_failures = 0
    planner_api_timeout_count = 0
    planner_api_latency_s = 0.0
    planner_api_success_latency_s = 0.0
    for row in metric_rows:
        provider = str(row.get("provider") or "").strip()
        model = str(row.get("model") or "").strip()
        if provider:
            provider_counter[provider] += 1
        if model:
            model_counter[model] += 1
        planner_api_latency_s += _safe_float(row.get("latency_s"))
        if bool(row.get("success")):
            planner_api_success_latency_s += _safe_float(row.get("latency_s"))
        else:
            planner_api_failures += 1
            if str(row.get("error_kind") or "").strip().lower() == "timeout":
                planner_api_timeout_count += 1

    planning_time_s = 0.0
    execution_time_s = 0.0
    step_total_time_s = 0.0
    total_steps = 0
    success_steps = 0
    failed_steps = 0
    error_counts: Counter[str] = Counter()
    error_category_counts: Counter[str] = Counter()
    final_submit_success = False
    wall_clock_time_s = 0.0
    for row in history_rows:
        total_steps += 1
        result_summary = str(row.get("result_summary") or "").strip().lower()
        if result_summary == "success":
            success_steps += 1
        elif result_summary:
            failed_steps += 1
        timing = row.get("diff", {}).get("timing_s") if isinstance(row.get("diff"), dict) else None
        if not isinstance(timing, dict):
            timing = row.get("timing_s")
        if isinstance(timing, dict):
            planning_time_s += _safe_float(timing.get("planning"))
            execution_time_s += _safe_float(timing.get("execution"))
            step_total_time_s += _safe_float(timing.get("total"))
        else:
            step_total_time_s += _safe_float(row.get("duration_s"))
        wall_clock_time_s = max(wall_clock_time_s, _safe_float(row.get("episode_elapsed_s")))
        error_text = _normalize_error_text(row.get("errors"))
        if error_text:
            error_counts[error_text] += 1
            error_category_counts[_classify_error(error_text)] += 1
        action_name = str(row.get("action_or_skill") or "").strip()
        if action_name == "gui_submit_dish_via_checkout_stand" and result_summary == "success":
            final_submit_success = True

    feedback_summary, missing_items = _extract_feedback_summary(feedback_obj)
    dish_name = str(feedback_obj.get("dishName") or "").strip()
    taste_score = feedback_obj.get("taste_score")
    if taste_score in ("", None):
        taste_score = None
    else:
        taste_score = _safe_float(taste_score, default=0.0)

    provider = provider_counter.most_common(1)[0][0] if provider_counter else ""
    model = model_counter.most_common(1)[0][0] if model_counter else ""
    final_success = bool(final_submit_success or feedback_path is not None)

    return {
        "run_name": run_root.name,
        "run_path": str(run_root),
        "method": method,
        "dish_id": dish_id,
        "provider": provider,
        "model": model,
        "method_model": f"{method}|{model}" if model else method,
        "dish_name": dish_name,
        "feedback_available": bool(feedback_path),
        "feedback_file": str(feedback_path) if feedback_path else "",
        "feedback_summary": feedback_summary,
        "missing_items": list(dict.fromkeys(missing_items)),
        "taste_score": taste_score,
        "final_success": final_success,
        "submit_success": final_submit_success,
        "total_steps": total_steps,
        "success_steps": success_steps,
        "failed_steps": failed_steps,
        "planning_time_s": round(planning_time_s, 3),
        "execution_time_s": round(execution_time_s, 3),
        "step_total_time_s": round(step_total_time_s, 3),
        "wall_clock_time_s": round(wall_clock_time_s, 3),
        "planner_api_calls": len(metric_rows),
        "planner_api_failures": planner_api_failures,
        "planner_api_timeout_count": planner_api_timeout_count,
        "planner_api_latency_s": round(planner_api_latency_s, 3),
        "planner_api_success_latency_s": round(planner_api_success_latency_s, 3),
        "top_error": error_counts.most_common(1)[0][0] if error_counts else "",
        "top_error_count": error_counts.most_common(1)[0][1] if error_counts else 0,
        "top_error_category": error_category_counts.most_common(1)[0][0] if error_category_counts else "",
        "top_error_category_count": error_category_counts.most_common(1)[0][1] if error_category_counts else 0,
        "error_counts": dict(error_counts),
        "error_category_counts": dict(error_category_counts),
    }


def _group_key_for(run_summary: dict[str, Any], group_by: str) -> str:
    if group_by == "method":
        return str(run_summary.get("method") or "unknown")
    if group_by == "model":
        return str(run_summary.get("model") or "unknown")
    if group_by == "provider":
        return str(run_summary.get("provider") or "unknown")
    if group_by == "method_model":
        return str(run_summary.get("method_model") or "unknown")
    raise ValueError(f"unsupported_group_by:{group_by}")


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(statistics.fmean(vals), 6) if vals else None


def _build_group_summary(
    *,
    key: str,
    rows: list[dict[str, Any]],
    expected_dishes: int,
) -> dict[str, Any]:
    error_counts: Counter[str] = Counter()
    error_category_counts: Counter[str] = Counter()
    dish_ids = {row.get("dish_id") for row in rows if row.get("dish_id") is not None}
    feedback_scores = [row.get("taste_score") for row in rows if row.get("taste_score") is not None]
    for row in rows:
        error_counts.update(row.get("error_counts") or {})
        error_category_counts.update(row.get("error_category_counts") or {})
    return {
        "group": key,
        "num_runs": len(rows),
        "num_unique_dishes": len(dish_ids),
        "dish_coverage_ratio": round((len(dish_ids) / expected_dishes), 6) if expected_dishes > 0 else None,
        "final_success_runs": sum(1 for row in rows if bool(row.get("final_success"))),
        "submit_success_runs": sum(1 for row in rows if bool(row.get("submit_success"))),
        "feedback_runs": sum(1 for row in rows if bool(row.get("feedback_available"))),
        "avg_taste_score": _mean([v for v in feedback_scores if v is not None]),
        "avg_total_steps": _mean([_safe_float(row.get("total_steps")) for row in rows]),
        "avg_failed_steps": _mean([_safe_float(row.get("failed_steps")) for row in rows]),
        "avg_planning_time_s": _mean([_safe_float(row.get("planning_time_s")) for row in rows]),
        "avg_execution_time_s": _mean([_safe_float(row.get("execution_time_s")) for row in rows]),
        "avg_step_total_time_s": _mean([_safe_float(row.get("step_total_time_s")) for row in rows]),
        "avg_wall_clock_time_s": _mean([_safe_float(row.get("wall_clock_time_s")) for row in rows]),
        "avg_planner_api_calls": _mean([_safe_float(row.get("planner_api_calls")) for row in rows]),
        "avg_planner_api_latency_s": _mean([_safe_float(row.get("planner_api_latency_s")) for row in rows]),
        "avg_planner_api_timeout_count": _mean([_safe_float(row.get("planner_api_timeout_count")) for row in rows]),
        "top_error_categories": error_category_counts.most_common(10),
        "top_errors": error_counts.most_common(10),
        "error_category_counts": dict(error_category_counts),
        "error_counts": dict(error_counts),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = dict(row)
        flat["missing_items"] = json.dumps(flat.get("missing_items") or [], ensure_ascii=False)
        flat["error_counts"] = json.dumps(flat.get("error_counts") or {}, ensure_ascii=False)
        flat["error_category_counts"] = json.dumps(flat.get("error_category_counts") or {}, ensure_ascii=False)
        flat_rows.append(flat)
    fieldnames = sorted({key for row in flat_rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize EPM experiment runs by method/model, including scores, timing, and failure categories.")
    parser.add_argument("--runs-root", default="epm/runs")
    parser.add_argument("--group-by", choices=("method", "model", "provider", "method_model"), default="method")
    parser.add_argument("--method", default="", help="Optional method filter, e.g. reflexion.")
    parser.add_argument("--model", default="", help="Optional model filter, e.g. gpt-5-mini.")
    parser.add_argument("--provider", default="", help="Optional provider filter, e.g. openai_compatible.")
    parser.add_argument("--expected-dishes", type=int, default=current_benchmark_size())
    parser.add_argument("--output-dir", default="epm/runs/analysis")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()] if runs_root.exists() else []
    run_dirs.sort(key=lambda p: p.name)

    run_summaries: list[dict[str, Any]] = []
    for run_root in run_dirs:
        summary = _summarize_run(run_root)
        if args.method and str(summary.get("method") or "").strip().lower() != args.method.strip().lower():
            continue
        if args.model and str(summary.get("model") or "").strip().lower() != args.model.strip().lower():
            continue
        if args.provider and str(summary.get("provider") or "").strip().lower() != args.provider.strip().lower():
            continue
        run_summaries.append(summary)

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in run_summaries:
        key = _group_key_for(row, args.group_by)
        groups.setdefault(key, []).append(row)

    grouped_rows = [
        _build_group_summary(key=key, rows=rows, expected_dishes=int(args.expected_dishes))
        for key, rows in sorted(groups.items(), key=lambda kv: kv[0])
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.group_by
    if args.method:
        suffix += f"__method_{args.method}"
    if args.model:
        suffix += f"__model_{args.model}"
    if args.provider:
        suffix += f"__provider_{args.provider}"

    runs_json = output_dir / f"run_summary__{suffix}.json"
    groups_json = output_dir / f"group_summary__{suffix}.json"
    runs_csv = output_dir / f"run_summary__{suffix}.csv"

    runs_json.write_text(json.dumps(run_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    groups_json.write_text(json.dumps(grouped_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(runs_csv, run_summaries)

    print(f"[summary] runs_root={runs_root}")
    print(f"[summary] num_runs={len(run_summaries)} group_by={args.group_by}")
    print(f"[summary] run_json={runs_json}")
    print(f"[summary] group_json={groups_json}")
    print(f"[summary] run_csv={runs_csv}")
    print()
    for group in grouped_rows:
        print(
            "[group] "
            f"{group['group']} "
            f"runs={group['num_runs']} "
            f"dishes={group['num_unique_dishes']}/{args.expected_dishes} "
            f"avg_score={group['avg_taste_score']} "
            f"avg_plan_s={group['avg_planning_time_s']} "
            f"avg_exec_s={group['avg_execution_time_s']} "
            f"top_error_categories={_counter_to_json(Counter(group['error_category_counts']))}"
        )


if __name__ == "__main__":
    main()

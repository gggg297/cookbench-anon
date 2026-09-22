import argparse
import json
from pathlib import Path
from typing import Any

'''
  新脚本

  它会根据：
  - memory/resume_state.json

  conda activate py310
  python epm/scripts/generate_run_time_summary.py --run-name 20260319_194345_825585-epm-1

  对最新 run 生成：

  conda activate py310
  python epm/scripts/generate_run_time_summary.py

  也支持直接给路径：

  conda activate py310
  python epm/scripts/generate_run_time_summary.py --run-root epm/runs/20260319_194345_825585-epm-1

  中断时的统计口径

  - executed_steps：按 long_horizon_history.txt 里已经落盘的步数算
  - total_time：max(sum(duration_s), max(episode_elapsed_s))
  - average_time_per_step：total_time / executed_steps

  额外说明

  - 如果 resume_state.json 不存在，脚本会把状态记成 partial
  - 所以即使是人工中断、事后补算，也能用

'''


SCRIPT_PATH = Path(__file__).resolve()
EPM_ROOT = SCRIPT_PATH.parents[1]
RUNS_ROOT = EPM_ROOT / "runs"


def _read_resume_state(memory_dir: Path) -> dict[str, Any]:
    path = memory_dir / "resume_state.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _find_latest_run_dir(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_history_records(memory_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    candidates = [memory_dir / "long_horizon_history.txt"]
    candidates.extend(sorted(memory_dir.glob("long_horizon_history.txt.fallback.*")))
    for path in candidates:
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                if isinstance(obj, dict):
                    records.append(obj)
        except Exception:
            continue

    dedup: dict[int, dict[str, Any]] = {}
    for rec in records:
        try:
            step_id = int(rec.get("step_id") or 0)
        except Exception:
            continue
        dedup[step_id] = rec
    return [dedup[k] for k in sorted(dedup.keys())]


def _format_minutes_text(seconds: float) -> str:
    minutes = max(0.0, float(seconds or 0.0)) / 60.0
    return f"{minutes:.2f} min"


def _infer_dish_id(run_name: str, resume_state: dict[str, Any]) -> str:
    dish_id = resume_state.get("dish_id")
    if isinstance(dish_id, int):
        return str(dish_id)
    if isinstance(dish_id, str) and dish_id.strip():
        return dish_id.strip()
    tail = str(run_name or "").rsplit("-", 1)
    if len(tail) == 2 and tail[1].isdigit():
        return tail[1]
    return "unknown"


def build_summary_text(*, run_root: Path) -> str:
    memory_dir = run_root / "memory"
    resume_state = _read_resume_state(memory_dir)
    records = _load_history_records(memory_dir)

    run_name = run_root.name
    dish_id = _infer_dish_id(run_name, resume_state)
    status = str(resume_state.get("status") or "").strip() or "partial"
    reason = str(resume_state.get("reason") or "").strip()

    executed_steps = len(records)
    planning_s = 0.0
    execution_s = 0.0
    summed_total_s = 0.0
    episode_elapsed_s = 0.0

    for rec in records:
        diff = rec.get("diff") if isinstance(rec.get("diff"), dict) else {}
        timing = diff.get("timing_s") if isinstance(diff.get("timing_s"), dict) else {}
        try:
            planning_s += float(timing.get("planning") or 0.0)
        except Exception:
            pass
        try:
            execution_s += float(timing.get("execution") or 0.0)
        except Exception:
            pass
        try:
            summed_total_s += float(rec.get("duration_s") or 0.0)
        except Exception:
            pass
        try:
            episode_elapsed_s = max(episode_elapsed_s, float(rec.get("episode_elapsed_s") or 0.0))
        except Exception:
            pass

    total_s = max(episode_elapsed_s, summed_total_s)
    avg_total_s = (total_s / executed_steps) if executed_steps > 0 else 0.0
    avg_plan_s = (planning_s / executed_steps) if executed_steps > 0 else 0.0
    avg_exec_s = (execution_s / executed_steps) if executed_steps > 0 else 0.0

    lines = [
        "Run Time Summary",
        f"run_name: {run_name}",
        f"dish_id: {dish_id}",
        f"status: {status}",
        f"reason: {reason}",
        f"executed_steps: {executed_steps}",
        f"planning_time: {_format_minutes_text(planning_s)}",
        f"execution_time: {_format_minutes_text(execution_s)}",
        f"total_time: {_format_minutes_text(total_s)}",
        f"average_time_per_step: {_format_minutes_text(avg_total_s)}",
        f"average_planning_time_per_step: {_format_minutes_text(avg_plan_s)}",
        f"average_execution_time_per_step: {_format_minutes_text(avg_exec_s)}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="", help="Run folder name under epm/runs.")
    parser.add_argument("--run-root", type=Path, default=None, help="Absolute or relative run folder path.")
    parser.add_argument("--latest", action="store_true", help="Use the latest run under epm/runs.")
    args = parser.parse_args()

    run_root: Path | None = None
    if args.run_root is not None:
        run_root = Path(args.run_root).resolve()
    elif (args.run_name or "").strip():
        run_root = (RUNS_ROOT / str(args.run_name).strip()).resolve()
    else:
        latest = _find_latest_run_dir(RUNS_ROOT)
        if latest is None:
            print("[EPM] no run directory found under epm/runs")
            return 2
        run_root = latest.resolve()

    if run_root is None or not run_root.exists():
        print(f"[EPM] run_root_not_found: {run_root}")
        return 2
    memory_dir = run_root / "memory"
    if not memory_dir.exists():
        print(f"[EPM] memory_dir_not_found: {memory_dir}")
        return 2

    text = build_summary_text(run_root=run_root)
    out_path = run_root / "run_time_summary.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"[EPM] run_time_summary_written: {out_path}")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

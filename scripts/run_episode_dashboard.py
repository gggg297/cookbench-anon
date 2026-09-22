from __future__ import annotations

"""
Interactive dashboard runner for EPM experiments.

Features:
- live dashboard for a single run
- optional auto-chain execution across multiple dish IDs
- dish range unions such as `1-5,8,10-12`
- optional planner model override without editing the source config
- feedback-file post-check after each run
- bilingual console reminders for missing feedback and next-run confirmation

Common usage:
    python epm/scripts/run_episode_dashboard.py --config epm/configs/pipelines/reflexion.json --dish-id 1
    python epm/scripts/run_episode_dashboard.py --config epm/configs/pipelines/reflexion.json --dish-range 1-131,137,138 --auto-chain
    python epm/scripts/run_episode_dashboard.py --config epm/configs/pipelines/reflexion.json --dish-range 1-20,41-60 --auto-chain
    python epm/scripts/run_episode_dashboard.py --config epm/configs/pipelines/reflexion.json --resume --refresh-memory --auto-chain --dish-range 1-131,137,138
    python epm/scripts/run_episode_dashboard.py --config epm/configs/pipelines/reflexion.json --dish-range 1-131,137,138 --auto-chain --planner-model gpt-5-mini
    
    回滚到指定步：
    python epm/scripts/run_episode.py --config epm/configs/pipelines/react.json --resume --run-name 20260413_xxx --resume-step-id 80

    相对当前恢复点回滚 5 步：
    python epm/scripts/run_episode.py --config epm/configs/pipelines/react.json --resume --run-name 20260413_xxx --rollback-steps 5

    - `budget30.level1`: `11, 13, 40, 41, 83, 88, 105, 112, 117`
    - `budget30.level2`: `1, 9, 15, 23, 35, 39, 45`
    - `budget30.level3`: `3, 4, 7, 10, 12, 14, 17`
    - `budget30.level4`: `2, 5, 6, 16`
    - `budget30.level5`: `8, 20, 30`



Behavior notes:
- `--dish-range` supports union syntax and preserves the given order.
- when feedback is missing, the console reminder includes both the manual dump command and the resume command.
- when feedback exists, the script prints the recommended next command; with `--auto-chain`, it directly jumps to the next dish.
- `--resume --refresh-memory --auto-chain` can continue an interrupted batch run.
- batch progress is persisted in `epm/runs/_batch_state/*.json`.
"""

import argparse
import hashlib
import json
import locale
import os
import re
import shutil
import subprocess
import sys
import threading
import textwrap
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
SRC_DIR = REPO_ROOT / "epm" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epm.core.prompt_ablation import resolve_ablation_reductions  # noqa: E402
from epm.core.settings import load_raw_settings  # noqa: E402
from _benchmark_defaults import current_benchmark_dish_ids  # noqa: E402

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except Exception:
    print("Missing dependency: rich. Install with: pip install rich")
    raise


_DASHBOARD_CONSOLE = Console()


def _safe_rich_style(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    text = str(value).strip()
    return text or None


def _decode_windows_exit_status(rc: int) -> dict[str, str] | None:
    try:
        code = int(rc)
    except Exception:
        return None
    if code < 0:
        code &= 0xFFFFFFFF
    if code <= 0x7FFFFFFF:
        return None

    hex_code = f"0x{code:08X}"
    known: dict[int, dict[str, str]] = {
        0xC0000005: {
            "name": "STATUS_ACCESS_VIOLATION",
            "diagnosis": "Windows 原生层访问非法内存；进程被系统直接终止，通常不会留下 Python traceback。",
            "probable_causes": "ctypes/pywin32/mss/cv2/native DLL/显卡或远程桌面图形链路异常",
            "suggested_action": "优先检查截图、窗口捕获、键鼠注入、OpenCV/pywin32/mss 链路；建议本地直连显示器复现，并查看 faulthandler 输出。",
        },
        0xC0000409: {
            "name": "STATUS_STACK_BUFFER_OVERRUN",
            "diagnosis": "Windows 检测到原生栈/快速失败保护触发；通常是某个 native 扩展、Win32 调用或底层 DLL 主动 fast-fail。",
            "probable_causes": "ctypes/pywin32/cv2/mss/native DLL 调用越界、驱动/overlay 注入冲突、远程桌面图形链路异常",
            "suggested_action": "重点检查 auto_navigation/截图/窗口捕获链路，关闭 overlay/录屏/远程桌面注入，并查看 faulthandler 是否留下崩前 Python 栈。",
        },
        0xC0000374: {
            "name": "STATUS_HEAP_CORRUPTION",
            "diagnosis": "Windows 检测到堆损坏；通常是原生扩展或 DLL 内存写坏。",
            "probable_causes": "native DLL、OpenCV/pywin32/mss、驱动或外部注入模块",
            "suggested_action": "优先排查最近触发的原生模块，必要时抓 dump 分析。",
        },
        0xC0000135: {
            "name": "STATUS_DLL_NOT_FOUND",
            "diagnosis": "进程依赖的 DLL 缺失。",
            "probable_causes": "VC++ Runtime 缺失、Python wheel 依赖不完整、系统环境不一致",
            "suggested_action": "检查本机运行时依赖与 Python 环境。",
        },
        0xC0000142: {
            "name": "STATUS_DLL_INIT_FAILED",
            "diagnosis": "某个 DLL 初始化失败。",
            "probable_causes": "图形/驱动/远程桌面/原生依赖初始化异常",
            "suggested_action": "检查显示驱动、远程桌面环境和相关原生依赖。",
        },
    }
    payload = dict(known.get(code) or {})
    payload["hex"] = hex_code
    if "name" not in payload:
        payload["name"] = "WINDOWS_NATIVE_FATAL_STATUS"
        payload["diagnosis"] = "Windows 返回了原生致命退出码；通常表示 Python 进程在 native 层被直接终止。"
        payload["probable_causes"] = "ctypes/native DLL/驱动/远程桌面或图形链路异常"
        payload["suggested_action"] = "记录 hex 退出码并结合崩前最后几行 Runner Logs / faulthandler 输出排查。"
    return payload


def _load_pipeline_from_config(config_path: str | Path) -> str:
    try:
        data = load_raw_settings(config_path)
        if isinstance(data, dict):
            brain = data.get("brain")
            if isinstance(brain, dict):
                pipeline = str(brain.get("pipeline") or "").strip().lower()
                if pipeline:
                    return pipeline
    except Exception:
        pass
    return "planner_executor"


def _load_planner_mode_from_config(config_path: str | Path) -> str:
    try:
        data = load_raw_settings(config_path)
        if isinstance(data, dict):
            brain = data.get("brain")
            if isinstance(brain, dict):
                mode = str(brain.get("planner_mode") or "").strip().lower()
                if mode == "scripted":
                    return "scripted"
                if "use_vlm" in brain:
                    return "vlm" if bool(brain.get("use_vlm", True)) else "llm"
                if mode in {"llm", "vlm"}:
                    return mode
    except Exception:
        pass
    return "vlm"


def _load_planner_model_from_config(config_path: str | Path) -> str:
    try:
        data = load_raw_settings(config_path)
        if isinstance(data, dict):
            mode = _load_planner_mode_from_config(config_path)
            assignments = data.get("api_model_assignments")
            if isinstance(assignments, dict):
                role = assignments.get(mode)
                if isinstance(role, dict):
                    model = str(role.get("model") or "").strip()
                    if model:
                        return model
    except Exception:
        pass
    return "model"


def _normalize_run_name_part(value: str, *, fallback: str) -> str:
    text = (value or "").strip()
    if not text:
        return fallback
    for ch in ('/', '\\', ':', '*', '?', '"', "<", ">", "|"):
        text = text.replace(ch, "_")
    text = re.sub(r"\s+", "_", text)
    return text or fallback


def _default_run_name(dish_id: int, *, pipeline: str, model: str, prompt_ablation_profile: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pipeline_part = _normalize_run_name_part(str(pipeline or ""), fallback="planner_executor")
    model_part = _normalize_run_name_part(str(model or ""), fallback="model")
    profile_part = _normalize_run_name_part(str(prompt_ablation_profile or ""), fallback="full")
    return f"{ts}-{pipeline_part}-{model_part}-{profile_part}-{dish_id}"


def _parse_dish_id_from_run_name(run_name: str) -> int | None:
    m = re.search(r"-(\d+)$", str(run_name or "").strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_dish_range(spec: str) -> list[int]:
    from _benchmark_defaults import parse_dish_range

    return parse_dish_range(spec)


def _format_dish_range(dish_ids: list[int]) -> str:
    from _benchmark_defaults import format_dish_range

    return format_dish_range(dish_ids)


def _next_dish_in_sequence(current_dish_id: int | None, dish_sequence: list[int]) -> int | None:
    if not dish_sequence:
        return None
    if current_dish_id is None:
        return dish_sequence[0]
    try:
        idx = dish_sequence.index(int(current_dish_id))
    except Exception:
        for dish_id in dish_sequence:
            if dish_id > int(current_dish_id):
                return dish_id
        return None
    next_idx = idx + 1
    if next_idx >= len(dish_sequence):
        return None
    return dish_sequence[next_idx]


def _trim_dish_sequence_from(start_dish_id: int, dish_sequence: list[int]) -> list[int]:
    if not dish_sequence:
        return []
    try:
        idx = dish_sequence.index(int(start_dish_id))
        return dish_sequence[idx:]
    except Exception:
        pass
    later = [dish_id for dish_id in dish_sequence if dish_id >= int(start_dish_id)]
    return later or [int(start_dish_id)]


def _absolutize_config_paths(data: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    def _resolve_in(section: dict[str, Any], key: str) -> None:
        raw = section.get(key)
        if not isinstance(raw, str):
            return
        text = raw.strip()
        if not text:
            return
        p = Path(text)
        if p.is_absolute():
            return
        section[key] = str((base_dir / p).resolve())

    paths = data.get("paths")
    if isinstance(paths, dict):
        for key in (
            "recipes_path",
            "memory_dir",
            "auto_nav_map_path",
            "screenshot_dir",
            "game_userdata_root",
            "realtime_products_path",
            "camera_info_path",
        ):
            _resolve_in(paths, key)

    brain = data.get("brain")
    if isinstance(brain, dict):
        for key in ("action_catalog_path", "prompt_layout_path", "scripted_plan_path"):
            _resolve_in(brain, key)

    return data


def _materialize_config_override(repo: Path, config_path: str | Path, planner_model: str) -> Path:
    src = Path(config_path).resolve()
    model = str(planner_model or "").strip()
    if not model:
        return src
    data = load_raw_settings(src)
    mode = _load_planner_mode_from_config(src)
    assignments = data.get("api_model_assignments")
    if not isinstance(assignments, dict):
        assignments = {}
        data["api_model_assignments"] = assignments
    role = assignments.get(mode)
    if not isinstance(role, dict):
        role = {}
        assignments[mode] = role
    role["model"] = model
    tmp_dir = repo / "epm" / "runs" / "_tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    base_stem = _normalize_run_name_part(src.stem, fallback="config")
    model_stem = _normalize_run_name_part(model, fallback="model")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dst = tmp_dir / f"{base_stem}-{model_stem}-{ts}.json"
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return dst


def _latest_recipe_feedback_file(run_root: Path) -> Path | None:
    if not run_root.exists():
        return None
    files = sorted(
        run_root.glob("recipe_feedback_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _norm_dish_name(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _read_json_with_fallback(path: Path) -> dict[str, Any]:
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            raw = path.read_text(encoding=enc, errors="ignore")
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return {}


def _load_expected_dish_name(config_path: str | Path, dish_id: int) -> str:
    try:
        data = load_raw_settings(config_path)
        paths = data.get("paths") if isinstance(data.get("paths"), dict) else {}
        recipes_path = str(paths.get("recipes_path") or "").strip()
        if not recipes_path:
            return ""
        from epm.kb.recipes import get_dish_by_id

        return str(get_dish_by_id(recipes_path, int(dish_id)).dish_name or "").strip()
    except Exception:
        return ""


def _find_matching_recipe_feedback(
    *,
    run_root: Path,
    expected_dish_name: str,
) -> tuple[Path | None, list[Path], list[str]]:
    files = sorted(
        run_root.glob("recipe_feedback_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    want_key = _norm_dish_name(expected_dish_name)
    mismatched_names: list[str] = []
    if not files:
        return None, [], mismatched_names
    if not want_key:
        return files[0], files, mismatched_names
    for path in files:
        try:
            data = _read_json_with_fallback(path)
        except Exception:
            continue
        got = str(data.get("dishName") or data.get("dish_name") or "").strip()
        if got:
            mismatched_names.append(got)
        if _norm_dish_name(got) == want_key:
            return path, files, mismatched_names
    return None, files, mismatched_names


def _poll_matching_recipe_feedback_on_mismatch(
    *,
    run_root: Path,
    expected_dish_name: str,
    interval_s: float = 2.0,
    max_wait_s: float = 10.0,
) -> tuple[Path | None, list[Path], list[str], int]:
    feedback_file, feedback_files, mismatch_names = _find_matching_recipe_feedback(
        run_root=run_root,
        expected_dish_name=expected_dish_name,
    )
    if feedback_file is not None or not feedback_files:
        return feedback_file, feedback_files, mismatch_names, 0

    try:
        total_attempts = max(0, int(max_wait_s // interval_s))
    except Exception:
        total_attempts = 8
    total_attempts = max(0, total_attempts)
    waited_s = 0
    for attempt in range(1, total_attempts + 1):
        time.sleep(max(0.0, float(interval_s)))
        waited_s = attempt
        feedback_file, feedback_files, mismatch_names = _find_matching_recipe_feedback(
            run_root=run_root,
            expected_dish_name=expected_dish_name,
        )
        if feedback_file is not None:
            return feedback_file, feedback_files, mismatch_names, waited_s
        if not feedback_files:
            break
    return feedback_file, feedback_files, mismatch_names, waited_s


def _quote_cmd(parts: list[str]) -> str:
    out: list[str] = []
    for part in parts:
        if re.search(r'[\s"]', part):
            out.append('"' + part.replace('"', '\\"') + '"')
        else:
            out.append(part)
    return " ".join(out)


def _validate_passthrough_unknown_args(parser: argparse.ArgumentParser, unknown: list[str]) -> list[str]:
    validated: list[str] = []
    i = 0
    while i < len(unknown):
        token = str(unknown[i] or "")
        if not token.startswith("--"):
            parser.error(f"unexpected extra positional argument: {token!r}")
        validated.append(token)
        if "=" in token:
            i += 1
            continue
        if i + 1 < len(unknown) and not str(unknown[i + 1] or "").startswith("--"):
            validated.append(str(unknown[i + 1]))
            i += 2
            continue
        i += 1
    return validated


def _batch_state_path(
    *,
    runs_root: Path,
    config_path: str | Path,
    pipeline: str,
    model: str,
    range_spec: str,
) -> Path:
    cfg = Path(config_path)
    cfg_stem = _normalize_run_name_part(cfg.stem, fallback="config")
    pipeline_part = _normalize_run_name_part(pipeline, fallback="pipeline")
    model_part = _normalize_run_name_part(model, fallback="model")
    range_part = _normalize_run_name_part(range_spec, fallback="range")
    digest = hashlib.md5(str(Path(config_path).resolve()).encode("utf-8")).hexdigest()[:8]
    filename = f"{cfg_stem}-{pipeline_part}-{model_part}-{range_part}-{digest}.json"
    return runs_root / "_batch_state" / filename


def _read_batch_state(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _write_batch_state(
    path: Path,
    *,
    source_config_path: str | Path,
    pipeline: str,
    planner_model: str,
    range_spec: str,
    dish_sequence: list[int],
    completed_ids: list[int],
    current_dish_id: int | None,
    next_dish_id: int | None,
    current_run_name: str,
    last_completed_run_name: str,
    status: str,
    run_status: str,
    run_reason: str,
) -> None:
    payload = {
        "source_config_path": str(Path(source_config_path).resolve()),
        "pipeline": str(pipeline or ""),
        "planner_model": str(planner_model or ""),
        "range_spec": str(range_spec or ""),
        "dish_sequence": [int(x) for x in dish_sequence],
        "completed_ids": [int(x) for x in completed_ids],
        "current_dish_id": (int(current_dish_id) if current_dish_id is not None else None),
        "next_dish_id": (int(next_dish_id) if next_dish_id is not None else None),
        "current_run_name": str(current_run_name or ""),
        "last_completed_run_name": str(last_completed_run_name or ""),
        "status": str(status or ""),
        "run_status": str(run_status or ""),
        "run_reason": str(run_reason or ""),
        "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _merge_completed_ids(existing: list[int], new_id: int | None) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for x in existing:
        try:
            value = int(x)
        except Exception:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    if new_id is not None:
        value = int(new_id)
        if value not in seen:
            out.append(value)
    return out


def _build_feedback_dump_command(*, config_path: str | Path, run_root: Path) -> str:
    return _quote_cmd(
        [
            sys.executable,
            "epm/scripts/dump_recipe_feedback.py",
            "--config",
            str(Path(config_path).resolve()),
            "--run-root",
            str(run_root.resolve()),
        ]
    )


def _build_resume_command(
    *,
    source_config_path: str | Path,
    effective_dish_range_spec: str,
    run_name: str,
    auto_chain: bool,
    planner_model: str,
    ablation_reduce: str,
    no_color: bool,
    log_level: str,
    unknown: list[str],
) -> str:
    cmd = [
        sys.executable,
        "epm/scripts/run_episode_dashboard.py",
        "--config",
        str(source_config_path),
        "--resume",
        "--refresh-memory",
    ]
    if run_name:
        cmd += ["--run-name", run_name]
    if effective_dish_range_spec:
        cmd += ["--dish-range", effective_dish_range_spec]
    if auto_chain:
        cmd.append("--auto-chain")
    if planner_model:
        cmd += ["--planner-model", planner_model]
    if ablation_reduce:
        cmd += ["--ablation-reduce", ablation_reduce]
    if no_color:
        cmd.append("--no-color")
    if log_level:
        cmd += ["--log-level", log_level]
    cmd.extend(unknown)
    return _quote_cmd(cmd)


def _build_next_dish_command(
    *,
    source_config_path: str | Path,
    next_dish_id: int,
    effective_dish_range_spec: str,
    auto_chain: bool,
    planner_model: str,
    ablation_reduce: str,
    steps: int | None,
    no_color: bool,
    alt_screen: bool,
    log_level: str,
    unknown: list[str],
) -> str:
    cmd = [
        sys.executable,
        "epm/scripts/run_episode_dashboard.py",
        "--config",
        str(source_config_path),
        "--dish-id",
        str(next_dish_id),
        "--restart_env",
    ]
    if effective_dish_range_spec:
        cmd += ["--dish-range", effective_dish_range_spec]
    if auto_chain:
        cmd.append("--auto-chain")
    if planner_model:
        cmd += ["--planner-model", planner_model]
    if ablation_reduce:
        cmd += ["--ablation-reduce", ablation_reduce]
    if steps is not None:
        cmd += ["--steps", str(steps)]
    if no_color:
        cmd.append("--no-color")
    if alt_screen:
        cmd.append("--alt-screen")
    if log_level:
        cmd += ["--log-level", log_level]
    cmd.extend(unknown)
    return _quote_cmd(cmd)


def _find_latest_run_name(runs_root: Path) -> str:
    if not runs_root.exists():
        return ""
    dirs = [
        p
        for p in runs_root.iterdir()
        if p.is_dir() and re.match(r"^\d{8}_\d{6}_\d{6}[-_].+", p.name)
    ]
    if not dirs:
        dirs = [p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if not dirs:
        return ""
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0].name


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_effective_resume_step(resume_state: dict[str, Any]) -> tuple[int, str]:
    state = resume_state if isinstance(resume_state, dict) else {}
    status = str(state.get("status") or "").strip().lower()
    reason = str(state.get("reason") or "").strip().lower()
    if status in {"waiting_network", "waiting_http_403", "fatal"} and (
        "network_" in reason or "http_403" in reason or "http_429" in reason
    ):
        for key, source in (
            ("last_completed_step_id", "resume_state.last_completed_step_id"),
            ("last_physical_step_id", "resume_state.last_physical_step_id"),
            ("last_network_resume_step_id", "resume_state.last_network_resume_step_id"),
            ("last_http_403_resume_step_id", "resume_state.last_http_403_resume_step_id"),
        ):
            raw = state.get(key)
            if isinstance(raw, int) and raw > 0:
                return int(raw), source
            if isinstance(raw, str) and raw.isdigit() and int(raw) > 0:
                return int(raw), source
    raw = state.get("last_step_id")
    if isinstance(raw, int) and raw > 0:
        return int(raw), "resume_state.last_step_id"
    if isinstance(raw, str) and raw.isdigit() and int(raw) > 0:
        return int(raw), "resume_state.last_step_id"
    return 0, "none"


_TRACE_RESULT_RE = re.compile(
    r"step=(?P<step>\d+)\s+"
    r"kind=(?P<kind>\w+)\s+"
    r"step_id=(?P<step_id>\S+)\s+"
    r"type=(?P<type>\S+)\s+"
    r"name=(?P<name>\S+)\s+"
    r"args=(?P<args>\{.*?\})\s+"
    r"result='(?P<result>[^']*)'\s+"
    r"error=(?P<error>.*)$"
)


def _dashboard_unknown_arg_value(unknown: list[str], flag: str) -> str | None:
    want = str(flag or "").strip()
    if not want:
        return None
    i = 0
    while i < len(unknown):
        token = str(unknown[i] or "")
        if token == want:
            if i + 1 < len(unknown) and not str(unknown[i + 1] or "").startswith("--"):
                return str(unknown[i + 1])
            return None
        prefix = want + "="
        if token.startswith(prefix):
            return token[len(prefix) :]
        i += 1
    return None


def _dashboard_unknown_arg_present(unknown: list[str], flag: str) -> bool:
    want = str(flag or "").strip()
    if not want:
        return False
    for token in unknown:
        raw = str(token or "")
        if raw == want or raw.startswith(want + "="):
            return True
    return False


def _dashboard_collect_post_resume_physical_actions(*, memory_dir: Path, resume_step_id: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    trace_path = memory_dir / "step_trace.log"
    if not trace_path.exists():
        return out
    try:
        for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _TRACE_RESULT_RE.search(line)
            if not m or str(m.group("kind") or "") != "result":
                continue
            step_no = int(m.group("step"))
            if step_no <= int(resume_step_id):
                continue
            step_type = str(m.group("type") or "").strip().lower()
            step_name = str(m.group("name") or "").strip().lower()
            if (step_type, step_name) in {
                ("skill", "query_scene_objects"),
                ("skill", "auto_perception"),
                ("skill", "list_supported_items"),
            }:
                continue
            out.append(
                {
                    "step": step_no,
                    "step_id": str(m.group("step_id") or ""),
                    "type": str(m.group("type") or ""),
                    "name": str(m.group("name") or ""),
                }
            )
    except Exception:
        return out
    return out


def _confirm_dashboard_manual_rewind(
    *,
    memory_dir: Path,
    detected_resume_step_id: int,
    target_resume_step_id: int,
) -> bool:
    if int(target_resume_step_id) >= int(detected_resume_step_id):
        return True
    actions = _dashboard_collect_post_resume_physical_actions(memory_dir=memory_dir, resume_step_id=target_resume_step_id)
    if not actions:
        return True
    if not sys.stdin or not sys.stdin.isatty():
        print("")
        _DASHBOARD_CONSOLE.print("[dashboard] [bold #ff9e64]MANUAL INPUT REQUIRED[/bold #ff9e64]")
        _DASHBOARD_CONSOLE.print("[dashboard] [#ffd166]rewind confirmation is required, but stdin is not interactive.[/#ffd166]")
        _DASHBOARD_CONSOLE.print("[dashboard] [bold #7bd88f]Re-run with `--yes-rewind` to continue.[/bold #7bd88f]")
        return False

    sample = ", ".join(
        f"step={int(item['step'])}:{item['type']}:{item['name']}"
        for item in actions[:6]
    )
    print("")
    _DASHBOARD_CONSOLE.print("[dashboard] [bold #ff9e64]" + "=" * 60 + "[/bold #ff9e64]")
    _DASHBOARD_CONSOLE.print("[dashboard] [bold #ff75c3]MANUAL INPUT REQUIRED: RESUME REWIND CONFIRMATION[/bold #ff75c3]")
    _DASHBOARD_CONSOLE.print("[dashboard] [bold #ff9e64]" + "=" * 60 + "[/bold #ff9e64]")
    _DASHBOARD_CONSOLE.print(f"[dashboard] [#ffd166]detected_resume_step_id={int(detected_resume_step_id)}[/#ffd166]")
    _DASHBOARD_CONSOLE.print(f"[dashboard] [#ffd166]target_resume_step_id={int(target_resume_step_id)}[/#ffd166]")
    _DASHBOARD_CONSOLE.print(f"[dashboard] [#ffd166]physical_actions_after_target={len(actions)}[/#ffd166]")
    if sample:
        _DASHBOARD_CONSOLE.print(f"[dashboard] [#ffd166]samples={sample}[/#ffd166]")
    _DASHBOARD_CONSOLE.print("[dashboard] [#ffd166]Logs and memory will be rewound, but the game world cannot be auto-restored.[/#ffd166]")
    _DASHBOARD_CONSOLE.print(
        "[dashboard] [bold #ff75c3]INPUT REQUIRED:[/bold #ff75c3] "
        "type [bold #7bd88f]yes[/bold #7bd88f] to continue rewind, "
        "or type [bold #ff75c3]no[/bold #ff75c3] to cancel."
    )
    print("")
    try:
        _DASHBOARD_CONSOLE.print("[dashboard] [bold #ff9e64]>>> [/bold #ff9e64]", end="")
        answer = input().strip().lower()
    except EOFError:
        answer = ""
    return answer in {"y", "yes"}


def _read_last_jsonl_obj(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {}
    for line in reversed(lines):
        s = (line or "").strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


class _TailReader:
    def __init__(self, path: Path, max_lines: int = 120) -> None:
        self.path = path
        self.lines: deque[str] = deque(maxlen=max_lines)
        self._pos = 0

    def poll(self) -> list[str]:
        if not self.path.exists():
            return list(self.lines)
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(self._pos)
                chunk = f.read()
                self._pos = f.tell()
        except Exception:
            return list(self.lines)
        if chunk:
            for line in chunk.splitlines():
                self.lines.append(line)
        return list(self.lines)


def _drain_process_output(proc: subprocess.Popen[str], logs: list[str], line_counter: dict[str, int]) -> None:
    assert proc.stdout is not None
    for raw in proc.stdout:
        logs.append(raw.rstrip("\r\n"))
        if len(logs) > 20000:
            del logs[:5000]
        try:
            line_counter["n"] = int(line_counter.get("n", 0)) + 1
        except Exception:
            line_counter["n"] = 0


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_TRACE_RE = re.compile(
    r"step=(?P<step>\d+)\s+"
    r"kind=(?P<kind>\w+)\s+"
    r"step_id=(?P<step_id>\S+)\s+"
    r"type=(?P<type>\S+)\s+"
    r"name=(?P<name>\S+)\s+"
    r"args=(?P<args>\{.*?\})\s+"
    r"result='(?P<result>[^']*)'\s+"
    r"error=(?P<error>.*)$"
)
_LOG_STEP_RE = re.compile(r"===== step\s+(?P<step>\d+)\s+=====")
_LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+\|\s+(?P<level>[A-Z]+)\s+\|\s+(?P<msg>.*)$"
)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def _compact_value(v: Any, max_len: int = 56) -> str:
    s = repr(v)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _compact_args(args: dict[str, Any], max_items: int = 3) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    picked: list[str] = []
    used: set[str] = set()
    for k in ("query", "target", "target_instance_id", "dish_name"):
        if k in args and k not in used:
            picked.append(k)
            used.add(k)
        if len(picked) >= max_items:
            break
    if len(picked) < max_items:
        for k in args.keys():
            if k in used:
                continue
            picked.append(k)
            if len(picked) >= max_items:
                break
    return " ".join(f"{k}={_compact_value(args.get(k))}" for k in picked)


def _clean_runner_message(msg: str) -> str:
    s = (msg or "").strip()
    s = s.lstrip()
    s = re.sub(r"^\[[A-Za-z0-9_-]+-\d+\]\s*", "", s)
    s = re.sub(r"^\[EPM\]\s*", "", s)
    s = re.sub(r"^\[EPM\]\[[^\]]+\]\s*", "", s)
    s = re.sub(r"^[A-Za-z0-9_.]+:[A-Za-z0-9_<>]+:\d+\s*-\s*", "", s)
    return s


def _parse_pipe_kv(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    parts: list[tuple[str, str]] = []
    extras: list[str] = []
    for chunk in re.split(r"\s+\|\s+", str(text or "").strip()):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            extras.append(item)
            continue
        k, v = item.split("=", 1)
        parts.append((str(k).strip(), str(v).strip()))
    return parts, extras


def _networkish_message(text: str) -> bool:
    fields, extras = _parse_pipe_kv(text)
    keys = {k for k, _ in fields}
    if keys & {
        "kind",
        "http_status",
        "probable_causes",
        "diagnosis",
        "common_triggers",
        "suggested_action",
        "api_key_masked",
        "api_key_slot",
        "provider",
        "source",
    }:
        return True
    raw = str(text or "").lower()
    if any(
        token in raw
        for token in (
            "network_pause_required",
            "network_abort_required",
            "http_401",
            "http_403",
            "connection_reset",
            "remote end closed connection",
            "ssl",
            "timeout",
            "quota",
            "api_key_pool_exhausted",
            "invalid_api_key",
            "unauthorized",
            "forbidden",
        )
    ):
        return True
    return any("network" in item.lower() for item in extras)


def _format_diagnostic_text(text: str, *, width: int = 68) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if not _networkish_message(raw):
        wrapped = textwrap.wrap(
            raw,
            width=max(24, int(width)),
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        return "\n".join(wrapped) if wrapped else raw

    pairs, extras = _parse_pipe_kv(raw)
    multi_keys = {"probable_causes", "common_triggers", "diagnosis", "suggested_action", "body", "error"}
    preferred = [
        "message",
        "kind",
        "http_status",
        "source",
        "episode_step",
        "rolled_back_to",
        "api_key_slot",
        "api_key_masked",
        "provider",
        "diagnosis",
        "probable_causes",
        "common_triggers",
        "suggested_action",
        "error",
        "body",
    ]
    label_map = {
        "message": "message",
        "kind": "kind",
        "http_status": "http_status",
        "source": "source",
        "episode_step": "episode_step",
        "rolled_back_to": "rolled_back_to",
        "api_key_slot": "api_key_slot",
        "api_key_masked": "api_key",
        "provider": "provider",
        "diagnosis": "diagnosis",
        "probable_causes": "probable_causes",
        "common_triggers": "common_triggers",
        "suggested_action": "suggested_action",
        "error": "error",
        "body": "body",
    }

    ordered: list[tuple[str, str]] = []
    used_idx: set[int] = set()
    for wanted in preferred:
        for idx, (k, v) in enumerate(pairs):
            if idx in used_idx or k != wanted:
                continue
            ordered.append((k, v))
            used_idx.add(idx)
    for idx, pair in enumerate(pairs):
        if idx not in used_idx:
            ordered.append(pair)

    lines: list[str] = []
    wrap_width = max(24, int(width))
    for k, v in ordered:
        label = label_map.get(k, k)
        value = str(v or "").strip()
        if not value:
            continue
        if k in multi_keys or len(value) > wrap_width - len(label) - 2:
            wrapped = textwrap.wrap(
                value,
                width=wrap_width,
                break_long_words=False,
                break_on_hyphens=False,
                replace_whitespace=False,
            )
            if not wrapped:
                lines.append(f"{label}:")
                continue
            lines.append(f"{label}: {wrapped[0]}")
            for cont in wrapped[1:]:
                lines.append(f"  {cont}")
        else:
            lines.append(f"{label}: {value}")
    for extra in extras:
        wrapped = textwrap.wrap(
            extra,
            width=wrap_width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        if wrapped:
            lines.append(f"note: {wrapped[0]}")
            for cont in wrapped[1:]:
                lines.append(f"  {cont}")
    return "\n".join(lines) if lines else raw


def _tail_wrapped_lines(lines: list[str], *, width: int, height: int) -> str:
    if not lines:
        return "(empty)"
    ww = max(24, int(width))
    hh = max(6, int(height))
    wrapped: list[str] = []
    for ln in lines:
        parts = textwrap.wrap(
            ln,
            width=ww,
            break_long_words=True,
            break_on_hyphens=False,
            drop_whitespace=False,
            replace_whitespace=False,
        )
        if not parts:
            wrapped.append("")
        else:
            wrapped.extend(parts)
    if len(wrapped) > hh:
        wrapped = wrapped[-hh:]
    return "\n".join(wrapped)



def _parse_trace_line(line: str) -> dict[str, Any] | None:
    m = _TRACE_RE.search(line)
    if not m:
        return None
    try:
        args = json.loads(str(m.group("args") or "{}"))
    except Exception:
        args = {}
    return {
        "step": int(m.group("step")),
        "kind": str(m.group("kind") or ""),
        "step_id": str(m.group("step_id") or ""),
        "type": str(m.group("type") or ""),
        "name": str(m.group("name") or ""),
        "args": args if isinstance(args, dict) else {},
        "result": str(m.group("result") or ""),
        "error": str(m.group("error") or ""),
    }


def _parse_lhh_line(line: str) -> dict[str, Any] | None:
    s = line.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _read_long_horizon_lines(history_path: Path, max_lines: int = 320) -> list[str]:
    lines: list[str] = []
    paths = [history_path]
    fallback_pattern = history_path.name + ".fallback.*"
    paths.extend(sorted(history_path.parent.glob(fallback_pattern), key=lambda p: p.stat().st_mtime))
    for p in paths:
        if not p.exists():
            continue
        try:
            chunk = p.read_text(encoding="utf-8", errors="replace").splitlines()
            lines.extend(chunk)
        except Exception:
            continue
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def _ensure_run_time_summary(*, repo: Path, run_name: str) -> None:
    try:
        cmd = [
            sys.executable,
            "epm/scripts/generate_run_time_summary.py",
            "--run-name",
            str(run_name),
        ]
        subprocess.run(
            cmd,
            cwd=str(repo),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except Exception:
        return


def _distill_lhh(history_lines: list[str], *, step_floor: int = 0, keep: int = 22) -> list[str]:
    out: list[str] = []
    for line in history_lines:
        rec = _parse_lhh_line(line)
        if rec is None:
            continue
        try:
            step = int(rec.get("step_id") or 0)
        except Exception:
            step = 0
        if step_floor > 0 and step <= int(step_floor):
            continue
        action = str(rec.get("action_or_skill") or "")
        params = rec.get("params") if isinstance(rec.get("params"), dict) else {}
        result = str(rec.get("result_summary") or "")
        plan_ref = rec.get("plan_ref") if isinstance(rec.get("plan_ref"), dict) else {}
        atomic = str(plan_ref.get("atomic_step_id") or "-")
        dur = rec.get("duration_s")
        params_preview = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
        if len(params_preview) > 88:
            params_preview = params_preview[:85] + "..."
        row = f"ep{step:03d} {atomic} {action} params={params_preview} result={result or '-'}"
        if isinstance(dur, (int, float)):
            row += f" ({dur:.1f}s)"
        err = str(rec.get("errors") or "").strip()
        if err:
            row += f" err={_compact_value(err, max_len=66)}"
        out.append(row)
    return out[-keep:]


def _distill_trace(trace_lines: list[str], *, step_floor: int = 0, keep: int = 12) -> list[str]:
    out: list[str] = []
    for line in trace_lines:
        rec = _parse_trace_line(line)
        if rec is None:
            continue
        step = int(rec.get("step") or 0)
        if step_floor > 0 and step <= int(step_floor):
            continue
        kind = str(rec.get("kind") or "").upper()
        step_id = str(rec.get("step_id") or "-")
        typ = str(rec.get("type") or "")
        name = str(rec.get("name") or "")
        args_preview = _compact_args(rec.get("args") or {})
        if kind == "RESULT":
            result = str(rec.get("result") or "")
            err = str(rec.get("error") or "")
            row = f"ep{step:03d} RESULT {step_id} {typ}:{name} => {result or '-'}"
            if result.lower() != "success" and err:
                row += f" err={_compact_value(err, max_len=66)}"
        else:
            row = f"ep{step:03d} PLAN   {step_id} {typ}:{name} {args_preview}".rstrip()
        out.append(row)
    return out[-keep:]


def _derive_current_state(
    *,
    status_obj: dict[str, Any],
    trace_lines: list[str],
    history_lines: list[str],
    logs: list[str],
) -> dict[str, Any]:
    step = status_obj.get("episode_step")
    logical_step = status_obj.get("logical_step")
    tentative_step = bool(status_obj.get("tentative_step"))
    phase = str(status_obj.get("phase") or "").strip()
    plan_step_id = str(status_obj.get("step_id") or "").strip()
    typ = str(status_obj.get("type") or "").strip()
    name = str(status_obj.get("name") or "").strip()
    args = status_obj.get("args") if isinstance(status_obj.get("args"), dict) else {}

    if step is None:
        for line in reversed(logs):
            m = _LOG_STEP_RE.search(_strip_ansi(line))
            if m:
                step = int(m.group("step"))
                break
    if step is None:
        for line in reversed(history_lines):
            rec = _parse_lhh_line(line)
            if rec and rec.get("step_id") is not None:
                try:
                    step = int(rec.get("step_id"))
                except Exception:
                    step = rec.get("step_id")
                break
    if step is None:
        for line in reversed(trace_lines):
            rec = _parse_trace_line(line)
            if rec:
                step = rec.get("step")
                break

    if not name:
        for line in reversed(trace_lines):
            rec = _parse_trace_line(line)
            if rec and str(rec.get("kind")) == "plan":
                plan_step_id = str(rec.get("step_id") or plan_step_id)
                typ = str(rec.get("type") or typ)
                name = str(rec.get("name") or name)
                args = rec.get("args") if isinstance(rec.get("args"), dict) else args
                if not phase:
                    phase = "planned"
                break
    if not name:
        for line in reversed(history_lines):
            rec = _parse_lhh_line(line)
            if not rec:
                continue
            action = str(rec.get("action_or_skill") or "").strip()
            if action:
                name = action
                typ = "step"
                args = rec.get("params") if isinstance(rec.get("params"), dict) else {}
                plan_ref = rec.get("plan_ref") if isinstance(rec.get("plan_ref"), dict) else {}
                plan_step_id = str(plan_ref.get("atomic_step_id") or plan_step_id)
                if not phase:
                    phase = str(rec.get("result_summary") or "running")
                break

    action_line = ""
    if name:
        action_line = f"{typ}:{name}"
        preview = _compact_args(args)
        if preview:
            action_line += f" {preview}"

    if step is None:
        step_display: Any = "-"
    else:
        step_display = step
        try:
            logical_int = int(logical_step) if logical_step is not None else None
        except Exception:
            logical_int = None
        try:
            step_int = int(step)
        except Exception:
            step_int = None
        if tentative_step and logical_int is not None and step_int is not None and logical_int != step_int:
            step_display = f"{step_int}~ (logical {logical_int})"

    return {
        "step": step_display,
        "phase": phase or "waiting",
        "plan_step_id": plan_step_id or "-",
        "action_line": action_line or "-",
    }


def _agent_state_lines(agent_state: dict[str, Any]) -> list[str]:
    if not agent_state:
        return ["(empty: {})"]
    lines: list[str] = []
    priority = [
        "mode",
        "is_held",
        "held_item",
        "held_item_name",
        "held_item_instance_id",
        "focus_name",
        "focus_instance_id",
        "force_submit_active",
        "force_submit_steps_left",
        "countdown",
    ]
    used: set[str] = set()
    for k in priority:
        if k in agent_state:
            used.add(k)
            lines.append(f"{k}: {_compact_value(agent_state.get(k), max_len=72)}")
    remaining = [k for k in sorted(agent_state.keys()) if k not in used]
    for k in remaining[:8]:
        lines.append(f"{k}: {_compact_value(agent_state.get(k), max_len=72)}")
    if len(remaining) > 8:
        lines.append(f"... (+{len(remaining)-8} keys)")
    return lines


def _planner_cache_rows(planner_metrics: dict[str, Any]) -> list[tuple[str, str]]:
    if not isinstance(planner_metrics, dict) or not planner_metrics:
        return []
    cache_status = str(planner_metrics.get("cache_status") or "").strip()
    cache_reason = str(planner_metrics.get("cache_reason") or "").strip()
    cache_mode = str(planner_metrics.get("cache_mode") or "").strip()
    cache_requested = bool(planner_metrics.get("cache_requested"))
    cache_enabled = bool(planner_metrics.get("cache_enabled"))
    cache_name = str(planner_metrics.get("cache_name") or "").strip()
    provider_hit = bool(planner_metrics.get("cache_hit"))
    try:
        provider_hit_tokens = int(planner_metrics.get("cache_hit_tokens") or 0)
    except Exception:
        provider_hit_tokens = 0
    if provider_hit_tokens > 0:
        provider_hit = True
    explicit_hit = cache_status in {"reused", "hit"}
    any_hit = provider_hit or explicit_hit

    source_bits: list[str] = []
    if provider_hit:
        source_bits.append("provider")
    if explicit_hit:
        source_bits.append("explicit")
    source_text = "+".join(source_bits) if source_bits else "-"

    hit_parts = ["hit" if any_hit else "miss", f"source={source_text}"]
    if any_hit:
        hit_parts.append(f"hit_tokens={provider_hit_tokens}")
    if explicit_hit and cache_name:
        hit_parts.append(f"cache={cache_name}")

    cfg_parts = [
        f"requested={cache_requested}",
        f"enabled={cache_enabled}",
    ]
    if cache_mode:
        cfg_parts.append(f"mode={cache_mode}")
    if cache_status:
        cfg_parts.append(f"status={cache_status}")
    if cache_reason:
        cfg_parts.append(f"reason={cache_reason}")

    return [
        ("planner_cache", " ".join(hit_parts)),
        ("planner_cache_cfg", " ".join(cfg_parts)),
    ]


def _make_header(
    run_name: str,
    dish_id: int,
    step: Any,
    plan_step_id: str,
    phase: Any,
    action_line: str,
    started: float,
    rc: int | None,
    balance_text: str = "",
) -> Panel:
    elapsed = max(0.0, time.time() - started)
    status = "RUNNING" if rc is None else f"EXIT={rc}"
    title = Text()
    title.append("EPM Dashboard", style=_safe_rich_style("bold cyan"))
    meta = Text()
    meta.append(f" run={run_name} ", style=_safe_rich_style("magenta"))
    meta.append(f"dish={dish_id} ", style=_safe_rich_style("green"))
    meta.append(f"ep_step={step} ", style=_safe_rich_style("yellow"))
    meta.append(f"plan_step={plan_step_id or '-'} ", style=_safe_rich_style("bright_yellow"))
    meta.append(f"phase={phase} ", style=_safe_rich_style("white"))
    meta.append(f"action={action_line} ", style=_safe_rich_style("bright_cyan"))
    if balance_text:
        meta.append(
            f"balance={balance_text} ",
            style=_safe_rich_style("bold red" if balance_text.startswith("LOW") else "bright_green"),
        )
    meta.append(f"elapsed={elapsed:0.1f}s ", style=_safe_rich_style("blue"))
    meta.append(status, style=_safe_rich_style("bold red" if rc is not None else "bold green"))
    return Panel(Group(title, meta), border_style="bright_blue")


def _make_status_panel(
    status_obj: dict[str, Any],
    agent_state: dict[str, Any],
    planner_metrics: dict[str, Any],
) -> Panel:
    table = Table(show_header=False, expand=True, box=None, pad_edge=False)
    table.add_column("k", style="cyan", no_wrap=True)
    table.add_column("v", style="white")
    if status_obj:
        for k in ("updated_at", "episode_step", "logical_step", "tentative_step", "phase", "step_id", "type", "name", "result", "error"):
            v = status_obj.get(k, "")
            if v != "":
                if k == "error":
                    table.add_row(k, _format_diagnostic_text(str(v), width=72))
                else:
                    table.add_row(k, str(v))
        args = status_obj.get("args")
        if isinstance(args, dict) and args:
            j = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
            if len(j) > 120:
                j = j[:117] + "..."
            table.add_row("args", j)
        planner_goal = str(status_obj.get("planner_goal") or "").strip()
        if planner_goal:
            table.add_row("planner_goal", planner_goal)
        planner_thoughts = str(status_obj.get("planner_thoughts") or "").strip()
        if planner_thoughts:
            table.add_row("planner_thoughts", planner_thoughts)
        planner_steps = status_obj.get("planner_steps")
        if isinstance(planner_steps, list) and planner_steps:
            s = json.dumps(planner_steps, ensure_ascii=False, separators=(",", ":"))
            table.add_row("planner_steps", s)
        ps = status_obj.get("planner_updated_at_episode_step")
        if isinstance(ps, int) and ps > 0:
            table.add_row("planner_step", str(ps))
    else:
        table.add_row("status", "waiting for current_step_status.json ...")
    if planner_metrics:
        call = str(planner_metrics.get("call") or "")
        latency = planner_metrics.get("latency_s")
        payload_bytes = planner_metrics.get("payload_bytes")
        table.add_row("planner_req", f"{call or '-'} latency={latency if latency is not None else '-'}s payload={payload_bytes if payload_bytes is not None else '-'}")
        input_tokens = planner_metrics.get("input_tokens")
        input_tokens_est = planner_metrics.get("input_tokens_est")
        output_tokens = planner_metrics.get("output_tokens")
        total_tokens = planner_metrics.get("total_tokens")
        token_text = []
        if input_tokens is not None:
            token_text.append(f"in={input_tokens}")
        elif input_tokens_est is not None:
            token_text.append(f"in~={input_tokens_est}")
        if output_tokens is not None:
            token_text.append(f"out={output_tokens}")
        if total_tokens is not None:
            token_text.append(f"total={total_tokens}")
        if token_text:
            table.add_row("planner_tok", " ".join(token_text))
        for cache_key, cache_value in _planner_cache_rows(planner_metrics):
            table.add_row(cache_key, cache_value)
        if planner_metrics.get("has_screenshot"):
            table.add_row(
                "image",
                (
                    f"fmt={planner_metrics.get('image_format') or '-'} "
                    f"src={planner_metrics.get('image_source_bytes') or '-'} "
                    f"enc={planner_metrics.get('image_encoded_bytes') or '-'} "
                    f"url={planner_metrics.get('image_data_url_bytes') or '-'}"
                ),
            )
            table.add_row(
                "",
                (
                    f"orig={_compact_value(planner_metrics.get('image_orig_size'), max_len=40)} "
                    f"final={_compact_value(planner_metrics.get('image_final_size'), max_len=40)} "
                    f"ratio={planner_metrics.get('image_compression_ratio') if planner_metrics.get('image_compression_ratio') is not None else '-'}"
                ),
            )
        if planner_metrics.get("error"):
            table.add_row("planner_err", _format_diagnostic_text(str(planner_metrics.get("error")), width=72))
    if planner_metrics:
        bal = planner_metrics.get("provider_balance_usd")
        used = planner_metrics.get("provider_used_usd")
        total = planner_metrics.get("provider_total_usd")
        step = planner_metrics.get("provider_balance_step")
        err = str(planner_metrics.get("provider_balance_error") or "").strip()
        if err:
            table.add_row("balance_err", err)
        elif bal is not None or used is not None or total is not None:
            table.add_row(
                "balance",
                f"${bal if bal is not None else '-'} / total={total if total is not None else '-'} used={used if used is not None else '-'} step={step if step is not None else '-'}",
            )
        key = str(planner_metrics.get("api_key_masked") or "").strip()
        if key:
            table.add_row("balance_key", key)
    table.add_row("agent_state", "")
    for ln in _agent_state_lines(agent_state):
        table.add_row("", ln)
    return Panel(table, title="Current Step / Agent State", border_style="bright_magenta")


def _planner_trace_lines(status_obj: dict[str, Any]) -> list[str]:
    if not isinstance(status_obj, dict) or not status_obj:
        return []
    goal = str(status_obj.get("planner_goal") or "").strip()
    thoughts = str(status_obj.get("planner_thoughts") or "").strip()
    steps = status_obj.get("planner_steps")
    if not goal and not thoughts and not (isinstance(steps, list) and steps):
        return []
    out: list[str] = ["[current_plan]"]
    ps = status_obj.get("planner_updated_at_episode_step")
    if isinstance(ps, int) and ps > 0:
        out.append(f"updated_at_ep_step={ps}")
    if goal:
        out.append(f"goal={goal}")
    if thoughts:
        out.append(f"thoughts={thoughts}")
    if isinstance(steps, list) and steps:
        out.append("steps=" + json.dumps(steps, ensure_ascii=False, separators=(",", ":")))
    return out


def _pe_active_plan_lines(pe_obj: dict[str, Any], *, max_remaining: int = 10) -> list[str]:
    if not isinstance(pe_obj, dict) or not pe_obj:
        return []
    action_list = pe_obj.get("plan")
    if not isinstance(action_list, list):
        action_list = pe_obj.get("action_list")
    if not isinstance(action_list, list):
        action_list = []
    try:
        cursor = int(pe_obj.get("cursor_index") or 0)
    except Exception:
        cursor = 0
    total_steps = int(pe_obj.get("total_steps") or len(action_list) or 0)
    cursor = max(0, min(cursor, max(0, total_steps)))
    current_step_id = ""
    if 0 <= cursor < len(action_list) and isinstance(action_list[cursor], dict):
        current_step_id = str(action_list[cursor].get("step_id") or "")
    out: list[str] = ["[pe_active_plan]"]
    out.append(f"cursor={cursor}/{total_steps} current_step_id={current_step_id or '-'}")
    out.append("remaining_queue:")
    remaining = action_list[cursor:] if cursor < len(action_list) else []
    if not remaining:
        out.append("(empty)")
        return out
    for i, row in enumerate(remaining[: max(1, int(max_remaining))], start=cursor):
        if not isinstance(row, dict):
            continue
        sid = str(row.get("step_id") or "-")
        typ = str(row.get("type") or "")
        name = str(row.get("name") or "")
        args = row.get("args") if isinstance(row.get("args"), dict) else {}
        row_status = str(row.get("status") or "pending")
        mark = "->" if i == cursor else "  "
        preview = _compact_args(args, max_items=2)
        suffix = f" {preview}" if preview else ""
        out.append(f"{mark} [{i}] {sid} {typ}:{name} status={row_status}{suffix}")
    if len(remaining) > max_remaining:
        out.append(f"... (+{len(remaining) - max_remaining} more)")
    return out


def _make_trace_panel(
    *,
    trace_lines: list[str],
    status_obj: dict[str, Any],
    pe_active_plan_obj: dict[str, Any],
    step_floor: int,
    keep: int,
    width: int,
    height: int,
) -> Panel:
    trc = _distill_trace(trace_lines, step_floor=step_floor, keep=keep)
    buf: list[str] = ["[step_trace]"]
    plan_block = _planner_trace_lines(status_obj)
    if plan_block:
        buf.extend(plan_block)
        buf.append("")
    pe_block = _pe_active_plan_lines(pe_active_plan_obj)
    if pe_block:
        buf.extend(pe_block)
        buf.append("")
    if trc:
        buf.extend(trc)
    else:
        buf.append("(no step_trace yet)")
    body = _tail_wrapped_lines(buf, width=width, height=height)
    return Panel(Text(body, overflow="fold", no_wrap=False), title="Step Trace", border_style="bright_yellow")


def _make_runner_logs_panel(lines: list[str], *, keep: int, width: int, height: int) -> Panel:
    level_style = {
        "TRACE": "bright_black",
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold red",
    }
    out_lines: list[str] = []
    for raw in lines[-max(1, int(keep)) :]:
        plain = _strip_ansi(raw).rstrip()
        if not plain:
            continue
        m = _LOG_LINE_RE.match(plain)
        if m:
            ts = str(m.group("ts") or "")
            level = str(m.group("level") or "INFO").strip().upper()
            msg = _clean_runner_message(str(m.group("msg") or ""))
            out_lines.append(f"{ts} | {level} | {msg}")
        else:
            out_lines.append(_clean_runner_message(plain))

    body = _tail_wrapped_lines(out_lines, width=width, height=height)
    # Re-apply level colors line-by-line (simple pass).
    txt = Text()
    for ln in body.splitlines():
        m = re.match(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (?P<level>[A-Z]+) \| (?P<msg>.*)$", ln)
        if m:
            level = str(m.group("level") or "INFO")
            txt.append(str(m.group("ts")), style=_safe_rich_style("green"))
            txt.append(" | ", style=_safe_rich_style("bright_black"))
            txt.append(level, style=_safe_rich_style(level_style.get(level, "white")))
            txt.append(" | ", style=_safe_rich_style("bright_black"))
            txt.append(str(m.group("msg")), style=_safe_rich_style("white"))
            txt.append("\n")
        else:
            txt.append(ln + "\n", style=_safe_rich_style("white"))
    if not txt.plain:
        txt = Text("(empty)")
    return Panel(txt, title="Runner Logs", border_style="bright_green")


def _make_error_panel(
    error_lines: list[str],
    *,
    status_obj: dict[str, Any],
    keep: int,
    width: int,
    height: int,
) -> Panel:
    out_lines: list[str] = []
    status_error = str((status_obj or {}).get("error") or "").strip()
    if status_error:
        phase = str((status_obj or {}).get("phase") or "").strip() or "-"
        step_id = str((status_obj or {}).get("step_id") or "").strip() or "-"
        name = str((status_obj or {}).get("name") or "").strip() or "-"
        out_lines.append(f"[current] phase={phase} step_id={step_id} name={name} error={status_error}")
    for raw in error_lines[-max(1, int(keep)) :]:
        plain = _strip_ansi(raw).rstrip()
        if plain:
            out_lines.append(plain)
    if not out_lines:
        out_lines.append("(no recent errors)")
    body = _tail_wrapped_lines(out_lines, width=width, height=height)
    return Panel(Text(body, overflow="fold", no_wrap=False, style="red"), title="Recent Errors", border_style="red")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="epm/configs/pipelines/planner_executor.json")
    parser.add_argument("--dish-id", type=int, default=1)
    parser.add_argument("--dish-range", default="", help="Dish ID union, e.g. 1-5,8,10-12")
    parser.add_argument("--max-dish-id", type=int, default=None, help="Used when --dish-range is empty.")
    parser.add_argument(
        "--auto-chain",
        action="store_true",
        help="Auto-run the next dish after success (enabled automatically for multi-dish sequences).",
    )
    parser.add_argument("--planner-model", default="", help="Override planner model for this dashboard run.")
    parser.add_argument(
        "--ablation-reduce",
        default="",
        metavar="NO_COMPONENT[,NO_COMPONENT...]",
        help="Comma-separated reductions forwarded to run_episode.py, e.g. no_body,no_strategy.",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--refresh-memory", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart-env", "--restart_env", "--restartenv", dest="restart_env", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--log-level", default="DEBUG", help="Runner log level (default: DEBUG).")
    parser.add_argument("--alt-screen", action="store_true", help="Use alternate screen buffer (clears dashboard after exit).")
    args, unknown = parser.parse_known_args()
    unknown = _validate_passthrough_unknown_args(parser, unknown)

    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)

    runs_root = repo / "epm" / "runs"
    source_config_path = str(Path(args.config))
    planner_model_override = str(args.planner_model or "").strip()
    try:
        effective_config_path = _materialize_config_override(repo, source_config_path, planner_model_override)
    except Exception as e:
        print(f"[dashboard] failed to prepare config override: {e}")
        return 1
    effective_pipeline = _load_pipeline_from_config(effective_config_path)
    effective_model = planner_model_override or _load_planner_model_from_config(effective_config_path)
    effective_raw_config = load_raw_settings(effective_config_path)
    effective_reduction = resolve_ablation_reductions(args.ablation_reduce)
    effective_ablation_reduce = ",".join(effective_reduction.reductions)
    effective_prompt_ablation_label = "-".join(effective_reduction.reductions) or "full"

    try:
        if args.dish_range:
            dish_sequence = _parse_dish_range(args.dish_range)
        else:
            if args.max_dish_id is not None:
                max_dish_id = args.max_dish_id
            elif args.auto_chain:
                dish_sequence = [dish_id for dish_id in current_benchmark_dish_ids() if dish_id >= int(args.dish_id)]
                max_dish_id = None
            else:
                max_dish_id = args.dish_id
            if max_dish_id is not None:
                if max_dish_id < args.dish_id:
                    raise ValueError("--max-dish-id must be >= --dish-id")
                dish_sequence = list(range(int(args.dish_id), int(max_dish_id) + 1))
    except ValueError as e:
        parser.error(str(e))
    if not dish_sequence:
        parser.error("dish sequence is empty")
    effective_dish_range_spec = str(args.dish_range or "").strip() or _format_dish_range(dish_sequence)
    if not args.resume:
        dish_sequence = _trim_dish_sequence_from(int(args.dish_id), dish_sequence)
        if not dish_sequence:
            parser.error("dish sequence is empty after applying --dish-id")

    requested_run_name = (args.run_name or "").strip()
    pass_run_name = True
    if requested_run_name:
        run_name = requested_run_name
    elif args.resume:
        run_name = _find_latest_run_name(runs_root)
        pass_run_name = False
    else:
        run_name = _default_run_name(
            dish_sequence[0],
            pipeline=effective_pipeline,
            model=effective_model,
            prompt_ablation_profile=effective_prompt_ablation_label,
        )
    if not run_name:
        print("[dashboard] no existing run found to resume")
        return 1

    effective_dish_id = int(dish_sequence[0])
    parsed_dish_id = _parse_dish_id_from_run_name(run_name)
    if args.resume and parsed_dish_id is not None:
        effective_dish_id = int(parsed_dish_id)
    if args.resume:
        dish_sequence = _trim_dish_sequence_from(int(effective_dish_id), dish_sequence)

    if len(dish_sequence) > 1 and not args.auto_chain:
        args.auto_chain = True
        print(f"[dashboard] auto_chain=enabled_for_multi_dish_sequence count={len(dish_sequence)}")

    run_root = runs_root / run_name
    memory_dir = run_root / "memory"
    expected_dish_name = _load_expected_dish_name(effective_config_path, effective_dish_id)
    status_path = memory_dir / "current_step_status.json"
    trace_path = memory_dir / "step_trace.log"
    history_path = memory_dir / "long_horizon_history.txt"
    agent_state_path = memory_dir / "agent_state.json"
    legacy_agent_state_path = repo / "epm" / "memory" / "agent_state.json"
    pe_active_plan_path = memory_dir / "pe_active_plan.json"
    planner_metrics_path = memory_dir / "planner_request_metrics.jsonl"
    error_events_path = memory_dir / "error_events.log"
    resume_state_path = memory_dir / "resume_state.json"
    resume_state_before = _read_json(resume_state_path)
    session_step_floor = 0
    session_step_floor_source = "none"
    try:
        if args.resume:
            session_step_floor, session_step_floor_source = _resolve_effective_resume_step(resume_state_before)
    except Exception:
        session_step_floor = 0
        session_step_floor_source = "none"

    if args.resume:
        detected_resume_step_id = int(session_step_floor)
        target_resume_step_id = int(detected_resume_step_id)
        resume_step_override = _dashboard_unknown_arg_value(unknown, "--resume-step-id")
        rollback_steps_raw = _dashboard_unknown_arg_value(unknown, "--rollback-steps")
        yes_rewind = _dashboard_unknown_arg_present(unknown, "--yes-rewind")
        if resume_step_override is not None:
            try:
                target_resume_step_id = int(resume_step_override)
                session_step_floor_source = f"cli.resume_step_id({int(target_resume_step_id)})"
            except Exception:
                print(f"[dashboard] invalid --resume-step-id: {resume_step_override!r}")
                return 1
        elif rollback_steps_raw is not None:
            try:
                target_resume_step_id = max(0, int(detected_resume_step_id) - int(rollback_steps_raw))
                session_step_floor_source = f"cli.rollback_steps({int(rollback_steps_raw)})"
            except Exception:
                print(f"[dashboard] invalid --rollback-steps: {rollback_steps_raw!r}")
                return 1
        if target_resume_step_id < 0:
            print(f"[dashboard] invalid target resume step: {target_resume_step_id}")
            return 1
        if target_resume_step_id > detected_resume_step_id:
            print(
                f"[dashboard] resume_step_id_out_of_range "
                f"requested={int(target_resume_step_id)} detected={int(detected_resume_step_id)}"
            )
            return 1
        if (
            int(target_resume_step_id) < int(detected_resume_step_id)
            and not yes_rewind
            and not _confirm_dashboard_manual_rewind(
                memory_dir=memory_dir,
                detected_resume_step_id=int(detected_resume_step_id),
                target_resume_step_id=int(target_resume_step_id),
            )
        ):
            print(f"[dashboard] resume_rewind=cancelled target_step={int(target_resume_step_id)}")
            return 1
        session_step_floor = int(target_resume_step_id)
        if int(target_resume_step_id) < int(detected_resume_step_id) and not yes_rewind:
            unknown = list(unknown) + ["--yes-rewind"]

    batch_state_path = _batch_state_path(
        runs_root=runs_root,
        config_path=source_config_path,
        pipeline=effective_pipeline,
        model=effective_model,
        range_spec=effective_dish_range_spec,
    )
    batch_state_before = _read_batch_state(batch_state_path)
    completed_ids_before = []
    if isinstance(batch_state_before.get("completed_ids"), list):
        for x in batch_state_before.get("completed_ids") or []:
            try:
                completed_ids_before.append(int(x))
            except Exception:
                continue
    next_dish_before = _next_dish_in_sequence(effective_dish_id, dish_sequence)
    _write_batch_state(
        batch_state_path,
        source_config_path=source_config_path,
        pipeline=effective_pipeline,
        planner_model=effective_model,
        range_spec=effective_dish_range_spec,
        dish_sequence=dish_sequence,
        completed_ids=completed_ids_before,
        current_dish_id=effective_dish_id,
        next_dish_id=next_dish_before,
        current_run_name=run_name,
        last_completed_run_name=str(batch_state_before.get("last_completed_run_name") or ""),
        status="running",
        run_status="running",
        run_reason="",
    )

    cmd: list[str] = [
        sys.executable,
        "epm/scripts/run_episode.py",
        "--config",
        str(effective_config_path),
        "--dish-id",
        str(effective_dish_id),
    ]
    if run_name:
        cmd += ["--run-name", run_name]
    if args.steps is not None:
        cmd += ["--steps", str(args.steps)]
    if args.refresh_memory:
        cmd.append("--refresh-memory")
    if args.resume:
        cmd.append("--resume")
    if args.restart_env:
        cmd.append("--restart_env")
    if args.no_color:
        cmd.append("--no-color")
    if effective_ablation_reduce:
        cmd += ["--ablation-reduce", effective_ablation_reduce]
    cmd.extend(unknown)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["LOGURU_LEVEL"] = str(args.log_level or "INFO").strip().upper()
    env["EPM_LOG_LEVEL"] = str(args.log_level or "INFO").strip().upper()
    stream_encoding = "utf-8"

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding=stream_encoding,
        errors="replace",
        env=env,
        bufsize=1,
    )

    logs: list[str] = []
    line_counter: dict[str, int] = {"n": 0}
    reader = threading.Thread(target=_drain_process_output, args=(proc, logs, line_counter), daemon=True)
    reader.start()

    trace_reader = _TailReader(trace_path, max_lines=140)
    error_reader = _TailReader(error_events_path, max_lines=120)
    started = time.time()

    layout = Layout()
    layout.split_column(Layout(name="header", size=4), Layout(name="main", ratio=1))
    # Make Runner Logs wider and give Step Trace more vertical space.
    layout["main"].split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=4))
    layout["left"].split_column(Layout(name="status", size=12), Layout(name="trace", ratio=1))
    layout["right"].split_column(Layout(name="errors", size=10), Layout(name="logs", ratio=1))

    last_sig: tuple[Any, ...] | None = None
    try:
        with Live(layout, refresh_per_second=2, screen=bool(args.alt_screen), transient=False, auto_refresh=False) as live:
            while True:
                rc = proc.poll()
                status_obj = _read_json(status_path)
                agent_state_obj = _read_json(agent_state_path) or _read_json(legacy_agent_state_path)
                pe_active_plan_obj = _read_json(pe_active_plan_path)
                planner_metrics_obj = _read_last_jsonl_obj(planner_metrics_path)
                trace_lines = trace_reader.poll()
                error_lines = error_reader.poll()
                log_lines = list(logs)
                # Adapt tail sizes to current terminal dimensions.
                term = shutil.get_terminal_size(fallback=(180, 52))
                log_keep = max(28, int(term.lines * 1.2))
                trace_keep = max(14, int(term.lines * 0.9))
                right_width = max(60, int(term.columns * 0.62) - 6)
                left_width = max(44, term.columns - right_width - 12)
                main_height = max(16, term.lines - 8)
                status_height = 12
                trace_height = max(8, main_height - status_height - 2)
                current = _derive_current_state(
                    status_obj=status_obj,
                    trace_lines=trace_lines,
                    history_lines=[],
                    logs=log_lines,
                )

                sig = (
                    rc,
                    status_obj.get("updated_at"),
                    current.get("step"),
                    current.get("plan_step_id"),
                    current.get("phase"),
                    trace_lines[-1] if trace_lines else "",
                    int(line_counter.get("n", 0)),
                    agent_state_obj.get("mode"),
                    agent_state_obj.get("is_held"),
                    pe_active_plan_obj.get("updated_at"),
                    pe_active_plan_obj.get("cursor_index"),
                    pe_active_plan_obj.get("current_step_id"),
                    planner_metrics_obj.get("ts"),
                    planner_metrics_obj.get("attempt"),
                    planner_metrics_obj.get("latency_s"),
                    planner_metrics_obj.get("payload_bytes"),
                    planner_metrics_obj.get("error"),
                    planner_metrics_obj.get("provider_balance_step"),
                    planner_metrics_obj.get("provider_balance_usd"),
                    planner_metrics_obj.get("provider_balance_error"),
                    error_lines[-1] if error_lines else "",
                    status_obj.get("error"),
                    int((time.time() - started) * 2),
                )
                balance_text = ""
                bal = planner_metrics_obj.get("provider_balance_usd")
                if bal is not None:
                    try:
                        bal_f = float(bal)
                        balance_text = f"${bal_f:.2f}"
                        if bal_f < 10.0:
                            balance_text = f"LOW ${bal_f:.2f}"
                    except Exception:
                        balance_text = f"${bal}"
                elif str(planner_metrics_obj.get("provider_balance_error") or "").strip():
                    balance_text = "ERR"
                if sig != last_sig:
                    layout["header"].update(
                        _make_header(
                            run_name=run_name,
                            dish_id=effective_dish_id,
                            step=current.get("step"),
                            plan_step_id=str(current.get("plan_step_id") or "-"),
                            phase=current.get("phase"),
                            action_line=str(current.get("action_line") or "-"),
                            started=started,
                            rc=rc,
                            balance_text=balance_text,
                        )
                    )
                    layout["status"].update(
                        _make_status_panel(
                            status_obj=status_obj,
                            agent_state=agent_state_obj,
                            planner_metrics=planner_metrics_obj,
                        )
                    )
                    layout["trace"].update(
                        _make_trace_panel(
                            trace_lines=trace_lines,
                            status_obj=status_obj,
                            pe_active_plan_obj=pe_active_plan_obj,
                            step_floor=int(session_step_floor),
                            keep=trace_keep,
                            width=left_width,
                            height=trace_height,
                        )
                    )
                    error_height = 10
                    logs_height = max(6, main_height - error_height - 1)
                    layout["errors"].update(
                        _make_error_panel(
                            error_lines,
                            status_obj=status_obj,
                            keep=max(6, min(20, log_keep // 3)),
                            width=right_width,
                            height=error_height,
                        )
                    )
                    layout["logs"].update(
                        _make_runner_logs_panel(
                            log_lines,
                            keep=log_keep,
                            width=right_width,
                            height=logs_height,
                        )
                    )
                    last_sig = sig
                    live.refresh()

                if rc is not None:
                    live.refresh()
                    break
                time.sleep(0.2)
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        _ensure_run_time_summary(repo=repo, run_name=run_name)
        return 130

    rc = int(proc.returncode or 0)
    _ensure_run_time_summary(repo=repo, run_name=run_name)
    print("")
    print(f"[dashboard] run finished rc={rc}")
    native_exit = _decode_windows_exit_status(rc)
    if native_exit:
        print("[dashboard] native_exit_detail:")
        print(f"[dashboard]   hex: {native_exit.get('hex')}")
        print(f"[dashboard]   name: {native_exit.get('name')}")
        print(f"[dashboard]   diagnosis: {native_exit.get('diagnosis')}")
        print(f"[dashboard]   probable_causes: {native_exit.get('probable_causes')}")
        print(f"[dashboard]   suggested_action: {native_exit.get('suggested_action')}")
        print("[dashboard]   note: child process likely exited before resume_state/error_events could be fully updated; existing run_reason may be stale")
    print(f"[dashboard] run_root={run_root}")
    print(f"[dashboard] trace={trace_path}")
    print(f"[dashboard] history={history_path}")
    print(f"[dashboard] agent_state={agent_state_path}")
    print(f"[dashboard] error_events={error_events_path}")

    print(f"[dashboard] batch_state={batch_state_path}")
    if args.resume:
        print(f"[dashboard] resume_step_id={int(session_step_floor)}")
        print(f"[dashboard] resume_step_source={session_step_floor_source}")

    resume_state_after = _read_json(resume_state_path)
    run_status = str(resume_state_after.get("status") or "").strip().lower()
    run_reason = str(resume_state_after.get("reason") or "").strip()
    if run_status:
        print(f"[dashboard] run_status={run_status}")
    if run_reason:
        print(f"[dashboard] run_reason={run_reason}")
        pretty_run_reason = _format_diagnostic_text(run_reason, width=96)
        if pretty_run_reason and pretty_run_reason != run_reason:
            print("[dashboard] run_reason_detail:")
            for line in pretty_run_reason.splitlines():
                print(f"[dashboard]   {line}")
    if native_exit:
        tail_lines = [str(line) for line in list(logs)[-12:] if str(line).strip()]
        if tail_lines:
            print("[dashboard] runner_log_tail:")
            for line in tail_lines:
                pretty_line = _format_diagnostic_text(line, width=120)
                for wrapped in (pretty_line.splitlines() if pretty_line else [line]):
                    print(f"[dashboard]   {wrapped}")

    resume_command = _build_resume_command(
        source_config_path=source_config_path,
        effective_dish_range_spec=effective_dish_range_spec,
        run_name=run_name,
        auto_chain=bool(args.auto_chain),
        planner_model=planner_model_override,
        ablation_reduce=effective_ablation_reduce,
        no_color=bool(args.no_color),
        log_level=str(args.log_level or ""),
        unknown=unknown,
    )
    interrupted_statuses = {
        "stopped",
        "fatal",
        "max_steps",
        "steps_limit",
        "running",
        "waiting_network",
        "waiting_http_403",
        "resume_rewind_cleaned",
    }
    interrupted_run = (rc != 0 or run_status in interrupted_statuses)
    feedback_file: Path | None = None
    feedback_files: list[Path] = []
    feedback_mismatch_names: list[str] = []
    if interrupted_run:
        feedback_file, feedback_files, feedback_mismatch_names = _find_matching_recipe_feedback(
            run_root=run_root,
            expected_dish_name=expected_dish_name,
        )
        if feedback_file is not None:
            print(f"[dashboard] feedback_check=weak_ok path={feedback_file}")
            print("[dashboard] weak_check_reason_cn=本轮未正常收尾，但已检测到匹配当前菜名的 recipe feedback 文件，按已完成提交处理")
            print("[dashboard] weak_check_reason_en=Run ended abnormally, but a matching archived recipe feedback file exists; treating submission as completed")
        elif feedback_files:
            feedback_file, feedback_files, feedback_mismatch_names, waited_s = _poll_matching_recipe_feedback_on_mismatch(
                run_root=run_root,
                expected_dish_name=expected_dish_name,
            )
            if feedback_file is not None:
                print(f"[dashboard] feedback_check=weak_ok path={feedback_file}")
                print(f"[dashboard] feedback_wait_seconds={waited_s}")
                print("[dashboard] weak_check_reason_cn=本轮未正常收尾，但轮询后检测到匹配当前菜名的 recipe feedback 文件，按已完成提交处理")
                print("[dashboard] weak_check_reason_en=Run ended abnormally, but a matching archived recipe feedback file appeared during polling; treating submission as completed")
            else:
                mismatch_preview = ", ".join(feedback_mismatch_names[:3]) if feedback_mismatch_names else "(unknown)"
                print("[dashboard] feedback_check=mismatch")
                print(f"[dashboard] feedback_wait_seconds={waited_s}")
                print(
                    f"[dashboard] error=feedback_dish_mismatch expected={expected_dish_name!r} "
                    f"seen={mismatch_preview!r}"
                )
        else:
            _write_batch_state(
                batch_state_path,
                source_config_path=source_config_path,
                pipeline=effective_pipeline,
                planner_model=effective_model,
                range_spec=effective_dish_range_spec,
                dish_sequence=dish_sequence,
                completed_ids=completed_ids_before,
                current_dish_id=effective_dish_id,
                next_dish_id=_next_dish_in_sequence(effective_dish_id, dish_sequence),
                current_run_name=run_name,
                last_completed_run_name=str(batch_state_before.get("last_completed_run_name") or ""),
                status="interrupted",
                run_status=run_status or f"rc_{rc}",
                run_reason=run_reason,
            )
            print("[dashboard] feedback_check=skipped")
            print("[dashboard] skip_reason_cn=\u672c\u8f6e\u5df2\u4e2d\u65ad\u6216\u672a\u6b63\u5e38\u5b8c\u6210\uff0c\u4e14\u672a\u68c0\u6d4b\u5230 archived feedback\uff0c\u8df3\u8fc7 feedback \u68c0\u67e5")
            print("[dashboard] skip_reason_en=Run interrupted or not completed normally, and no archived feedback was found; skipping feedback check")
            print(f"[dashboard] resume_command={resume_command}")
            return rc

    if feedback_file is None:
        feedback_file, feedback_files, feedback_mismatch_names = _find_matching_recipe_feedback(
            run_root=run_root,
            expected_dish_name=expected_dish_name,
        )

    if interrupted_run and run_status == "stopped":
        print("[dashboard] auto_chain_guard=manual_stop_detected")
        print("[dashboard] auto_chain_guard_cn=检测到手动停止；即使 feedback 已存在，也不会自动进入下一道菜")
        print("[dashboard] auto_chain_guard_en=Manual stop detected; auto-chain will not continue even though feedback exists")
    feedback_dump_command = _build_feedback_dump_command(
        config_path=source_config_path,
        run_root=run_root,
    )

    if feedback_file is None:
        if feedback_files:
            feedback_file, feedback_files, feedback_mismatch_names, waited_s = _poll_matching_recipe_feedback_on_mismatch(
                run_root=run_root,
                expected_dish_name=expected_dish_name,
            )
            if feedback_file is not None:
                print(f"[dashboard] feedback_status=ok path={feedback_file}")
                print(f"[dashboard] feedback_wait_seconds={waited_s}")
                next_dish_id = _next_dish_in_sequence(effective_dish_id, dish_sequence)
                completed_ids_after = _merge_completed_ids(completed_ids_before, effective_dish_id)
                _write_batch_state(
                    batch_state_path,
                    source_config_path=source_config_path,
                    pipeline=effective_pipeline,
                    planner_model=effective_model,
                    range_spec=effective_dish_range_spec,
                    dish_sequence=dish_sequence,
                    completed_ids=completed_ids_after,
                    current_dish_id=effective_dish_id,
                    next_dish_id=next_dish_id,
                    current_run_name="",
                    last_completed_run_name=run_name,
                    status=("completed" if next_dish_id is not None else "finished"),
                    run_status="feedback_ok",
                    run_reason="",
                )
                if bool(args.auto_chain):
                    if interrupted_run and run_status == "stopped":
                        print("[dashboard] auto_chain=blocked_manual_stop")
                    elif next_dish_id is not None:
                        continue_command = _build_next_dish_command(
                            source_config_path=source_config_path,
                            next_dish_id=next_dish_id,
                            effective_dish_range_spec=effective_dish_range_spec,
                            auto_chain=bool(args.auto_chain),
                            planner_model=planner_model_override,
                            ablation_reduce=effective_ablation_reduce,
                            steps=args.steps,
                            no_color=bool(args.no_color),
                            alt_screen=bool(args.alt_screen),
                            log_level=str(args.log_level or ""),
                            unknown=unknown,
                        )
                        print(f"[dashboard] auto_chain_command={continue_command}")
                return 0
        next_dish_id = _next_dish_in_sequence(effective_dish_id, dish_sequence)
        _write_batch_state(
            batch_state_path,
            source_config_path=source_config_path,
            pipeline=effective_pipeline,
            planner_model=effective_model,
            range_spec=effective_dish_range_spec,
            dish_sequence=dish_sequence,
            completed_ids=completed_ids_before,
            current_dish_id=effective_dish_id,
            next_dish_id=next_dish_id,
            current_run_name=run_name,
            last_completed_run_name=str(batch_state_before.get("last_completed_run_name") or ""),
            status=("feedback_mismatch" if feedback_files else "awaiting_feedback"),
            run_status=run_status or "done",
            run_reason=run_reason,
        )
        if feedback_files:
            mismatch_preview = ", ".join(feedback_mismatch_names[:3]) if feedback_mismatch_names else "(unknown)"
            print("[dashboard] feedback_status=mismatch")
            if 'waited_s' in locals():
                print(f"[dashboard] feedback_wait_seconds={waited_s}")
            print(
                f"[dashboard] error=feedback_dish_mismatch expected={expected_dish_name!r} "
                f"seen={mismatch_preview!r}"
            )
        else:
            print("[dashboard] feedback_status=missing")
            print("[dashboard] error=feedback_missing_after_run")
        if next_dish_id is not None:
            continue_command = _build_next_dish_command(
                source_config_path=source_config_path,
                next_dish_id=next_dish_id,
                effective_dish_range_spec=effective_dish_range_spec,
                auto_chain=bool(args.auto_chain),
                planner_model=planner_model_override,
                ablation_reduce=effective_ablation_reduce,
                steps=args.steps,
                no_color=bool(args.no_color),
                alt_screen=bool(args.alt_screen),
                log_level=str(args.log_level or ""),
                unknown=unknown,
            )
            print(f"[dashboard] continue_after_feedback_command={continue_command}")
        if feedback_files:
            print("[dashboard] reminder_cn=已检测到 archived feedback，但菜品名称与当前目标不匹配；只要存在任意一个匹配当前菜名的 feedback 文件才算通过")
            print("[dashboard] reminder_en=Archived feedback exists, but the dish name does not match the current target; only a matching feedback file counts as success")
        else:
            print("[dashboard] reminder_cn=\u672a\u68c0\u6d4b\u5230 feedback \u6587\u4ef6\uff0c\u8bf7\u5148\u624b\u52a8\u5bfc\u51fa feedback\uff1b\u82e5\u8be5\u8f6e\u5df2\u6b63\u5e38\u5b8c\u6210\uff0c\u518d\u8fd0\u884c continue_after_feedback_command \u8fdb\u5165\u4e0b\u4e00\u9053\u83dc")
            print("[dashboard] reminder_en=No archived feedback file was found; dump feedback first, then use continue_after_feedback_command to move to the next dish if this run finished normally")
        print(f"[dashboard] manual_feedback_command={feedback_dump_command}")
        return 2

    print(f"[dashboard] feedback_status=ok path={feedback_file}")
    next_dish_id = _next_dish_in_sequence(effective_dish_id, dish_sequence)
    completed_ids_after = _merge_completed_ids(completed_ids_before, effective_dish_id)
    _write_batch_state(
        batch_state_path,
        source_config_path=source_config_path,
        pipeline=effective_pipeline,
        planner_model=effective_model,
        range_spec=effective_dish_range_spec,
        dish_sequence=dish_sequence,
        completed_ids=completed_ids_after,
        current_dish_id=effective_dish_id,
        next_dish_id=next_dish_id,
        current_run_name=run_name,
        last_completed_run_name=run_name,
        status="completed" if next_dish_id is not None else "sequence_completed",
        run_status=run_status or "done",
        run_reason=run_reason,
    )
    if next_dish_id is None:
        print(f"[dashboard] dish_sequence_completed={effective_dish_range_spec}")
        return rc

    next_display_cmd: list[str] = [
        sys.executable,
        "epm/scripts/run_episode_dashboard.py",
        "--config",
        str(source_config_path),
        "--dish-id",
        str(next_dish_id),
        "--restart_env",
    ]
    next_exec_argv: list[str] = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(source_config_path),
        "--dish-id",
        str(next_dish_id),
        "--restart_env",
    ]
    if effective_dish_range_spec:
        next_display_cmd += ["--dish-range", effective_dish_range_spec]
        next_exec_argv += ["--dish-range", effective_dish_range_spec]
    if args.auto_chain:
        next_display_cmd.append("--auto-chain")
        next_exec_argv.append("--auto-chain")
    if effective_ablation_reduce:
        # Keep the same reduction when os.execv starts the next dish. Without
        # this, only the first dish receives --ablation-reduce and later dishes
        # silently fall back to the full prompt.
        next_display_cmd += ["--ablation-reduce", effective_ablation_reduce]
        next_exec_argv += ["--ablation-reduce", effective_ablation_reduce]
    if planner_model_override:
        next_display_cmd += ["--planner-model", planner_model_override]
        next_exec_argv += ["--planner-model", planner_model_override]
    if args.max_dish_id is not None:
        next_display_cmd += ["--max-dish-id", str(args.max_dish_id)]
        next_exec_argv += ["--max-dish-id", str(args.max_dish_id)]
    if args.steps is not None:
        next_display_cmd += ["--steps", str(args.steps)]
        next_exec_argv += ["--steps", str(args.steps)]
    if args.no_color:
        next_display_cmd.append("--no-color")
        next_exec_argv.append("--no-color")
    if args.alt_screen:
        next_display_cmd.append("--alt-screen")
        next_exec_argv.append("--alt-screen")
    if args.log_level:
        next_display_cmd += ["--log-level", str(args.log_level)]
        next_exec_argv += ["--log-level", str(args.log_level)]
    next_display_cmd.extend(unknown)
    next_exec_argv.extend(unknown)

    recommended_next_command = _quote_cmd(next_display_cmd)
    print(f"[dashboard] recommended_next_command={recommended_next_command}")

    if args.auto_chain and not (interrupted_run and run_status == "stopped"):
        print(f"[dashboard] auto_chain_next_dish={next_dish_id}")
        print("[dashboard] auto_chain_delay_s=3")
        time.sleep(3.0)
        os.execv(sys.executable, next_exec_argv)
        return rc

    print(f"[dashboard] next_dish={next_dish_id}")
    print("[dashboard] reminder_cn=\u5df2\u751f\u6210 feedback \u6587\u4ef6\uff0c\u5982\u9700\u7ee7\u7eed\u53ef\u76f4\u63a5\u8fd0\u884c\u4e0a\u9762\u7684 recommended_next_command")
    print("[dashboard] reminder_en=Feedback archived; run the recommended_next_command above to continue")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())

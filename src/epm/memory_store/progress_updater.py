from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class NeedReplan(RuntimeError):
    """
    Raised when the executor cannot align execution records to the current plan.

    Typical triggers:
    - missing plan_ref in long_horizon_history.jsonl
    - plan_ref not found in task_progress file
    - malformed task_progress format (missing step_id/checked fields)
    - repeated failures / loop detected (policy choice)
    """


@dataclass(frozen=True)
class PlanRef:
    high_level_id: str
    atomic_step_id: str


def read_last_jsonl_record(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    lines = p.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return json.loads(line)
    raise ValueError(f"No JSONL records in {p}")


_HIGH_LEVEL_ID_RE = re.compile(r'^\s*-\s*id:\s*"?(?P<id>[^"]+)"?\s*$')
_STATUS_RE = re.compile(r'^\s*status:\s*(?P<status>todo|doing|done|blocked)\s*$')

# step line is single-line by design:
# - step_id: "H1.A1" action: "goto" args: {...} checked: false last_result: "" last_error: ""
_STEP_LINE_RE = re.compile(
    r'^(?P<indent>\s*)-\s*step_id:\s*"?(?P<step_id>[^"]+)"?\s+action:\s*"(?P<action>[^"]+)"\s+args:\s*(?P<args>\{.*?\})\s+checked:\s*(?P<checked>true|false)\s*(?P<tail>.*)$'
)
_PLAN_STATUS_RE = re.compile(r'^\s*plan_status:\s*"(?:[^"]*)"\s*$')
_STEP_CURSOR_RE = re.compile(r'^\s*step_cursor:\s*\d+\s*$')


def _find_block(lines: list[str], *, header: str) -> tuple[Optional[int], Optional[int]]:
    start = None
    end = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            break
        if line.strip().startswith(header + " "):
            start = i
            end = i + 1
            return start, end
    if start is None:
        return None, None
    for j in range(start + 1, len(lines)):
        if lines[j].strip() and not lines[j].startswith("  "):
            end = j
            break
    if end is None:
        end = len(lines)
    return start, end


def _parse_plan_ref(record: Dict[str, Any]) -> PlanRef:
    plan_ref = record.get("plan_ref")
    if not isinstance(plan_ref, dict):
        raise NeedReplan("missing plan_ref in execution record")
    high_level_id = plan_ref.get("high_level_id")
    atomic_step_id = plan_ref.get("atomic_step_id")
    if not isinstance(high_level_id, str) or not isinstance(atomic_step_id, str):
        raise NeedReplan("invalid plan_ref fields")
    return PlanRef(high_level_id=high_level_id, atomic_step_id=atomic_step_id)


def apply_execution_result_to_task_progress(
    *,
    task_progress_path: str | Path,
    plan_ref: PlanRef,
    result_summary: str,
    error: str,
) -> None:
    """
    Mark the referenced atomic step as checked (success) or update last_error (failure).

    This function performs minimal, line-based updates to keep LLM-authored content intact.
    """
    path = Path(task_progress_path)
    if not path.exists():
        raise FileNotFoundError(path)

    lines = path.read_text(encoding="utf-8").splitlines(keepends=False)

    # 1) Find the target high-level block range [start, end)
    high_start = None
    high_end = None
    for i, line in enumerate(lines):
        m = _HIGH_LEVEL_ID_RE.match(line)
        if m and m.group("id") == plan_ref.high_level_id:
            high_start = i
            continue
        if high_start is not None and m and m.group("id") != plan_ref.high_level_id:
            high_end = i
            break
    if high_start is None:
        raise NeedReplan(f"high_level_id not found: {plan_ref.high_level_id}")
    if high_end is None:
        high_end = len(lines)

    # 2) Update the referenced step line
    target_idx = None
    for i in range(high_start, high_end):
        m = _STEP_LINE_RE.match(lines[i])
        if m and m.group("step_id") == plan_ref.atomic_step_id:
            target_idx = i
            break
    if target_idx is None:
        raise NeedReplan(f"atomic_step_id not found: {plan_ref.atomic_step_id}")

    current = lines[target_idx]
    m = _STEP_LINE_RE.match(current)
    if not m:
        raise NeedReplan("malformed step line (expected step_id/action/args/checked on same line)")

    checked = "true" if str(result_summary).lower() == "success" else "false"
    # Update checked + last_result + last_error in tail (append if absent)
    tail = m.group("tail") or ""

    def _set_kv(text: str, key: str, value: str) -> str:
        pattern = re.compile(rf'(\b{re.escape(key)}:\s*)(\"[^\"]*\"|[^ \t]+)')
        if pattern.search(text):
            return pattern.sub(rf"\1{value}", text, count=1)
        sep = "" if text.strip() == "" else " "
        return f"{text}{sep}{key}: {value}"

    tail = _set_kv(tail, "last_result", json.dumps(result_summary, ensure_ascii=False))
    tail = _set_kv(tail, "last_error", json.dumps(error or "", ensure_ascii=False))

    updated = (
        f'{m.group("indent")}- step_id: "{m.group("step_id")}" '
        f'action: "{m.group("action")}" '
        f"args: {m.group('args')} "
        f"checked: {checked}{tail}"
    )
    lines[target_idx] = updated

    # 3) If all steps in this high-level todo are checked, set status: done
    all_steps_checked = True
    saw_any_step = False
    for i in range(high_start, high_end):
        sm = _STEP_LINE_RE.match(lines[i])
        if not sm:
            continue
        saw_any_step = True
        if sm.group("checked") != "true":
            all_steps_checked = False
            break

    if saw_any_step and all_steps_checked:
        # find first status line inside high-level block and set to done
        for i in range(high_start, high_end):
            if _STATUS_RE.match(lines[i]):
                lines[i] = re.sub(r"status:\s*(todo|doing|done|blocked)", "status: done", lines[i])
                break

    content = "\n".join(lines) + "\n"

    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(content, encoding="utf-8")
    except Exception:
        # If we can't even write tmp, give up silently.
        return

    for sleep_s in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
        if sleep_s:
            time.sleep(sleep_s)
        try:
            os.replace(tmp, path)
            return
        except Exception:
            continue

    # Best-effort fallback.
    try:
        path.write_text(content, encoding="utf-8")
    except Exception:
        return


def apply_execution_result_to_task_progress_pe(
    *,
    task_progress_path: str | Path,
    plan_ref: PlanRef,
    result_summary: str,
    error: str,
) -> None:
    """
    Update a flat PE task_progress file:
    - locate step_id in plan_steps
    - update checked / last_result / last_error
    - advance step_cursor on success
    - set plan_status=done if all checked
    """
    path = Path(task_progress_path)
    if not path.exists():
        raise FileNotFoundError(path)

    lines = path.read_text(encoding="utf-8").splitlines(keepends=False)

    start, end = _find_block(lines, header="plan_steps:")
    if start is None:
        raise NeedReplan("plan_steps block not found")

    target_idx = None
    step_pos = -1
    step_count = 0
    for i in range(start + 1, end):
        m = _STEP_LINE_RE.match(lines[i])
        if not m:
            continue
        step_pos += 1
        step_count += 1
        if m.group("step_id") == plan_ref.atomic_step_id:
            target_idx = i
            break
    if target_idx is None:
        raise NeedReplan(f"atomic_step_id not found: {plan_ref.atomic_step_id}")

    current = lines[target_idx]
    m = _STEP_LINE_RE.match(current)
    if not m:
        raise NeedReplan("malformed step line (expected step_id/action/args/checked on same line)")

    checked = "true" if str(result_summary).lower() == "success" else "false"
    tail = m.group("tail") or ""

    def _set_kv(text: str, key: str, value: str) -> str:
        pattern = re.compile(rf'(\b{re.escape(key)}:\s*)(\"[^\"]*\"|[^ \t]+)')
        if pattern.search(text):
            return pattern.sub(rf"\1{value}", text, count=1)
        sep = "" if text.strip() == "" else " "
        return f"{text}{sep}{key}: {value}"

    tail = _set_kv(tail, "last_result", json.dumps(result_summary, ensure_ascii=False))
    tail = _set_kv(tail, "last_error", json.dumps(error or "", ensure_ascii=False))

    updated = (
        f'{m.group("indent")}- step_id: "{m.group("step_id")}" '
        f'action: "{m.group("action")}" '
        f"args: {m.group('args')} "
        f"checked: {checked}{tail}"
    )
    lines[target_idx] = updated

    # Update step_cursor on success (best-effort).
    if str(result_summary).lower() == "success" and step_pos >= 0:
        for i, line in enumerate(lines):
            if _STEP_CURSOR_RE.match(line):
                indent = line.split("step_cursor:", 1)[0]
                lines[i] = f"{indent}step_cursor: {step_pos + 1}"
                break

    # Update plan_status on failure.
    if str(result_summary).lower() != "success":
        for i, line in enumerate(lines):
            if _PLAN_STATUS_RE.match(line):
                indent = line.split("plan_status:", 1)[0]
                lines[i] = f'{indent}plan_status: "blocked"'
                break

    # If all steps are checked, mark plan_status=done.
    all_steps_checked = True
    saw_any_step = False
    for i in range(start + 1, end):
        sm = _STEP_LINE_RE.match(lines[i])
        if not sm:
            continue
        saw_any_step = True
        if sm.group("checked") != "true":
            all_steps_checked = False
            break
    if saw_any_step and all_steps_checked:
        for i, line in enumerate(lines):
            if _PLAN_STATUS_RE.match(line):
                indent = line.split("plan_status:", 1)[0]
                lines[i] = f'{indent}plan_status: "done"'
                break

    content = "\n".join(lines) + "\n"
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(content, encoding="utf-8")
    except Exception:
        return
    for sleep_s in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
        if sleep_s:
            time.sleep(sleep_s)
        try:
            os.replace(tmp, path)
            return
        except Exception:
            continue
    try:
        path.write_text(content, encoding="utf-8")
    except Exception:
        return


def append_off_plan_event(
    *,
    task_progress_path: str | Path,
    step_id: int,
    action_or_skill: str,
    reason: str,
    plan_ref: Optional[PlanRef] = None,
    time_iso: Optional[str] = None,
) -> None:
    """
    Append an off-plan event to task_progress (PE or EPM).
    """
    path = Path(task_progress_path)
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=False)
    time_iso = time_iso or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "high_level_id": plan_ref.high_level_id if plan_ref else "",
        "atomic_step_id": plan_ref.atomic_step_id if plan_ref else "",
    }
    event_lines = [
        f"  - step_id: {int(step_id)}",
        f"    action: {json.dumps(str(action_or_skill), ensure_ascii=False)}",
        f"    reason: {json.dumps(str(reason), ensure_ascii=False)}",
        f"    plan_ref: {json.dumps(payload, ensure_ascii=False)}",
        f"    time: {json.dumps(str(time_iso), ensure_ascii=False)}",
    ]

    start, end = _find_block(lines, header="off_plan_events:")
    if start is None:
        lines.append("off_plan_events:")
        lines.extend(event_lines)
    else:
        insert_at = end if end is not None else len(lines)
        lines[insert_at:insert_at] = event_lines

    content = "\n".join(lines) + "\n"
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(content, encoding="utf-8")
    except Exception:
        return
    for sleep_s in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
        if sleep_s:
            time.sleep(sleep_s)
        try:
            os.replace(tmp, path)
            return
        except Exception:
            continue
    try:
        path.write_text(content, encoding="utf-8")
    except Exception:
        return

def auto_update_task_progress_from_long_horizon(
    *,
    long_horizon_path: str | Path,
    task_progress_path: str | Path,
) -> None:
    """
    One-shot updater:
    - reads the last JSONL execution record
    - aligns it via plan_ref
    - updates task_progress (check-off / last_result / last_error)
    """
    record = read_last_jsonl_record(long_horizon_path)
    plan_ref = _parse_plan_ref(record)
    apply_execution_result_to_task_progress(
        task_progress_path=task_progress_path,
        plan_ref=plan_ref,
        result_summary=str(record.get("result_summary", "")),
        error=str(record.get("errors", "")),
    )

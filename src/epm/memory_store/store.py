from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Optional

from epm.core.epm_types import Decision, LoopDetection, PlanRef, StepRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    # Windows + cloud sync tools may briefly lock files and cause rename to fail.
    # We retry and then best-effort to avoid crashing the agent.
    last_err: Exception | None = None
    for sleep_s in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
        if sleep_s:
            time.sleep(sleep_s)
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last_err = e
        except OSError as e:
            last_err = e
    try:
        path.write_text(content, encoding="utf-8")
    except Exception:
        # Best-effort only (common on Windows when sync tools/AV temporarily lock the file).
        # We intentionally do NOT raise here to keep the agent running.
        return


def _ensure_parameter_heuristics_template(path: Path) -> None:
    if path.exists():
        return
    template: Dict[str, Any] = {
        "version": 2,
        "update_policy": "manual_or_low_frequency_from_long_horizon_history",
        "last_updated": "",
        "notes": "Parameter heuristics for the planner. Prefer stable defaults and ranges. Keep entries small.",
        "by_action": {
            # Example:
            # "move_forward": {"defaults": {"duration": 0.3}, "ranges": {"duration": [0.1, 1.5]}, "notes": "Short taps are safer."}
        },
        "by_skill": {
            # Example:
            # "Navigation": {"defaults": {"duration": 0.3}, "notes": "Avoid long moves to reduce overshoot."}
        },
        "legacy_heuristics": [],
    }
    _atomic_write_text(path, json.dumps(template, ensure_ascii=False, indent=2) + "\n")


def _ensure_body_rules_template(path: Path) -> None:
    if path.exists():
        return
    text = (
        "# Body Rules (Global)\n"
        "# This file is injected into the planner SYSTEM prompt.\n"
        "# Add safety / style / interface constraints here.\n"
        "\n"
        "- Use ONLY allowed actions/skills.\n"
        "- Do NOT invent new actions.\n"
        "- Output STRICT JSON only.\n"
        "\n"
    )
    _atomic_write_text(path, text)


def _ensure_strategy_notes_template(path: Path) -> None:
    if path.exists():
        return
    text = (
        "# Strategy Notes\n"
        "# This file is injected into the planner SYSTEM prompt.\n"
        "# Add planning guidance, heuristics, and experience notes here.\n"
        "\n"
        "- Prefer stable recovery over blind retries.\n"
        "- Reuse recent query results when available.\n"
        "- Optimize for task score under the step budget.\n"
        "\n"
    )
    _atomic_write_text(path, text)


def _maybe_seed_from_stable_repo_file(*, dst: Path, stable_rel: str) -> None:
    """
    If the per-run memory file doesn't exist yet, seed it from the repo-level stable
    epm/memory/<file> when available.
    """
    if dst.exists():
        return
    try:
        epm_dir = Path(__file__).resolve().parents[3]  # <repo>/epm
        src = epm_dir / "memory" / stable_rel
        if not src.exists():
            return
        _atomic_write_text(dst, src.read_text(encoding="utf-8-sig", errors="replace"))
    except Exception:
        return


def _ensure_reflexion_memory_template(path: Path) -> None:
    if path.exists():
        return
    text = (
        "# Reflexion Memory (Long-term)\n"
        "# Sliding window of recent reusable lessons from Reflexion.\n"
        "# Each entry should summarize failure_pattern / cause / fix / applicability.\n"
        "\n"
    )
    _atomic_write_text(path, text)


def _ensure_reflexion_progress_template(path: Path) -> None:
    if path.exists():
        return
    text = (
        "# Reflexion Progress Memory\n"
        "# Recent stable progress facts for the current episode.\n"
        "\n"
    )
    _atomic_write_text(path, text)

def _read_repo_prompt_module(epm_dir: Path, name: str) -> str:
    path = epm_dir / "memory" / "prompt_modules" / name
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace").strip() if path.exists() else ""
    except Exception:
        return ""


def _format_prompt_module(template: str, *, values: dict[str, Any]) -> str:
    if not template:
        return ""
    try:
        return template.format(**values).strip()
    except Exception:
        return template.strip()


class FileBackedMemoryStore:
    """
    Maintains the file-based memory docs under `epm/memory/`.

    - `stm_window.txt`: overwrite each step (YAML-like)
    - `long_horizon_history.txt`: append-only JSONL, same fields as STM step records
    - `agent_state.json`: overwrite each step
    """

    def __init__(
        self,
        memory_dir: str | Path,
        window_size: int = 200,
        *,
        agent_state_path: Optional[Path] = None,
        task_progress_path: Optional[Path] = None,
        enable_reflexion_memory: bool = False,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.window_size = int(window_size)
        self.enable_reflexion_memory = bool(enable_reflexion_memory)
        self._recent: Deque[StepRecord] = deque(maxlen=self.window_size)
        epm_dir = Path(__file__).resolve().parents[3]  # <repo>/epm

        self.paths = {
            "stm_window": self.memory_dir / "stm_window.txt",
            "task_progress": (Path(task_progress_path) if task_progress_path is not None else None),
            "long_horizon": self.memory_dir / "long_horizon_history.txt",
            "resume_state": self.memory_dir / "resume_state.json",
            "latest_precondition_feedback": self.memory_dir / "latest_precondition_feedback.txt",
            "latest_task_progress_feedback": self.memory_dir / "latest_task_progress_feedback.txt",
            "latest_visual_anomaly_feedback": self.memory_dir / "latest_visual_anomaly_feedback.txt",
            "cap_runtime_state": self.memory_dir / "cap_runtime_state.json",
            "body_rules": epm_dir / "memory" / "body_rules_en.txt",
            "strategy_notes": epm_dir / "memory" / "strategy_notes_en.txt",
            "reflexion_memory": self.memory_dir / "reflexion_memory.txt",
            "reflexion_progress_memory": self.memory_dir / "reflexion_progress_memory.txt",
            "agent_state": (Path(agent_state_path) if agent_state_path is not None else (self.memory_dir / "agent_state.json")),
            "current_step_status": self.memory_dir / "current_step_status.json",
            "action_memory": epm_dir / "memory" / "action_memory.json",
            "parameter_heuristics": epm_dir / "memory" / "parameter_heuristics.json",
            "put_place_occupancy": self.memory_dir / "put_place_occupancy.json",
            "action_catalog": self.memory_dir / "action_catalog.txt",
            "action_specs": self.memory_dir / "action_specs.txt",
            "skills_catalog": self.memory_dir / "skills_catalog.json",
            "skill_specs": self.memory_dir / "skill_specs.txt",
            "supported_items": epm_dir / "memory" / "supported_items.txt",
        }
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        # Seed stable templates first (so users can edit epm/memory/*.txt once).
        _maybe_seed_from_stable_repo_file(dst=self.paths["body_rules"], stable_rel="body_rules_en.txt")
        _maybe_seed_from_stable_repo_file(dst=self.paths["strategy_notes"], stable_rel="strategy_notes_en.txt")
        _maybe_seed_from_stable_repo_file(dst=self.paths["parameter_heuristics"], stable_rel="parameter_heuristics.json")
        _ensure_parameter_heuristics_template(self.paths["parameter_heuristics"])
        _ensure_body_rules_template(self.paths["body_rules"])
        _ensure_strategy_notes_template(self.paths["strategy_notes"])
        if self.enable_reflexion_memory:
            _maybe_seed_from_stable_repo_file(dst=self.paths["reflexion_memory"], stable_rel="reflexion_memory.txt")
            _ensure_reflexion_memory_template(self.paths["reflexion_memory"])
            _ensure_reflexion_progress_template(self.paths["reflexion_progress_memory"])
        else:
            try:
                self.paths["reflexion_memory"].unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
            try:
                self.paths["reflexion_progress_memory"].unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    # ----------------------------
    # Write APIs
    # ----------------------------
    def record_step(
        self,
        *,
        step_id: int,
        observation_summary: str,
        executed: Decision,
        result_summary: str,
        screenshot_path: Optional[str] = None,
        duration_s: Optional[float] = None,
        episode_elapsed_s: Optional[float] = None,
        diff: Optional[Dict[str, Any]] = None,
        errors: str = "",
        time_iso: Optional[str] = None,
        planned: Optional[Decision] = None,
        loop_detection_override: Optional[Dict[str, Any]] = None,
    ) -> StepRecord:
        time_iso = time_iso or _utc_now_iso()
        loop = self._compute_loop_detection(
            executed_action=executed.action_or_skill,
            result_summary=result_summary,
            errors=errors,
            override=loop_detection_override,
        )

        record = StepRecord(
            step_id=int(step_id),
            time=time_iso,
            observation_summary=str(observation_summary),
            action_or_skill=str(executed.action_or_skill),
            params=dict(executed.params),
            result_summary=str(result_summary),
            errors=str(errors),
            loop_detection=loop,
            screenshot_path=str(screenshot_path) if screenshot_path else None,
            duration_s=(float(duration_s) if duration_s is not None else None),
            episode_elapsed_s=(float(episode_elapsed_s) if episode_elapsed_s is not None else None),
            plan_ref=executed.plan_ref,
            planned=({"action_or_skill": planned.action_or_skill, "params": planned.params} if planned else None),
            executed={"action_or_skill": executed.action_or_skill, "params": executed.params},
            diff=(dict(diff) if isinstance(diff, dict) else None),
        )

        self._recent.append(record)
        self._write_stm_window(last_updated=time_iso)
        self._append_long_horizon_jsonl(record)
        return record

    def update_agent_state(self, **fields: Any) -> Dict[str, Any]:
        path = self.paths["agent_state"]
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                current = {}
        else:
            current = {}

        current.update(fields)
        current["last_updated"] = _utc_now_iso()
        _atomic_write_text(path, json.dumps(current, ensure_ascii=False, indent=2) + "\n")
        return current

    def write_text(self, key: str, content: str) -> None:
        path = self.paths.get(key)
        if path is None:
            return
        _atomic_write_text(path, str(content or ""))

    def write_json(self, key: str, payload: Dict[str, Any]) -> None:
        path = self.paths.get(key)
        if path is None:
            return
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def save_precondition_check(self, *, step_id: int, payload: Dict[str, Any]) -> Path:
        path = self.memory_dir / "precondition_checks" / f"step_{int(step_id):06d}.json"
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return path

    def save_task_progress_update(self, *, step_id: int, payload: Dict[str, Any]) -> Path:
        path = self.memory_dir / "task_progress_updates" / f"step_{int(step_id):06d}.json"
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return path

    # ----------------------------
    # Read APIs (prompt snippets)
    # ----------------------------
    def read_text(self, key: str) -> str:
        path = self.paths.get(key)
        if path is None:
            return ""
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def read_json(self, key: str) -> Dict[str, Any]:
        path = self.paths[key]
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def bundle_for_brain(self) -> Dict[str, str]:
        """
        Returns prompt-friendly snippets. This intentionally excludes huge histories.
        """
        bundle: Dict[str, str] = {}
        bundle["body_rules"] = self.read_text("body_rules").strip()
        bundle["strategy_notes"] = self.read_text("strategy_notes").strip()
        bundle["action_memory"] = self.read_text("action_memory").strip()
        bundle["reflexion_memory"] = self.read_text("reflexion_memory").strip()
        bundle["reflexion_progress_memory"] = self.read_text("reflexion_progress_memory").strip()
        bundle["stm_window"] = self.read_text("stm_window").strip()
        bundle["task_progress"] = self.read_text("task_progress").strip()
        bundle["latest_precondition_feedback"] = self.read_text("latest_precondition_feedback").strip()
        bundle["latest_task_progress_feedback"] = self.read_text("latest_task_progress_feedback").strip()
        bundle["latest_visual_anomaly_feedback"] = self.read_text("latest_visual_anomaly_feedback").strip()
        bundle["action_catalog"] = self.read_text("action_catalog").strip()
        bundle["supported_items"] = self.read_text("supported_items").strip()
        bundle["put_place_occupancy"] = self.read_text("put_place_occupancy").strip()
        # Stable, repo-level tool manifest (OpenAI `tools=[...]` JSON).
        # This is used for tool-schema prompting even when the runtime planner is not
        # using OpenAI tool calling.
        try:
            epm_dir = Path(__file__).resolve().parents[3]  # <repo>/epm
            tools_path_en = epm_dir / "memory" / "tools_manifest_openai_en.json"
            tools_path = epm_dir / "memory" / "tools_manifest_openai.json"
            if tools_path_en.exists():
                bundle["tools_manifest_openai"] = tools_path_en.read_text(encoding="utf-8")
            else:
                bundle["tools_manifest_openai"] = tools_path.read_text(encoding="utf-8") if tools_path.exists() else ""
        except Exception:
            bundle["tools_manifest_openai"] = ""
        # Stable, repo-level SYSTEM role prompt module.
        try:
            epm_dir = Path(__file__).resolve().parents[3]  # <repo>/epm
            bundle["system_role"] = _read_repo_prompt_module(epm_dir, "system_role.txt")
        except Exception:
            bundle["system_role"] = ""
        # Force-submit deadline module (templated).
        try:
            from epm.cerebellum.skills._shared_paths import load_epm_config

            epm_dir = Path(__file__).resolve().parents[3]
            template = _read_repo_prompt_module(epm_dir, "force_submit_deadline.txt")
            cfg = load_epm_config() or {}
            runtime = cfg.get("runtime") or {}
            threshold = int(runtime.get("force_submit_step_threshold", 0) or 0)
            within = int(runtime.get("force_submit_within_steps", 0) or 0)
            if template and threshold > 0 and within > 0:
                bundle["force_submit_deadline"] = _format_prompt_module(
                    template,
                    values={
                        "force_submit_step_threshold": int(threshold),
                        "force_submit_within_steps": int(within),
                    },
                )
            else:
                bundle["force_submit_deadline"] = ""
        except Exception:
            bundle["force_submit_deadline"] = ""
        bundle["parameter_heuristics"] = ""
        bundle["action_specs"] = ""
        bundle["skill_specs"] = ""
        bundle["skills_catalog"] = ""
        try:
            bundle["agent_state"] = json.dumps(self.read_json("agent_state"), ensure_ascii=False, indent=2).strip()
        except Exception:
            bundle["agent_state"] = ""
        try:
            payload = self.read_json("current_step_status")
            bundle["current_step_status"] = json.dumps(payload, ensure_ascii=False, indent=2).strip() if payload else ""
        except Exception:
            bundle["current_step_status"] = ""
        try:
            payload = self.read_json("resume_state")
            bundle["resume_state"] = json.dumps(payload, ensure_ascii=False, indent=2).strip() if payload else ""
        except Exception:
            bundle["resume_state"] = ""
        try:
            payload = self.read_json("cap_runtime_state")
            bundle["cap_runtime_state"] = json.dumps(payload, ensure_ascii=False, indent=2).strip() if payload else ""
        except Exception:
            bundle["cap_runtime_state"] = ""
        try:
            epm_dir = Path(__file__).resolve().parents[3]  # <repo>/epm
            put_place_path = epm_dir / "data" / "put_place_list.json"
            bundle["put_place_list"] = put_place_path.read_text(encoding="utf-8") if put_place_path.exists() else ""
        except Exception:
            bundle["put_place_list"] = ""
        try:
            epm_dir = Path(__file__).resolve().parents[3]  # <repo>/epm
            tip_path = epm_dir / "data" / "tool_interaction_point_list.json"
            bundle["tool_interaction_point_list"] = tip_path.read_text(encoding="utf-8") if tip_path.exists() else ""
        except Exception:
            bundle["tool_interaction_point_list"] = ""
        return bundle

    # ----------------------------
    # Internals
    # ----------------------------
    def _compute_loop_detection(
        self,
        *,
        executed_action: str,
        result_summary: str,
        errors: str,
        override: Optional[Dict[str, Any]],
    ) -> LoopDetection:
        if override is not None:
            return LoopDetection(
                loop_detected=bool(override.get("loop_detected", False)),
                consecutive_failures=int(override.get("consecutive_failures", 0) or 0),
                last_error_types=list(override.get("last_error_types") or []),
            )

        steps = list(self._recent)
        consecutive_failures = 0
        for s in reversed(steps):
            if str(s.result_summary).lower() == "success":
                break
            consecutive_failures += 1

        last_error_types = [str(s.errors) for s in steps[-self.window_size :] if s.errors]

        last_actions = [s.action_or_skill for s in steps[-2:]] + [executed_action]
        loop_detected = consecutive_failures >= 3 and len(set(last_actions[-3:])) == 1
        return LoopDetection(
            loop_detected=loop_detected,
            consecutive_failures=consecutive_failures,
            last_error_types=last_error_types[-5:],
        )

    def _write_stm_window(self, *, last_updated: str) -> None:
        path = self.paths["stm_window"]
        lines: list[str] = []
        lines.append("# STM Window (Short-Term Memory)")
        lines.append("update_policy: overwrite_each_step")
        lines.append(f"window_size: {self.window_size}")
        lines.append(f"last_updated: \"{last_updated}\"")
        lines.append("")

        if len(self._recent) == 0:
            lines.append("steps: []")
        else:
            lines.append("steps:")
            for rec in self._recent:
                lines.append(f"  - step_id: {rec.step_id}")
                lines.append(f"    action_or_skill: {json.dumps(rec.action_or_skill, ensure_ascii=False)}")
                lines.append(f"    params: {json.dumps(rec.params, ensure_ascii=False)}")
                lines.append(f"    result_summary: {json.dumps(rec.result_summary, ensure_ascii=False)}")
                lines.append(f"    errors: {json.dumps(rec.errors, ensure_ascii=False)}")
        lines.append("")
        _atomic_write_text(path, "\n".join(lines))

    def _append_long_horizon_jsonl(self, record: StepRecord) -> None:
        path = self.paths["long_horizon"]
        payload: Dict[str, Any] = {
            "step_id": record.step_id,
            "time": record.time,
            "observation_summary": record.observation_summary,
            "screenshot_path": record.screenshot_path,
            "duration_s": record.duration_s,
            "episode_elapsed_s": record.episode_elapsed_s,
            "action_or_skill": record.action_or_skill,
            "params": record.params,
            "result_summary": record.result_summary,
            "errors": record.errors,
            "loop_detection": asdict(record.loop_detection),
        }
        if record.plan_ref:
            payload["plan_ref"] = asdict(record.plan_ref)
        if record.planned:
            payload["planned"] = record.planned
        if record.executed:
            payload["executed"] = record.executed
        if record.diff:
            payload["diff"] = record.diff

        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = line + "\n"

        # Windows + sync tools/AV may temporarily lock the file. Retry a bit; if still blocked,
        # fall back to a per-process sidecar file so we don't crash the episode.
        last_err: Exception | None = None
        for sleep_s in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
            if sleep_s:
                time.sleep(sleep_s)
            try:
                with path.open("a", encoding="utf-8") as f:
                    f.write(content)
                return
            except PermissionError as e:
                last_err = e
            except OSError as e:
                last_err = e

        fallback = path.with_name(path.name + f".fallback.{os.getpid()}.txt")
        try:
            with fallback.open("a", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            # Best-effort only.
            return

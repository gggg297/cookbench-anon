from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from epm.brain.chat_client import chat_complete_text
from epm.brain.model_output_trace import write_trace_files_for_raw_dir
from epm.brain.modules.prompt_assets import load_asset
from epm.brain.plan_schema import extract_json_object
from epm.brain.interfaces import PlannerContext
from epm.core.prompt_ablation import assert_prompt_ablation, is_prompt_section_enabled


@dataclass(frozen=True)
class EPMConfig:
    max_subgoals: int = 12
    check_done_every_n_steps: int = 1
    subgoal_done_judge_enabled: bool = True
    save_raw: bool = True
    prompt_ablation_profile: str = "full"
    prompt_disabled_groups: frozenset[str] | None = None

    def section_enabled(self, section: str) -> bool:
        return is_prompt_section_enabled(
            profile=self.prompt_ablation_profile,
            section=section,
            disabled_groups=self.prompt_disabled_groups,
        )


class GoalDecomposer:
    def __init__(self, *, chat_cfg: Any, cfg: EPMConfig, memory_dir: Path) -> None:
        self.chat_cfg = chat_cfg
        self.cfg = cfg
        self.memory_dir = Path(memory_dir)

    @staticmethod
    def _split_recipe_to_steps(recipe_text: str) -> list[str]:
        text = str(recipe_text or "").replace("\n", " ").strip()
        if not text:
            return []
        parts = [p.strip() for p in text.split(".") if p.strip()]
        return parts[:50]

    def _load_task_progress_recipe_steps(self, context: PlannerContext) -> list[str]:
        if not self.cfg.section_enabled("task_progress"):
            return []
        raw = str((context.memory_bundle or {}).get("task_progress", "") or "").strip()
        if not raw:
            return []
        try:
            obj = json.loads(raw)
        except Exception:
            return []
        if not isinstance(obj, dict):
            return []
        recipe = obj.get("recipe")
        if not isinstance(recipe, dict):
            return []
        steps = recipe.get("steps")
        if not isinstance(steps, list):
            return []
        clean: list[str] = []
        for item in steps:
            if not isinstance(item, str):
                continue
            step = " ".join(item.strip().split())
            if step:
                clean.append(step)
        return clean

    @staticmethod
    def _normalize_recipe_step(step: str) -> str:
        text = " ".join(str(step or "").strip().split())
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        return text.rstrip(".")

    def _deterministic_recipe_order_subgoals(self, *, context: PlannerContext) -> list[str]:
        steps = self._load_task_progress_recipe_steps(context)
        if not steps:
            steps = self._split_recipe_to_steps(context.recipe_text)
        clean: list[str] = []
        for step in steps:
            norm = self._normalize_recipe_step(step)
            if norm:
                clean.append(norm)
        return clean[: int(self.cfg.max_subgoals)]

    def decompose(self, *, context: PlannerContext) -> list[str]:
        raw_dir = self.memory_dir / "epm_raw"
        if bool(self.cfg.save_raw):
            raw_dir.mkdir(parents=True, exist_ok=True)

        deterministic = self._deterministic_recipe_order_subgoals(context=context)
        if deterministic:
            if bool(self.cfg.save_raw):
                payload = {
                    "source": "deterministic_recipe_order",
                    "high_level_goal": str(context.high_level_goal or "").strip(),
                    "subgoals": deterministic,
                }
                (raw_dir / "decompose_last.txt").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            return deterministic

        role_text = load_asset(
            "epm/goal_decomposer",
            "role.txt",
            "You are an EPM-style hierarchical planner. Decompose the task into a small ordered list of subgoals. Return STRICT JSON only.",
        )
        rules_text = load_asset(
            "epm/goal_decomposer",
            "rules.txt",
            (
                "Rules:\n"
                "- Each subgoal should be actionable and short.\n"
                "- Use strict recipe order.\n"
                "- Prefer one subgoal per recipe step.\n"
                "- Do NOT create broad setup subgoals such as Gather ingredients and equipment or Prepare the workspace.\n"
                "- Do NOT merge non-adjacent recipe steps.\n"
                "- Do not include tool calls or code."
            ),
        )
        output_text = load_asset(
            "epm/goal_decomposer",
            "output_format.txt",
            'Response JSON schema:\n{ "subgoals": ["...", "..."], "notes": "optional" }',
        )
        prompt = (
            "SYSTEM:\n"
            f"{role_text}\n"
            "\n"
            f"{rules_text}\n"
            "\n"
            f"{output_text}\n"
            "\n"
            "USER:\n"
            f"High-level goal: {context.high_level_goal}\n"
            "\n"
            "Recipe (raw):\n"
            f"{(context.recipe_text or '').strip()}\n"
            "\n"
            + (
                "Task progress snapshot:\n"
                f"{(context.memory_bundle.get('task_progress', '') or '').strip()}\n"
                if self.cfg.section_enabled("task_progress")
                else ""
            )
        )

        assert_prompt_ablation(
            profile=self.cfg.prompt_ablation_profile,
            disabled_groups=self.cfg.prompt_disabled_groups,
            source="epm.goal_decomposer",
            prompt_text=prompt,
            memory_dir=self.memory_dir,
        )

        out = chat_complete_text(
            cfg=self.chat_cfg,
            prompt=prompt,
            screenshot_path=context.observation.screenshot_path,
            force_use_vision=bool(getattr(self.chat_cfg, "use_vision", False)),
        )
        if bool(self.cfg.save_raw):
            (raw_dir / "decompose_last.txt").write_text(out.content, encoding="utf-8")
            write_trace_files_for_raw_dir(raw_dir=raw_dir, filename="decompose_last.txt", text=out.content)

        obj = extract_json_object(out.content, strip_think_tags=True)
        subgoals = obj.get("subgoals")
        if not isinstance(subgoals, list):
            return []
        clean: list[str] = []
        for s in subgoals:
            if not isinstance(s, str):
                continue
            t = s.strip()
            if not t:
                continue
            clean.append(t)
        return clean[: int(self.cfg.max_subgoals)]


class SubgoalDoneJudge:
    def __init__(self, *, chat_cfg: Any, cfg: EPMConfig, memory_dir: Path) -> None:
        self.chat_cfg = chat_cfg
        self.cfg = cfg
        self.memory_dir = Path(memory_dir)

    def is_done(self, *, context: PlannerContext, subgoal: str) -> tuple[bool, str]:
        raw_dir = self.memory_dir / "epm_raw"
        if bool(self.cfg.save_raw):
            raw_dir.mkdir(parents=True, exist_ok=True)

        role_text = load_asset(
            "epm/subgoal_done_judge",
            "role.txt",
            "You are a strict checker. Decide if the current subgoal is already satisfied. Return STRICT JSON only.",
        )
        output_text = load_asset(
            "epm/subgoal_done_judge",
            "output_format.txt",
            'Response JSON schema:\n{ "done": true|false, "reason": "one sentence" }',
        )
        feedback_text = (context.feedback or '').strip() if self.cfg.section_enabled("feedback") else ""
        on_screen_names = (
            json.dumps((context.observation.state.get('_on_screen_names', []) or [])[:20], ensure_ascii=False)
            if self.cfg.section_enabled("oracle_observation")
            else ""
        )
        on_screen_objects = (
            json.dumps((context.observation.state.get('_on_screen_objects', []) or [])[:10], ensure_ascii=False)
            if self.cfg.section_enabled("oracle_observation")
            else ""
        )
        prompt = (
            "SYSTEM:\n"
            f"{role_text}\n"
            "\n"
            f"{output_text}\n"
            "\n"
            "USER:\n"
            f"Subgoal: {subgoal}\n"
            "\n"
            + ("Latest feedback:\n" + feedback_text + "\n\n" if feedback_text else "")
            + ("On-screen items (top20):\n" + on_screen_names + "\n\n" if on_screen_names else "")
            + ("On-screen objects (top10):\n" + on_screen_objects + "\n" if on_screen_objects else "")
        )

        assert_prompt_ablation(
            profile=self.cfg.prompt_ablation_profile,
            disabled_groups=self.cfg.prompt_disabled_groups,
            source="epm.subgoal_done_judge",
            prompt_text=prompt,
            memory_dir=self.memory_dir,
        )

        out = chat_complete_text(
            cfg=self.chat_cfg,
            prompt=prompt,
            screenshot_path=context.observation.screenshot_path,
            force_use_vision=bool(getattr(self.chat_cfg, "use_vision", False)),
        )
        if bool(self.cfg.save_raw):
            (raw_dir / "judge_last.txt").write_text(out.content, encoding="utf-8")
            write_trace_files_for_raw_dir(raw_dir=raw_dir, filename="judge_last.txt", text=out.content)

        obj = extract_json_object(out.content, strip_think_tags=True)
        done = bool(obj.get("done", False))
        reason = obj.get("reason")
        if not isinstance(reason, str):
            reason = ""
        return done, reason.strip()


def save_goal_tree(*, path: Path, high_level_goal: str, subgoals: list[str], current_index: int) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "goal": high_level_goal, "subgoals": subgoals, "current_index": int(current_index)}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        return


def load_goal_tree(*, path: Path) -> Optional[dict[str, Any]]:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None

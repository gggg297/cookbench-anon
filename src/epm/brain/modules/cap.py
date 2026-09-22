from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from epm.brain.chat_client import chat_complete_text
from epm.brain.interfaces import PlannerContext
from epm.brain.model_output_trace import split_think_and_final, write_trace_files_for_raw_dir
from epm.brain.plan_schema import extract_json_object
from epm.brain.modules.prompt_assets import load_asset
from epm.brain.prompt_builder import PromptBuilder, PromptBuilderConfig
from epm.brain.tools_manifest import build_tool_manifest, to_openai_tools
from epm.core.prompt_ablation import (
    assert_prompt_ablation,
    is_prompt_section_enabled,
    restrict_to_actions_only,
    restrict_to_raw_input_actions,
)
from epm.cerebellum.action_specs import to_prompt_text as action_specs_to_prompt_text
from epm.cerebellum.cookbench_api import LocalActionAPI
from epm.cerebellum.skills.registry import list_skills, skill_specs_to_prompt_text
from epm.cerebellum.skills._shared_paths import load_epm_config


@dataclass(frozen=True)
class CaPConfig:
    max_code_chars: int = 8000
    max_code_lines: int = 120
    max_yields_before_regen: int = 32
    save_raw: bool = True
    prompt_ablation_profile: str = "full"
    prompt_disabled_groups: frozenset[str] | None = None


def _is_instance_disambiguation_feedback(feedback: str) -> bool:
    return "blocking=instance_disambiguation" in str(feedback or "").strip().lower()


def _feedback_line_value(feedback: str, key: str) -> str:
    prefix = f"{key}="
    for raw in str(feedback or "").splitlines():
        line = raw.strip()
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _extract_failed_step_type_and_name(feedback: str) -> tuple[str, str]:
    raw = str(feedback or "")
    for line in raw.splitlines():
        s = line.strip()
        if not s.startswith("type="):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        step_type = parts[0].split("=", 1)[-1].strip().lower()
        step_name = parts[1].split("=", 1)[-1].strip()
        if step_type in ("action", "skill") and step_name:
            return step_type, step_name
    return "", ""


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _is_query_result_feedback(feedback: str) -> bool:
    text = str(feedback or "")
    if "type=skill name=query_scene_objects" not in text:
        return False
    try:
        return int(_feedback_line_value(text, "results_len")) > 0
    except Exception:
        return False


def _validate_plan_ast(tree: ast.AST) -> None:
    banned_nodes = (
        ast.Global,
        ast.Nonlocal,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.While,
        ast.For,
        ast.AsyncFor,
        ast.Match,
        ast.ClassDef,
        ast.Lambda,
    )
    for node in ast.walk(tree):
        if isinstance(node, banned_nodes):
            raise ValueError(f"cap_banned_plan_syntax:{type(node).__name__}")
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in ("open", "exec", "eval", "compile", "__import__"):
                raise ValueError(f"cap_banned_call:{fn.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("cap_banned_dunder_attribute")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("cap_banned_dunder_name")


def _literal_from_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal_from_ast(node.operand)
        if not isinstance(value, (int, float)):
            raise ValueError("cap_non_literal_numeric")
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.List):
        return [_literal_from_ast(it) for it in node.elts]
    if isinstance(node, ast.Tuple):
        return [_literal_from_ast(it) for it in node.elts]
    if isinstance(node, ast.Dict):
        out: dict[str, Any] = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise ValueError("cap_kwargs_unpack_not_allowed")
            out[str(_literal_from_ast(key))] = _literal_from_ast(value)
        return out
    if isinstance(node, ast.Subscript):
        raise ValueError(
            "cap_non_literal_arg:Subscript:yield args cannot use indexing like ctx[...] / results[...] / x[...]"
        )
    if isinstance(node, ast.Attribute):
        raise ValueError("cap_non_literal_arg:Attribute:yield args cannot use attribute access like obj.attr")
    if isinstance(node, ast.Call):
        raise ValueError("cap_non_literal_arg:Call:yield args cannot use function calls")
    raise ValueError(f"cap_non_literal_arg:{type(node).__name__}")


def _humanize_plan_extract_error(error: Exception) -> str:
    text = str(error or "").strip()
    if "cap_non_literal_arg:Subscript" in text:
        return (
            "yield_args_must_be_literal_only: found indexing expression inside a yield argument "
            "(for example ctx[...] / results[...] / x[...])"
        )
    if "cap_non_literal_arg:Attribute" in text:
        return "yield_args_must_be_literal_only: found attribute access inside a yield argument (for example obj.attr)"
    if "cap_non_literal_arg:Call" in text:
        return "yield_args_must_be_literal_only: found function call inside a yield argument"
    if "cap_non_literal_arg:" in text:
        return "yield_args_must_be_literal_only: found a non-literal expression inside a yield argument"
    if "cap_plan_statement_not_allowed" in text:
        return "policy_body_must_be_straight_line_only: only comments, pass, and sequential yield statements are allowed"
    if "cap_banned_plan_syntax:" in text:
        return "policy_contains_banned_python_syntax: loops/try/helper constructs are not allowed in policy(ctx)"
    return text or repr(error)


def _extract_step_from_call(call: ast.Call) -> dict[str, Any]:
    if not isinstance(call.func, ast.Name) or call.func.id not in {"action", "skill"}:
        raise ValueError("cap_step_not_action_or_skill")
    if not call.args:
        raise ValueError("cap_step_missing_name")
    step_name = _literal_from_ast(call.args[0])
    if not isinstance(step_name, str) or not step_name.strip():
        raise ValueError("cap_step_invalid_name")
    args: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            raise ValueError("cap_kwargs_unpack_not_allowed")
        args[str(kw.arg)] = _literal_from_ast(kw.value)
    return {
        "type": str(call.func.id),
        "name": step_name.strip(),
        "args": args,
    }


def _extract_plan_steps_from_code(code: str, *, max_steps: int) -> list[dict[str, Any]]:
    tree = ast.parse(code, mode="exec")
    _validate_plan_ast(tree)
    policy_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "policy":
            policy_fn = node
            break
    if policy_fn is None:
        raise ValueError("cap_missing_policy_function")

    steps: list[dict[str, Any]] = []
    for stmt in policy_fn.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
            call = stmt.value.value
            if not isinstance(call, ast.Call):
                raise ValueError("cap_yield_non_call")
            steps.append(_extract_step_from_call(call))
        else:
            raise ValueError(f"cap_plan_statement_not_allowed:{type(stmt).__name__}")
        if len(steps) >= int(max_steps):
            break
    if not steps:
        raise ValueError("cap_empty_plan")
    return steps


@dataclass
class CapProgram:
    code: str
    plan_steps: list[dict[str, Any]]
    ctx: dict[str, Any]


class CaPCodegen:
    def __init__(self, *, chat_cfg: Any, cfg: CaPConfig, memory_dir: Path) -> None:
        self.chat_cfg = chat_cfg
        self.cfg = cfg
        self.memory_dir = Path(memory_dir)

    def _skill_interface_enabled(self) -> bool:
        return not restrict_to_actions_only(
            self.cfg.prompt_ablation_profile,
            disabled_groups=self.cfg.prompt_disabled_groups,
        ) and not restrict_to_raw_input_actions(
            self.cfg.prompt_ablation_profile,
            disabled_groups=self.cfg.prompt_disabled_groups,
        )

    @staticmethod
    def _strip_skill_interface_references(text: str) -> str:
        """Remove high-level-skill knowledge from a no-skill CAP request.

        CAP receives static assets and shared planner context from independent
        paths.  Filtering the tool manifest alone therefore does not prevent a
        template or a system rule from teaching an unavailable skill API.
        """
        replacements = (
            ("`yield action(...)` / `yield skill(...)`", "`yield action(...)`"),
            ("yield action(...) / yield skill(...)", "yield action(...)"),
            ("action/skill", "action"),
            ("action / skill", "action"),
            ("action or skill", "action"),
            ("actions and skills", "actions"),
            ("actions/skills", "actions"),
            ("action/skills", "actions"),
        )
        skill_names = {str(name or "").strip().lower() for name in list_skills()}
        skill_names.discard("")
        kept: list[str] = []
        for raw_line in str(text or "").splitlines():
            line = raw_line
            for source, replacement in replacements:
                line = line.replace(source, replacement)
            lowered = line.lower()
            # A skill name may occur in static guidance even when the manifest
            # has already been filtered.  `auto_*` covers skill aliases from
            # legacy prompt assets that are absent from the current registry.
            has_skill_name = any(name in lowered for name in skill_names)
            has_legacy_skill_name = bool(re.search(r"\b(?:auto_[a-z0-9_]+|query_scene_objects)\b", lowered))
            if (
                "skill" in lowered
                or has_skill_name
                or has_legacy_skill_name
            ):
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    @staticmethod
    def _force_submit_hint() -> str:
        try:
            cfg = load_epm_config() or {}
            runtime = cfg.get("runtime") or {}
            threshold = int(runtime.get("force_submit_step_threshold", 0) or 0)
            within = int(runtime.get("force_submit_within_steps", 0) or 0)
            if threshold > 0 and within > 0:
                try:
                    epm_dir = Path(__file__).resolve().parents[3]
                    template = (epm_dir / "memory" / "prompt_modules" / "force_submit_deadline.txt").read_text(
                        encoding="utf-8-sig", errors="replace"
                    )
                    template = template.strip()
                    if template:
                        return template.format(
                            force_submit_step_threshold=int(threshold),
                            force_submit_within_steps=int(within),
                        ).strip()
                except Exception:
                    pass
                return (
                    f"Deadline: finish cooking and submit BEFORE step {threshold}. "
                    f"After that, the system will override planning and force-submit within {within} steps "
                    "using the tracked active container (often the last used container)."
                )
        except Exception:
            pass
        return ""

    @staticmethod
    def _filter_skill_specs(*, allowed_skills: list[str]) -> str:
        base = skill_specs_to_prompt_text()
        allow = set(allowed_skills or [])
        if not allow:
            return ""
        out: list[str] = []
        for line in base.splitlines():
            s = line.strip()
            if not s:
                out.append(line)
                continue
            if s.startswith("Builtin Skills"):
                out.append(line)
                continue
            if s.startswith("- "):
                name = s[2:].split("(", 1)[0].strip()
                if name in allow:
                    out.append(line)
                continue
            out.append(line)
        return "\n".join(out).strip() + "\n"

    @staticmethod
    def _build_interface_reference(
        *,
        prompt_ablation_profile: str = "full",
        prompt_disabled_groups: frozenset[str] | None = None,
    ) -> str:
        try:
            allowed_actions = LocalActionAPI(
                raw_input_only=restrict_to_raw_input_actions(
                    prompt_ablation_profile, disabled_groups=prompt_disabled_groups
                )
            ).list_actions()
            allowed_skills = list_skills()
            if restrict_to_raw_input_actions(
                prompt_ablation_profile, disabled_groups=prompt_disabled_groups
            ) or restrict_to_actions_only(prompt_ablation_profile, disabled_groups=prompt_disabled_groups):
                allowed_skills = []
            manifest = build_tool_manifest(allowed_actions=allowed_actions, allowed_skills=allowed_skills)
            tools_manifest = json.dumps(to_openai_tools(manifest), ensure_ascii=False, indent=2)
            action_specs = action_specs_to_prompt_text(action_names=allowed_actions).strip()
            skill_specs = CaPCodegen._filter_skill_specs(allowed_skills=allowed_skills).strip()

            parts: list[str] = []
            parts.append("Authoritative tool manifest (exact callable names and parameters; use these exact names only):")
            parts.append(tools_manifest)
            if skill_specs:
                parts.append("")
                parts.append(skill_specs)
            if action_specs:
                parts.append("")
                parts.append("Atomic Actions (exact names/signatures):")
                parts.append(action_specs)
            return "\n".join(parts).strip()
        except Exception:
            return ""

    @staticmethod
    def _load_prompt_layout() -> Optional[dict[str, Any]]:
        try:
            cfg = load_epm_config() or {}
            brain = cfg.get("brain") if isinstance(cfg, dict) else {}
            raw_path = brain.get("prompt_layout_path") if isinstance(brain, dict) else None
            epm_dir = Path(__file__).resolve().parents[3]
            if isinstance(raw_path, str) and raw_path.strip():
                p = Path(raw_path)
                if not p.is_absolute():
                    p = (epm_dir / p).resolve()
            else:
                p = epm_dir / "memory" / "prompt_layout.json"
            if not p.exists():
                return None
            raw = json.loads(p.read_text(encoding="utf-8-sig", errors="replace"))
            return raw if isinstance(raw, dict) else None
        except Exception:
            return None

    def _build_shared_prompt_sections(self, *, context: PlannerContext) -> tuple[str, str]:
        memory_bundle = dict(context.memory_bundle) if isinstance(context.memory_bundle, dict) else {}
        # CAP rebuilds the planner prompt itself, so preserve the run-level ablation
        # profile even if an intermediate context omitted its private bundle field.
        memory_bundle["_prompt_ablation_profile"] = str(self.cfg.prompt_ablation_profile or "full")
        if restrict_to_raw_input_actions(
            self.cfg.prompt_ablation_profile, disabled_groups=self.cfg.prompt_disabled_groups
        ) or restrict_to_actions_only(
            self.cfg.prompt_ablation_profile, disabled_groups=self.cfg.prompt_disabled_groups
        ):
            allowed_actions = LocalActionAPI(
                raw_input_only=restrict_to_raw_input_actions(
                    self.cfg.prompt_ablation_profile, disabled_groups=self.cfg.prompt_disabled_groups
                )
            ).list_actions()
            manifest = build_tool_manifest(allowed_actions=allowed_actions, allowed_skills=[])
            memory_bundle["tools_manifest_openai"] = json.dumps(to_openai_tools(manifest), ensure_ascii=False, indent=2)
            memory_bundle["action_specs"] = action_specs_to_prompt_text(action_names=allowed_actions).strip()
            memory_bundle["skill_specs"] = ""
            memory_bundle["skills_catalog"] = ""
        builder = PromptBuilder(
            PromptBuilderConfig(
                inject_memory=bool(memory_bundle.get("_inject_memory", True)),
                inject_tool_schemas=bool(memory_bundle.get("_inject_tool_schemas", True)),
                prompt_layout=self._load_prompt_layout(),
                prompt_ablation_profile=str(self.cfg.prompt_ablation_profile or "full"),
                prompt_disabled_groups=self.cfg.prompt_disabled_groups,
            )
        )

        system_parts: list[str] = []
        builder._append_system_injected_sections(
            system_parts=system_parts,
            memory_bundle=memory_bundle,
            inject_tool_schemas=bool(memory_bundle.get("_inject_tool_schemas", True)),
            high_level_goal=context.high_level_goal,
            recipe_text=memory_bundle.get("recipe_text", ""),
            feedback=context.feedback,
            instance_id_selection_mode=False,
        )

        user_parts: list[str] = []
        builder._append_user_sections(
            user_parts=user_parts,
            observation=context.observation,
            high_level_id=context.high_level_id,
            high_level_goal=context.high_level_goal,
            memory_bundle=memory_bundle,
            feedback=context.feedback,
            percept_text=(context.percept.text if context.percept else ""),
            inject_memory=bool(memory_bundle.get("_inject_memory", True)),
            include_task_progress=bool(memory_bundle.get("_include_task_progress", True)),
            instance_id_selection_mode=False,
        )

        return "\n".join(system_parts).strip(), "\n".join(user_parts).strip()

    def _prompt_section_enabled(self, section: str) -> bool:
        return is_prompt_section_enabled(
            profile=self.cfg.prompt_ablation_profile,
            section=section,
            disabled_groups=self.cfg.prompt_disabled_groups,
        )

    def _audit_prompt(self, *, source: str, prompt: str) -> None:
        assert_prompt_ablation(
            profile=self.cfg.prompt_ablation_profile,
            disabled_groups=self.cfg.prompt_disabled_groups,
            source=source,
            prompt_text=prompt,
            memory_dir=self.memory_dir,
        )

    @staticmethod
    def _without_feedback_runtime_fields(runtime_state: dict[str, Any]) -> dict[str, Any]:
        blocked_tokens = ("error", "feedback", "last_result", "last_yield", "recent", "repair")
        return {
            str(key): value
            for key, value in runtime_state.items()
            if not any(token in str(key).lower() for token in blocked_tokens)
        }

    @staticmethod
    def _load_json_dict(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def _build_resume_summary(self, *, context: PlannerContext) -> str:
        if not self._prompt_section_enabled("resume_state"):
            return ""

        include_feedback = self._prompt_section_enabled("feedback")
        runtime_payload = self._load_json_dict((context.memory_bundle or {}).get("cap_runtime_state", ""))
        runtime_state = runtime_payload.get("runtime_state")
        if not isinstance(runtime_state, dict):
            runtime_state = {}
        if not include_feedback:
            runtime_state = self._without_feedback_runtime_fields(runtime_state)
        agent_state = self._load_json_dict((context.memory_bundle or {}).get("agent_state", ""))
        lines = [
            "Continue from the existing execution prefix without restarting completed substeps.",
        ]
        stage = runtime_state.get("stage")
        if stage is not None:
            lines.append(f"runtime_stage={stage!r}")
        postconditions = runtime_payload.get("postconditions")
        if isinstance(postconditions, dict) and postconditions:
            lines.append("postconditions=" + json.dumps(postconditions, ensure_ascii=False, sort_keys=True))
        ordered_names = agent_state.get("ordered_dish_names")
        if isinstance(ordered_names, list) and ordered_names:
            lines.append("dish_already_ordered=true")
            lines.append("ordered_dish_names=" + json.dumps([str(it) for it in ordered_names[:8]], ensure_ascii=False))
        if include_feedback:
            last_result = runtime_payload.get("last_step_result")
            if isinstance(last_result, dict) and last_result:
                lines.append("last_step_result=" + json.dumps(last_result, ensure_ascii=False, sort_keys=True))
            last_yield = runtime_payload.get("last_yield")
            if isinstance(last_yield, dict) and last_yield:
                lines.append("last_yield=" + json.dumps(last_yield, ensure_ascii=False, sort_keys=True))
            recent_steps = runtime_payload.get("recent_steps")
            if isinstance(recent_steps, list) and recent_steps:
                compact = []
                for item in recent_steps[-4:]:
                    if not isinstance(item, dict):
                        continue
                    compact.append(
                        {
                            "name": str(item.get("name") or ""),
                            "type": str(item.get("type") or ""),
                            "success": bool(item.get("success")),
                            "args": dict(item.get("args") or {}) if isinstance(item.get("args"), dict) else {},
                        }
                    )
                if compact:
                    lines.append("recent_steps=" + json.dumps(compact, ensure_ascii=False, sort_keys=True))
        lines.append(
            "Fresh-scope rule: after regeneration, the new code starts in a fresh Python module. "
            "Do not reference variables, helper state, or locals from any previous generated program."
        )
        lines.append(
            "Only continue from durable data exposed through ctx['resume_state'], ctx['runtime_state'], "
            "and agent_state/postconditions."
        )
        return "\n".join(lines).strip()

    def _build_suffix_repair_summary(self, *, context: PlannerContext) -> str:
        if not self._prompt_section_enabled("resume_state") or not self._prompt_section_enabled("feedback"):
            return ""

        runtime_payload = self._load_json_dict((context.memory_bundle or {}).get("cap_runtime_state", ""))
        lines = [
            "Suffix repair mode: keep the successful executed prefix fixed and repair only the remaining continuation.",
        ]
        last_result = runtime_payload.get("last_step_result")
        if isinstance(last_result, dict) and last_result:
            lines.append("last_step_result=" + json.dumps(last_result, ensure_ascii=False, sort_keys=True))
        recent_steps = runtime_payload.get("recent_steps")
        if isinstance(recent_steps, list) and recent_steps:
            compact: list[dict[str, Any]] = []
            for item in recent_steps[-6:]:
                if not isinstance(item, dict):
                    continue
                compact.append(
                    {
                        "name": str(item.get("name") or ""),
                        "type": str(item.get("type") or ""),
                        "success": bool(item.get("success")),
                        "error": str(item.get("error") or ""),
                        "args": dict(item.get("args") or {}) if isinstance(item.get("args"), dict) else {},
                    }
                )
            if compact:
                lines.append("recent_steps=" + json.dumps(compact, ensure_ascii=False, sort_keys=True))
        runtime_state = runtime_payload.get("runtime_state")
        if isinstance(runtime_state, dict):
            local_repair = runtime_state.get("_local_repair")
            if isinstance(local_repair, dict) and local_repair:
                lines.append("local_repair=" + json.dumps(local_repair, ensure_ascii=False, sort_keys=True))
        lines.append(
            "Repair code must be self-contained. Do not reuse variable names or helper locals from the previous policy version."
        )
        lines.append(
            "If you need old progress, read it again from ctx['resume_state'] / ctx['runtime_state']; "
            "never assume previous Python locals still exist."
        )
        return "\n".join(lines).strip()

    @staticmethod
    def _prompt_generation_reason(*, reason: str, feedback_enabled: bool) -> str:
        """Do not expose an execution failure through a no-feedback regeneration label."""
        raw = str(reason or "").strip()
        if not feedback_enabled and raw.lower().startswith("repair_suffix:"):
            return "continuation_regeneration"
        return raw or "initial"

    def _build_durable_runtime_summary(self, *, context: PlannerContext) -> str:
        if not self._prompt_section_enabled("resume_state"):
            return ""

        include_feedback = self._prompt_section_enabled("feedback")
        runtime_payload = self._load_json_dict((context.memory_bundle or {}).get("cap_runtime_state", ""))
        lines = [
            "Durable carry-over state only. Old temporary locals/custom_ctx from previous planning drafts are NOT available after regeneration.",
        ]
        runtime_state = runtime_payload.get("runtime_state")
        if isinstance(runtime_state, dict) and runtime_state:
            if not include_feedback:
                runtime_state = self._without_feedback_runtime_fields(runtime_state)
            if runtime_state:
                lines.append("runtime_state=" + json.dumps(runtime_state, ensure_ascii=False, sort_keys=True))
        postconditions = runtime_payload.get("postconditions")
        if isinstance(postconditions, dict) and postconditions:
            lines.append("postconditions=" + json.dumps(postconditions, ensure_ascii=False, sort_keys=True))
        if include_feedback:
            last_error = str(runtime_payload.get("last_error") or "").strip()
            if last_error:
                lines.append(f"last_error={last_error}")
        return "\n".join(lines).strip()

    @staticmethod
    def _candidate_name(item: dict[str, Any]) -> str:
        for key in ("name_en", "name", "name_cn"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _parse_instance_candidates(self, *, context: PlannerContext) -> list[dict[str, Any]]:
        candidates_raw = _feedback_line_value(context.feedback, "candidates")
        for raw in (
            candidates_raw,
            _feedback_line_value(str(context.memory_bundle.get("instance_query_snapshot", "") or ""), "auto_query_results"),
        ):
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            if isinstance(data, list):
                return [it for it in data if isinstance(it, dict)]
        return []

    def _parse_query_results(self, *, context: PlannerContext) -> list[dict[str, Any]]:
        raw = _feedback_line_value(context.feedback, "results")
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except Exception:
            return []
        if isinstance(data, list):
            return [it for it in data if isinstance(it, dict)]
        return []

    def _choose_query_result_fallback(self, *, context: PlannerContext) -> tuple[str, int]:
        candidates = self._parse_query_results(context=context)
        if not candidates:
            raise ValueError("cap_query_binding_no_candidates")

        ranked: list[tuple[tuple[int, int, float], str, int]] = []
        for item in candidates:
            name = self._candidate_name(item)
            instance_id = _safe_int(item.get("instance_id"))
            if not name or instance_id is None:
                continue
            try:
                distance = float(item.get("distance", 9999))
            except Exception:
                distance = 9999.0
            rank = (
                0 if bool(item.get("is_on_screen", False)) else 1,
                0 if str(item.get("container") or "").strip().lower() == "scene" else 1,
                distance,
            )
            ranked.append((rank, name, int(instance_id)))
        if not ranked:
            raise ValueError("cap_query_binding_no_valid_candidate")
        _, name, instance_id = min(ranked, key=lambda item: item[0])
        return name, instance_id

    def _choose_instance_fallback(self, *, context: PlannerContext) -> tuple[str, int]:
        target_name = _feedback_line_value(context.feedback, "instance_meta.name").lower()
        candidates = self._parse_instance_candidates(context=context)
        if not candidates:
            raise ValueError("cap_instance_selection_no_candidates")

        best_item: Optional[dict[str, Any]] = None
        best_score: Optional[tuple[int, float]] = None
        for item in candidates:
            name = self._candidate_name(item)
            instance_id = _safe_int(item.get("instance_id"))
            if not name or instance_id is None:
                continue
            score = 0
            if target_name and target_name in name.lower():
                score += 20
            if item.get("is_on_screen") is True:
                score += 100
            if str(item.get("container") or "").strip().lower() == "scene":
                score += 10
            distance = 9999.0
            try:
                distance = float(item.get("distance", 9999.0))
            except Exception:
                distance = 9999.0
            candidate_score = (score, -distance)
            if best_item is None or candidate_score > best_score:
                best_item = item
                best_score = candidate_score
        if best_item is None:
            raise ValueError("cap_instance_selection_no_valid_candidate")
        return self._candidate_name(best_item), int(best_item["instance_id"])

    @staticmethod
    def _python_literal(value: Any) -> str:
        return repr(value)

    @classmethod
    def _build_retry_code(*, context: PlannerContext, selected_name: str, selected_instance_id: int) -> str:
        step_type, step_name = _extract_failed_step_type_and_name(context.feedback)
        if step_type not in ("action", "skill") or not step_name:
            raise ValueError("cap_instance_selection_missing_failed_step")

        args_raw = _feedback_line_value(context.feedback, "args")
        args: dict[str, Any] = {}
        if args_raw:
            try:
                parsed = json.loads(args_raw)
                if isinstance(parsed, dict):
                    args = dict(parsed)
            except Exception:
                args = {}

        instance_arg = _feedback_line_value(context.feedback, "instance_meta.instance_arg")
        name_arg = _feedback_line_value(context.feedback, "instance_meta.name_arg")
        if not instance_arg:
            for key in list(args.keys()):
                if isinstance(key, str) and key.endswith("_instance_id"):
                    instance_arg = key
                    break
        if not instance_arg:
            raise ValueError("cap_instance_selection_missing_instance_arg")

        args[instance_arg] = int(selected_instance_id)
        if name_arg:
            args[name_arg] = str(selected_name)

        callee = "skill" if step_type == "skill" else "action"
        kw_parts = [f"{key}={CaPCodegen._python_literal(value)}" for key, value in args.items()]
        kwargs_text = (", " + ", ".join(kw_parts)) if kw_parts else ""
        return "def policy(ctx):\n" f"    yield {callee}({step_name!r}{kwargs_text})\n"

    def _generate_instance_selection_code(self, *, context: PlannerContext, reason: str) -> tuple[str, str]:
        if not self._prompt_section_enabled("feedback"):
            raise ValueError("cap_instance_selection_disabled_by_no_feedback")

        prompt_template = load_asset(
            "cap/codegen",
            "instance_selection.txt",
            (
                "You are in CaP instance selection mode.\n"
                "The previous step already failed because `instance_id` is missing or invalid.\n"
                "Choose exactly one candidate and return JSON only:\n"
                '{"name":"candidate name","instance_id":123}\n'
                "Selection rules:\n"
                "- Choose only from `candidates` or `auto_query_results`.\n"
                "- Prefer exact/near-exact name match to the requested target.\n"
                "- Prefer `is_on_screen=true`, then `container=Scene`, then smaller `distance`.\n"
                "- Do not output code, markdown, or a new plan.\n"
            ),
        ).strip()
        prompt_parts = [
            prompt_template,
            "",
            f"reason={reason}",
            "feedback:",
            str(context.feedback or "").strip(),
            "",
            "instance_query_snapshot:",
            str(context.memory_bundle.get("instance_query_snapshot", "") or "").strip() or "(none)",
        ]
        if self._prompt_section_enabled("query_memory"):
            prompt_parts.extend(
                [
                    "",
                    "query_memory:",
                    str(context.memory_bundle.get("query_memory", "") or "").strip() or "(none)",
                ]
            )
        prompt = "\n".join(prompt_parts).strip() + "\n"
        self._audit_prompt(source="cap.instance_selection", prompt=prompt)
        out = chat_complete_text(
            cfg=self.chat_cfg,
            prompt=prompt,
            screenshot_path=context.observation.screenshot_path,
            force_use_vision=bool(getattr(self.chat_cfg, "use_vision", False)),
        )
        raw_output = out.content
        try:
            obj = extract_json_object(raw_output, strip_think_tags=True)
            selected_name = str(obj.get("name") or "").strip()
            selected_instance_id = _safe_int(obj.get("instance_id"))
            if not selected_name or selected_instance_id is None:
                raise ValueError("cap_instance_selection_bad_json")
        except Exception:
            selected_name, selected_instance_id = self._choose_instance_fallback(context=context)
            raw_output = str(raw_output or "").rstrip() + (
                "\n\n[fallback] selected from heuristic\n"
                + json.dumps({"name": selected_name, "instance_id": selected_instance_id}, ensure_ascii=False)
            )
        code = self._build_retry_code(
            context=context,
            selected_name=selected_name,
            selected_instance_id=int(selected_instance_id),
        )
        return code, raw_output

    def select_query_binding(self, *, context: PlannerContext, reason: str) -> tuple[str, int, str]:
        if not self._prompt_section_enabled("feedback"):
            raise ValueError("cap_query_binding_disabled_by_no_feedback")

        prompt_template = load_asset(
            "cap/codegen",
            "query_binding_selection.txt",
            (
                "You are in CaP query-result binding mode.\n"
                "The previous step was `query_scene_objects` and it already returned non-empty results.\n"
                "Choose exactly one result that best matches the current target and return JSON only:\n"
                '{"name":"candidate name","instance_id":123}\n'
                "Selection rules:\n"
                "- Choose from `results` only.\n"
                "- Prefer the exact target name from the previous query when present.\n"
                "- Prefer visible / scene-level / more plausible task objects when needed.\n"
                "- Do not output code, markdown, or explanations.\n"
            ),
        ).strip()
        query_args_raw = _feedback_line_value(context.feedback, "args")
        current_query = ""
        if query_args_raw:
            try:
                parsed = json.loads(query_args_raw)
                if isinstance(parsed, dict):
                    current_query = str(parsed.get("query", "") or "").strip()
            except Exception:
                current_query = ""
        prompt = (
            prompt_template
            + "\n\n"
            + f"reason={reason}\n"
            + f"last_query={current_query!r}\n"
            + "feedback:\n"
            + str(context.feedback or "").strip()
            + "\n"
        )
        self._audit_prompt(source="cap.query_binding", prompt=prompt)
        out = chat_complete_text(
            cfg=self.chat_cfg,
            prompt=prompt,
            screenshot_path=context.observation.screenshot_path,
            force_use_vision=bool(getattr(self.chat_cfg, "use_vision", False)),
        )
        raw_output = out.content
        obj = extract_json_object(raw_output, strip_think_tags=True)
        selected_name = str(obj.get("name") or "").strip()
        selected_instance_id = _safe_int(obj.get("instance_id"))
        if not selected_name or selected_instance_id is None:
            raise ValueError("cap_query_binding_bad_json")
        return selected_name, int(selected_instance_id), raw_output

    @staticmethod
    def _comment_block(title: str, text: str) -> str:
        body = str(text or "").strip()
        lines = [f"# {title}"]
        if not body:
            lines.append("# (none)")
            return "\n".join(lines)
        for line in body.splitlines():
            if line.strip():
                lines.append(f"# {line}")
            else:
                lines.append("#")
        return "\n".join(lines)

    @staticmethod
    def _slugify_reason(reason: str) -> str:
        raw = str(reason or "").strip().lower()
        if not raw:
            return "unspecified"
        chars: list[str] = []
        prev_sep = False
        for ch in raw:
            if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
                chars.append(ch)
                prev_sep = False
                continue
            if not prev_sep:
                chars.append("_")
                prev_sep = True
        out = "".join(chars).strip("_")
        return out or "unspecified"

    @staticmethod
    def _strip_code_fences(code: str) -> str:
        text = str(code or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _write_generation_history(
        self,
        *,
        code: str,
        raw_output: str,
        reason: str,
        context: PlannerContext,
    ) -> None:
        try:
            hist_dir = self.memory_dir / "cap_history"
            hist_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            slug = self._slugify_reason(reason)
            policy_name = f"policy_{stamp}_{slug}.py"
            policy_path = hist_dir / policy_name
            policy_path.write_text(code.rstrip() + "\n", encoding="utf-8")

            raw_name = ""
            think_name = ""
            final_name = ""
            if bool(self.cfg.save_raw):
                raw_name = f"raw_{stamp}_{slug}.txt"
                raw_text = str(raw_output or "")
                (hist_dir / raw_name).write_text(raw_text, encoding="utf-8")
                think_name = f"think_{stamp}_{slug}.txt"
                final_name = f"final_{stamp}_{slug}.txt"
                think_text, final_text = split_think_and_final(raw_text)
                (hist_dir / think_name).write_text(think_text, encoding="utf-8")
                (hist_dir / final_name).write_text((final_text or raw_text.strip()), encoding="utf-8")

            event = {
                "timestamp": stamp,
                "reason": str(reason or ""),
                "policy_file": policy_name,
                "raw_file": raw_name,
                "think_file": think_name,
                "final_file": final_name,
                "goal": str(context.high_level_goal or ""),
                "high_level_id": str(context.high_level_id or ""),
            }
            with (hist_dir / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _write_prompt_snapshot(
        self,
        *,
        prompt_text: str,
        reason: str,
        attempt: int,
    ) -> None:
        try:
            latest_dir = self.memory_dir / "cap_prompts"
            latest_dir.mkdir(parents=True, exist_ok=True)
            latest_path = latest_dir / "codegen_last_prompt.txt"
            latest_path.write_text(str(prompt_text or ""), encoding="utf-8")

            hist_dir = self.memory_dir / "cap_history"
            hist_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            slug = self._slugify_reason(reason)
            hist_name = f"prompt_{stamp}_{slug}.attempt_{int(attempt)}.txt"
            (hist_dir / hist_name).write_text(str(prompt_text or ""), encoding="utf-8")
        except Exception:
            pass

    def generate(self, *, context: PlannerContext, reason: str = "initial", max_steps: int = 10) -> CapProgram:
        raw_dir = self.memory_dir / "cap_raw"
        if bool(self.cfg.save_raw):
            raw_dir.mkdir(parents=True, exist_ok=True)

        if _is_instance_disambiguation_feedback(context.feedback) and self._prompt_section_enabled("feedback"):
            code, content = self._generate_instance_selection_code(context=context, reason=reason)
            if bool(self.cfg.save_raw):
                (raw_dir / "codegen_last.txt").write_text(content, encoding="utf-8")
                write_trace_files_for_raw_dir(raw_dir=raw_dir, filename="codegen_last.txt", text=content)
        else:
            deadline_hint = self._force_submit_hint()
            interface_text = self._build_interface_reference(
                prompt_ablation_profile=self.cfg.prompt_ablation_profile,
                prompt_disabled_groups=self.cfg.prompt_disabled_groups,
            )
            shared_system_text, shared_user_text = self._build_shared_prompt_sections(context=context)
            feedback_enabled = self._prompt_section_enabled("feedback")
            prompt_reason = self._prompt_generation_reason(reason=reason, feedback_enabled=feedback_enabled)
            resume_text = self._build_resume_summary(context=context)
            repair_mode = feedback_enabled and str(reason or "").strip().lower().startswith("repair_suffix:")
            repair_text = self._build_suffix_repair_summary(context=context) if repair_mode else ""
            role_text = load_asset(
                "cap/codegen",
                "role.txt",
                (
                    "You are CaP (Code-as-Policies). Write Python planning code that expresses a short future action sequence.\n"
                    "This code is used for planning only; it is NOT executed as Python at runtime.\n"
                    "Write a single straight-line `policy(ctx)` function whose body is only comments plus sequential `yield action(...)` / `yield skill(...)` lines.\n"
                    "Every regeneration starts from a fresh Python module scope.\n"
                    "Never reference locals, helper variables, or temporary names from a previous generated policy.\n"
                    "Only use values defined in the current code block or read explicitly from ctx.\n"
                    "Prefer direct action/skill calls with explicit arguments over building many intermediate variables.\n"
                    "Use comments to express phase structure. Do NOT use loops, runtime branches, or stage variables.\n"
                    "Every yield argument must be a plain Python literal only: string, number, bool, null-like literal, list, tuple, or dict of literals.\n"
                    "Do NOT put ctx[...] / results[...] / runtime_state[...] / variable indexing / attribute access / function calls inside any yield argument.\n"
                    "Do not invent placeholder instance ids or binding syntax inside the code.\n"
                    "Do not assume the executor will auto-fill container instance ids for you.\n"
                    "Return ONLY a Python code block and no extra text."
                ),
            )
            api_text = load_asset(
                "cap/codegen",
                "api.txt",
                (
                    "API:\n"
                    "- yield action(name, **args)\n"
                    "- yield skill(name, **args)\n"
                    "- ctx is a dict with keys like: goal, feedback, on_screen_names, on_screen_objects, agent_state.\n"
                    "- The body of policy(ctx) must be a straight-line plan sketch, not executable control flow."
                ),
            )
            rules_text = load_asset(
                "cap/codegen",
                "rules.txt",
                (
                    "Rules:\n"
                    "- Do NOT import anything.\n"
                    "- Do NOT access files.\n"
                    "- Treat the policy as high-level control logic. Keep it simple and declarative.\n"
                    "- Every regeneration is a fresh Python file. Variables from older generated policies do NOT exist now.\n"
                    "- Never reference a name unless it is defined in the current code, provided as a function argument/local, or read from ctx.\n"
                    "- Never rely on previous temporary locals, previous helper functions, or previous convenience variables.\n"
                    "- Do NOT use `if`, `for`, `while`, `try`, helper functions, or any runtime control flow inside `policy(ctx)`.\n"
                    "- Do NOT introduce `stage` or many branch-control variables. Use comments instead.\n"
                    "- Use comments to mark phases/subtasks instead of storing those phase names in runtime variables.\n"
                    f"- Output a compact straight-line plan with about {int(max_steps)} steps when the task allows it.\n"
                    "- Every `yield action(...)` / `yield skill(...)` argument must be a plain literal. No expressions.\n"
                    "- Forbidden inside yield args: `ctx[...]`, `results[...]`, variable indexing, attribute access, arithmetic, function calls, or any computed expression.\n"
                    "- If you need information from feedback/query results, convert it into a comment and keep the yielded call itself literal-only.\n"
                    "- Do NOT invent placeholder instance ids, binding markers, or synthetic variable protocols.\n"
                    "- Do NOT assume agent_state/runtime_state container ids will be auto-filled by the executor.\n"
                    "- Keep the plan natural and let runtime disambiguation resolve missing instance ids when needed.\n"
                    "- Do NOT create convenience variables that only rename fixed strings/numbers if those values can be written directly in the call.\n"
                    "- Use ONLY exact action/skill names from the provided interface reference.\n"
                    "- Use ONLY kwargs that appear in the provided signatures/schema.\n"
                    "- Never invent high-level verbs like find/take/add_to; use the provided query/action/skill names instead.\n"
                    "- Define `policy(ctx)` and place only sequential yield statements inside it."
                ),
            )
            prefix_text = load_asset(
                "cap/codegen",
                "prefix.py.txt",
                (
                    "# Write Python code only.\n"
                    "# Define exactly one function: policy(ctx).\n"
                    "# policy(ctx) must contain a straight-line sequence of yield action(...) / yield skill(...).\n"
                    "# This is planning code, not executable runtime control flow.\n"
                    "# Regeneration always starts from a fresh Python scope.\n"
                    "# Never use variables from a previous generated policy; only use locals defined below or values read from ctx.\n"
                    "# Use comments for phase structure; avoid explicit stage variables.\n"
                    "# Do not use loops, branches, try/except, or helper functions.\n"
                    "# Prefer direct calls with explicit args over many convenience variables.\n"
                    "# Every yield arg must be a literal only. Do not use ctx[...] / results[...] / name[index] / obj.attr / function_call(...).\n"
                    "# Do not invent placeholder instance ids or custom binding syntax in the code.\n"
                ),
            )
            examples_text = load_asset(
                "cap/codegen",
                "examples.py.txt",
                (
                    "# Example 1\n"
                    "# Query, navigate, pick up, navigate, and put down.\n"
                    "def policy(ctx):\n"
                    "    # Acquire the target object.\n"
                    "    yield skill(\"query_scene_objects\", query=\"Chicken Breast\", only_on_screen=False)\n"
                    "    yield skill(\"auto_navigation\", target=\"Chicken Breast\")\n"
                    "    yield action(\"pick_up\")\n"
                    "    # Move to the placement point and release it.\n"
                    "    yield skill(\"auto_navigation\", target=\"Side Table\")\n"
                    "    yield action(\"put_down\")\n"
                    "\n"
                    "# Example 2\n"
                    "# Query, navigate, pick up, navigate, and pour.\n"
                    "def policy(ctx):\n"
                    "    # Acquire the liquid container.\n"
                    "    yield skill(\"query_scene_objects\", query=\"Olive Oil\", only_on_screen=False)\n"
                    "    yield skill(\"auto_navigation\", target=\"Olive Oil\")\n"
                    "    yield action(\"pick_up\")\n"
                    "    # Move to the pot and pour.\n"
                    "    yield skill(\"auto_navigation\", target=\"Big Pot\")\n"
                    "    yield skill(\"auto_pour\", container_name=\"Big Pot\", target_ml=\"10\")\n"
                ),
            )
            if restrict_to_raw_input_actions(
                self.cfg.prompt_ablation_profile,
                disabled_groups=self.cfg.prompt_disabled_groups,
            ) or restrict_to_actions_only(
                self.cfg.prompt_ablation_profile,
                disabled_groups=self.cfg.prompt_disabled_groups,
            ):
                # The bundled examples demonstrate high-level skills. Keeping
                # them would reintroduce a removed interface despite a filtered
                # schema, so omit them for no_skill/no_action variants.
                examples_text = ""

            current_task_lines = [
                "# Current task",
                f"# regeneration_reason: {prompt_reason}",
                f"# high_level_id: {context.high_level_id}",
                f"# high_level_goal: {context.high_level_goal}",
                "# Write the final code below.",
                "# Return only Python code. No markdown fences.",
                "# The final code must define only policy(ctx).",
                "# Fresh-scope rule: this regeneration does NOT inherit Python locals/helpers from any earlier generated policy.",
                "# Never reference previous variable names unless you redefine them in this code block.",
            ]
            if self._prompt_section_enabled("resume_state"):
                current_task_lines.append("# If old progress matters, read it from ctx['resume_state'], ctx['runtime_state'], or agent_state.")
            if self._prompt_section_enabled("feedback"):
                current_task_lines.append("# Feedback may describe the latest execution outcome; use it only as evidence.")
            if int(self.cfg.max_code_lines) > 0:
                current_task_lines.append(f"# Soft target: keep the final code compact, ideally within {int(self.cfg.max_code_lines)} lines.")
            if int(self.cfg.max_code_chars) > 0:
                current_task_lines.append(f"# Soft target: keep the final code compact, ideally within {int(self.cfg.max_code_chars)} characters.")
            current_task_lines.append(f"# Plan for up to {int(max_steps)} meaningful future steps when the task allows it.")
            current_task_lines.append("# Use only sequential yield statements; do not rely on runtime execution or branching.")
            if self._prompt_section_enabled("resume_state"):
                current_task_lines.append("# Resume from the existing execution prefix; do not restart the recipe from the beginning after regeneration.")
                current_task_lines.append("# If the dish is already ordered in resume_state/postconditions/agent_state, do NOT call gui_order_dish_via_computer again.")
            current_task_lines.append("# Use countdown_time only for explicit waiting/timer semantics, never as a generic fallback.")
            if self._prompt_section_enabled("feedback"):
                current_task_lines.append("# Do not repeat the exact same successful step unless the latest feedback proves the world state changed.")
            current_task_lines.append("# Use comments for phase structure; do not store phase names in runtime state unless unavoidable.")
            current_task_lines.append("# Prefer direct action/skill calls with explicit args; avoid convenience variables that merely rename fixed strings/numbers.")
            current_task_lines.append("# Avoid explicit stage variables and avoid helper locals unless they are absolutely necessary.")
            if self._prompt_section_enabled("resume_state") or self._prompt_section_enabled("feedback"):
                current_task_lines.append("# Reconstruct available progress from the supplied runtime context instead of maintaining many fragile locals.")
            current_task_lines.append("# Every yield argument must be a plain literal only.")
            current_task_lines.append("# Forbidden in yield args: ctx[...], results[...], runtime_state[...], variable indexing, obj.attr, function calls, arithmetic, or any computed expression.")
            current_task_lines.append("# If you want to mention supplied context, put it in a comment, not in the yielded call arguments.")
            current_task_lines.append("# Do not invent placeholder instance ids or custom binding syntax.")
            current_task_lines.append("# Do not assume the executor will auto-fill missing container instance ids from agent_state.")
            current_task_lines.append("# Keep the plan natural and let runtime disambiguation handle missing instance ids when needed.")
            if repair_mode:
                current_task_lines.append("# Suffix repair mode: do NOT rewrite the successful executed prefix.")
                current_task_lines.append("# Only repair the continuation after the latest failed step.")
                current_task_lines.append("# The first new yield should address the immediate blocker/error from the latest failed step.")
                current_task_lines.append("# Do not re-emit recent successful steps unless the latest feedback proves they must be retried.")
            if deadline_hint:
                current_task_lines.append(f"# deadline_hint: {deadline_hint}")
            prompt_parts = [
                    self._comment_block("Shared planner context", shared_system_text),
                    self._comment_block("Exact tool reference", interface_text),
                    self._comment_block("Role", role_text),
                    self._comment_block("API", api_text),
                    self._comment_block("Rules", rules_text),
                    prefix_text.strip(),
                    examples_text.strip(),
                    self._comment_block("Runtime observation and feedback", shared_user_text or f"Goal: {context.high_level_goal}\nFull recipe:\n{(context.recipe_text or '').strip()}"),
            ]
            if self._prompt_section_enabled("resume_state"):
                prompt_parts.append(self._comment_block("Resume and progress state", resume_text))
                prompt_parts.append(self._comment_block("Durable persisted state", self._build_durable_runtime_summary(context=context)))
            if repair_mode and self._prompt_section_enabled("feedback"):
                prompt_parts.append(self._comment_block("Suffix repair context", repair_text))
            prompt_parts.append("\n".join(current_task_lines))
            prompt = "\n\n".join(part for part in prompt_parts if str(part or "").strip()).strip() + "\n"
            if not self._skill_interface_enabled():
                prompt = self._strip_skill_interface_references(prompt) + "\n"

            last_error = ""
            code = ""
            content = ""
            extracted_steps: list[dict[str, Any]] | None = None
            prompt_for_attempt = prompt
            for attempt in range(2):
                if bool(self.cfg.save_raw):
                    self._write_prompt_snapshot(
                        prompt_text=prompt_for_attempt,
                        reason=reason,
                        attempt=(attempt + 1),
                    )
                self._audit_prompt(source="cap.codegen", prompt=prompt_for_attempt)
                out = chat_complete_text(
                    cfg=self.chat_cfg,
                    prompt=prompt_for_attempt,
                    screenshot_path=context.observation.screenshot_path,
                    force_use_vision=bool(getattr(self.chat_cfg, "use_vision", False)),
                )
                content = out.content
                if bool(self.cfg.save_raw):
                    (raw_dir / "codegen_last.txt").write_text(content, encoding="utf-8")
                    write_trace_files_for_raw_dir(raw_dir=raw_dir, filename="codegen_last.txt", text=content)
                code = self._strip_code_fences(content)
                try:
                    extracted_steps = _extract_plan_steps_from_code(code, max_steps=max_steps)
                    break
                except Exception as e:
                    last_error = _humanize_plan_extract_error(e)
                    prompt_for_attempt = (
                        prompt
                        + "\n# Previous draft was invalid for plan extraction.\n"
                        + f"# validation_error: {last_error}\n"
                        + "# Rewrite policy(ctx) as straight-line planning code with only comments and sequential yield statements.\n"
                    )
            if extracted_steps is None:
                raise ValueError(f"cap_plan_extract_failed:{last_error}")
        code = self._strip_code_fences(code)
        if _is_instance_disambiguation_feedback(context.feedback) and self._prompt_section_enabled("feedback"):
            extracted_steps = _extract_plan_steps_from_code(code, max_steps=max_steps)
        assert extracted_steps is not None
        ctx: dict[str, Any] = {}

        # Save the planning code for reproducibility.
        try:
            (self.memory_dir / "cap_policy.py").write_text(code.rstrip() + "\n", encoding="utf-8")
        except Exception:
            pass
        self._write_generation_history(code=code, raw_output=content, reason=reason, context=context)

        return CapProgram(code=code, plan_steps=extracted_steps, ctx=ctx)

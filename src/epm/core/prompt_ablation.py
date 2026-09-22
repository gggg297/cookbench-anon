from __future__ import annotations

"""Named, reproducible prompt-ablation profiles for the main EPM experiment."""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable


PROMPT_ABLATION_PROFILES: Final[tuple[str, ...]] = (
    "full",
    "no_body",
    "no_perception_sup",
    "no_strategy",
    "no_history",
    "no_feedback",
    "no_rgb",
    "no_skill",
    "no_action",
)

# The no_action baseline may control only the game-facing raw input primitives.
# Keep this deliberately smaller than RawInputController's full Windows surface:
# no absolute cursor positioning, text entry, window control, or stop hotkeys.
RAW_INPUT_ACTIONS: Final[tuple[str, ...]] = (
    "press_keyboard",
    "hold_keyboard",
    "leave_keyboard",
    "click_mouse",
    "hold_mouse",
    "leave_mouse",
    "move_related_mouse",
    "scroll_down_the_wheel",
    "scroll_up_the_wheel",
)

_DISABLED_GROUP_BY_PROFILE: Final[dict[str, frozenset[str]]] = {
    "full": frozenset(),
    "no_body": frozenset({"body"}),
    "no_perception_sup": frozenset({"perception_support"}),
    "no_strategy": frozenset({"strategy"}),
    "no_history": frozenset({"history"}),
    "no_feedback": frozenset({"feedback"}),
    # Keep text and structured prompt support, but never attach an RGB
    # screenshot to any model request.
    "no_rgb": frozenset({"rgb_input"}),
    "no_skill": frozenset({"skill_interface"}),
    # `no_action` removes the semantic action/skill interface. The runner still
    # exposes a deliberately small raw keyboard/mouse control interface.
    "no_action": frozenset({"semantic_action_interface"}),
}

# Task, recipe, and output protocol are always retained. For `no_action`, semantic
# actions and skills are removed by the executor allowlist; the raw-control manifest
# remains visible so the model can issue primitive keyboard/mouse commands.
SECTION_GROUPS: Final[dict[str, str]] = {
    "body_rules": "body",
    "strategy_notes": "strategy",
    "tool_interaction_point_list": "perception_support",
    "put_place_occupancy": "perception_support",
    "oracle_observation": "perception_support",
    "query_memory": "history",
    "resume_state": "history",
    "feedback": "feedback",
    "precondition_feedback": "feedback",
    "tools_manifest_openai": "action_interface",
    "skill_cards": "action_interface",
    "parameter_heuristics": "action_interface",
    "skill_specs": "action_interface",
    "stm_window": "history",
    "task_progress": "history",
    "task_progress_feedback": "history",
    "reflexion_memory": "history",
    "reflexion_progress_memory": "history",
}


@dataclass(frozen=True)
class ResolvedPromptAblation:
    """Immutable reduction policy resolved once for an experiment run."""

    profile: str
    disabled_groups: frozenset[str]
    reductions: tuple[str, ...] = ()


def _normalize_component_name(value: object) -> str:
    name = str(value or "").strip().lower().replace("-", "_")
    try:
        return _COMPONENT_ALIASES[name]
    except KeyError as exc:
        allowed = ", ".join(PROMPT_COMPONENT_GROUPS)
        raise ValueError(f"invalid prompt component {value!r}; expected one of: {allowed}") from exc


def resolve_ablation_reductions(value: object = None) -> ResolvedPromptAblation:
    """Resolve the sole CLI form: ``--ablation-reduce no_body,no_strategy``."""
    raw_tokens = [token.strip() for token in str(value or "").split(",") if token.strip()]
    reductions: list[str] = []
    disabled: set[str] = set()
    for raw in raw_tokens:
        name = normalize_prompt_ablation_profile(raw)
        if name == "full":
            raise ValueError("--ablation-reduce accepts only no_* reduction labels, not 'full'")
        if name not in reductions:
            reductions.append(name)
            disabled.update(_DISABLED_GROUP_BY_PROFILE[name])
    return ResolvedPromptAblation(
        profile="full",
        disabled_groups=frozenset(disabled),
        reductions=tuple(reductions),
    )


def normalize_disabled_prompt_groups(groups: Iterable[object] | None) -> frozenset[str]:
    if groups is None:
        return frozenset()
    return frozenset(_normalize_component_name(group) for group in groups)


def is_prompt_group_enabled(
    *,
    profile: object,
    group: object,
    disabled_groups: Iterable[object] | None = None,
) -> bool:
    disabled = (
        normalize_disabled_prompt_groups(disabled_groups)
        if disabled_groups is not None
        else disabled_prompt_groups(profile)
    )
    return _normalize_component_name(group) not in disabled

# Experiment-level components. prompt_layout.json controls ordering, while this
# policy determines whether a component is permitted to appear at all.
PROMPT_COMPONENT_GROUPS: Final[tuple[str, ...]] = (
    "body",
    "perception_support",
    "strategy",
    "history",
    "feedback",
    "rgb_input",
    "skill_interface",
    "semantic_action_interface",
)

_COMPONENT_ALIASES: Final[dict[str, str]] = {
    "body": "body",
    "perception": "perception_support",
    "perception_support": "perception_support",
    "spatial": "perception_support",
    "strategy": "strategy",
    "history": "history",
    "feedback": "feedback",
    "rgb": "rgb_input",
    "rgb_input": "rgb_input",
    "skill": "skill_interface",
    "skills": "skill_interface",
    "skill_interface": "skill_interface",
    "action": "semantic_action_interface",
    "actions": "semantic_action_interface",
    "semantic_action": "semantic_action_interface",
    "semantic_action_interface": "semantic_action_interface",
    # Internal PromptBuilder section: interface visibility remains enabled for
    # no_action because it must expose the raw keyboard/mouse schema.
    "action_interface": "action_interface",
}


def normalize_prompt_ablation_profile(value: object) -> str:
    profile = str(value or "full").strip().lower().replace("-", "_")
    # Preserve existing experiment configs while exposing the clearer profile name.
    if profile == "no_spatial":
        profile = "no_perception_sup"
    if profile not in _DISABLED_GROUP_BY_PROFILE:
        allowed = ", ".join(PROMPT_ABLATION_PROFILES)
        raise ValueError(f"invalid prompt_ablation profile {value!r}; expected one of: {allowed}")
    return profile


def disabled_prompt_groups(profile: object) -> frozenset[str]:
    return _DISABLED_GROUP_BY_PROFILE[normalize_prompt_ablation_profile(profile)]


def restrict_to_raw_input_actions(
    profile: object,
    *,
    disabled_groups: Iterable[object] | None = None,
) -> bool:
    """Whether the profile replaces semantic actions/skills with raw input primitives."""
    return not is_prompt_group_enabled(
        profile=profile,
        group="semantic_action_interface",
        disabled_groups=disabled_groups,
    )


def restrict_to_actions_only(
    profile: object,
    *,
    disabled_groups: Iterable[object] | None = None,
) -> bool:
    """Whether the profile removes all high-level skills but keeps semantic actions."""
    return (
        not is_prompt_group_enabled(
            profile=profile,
            group="skill_interface",
            disabled_groups=disabled_groups,
        )
        and not restrict_to_raw_input_actions(profile, disabled_groups=disabled_groups)
    )


def is_prompt_section_enabled(
    *,
    profile: object,
    section: str,
    disabled_groups: Iterable[object] | None = None,
) -> bool:
    group = SECTION_GROUPS.get(str(section or "").strip().lower())
    return group is None or is_prompt_group_enabled(
        profile=profile,
        group=group,
        disabled_groups=disabled_groups,
    )


# Structural markers are deliberately narrow. They catch a disabled prompt block
# being reintroduced by a bypass path without treating ordinary words such as
# "history" or "feedback" in generic instructions as a false violation.
_AUDIT_MARKERS: Final[dict[str, tuple[str, ...]]] = {
    "body": ("Body rules (global constraints):", "# Body Rules (Embodiment Rules)"),
    "strategy": ("Strategy notes (planning guidance and heuristics):", "# Strategy Notes"),
    "perception_support": (
        "Tool interaction point list:",
        "Placement occupancy (oracle snapshot):",
        "Oracle visible objects (count=",
    ),
    "history": (
        "Resume state snapshot:",
        "STM window snapshot:",
        "Task progress hard constraints:",
        "Task progress snapshot:",
        "query_memory:",
        "Recent STM window:",
        "Recent stable progress facts:",
        "Existing reflexion memory:",
        "Resume and progress state",
        "Durable persisted state",
    ),
    "feedback": (
        "Feedback from last execution:",
        "Precondition feedback:",
        "Latest feedback:",
        "Last feedback string:",
        "feedback:\n",
        "    result_summary:",
        "    errors:",
        "regeneration_reason: repair_suffix:",
    ),
    # `no_skill` is an interface-availability ablation, not an executor-only
    # denylist. Any remaining skill wording teaches the unavailable API.
    "skill_interface": ("skill",),
}


def audit_prompt_text(
    *,
    profile: object,
    disabled_groups: Iterable[object] | None,
    source: str,
    prompt_text: str,
) -> dict[str, object]:
    """Return a serializable audit of the final prompt sent to a model."""
    resolved_profile = normalize_prompt_ablation_profile(profile)
    effective_disabled = (
        normalize_disabled_prompt_groups(disabled_groups)
        if disabled_groups is not None
        else disabled_prompt_groups(resolved_profile)
    )
    prompt = str(prompt_text or "")
    violations: list[dict[str, str]] = []
    for group in sorted(effective_disabled):
        for marker in _AUDIT_MARKERS.get(group, ()):
            if marker in prompt:
                violations.append({"group": group, "marker": marker})
    return {
        "timestamp": time.time(),
        "profile": resolved_profile,
        "disabled_groups": sorted(effective_disabled),
        "source": str(source or "unknown"),
        "prompt_chars": len(prompt),
        "violations": violations,
        "ok": not violations,
    }


def record_prompt_audit(result: dict[str, object], *, memory_dir: Path | str | None = None) -> None:
    """Append a prompt-audit event and retain a run-level invalid marker."""
    root_text = str(memory_dir or os.environ.get("EPM_RUN_MEMORY_DIR", "") or "").strip()
    if not root_text:
        return
    try:
        root = Path(root_text).resolve()
        root.mkdir(parents=True, exist_ok=True)
        with (root / "prompt_ablation_audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        if not bool(result.get("ok", False)):
            (root / "prompt_ablation_invalid.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    except Exception:
        # Prompt generation must not fail solely because diagnostic storage is unavailable.
        return


def assert_prompt_ablation(
    *,
    profile: object,
    disabled_groups: Iterable[object] | None,
    source: str,
    prompt_text: str,
    memory_dir: Path | str | None = None,
) -> None:
    result = audit_prompt_text(
        profile=profile,
        disabled_groups=disabled_groups,
        source=source,
        prompt_text=prompt_text,
    )
    record_prompt_audit(result, memory_dir=memory_dir)
    if not bool(result["ok"]):
        details = ", ".join(
            f"{item['group']}:{item['marker']}" for item in result["violations"]  # type: ignore[index]
        )
        raise RuntimeError(f"prompt_ablation_violation source={source} details={details}")


def audit_prompt_trace_directory(
    *,
    memory_dir: Path | str,
    profile: object,
    disabled_groups: Iterable[object] | None,
) -> dict[str, object]:
    """Scan all recorded final prompts and return a run-level audit summary."""
    root = Path(memory_dir).resolve()
    trace_dir = root / "model_call_traces"
    events: list[dict[str, object]] = []
    if trace_dir.exists():
        for path in sorted(trace_dir.glob("*.prompt.txt")):
            result = audit_prompt_text(
                profile=profile,
                disabled_groups=disabled_groups,
                source=f"trace:{path.name}",
                prompt_text=path.read_text(encoding="utf-8", errors="replace"),
            )
            if "rgb_input" in normalize_disabled_prompt_groups(disabled_groups):
                meta_path = path.with_name(path.name.replace(".prompt.txt", ".meta.json"))
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
                screenshot_paths = meta.get("screenshot_paths") if isinstance(meta, dict) else []
                if isinstance(screenshot_paths, list) and any(str(item or "").strip() for item in screenshot_paths):
                    result["violations"].append({"group": "rgb_input", "marker": "trace.screenshot_paths"})
                    result["ok"] = False
            events.append(result)
    violations = [item for item in events if not bool(item.get("ok", False))]
    summary = {
        "profile": normalize_prompt_ablation_profile(profile),
        "disabled_groups": sorted(normalize_disabled_prompt_groups(disabled_groups)),
        "trace_count": len(events),
        "violation_count": len(violations),
        "ok": not violations,
        "violations": violations,
    }
    try:
        (root / "prompt_ablation_trace_audit.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass
    return summary

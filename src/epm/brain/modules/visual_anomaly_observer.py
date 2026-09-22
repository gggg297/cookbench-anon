from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from epm.brain.chat_client import _normalize_timeout, chat_complete_text
from epm.brain.http_qwen3vl import Qwen3VlHttpClient
from epm.brain.model_output_trace import default_model_trace_dir, write_model_call_trace, write_trace_files_for_raw_dir
from epm.brain.modules.prompt_assets import load_asset
from epm.brain.plan_schema import PlanStep, extract_json_object
from epm.brain.provider_compat import ANTHROPIC_PROVIDER_TYPES, OPENAI_PROVIDER_TYPES
from epm.core.epm_types import Observation
from epm.core.settings import VlmSettings


def _slice_stm_window_text_by_steps(text: str, *, history_window_steps: int) -> str:
    raw = str(text or "").strip()
    limit = max(0, int(history_window_steps or 0))
    if limit <= 0 or not raw:
        return raw
    lines = raw.splitlines()
    step_start_indices: list[int] = []
    for idx, line in enumerate(lines):
        if line.startswith("  - step_id: "):
            step_start_indices.append(idx)
    if len(step_start_indices) <= limit:
        return raw
    keep_from = step_start_indices[-limit]
    header = lines[:keep_from]
    kept = lines[keep_from:]
    return "\n".join(header + kept).strip()


def _truncate(text: Any, *, limit: int) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: limit - 3] + "..."


@dataclass(frozen=True)
class VisualAnomalyObservation:
    has_anomaly: bool
    severity: str = "none"
    tags: list[str] | None = None
    feedback: str = ""
    expected_visible: str = ""
    observed_issue: str = ""
    confidence: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class VisualAnomalyObserverConfig:
    enabled: bool = False
    vlm: Optional[VlmSettings] = None
    max_retries: int = 0
    strip_think_tags: bool = True
    history_window_steps: int = 4
    save_prompt_dir: Optional[Path] = None
    save_raw_dir: Optional[Path] = None


class VisualAnomalyObserver:
    """
    Lightweight post-step multimodal anomaly observer.

    The observer uses the current post screenshot plus recent history text to detect
    visually evident issues that matter for the next planning round, such as spill,
    drop, occlusion, unstable placement, or target mismatch.
    """

    def __init__(self, cfg: VisualAnomalyObserverConfig) -> None:
        self.cfg = cfg
        if not self.cfg.enabled:
            return
        if self.cfg.vlm is None:
            raise ValueError("VisualAnomalyObserver enabled but vlm settings missing")
        provider = str(self.cfg.vlm.provider or "").strip().lower()
        if provider != "qwen3vl_http" and provider not in OPENAI_PROVIDER_TYPES and provider not in ANTHROPIC_PROVIDER_TYPES:
            raise ValueError(f"VisualAnomalyObserver unsupported provider: {provider!r}")
        self._provider = provider
        self.client: Optional[Qwen3VlHttpClient] = None
        if provider == "qwen3vl_http":
            self.client = Qwen3VlHttpClient(
                base_url=self.cfg.vlm.base_url,
                connect_timeout_s=5.0,
                read_timeout_s=_normalize_timeout(float(self.cfg.vlm.timeout_s)),
            )

    @staticmethod
    def _step_tag(obs: Observation) -> str:
        fid = str(getattr(obs, "frame_id", "") or "").strip()
        if fid.isdigit():
            return f"step_{int(fid):06d}"
        return fid or "step_unknown"

    @staticmethod
    def _on_screen(obs: Observation) -> list[dict[str, Any]]:
        state = getattr(obs, "state", {}) or {}
        items = state.get("_on_screen_objects", [])
        return items if isinstance(items, list) else []

    def observe(
        self,
        *,
        step: PlanStep,
        pre: Observation,
        post: Observation,
        recent_history_text: str,
        current_high_level_goal: str,
        last_feedback: str,
        agent_state_text: str,
    ) -> VisualAnomalyObservation:
        if not self.cfg.enabled:
            return VisualAnomalyObservation(has_anomaly=False, severity="none", tags=[], feedback="", raw=None)

        prompt = self._build_prompt(
            step=step,
            pre=pre,
            post=post,
            recent_history_text=recent_history_text,
            current_high_level_goal=current_high_level_goal,
            last_feedback=last_feedback,
            agent_state_text=agent_state_text,
        )
        tag = self._step_tag(post)

        if self.cfg.save_prompt_dir is not None:
            try:
                self.cfg.save_prompt_dir.mkdir(parents=True, exist_ok=True)
                (self.cfg.save_prompt_dir / f"{tag}.txt").write_text(prompt, encoding="utf-8")
            except Exception:
                pass

        image_paths = []
        if self.cfg.vlm and bool(self.cfg.vlm.use_vision):
            if pre.screenshot_path:
                image_paths.append(pre.screenshot_path)
            if post.screenshot_path and post.screenshot_path not in image_paths:
                image_paths.append(post.screenshot_path)

        last_err: Optional[Exception] = None
        last_text = ""
        for attempt in range(int(self.cfg.max_retries) + 1):
            try:
                current_prompt = prompt if attempt == 0 else self._repair_prompt(previous_output=last_text)
                if self._provider == "qwen3vl_http":
                    assert self.client is not None
                    last_text = self.client.chat(
                        prompt=current_prompt,
                        image_paths=image_paths,
                        max_new_tokens=int(self.cfg.vlm.max_tokens) if self.cfg.vlm else None,
                    )
                    try:
                        write_model_call_trace(
                            trace_dir=default_model_trace_dir(request_metrics_path=(self.cfg.vlm.request_metrics_path if self.cfg.vlm else None)),
                            call_name="visual_anomaly_observer_qwen3vl_http",
                            prompt_text=current_prompt,
                            response_text=last_text,
                            provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                            model=str(getattr(self.cfg.vlm, "model", "") or ""),
                            attempt=int(attempt),
                            screenshot_paths=image_paths,
                        )
                    except Exception:
                        pass
                else:
                    completion = chat_complete_text(
                        cfg=self.cfg.vlm,
                        prompt=current_prompt,
                        screenshot_paths=image_paths,
                    )
                    last_text = completion.content

                if self.cfg.save_raw_dir is not None:
                    try:
                        self.cfg.save_raw_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"{tag}.txt"
                        (self.cfg.save_raw_dir / filename).write_text(last_text, encoding="utf-8")
                        write_trace_files_for_raw_dir(raw_dir=self.cfg.save_raw_dir, filename=filename, text=last_text)
                    except Exception:
                        pass

                obj = extract_json_object(last_text, strip_think_tags=bool(self.cfg.strip_think_tags))
                return self._parse(obj)
            except Exception as e:
                try:
                    write_model_call_trace(
                        trace_dir=default_model_trace_dir(request_metrics_path=(self.cfg.vlm.request_metrics_path if self.cfg.vlm else None)),
                        call_name="visual_anomaly_observer_qwen3vl_http" if self._provider == "qwen3vl_http" else "visual_anomaly_observer_chat_complete",
                        prompt_text=current_prompt if "current_prompt" in locals() else prompt,
                        response_text="",
                        error_text=repr(e),
                        provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                        model=str(getattr(self.cfg.vlm, "model", "") or ""),
                        attempt=int(attempt),
                        screenshot_paths=image_paths,
                    )
                except Exception:
                    pass
                last_err = e
                time.sleep(0.2 * (attempt + 1))

        return VisualAnomalyObservation(
            has_anomaly=False,
            severity="none",
            tags=[],
            feedback="",
            raw={"raw_text": last_text, "error": repr(last_err)},
        )

    @staticmethod
    def _parse(obj: Dict[str, Any]) -> VisualAnomalyObservation:
        has_anomaly = bool(obj.get("has_anomaly", False))
        severity = str(obj.get("severity") or ("low" if has_anomaly else "none")).strip().lower()
        if severity not in {"none", "low", "medium", "high"}:
            severity = "low" if has_anomaly else "none"
        tags_raw = obj.get("tags", [])
        tags = [str(x).strip() for x in (tags_raw if isinstance(tags_raw, list) else []) if str(x).strip()]
        feedback = _truncate(obj.get("feedback"), limit=240)
        expected_visible = _truncate(obj.get("expected_visible"), limit=200)
        observed_issue = _truncate(obj.get("observed_issue"), limit=200)
        conf = obj.get("confidence", None)
        if conf is not None and not isinstance(conf, (int, float)):
            conf = None
        if not has_anomaly:
            severity = "none"
            if not feedback:
                feedback = ""
        return VisualAnomalyObservation(
            has_anomaly=has_anomaly,
            severity=severity,
            tags=tags,
            feedback=feedback,
            expected_visible=expected_visible,
            observed_issue=observed_issue,
            confidence=(float(conf) if conf is not None else None),
            raw=obj,
        )

    def _build_prompt(
        self,
        *,
        step: PlanStep,
        pre: Observation,
        post: Observation,
        recent_history_text: str,
        current_high_level_goal: str,
        last_feedback: str,
        agent_state_text: str,
    ) -> str:
        pre_items = self._on_screen(pre)
        post_items = self._on_screen(post)
        recent_history = _slice_stm_window_text_by_steps(
            _truncate(recent_history_text, limit=6000),
            history_window_steps=int(self.cfg.history_window_steps),
        )
        agent_state = _truncate(agent_state_text, limit=2000)
        last_fb = _truncate(last_feedback, limit=1200)
        role_text = load_asset(
            "visual_anomaly_observer",
            "role.txt",
            (
                "You are a visual anomaly observer for an embodied CookingSimulator agent.\n"
                "Infer what should probably be visible after the executed step from recent history and the step intent, then compare it with the current post-step evidence."
            ),
        )
        rules_text = load_asset(
            "visual_anomaly_observer",
            "rules.txt",
            (
                "Focus only on anomalies that matter for the next planning round, such as:\n"
                "- spill, overflow, bottle tipped or protruding from container\n"
                "- item fell, missing after transport, or unstable placement risk\n"
                "- held item or camera occludes the needed workspace\n"
                "- target/workspace not really aligned even if the action nominally succeeded\n"
                "- interaction mode appears visually inconsistent with the intended state\n"
                "\n"
                "Rules:\n"
                "- Prefer has_anomaly=false when evidence is weak.\n"
                "- Keep feedback short and actionable for the planner.\n"
                "- Use the screenshot plus provided textual context only.\n"
                "- Output MUST be a single JSON object and NOTHING else."
            ),
        )
        output_text = load_asset(
            "visual_anomaly_observer",
            "output_format.txt",
            (
                "Output schema:\n"
                "{\n"
                '  "has_anomaly": false,\n'
                '  "severity": "none",\n'
                '  "tags": [],\n'
                '  "feedback": "",\n'
                '  "expected_visible": "",\n'
                '  "observed_issue": "",\n'
                '  "confidence": 0.0\n'
                "}"
            ),
        )
        return (
            "SYSTEM:\n"
            f"{role_text}\n"
            "\n"
            f"{rules_text}\n"
            "\n"
            f"{output_text}\n"
            "\n"
            "USER:\n"
            f"Current high-level goal: {current_high_level_goal!r}\n"
            f"Executed step: name={step.name} type={step.type} args={json.dumps(step.args or {}, ensure_ascii=False)} expectation={step.expectation!r}\n"
            "\n"
            "Recent history (latest window):\n"
            f"{recent_history or '(none)'}\n"
            "\n"
            "Last feedback before this step:\n"
            f"{last_fb or '(none)'}\n"
            "\n"
            "Agent state snapshot:\n"
            f"{agent_state or '(none)'}\n"
            "\n"
            "Pre on-screen objects:\n"
            f"{pre_items}\n"
            "\n"
            "Post on-screen objects:\n"
            f"{post_items}\n"
            "\n"
            "Two screenshot images are provided in this exact order when available:\n"
            "1. pre-step screenshot\n"
            "2. post-step screenshot\n"
            "\n"
            f"Pre screenshot_path: {pre.screenshot_path}\n"
            f"Post screenshot_path: {post.screenshot_path}\n"
        )

    @staticmethod
    def _repair_prompt(*, previous_output: str) -> str:
        prev = (previous_output or "").strip()
        if len(prev) > 2000:
            prev = prev[:2000] + "..."
        return (
            "Rewrite your answer to be a SINGLE JSON object and NOTHING else.\n"
            "- No markdown.\n"
            "- No commentary.\n"
            "- No <think>...</think>.\n"
            "- Use valid JSON (double quotes, no trailing commas).\n"
            "\n"
            "Schema:\n"
            "{\n"
            '  "has_anomaly": false,\n'
            '  "severity": "none",\n'
            '  "tags": [],\n'
            '  "feedback": "",\n'
            '  "expected_visible": "",\n'
            '  "observed_issue": "",\n'
            '  "confidence": 0.0\n'
            "}\n"
            "\n"
            "Previous response (invalid):\n"
            f"{prev}\n"
        )

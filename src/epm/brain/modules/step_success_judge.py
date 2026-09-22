from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from epm.brain.chat_client import _normalize_timeout, chat_complete_text
from epm.brain.http_qwen3vl import Qwen3VlHttpClient
from epm.brain.model_output_trace import default_model_trace_dir, write_model_call_trace, write_trace_files_for_raw_dir
from epm.brain.modules.prompt_assets import load_asset
from epm.brain.plan_schema import extract_json_object
from epm.brain.plan_schema import PlanStep
from epm.brain.provider_compat import ANTHROPIC_PROVIDER_TYPES, OPENAI_PROVIDER_TYPES
from epm.core.epm_types import Observation
from epm.core.settings import VlmSettings


@dataclass(frozen=True)
class StepSuccessJudgement:
    success: bool
    reason: str = ""
    confidence: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class StepSuccessJudgeConfig:
    enabled: bool = False
    vlm: Optional[VlmSettings] = None
    max_retries: int = 1
    strip_think_tags: bool = True
    save_prompt_dir: Optional[Path] = None
    save_raw_dir: Optional[Path] = None


class StepSuccessJudge:
    """
    Let the model judge whether the last action succeeded based on realtime on-screen feedback.

    Intended semantics:
    - Execution API success means "the call did not crash".
    - The judge decides "did it achieve the intended effect?" using:
      - last step (name/args/expectation)
      - pre/post `state._on_screen_objects` (is_on_screen==true)
      - pre/post state_keys (optional)
      - post screenshot (optional, if available)
    """

    def __init__(self, cfg: StepSuccessJudgeConfig) -> None:
        self.cfg = cfg
        if not self.cfg.enabled:
            return
        if self.cfg.vlm is None:
            raise ValueError("StepSuccessJudge enabled but vlm settings missing")
        provider = str(self.cfg.vlm.provider or "").strip().lower()
        if provider != "qwen3vl_http" and provider not in OPENAI_PROVIDER_TYPES and provider not in ANTHROPIC_PROVIDER_TYPES:
            raise ValueError(f"StepSuccessJudge unsupported provider: {provider!r}")
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

    def judge(
        self,
        *,
        step: PlanStep,
        pre: Observation,
        post: Observation,
        api_success: bool,
        api_error: str,
    ) -> StepSuccessJudgement:
        if not self.cfg.enabled:
            return StepSuccessJudgement(success=bool(api_success), reason=api_error or "", raw=None)

        prompt = self._build_prompt(step=step, pre=pre, post=post, api_success=api_success, api_error=api_error)
        tag = self._step_tag(pre)

        if self.cfg.save_prompt_dir is not None:
            try:
                self.cfg.save_prompt_dir.mkdir(parents=True, exist_ok=True)
                (self.cfg.save_prompt_dir / f"{tag}.txt").write_text(prompt, encoding="utf-8")
            except Exception:
                pass

        last_err: Optional[Exception] = None
        last_text: str = ""
        for attempt in range(int(self.cfg.max_retries) + 1):
            try:
                current_prompt = prompt if attempt == 0 else self._repair_prompt(previous_output=last_text)
                if self._provider == "qwen3vl_http":
                    last_text = self.client.chat(
                        prompt=current_prompt,
                        image_path=(post.screenshot_path if (self.cfg.vlm and bool(self.cfg.vlm.use_vision)) else None),
                        max_new_tokens=int(self.cfg.vlm.max_tokens) if self.cfg.vlm else None,
                    )
                    try:
                        write_model_call_trace(
                            trace_dir=default_model_trace_dir(request_metrics_path=(self.cfg.vlm.request_metrics_path if self.cfg.vlm else None)),
                            call_name="step_success_judge_qwen3vl_http",
                            prompt_text=current_prompt,
                            response_text=last_text,
                            provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                            model=str(getattr(self.cfg.vlm, "model", "") or ""),
                            attempt=int(attempt),
                            screenshot_paths=([post.screenshot_path] if (post.screenshot_path and self.cfg.vlm and bool(self.cfg.vlm.use_vision)) else []),
                        )
                    except Exception:
                        pass
                else:
                    completion = chat_complete_text(
                        cfg=self.cfg.vlm,
                        prompt=current_prompt,
                        screenshot_path=(post.screenshot_path if (self.cfg.vlm and bool(self.cfg.vlm.use_vision)) else None),
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
                        call_name="step_success_judge_qwen3vl_http" if self._provider == "qwen3vl_http" else "step_success_judge_chat_complete",
                        prompt_text=current_prompt if "current_prompt" in locals() else prompt,
                        response_text="",
                        error_text=repr(e),
                        provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                        model=str(getattr(self.cfg.vlm, "model", "") or ""),
                        attempt=int(attempt),
                        screenshot_paths=([post.screenshot_path] if (post.screenshot_path and self.cfg.vlm and bool(self.cfg.vlm.use_vision)) else []),
                    )
                except Exception:
                    pass
                last_err = e
                time.sleep(0.3 * (attempt + 1))

        # Fallback: if judge fails, do not block the run; assume API result.
        return StepSuccessJudgement(success=bool(api_success), reason=f"judge_failed:{last_err!r}", raw={"raw_text": last_text})

    @staticmethod
    def _parse(obj: Dict[str, Any]) -> StepSuccessJudgement:
        succ = obj.get("success")
        if not isinstance(succ, bool):
            raise ValueError("missing_success_bool")
        reason = obj.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)
        conf = obj.get("confidence", None)
        if conf is not None and not isinstance(conf, (int, float)):
            conf = None
        return StepSuccessJudgement(success=succ, reason=reason.strip(), confidence=float(conf) if conf is not None else None, raw=obj)

    @staticmethod
    def _on_screen(pre_or_post: Observation) -> list[dict[str, Any]]:
        s = getattr(pre_or_post, "state", {}) or {}
        items = s.get("_on_screen_objects", [])
        return items if isinstance(items, list) else []

    def _build_prompt(
        self,
        *,
        step: PlanStep,
        pre: Observation,
        post: Observation,
        api_success: bool,
        api_error: str,
    ) -> str:
        pre_items = self._on_screen(pre)
        post_items = self._on_screen(post)
        role_text = load_asset(
            "epm/step_success_judge",
            "role.txt",
            (
                "You are a success judge for an embodied agent in CookingSimulator.\n"
                "Your job: decide if the last executed step achieved its intended effect."
            ),
        )
        rules_text = load_asset(
            "epm/step_success_judge",
            "rules.txt",
            (
                "IMPORTANT:\n"
                "- Use ONLY the provided on-screen object snapshots (is_on_screen==true) and execution feedback.\n"
                "- Do not assume hidden state.\n"
                "- Output MUST be a single JSON object and NOTHING else."
            ),
        )
        output_text = load_asset(
            "epm/step_success_judge",
            "output_format.txt",
            (
                "Output schema:\n"
                "{\n"
                '  "success": true,\n'
                '  "reason": "short",\n'
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
            f"Executed step: name={step.name} args={step.args} expectation={step.expectation!r}\n"
            f"Executor api_success={api_success} api_error={api_error!r}\n"
            "\n"
            "Pre on-screen objects (from realtime, is_on_screen==true):\n"
            f"{pre_items}\n"
            "\n"
            "Post on-screen objects (from realtime, is_on_screen==true):\n"
            f"{post_items}\n"
            "\n"
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
            '  "success": true,\n'
            '  "reason": "short",\n'
            '  "confidence": 0.0\n'
            "}\n"
            "\n"
            "Previous response (invalid):\n"
            f"{prev}\n"
        )

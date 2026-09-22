from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from epm.brain.chat_client import _normalize_timeout, chat_complete_text
from epm.brain.http_qwen3vl import Qwen3VlHttpClient
from epm.brain.interfaces import Percept, PerceptionModule
from epm.brain.model_output_trace import default_model_trace_dir, write_model_call_trace, write_trace_files_for_raw_dir
from epm.brain.modules.prompt_assets import load_asset
from epm.brain.provider_compat import ANTHROPIC_PROVIDER_TYPES, OPENAI_PROVIDER_TYPES
from epm.core.epm_types import Observation
from epm.core.settings import VlmSettings


@dataclass(frozen=True)
class VlmPerceptionConfig:
    vlm: VlmSettings
    image_max_side: int = 1024
    image_format: str = "jpeg"
    jpeg_quality: int = 85
    save_raw_dir: Optional[Path] = None


class VlmPerception(PerceptionModule):
    """
    VLM perception as plain text only.

    The planner only consumes `percept.text`, so do not force structured JSON here.
    """

    def __init__(self, cfg: VlmPerceptionConfig) -> None:
        self.cfg = cfg
        if self.cfg.vlm.provider not in OPENAI_PROVIDER_TYPES and self.cfg.vlm.provider not in ANTHROPIC_PROVIDER_TYPES and self.cfg.vlm.provider != "qwen3vl_http":
            raise ValueError(f"unsupported_vlm_provider:{self.cfg.vlm.provider}")
        if self.cfg.vlm.provider != "qwen3vl_http" and not self.cfg.vlm.model:
            raise ValueError("vlm.model is required when perception_mode=vlm")
        self._last_screenshot_path: Optional[str] = None

    def run(self, *, observation: Observation) -> Percept:
        if not observation.screenshot_path:
            return Percept(text="(no screenshot)", slots={})
        self._last_screenshot_path = observation.screenshot_path

        last_text = ""
        last_err: Optional[str] = None
        last_raw: Optional[Dict[str, Any]] = None
        step_tag = self._step_tag(observation)
        for attempt in range(int(self.cfg.vlm.max_retries) + 1):
            prompt_text = self._base_prompt() if attempt == 0 else self._repair_prompt(previous_output=last_text, issue=str(last_err or "empty_text"))
            last_text, last_raw = self._generate_text(prompt_text=prompt_text, screenshot_path=observation.screenshot_path)
            self._save_raw_text(step_tag=step_tag, attempt=attempt, text=last_text)
            self._save_raw_payload(step_tag=step_tag, attempt=attempt, payload=last_raw)
            cleaned = self._normalize_text(last_text)
            if cleaned:
                return Percept(text=cleaned, slots={})
            last_err = "empty_text"

        excerpt = self._short_excerpt(last_text)
        raise RuntimeError(f"vlm_perception_text_invalid:{last_err}; last_text={excerpt}")

    @staticmethod
    def _step_tag(observation: Observation) -> str:
        fid = str(getattr(observation, "frame_id", "") or "").strip()
        if fid.isdigit():
            return f"step_{int(fid):06d}"
        return fid or "step_unknown"

    def _save_raw_text(self, *, step_tag: str, attempt: int, text: str) -> None:
        if self.cfg.save_raw_dir is None:
            return
        try:
            self.cfg.save_raw_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{step_tag}_attempt_{int(attempt)}.txt"
            path = self.cfg.save_raw_dir / filename
            path.write_text(str(text or ""), encoding="utf-8")
            write_trace_files_for_raw_dir(raw_dir=self.cfg.save_raw_dir, filename=filename, text=str(text or ""))
        except Exception:
            pass

    def _save_raw_payload(self, *, step_tag: str, attempt: int, payload: Optional[Dict[str, Any]]) -> None:
        if self.cfg.save_raw_dir is None or not isinstance(payload, dict) or not payload:
            return
        try:
            self.cfg.save_raw_dir.mkdir(parents=True, exist_ok=True)
            path = self.cfg.save_raw_dir / f"{step_tag}_attempt_{int(attempt)}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _short_excerpt(text: str, *, limit: int = 160) -> str:
        raw = str(text or "").replace("\r", " ").replace("\n", " ").strip()
        if not raw:
            return repr("")
        if len(raw) <= int(limit):
            return repr(raw)
        return repr(raw[: int(limit) - 3] + "...")

    @staticmethod
    def _normalize_text(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        out = "\n".join(lines).strip()
        if out.startswith("```") and out.endswith("```"):
            out = out.strip("`").strip()
        return out

    @staticmethod
    def _base_prompt() -> str:
        role_text = load_asset(
            "vlm_perception",
            "role.txt",
            (
                "You are a vision summarizer for an embodied CookingSimulator agent.\n"
                "Look at the screenshot and output only a short plain-text summary for planning."
            ),
        )
        rules_text = load_asset(
            "vlm_perception",
            "rules.txt",
            (
                "Requirements:\n"
                "- Output plain text only. No JSON. No markdown.\n"
                "- Keep it concise but concrete.\n"
                "- Prioritize these points in order:\n"
                "  1) whether the camera view or foreground object causes occlusion,\n"
                "  2) whether the current target workspace/container is visible, centered, and aligned for interaction,\n"
                "  3) the immediate blocker for the next interaction,\n"
                "  4) whether the current execution already shows an anomaly or a likely near-term failure risk.\n"
                "- Mention only the most decision-relevant facts.\n"
                "- Use at most 4 short lines."
            ),
        )
        return f"{role_text}\n\n{rules_text}\n"

    @staticmethod
    def _repair_prompt(*, previous_output: str, issue: str) -> str:
        prev = str(previous_output or "").strip()
        if len(prev) > 1000:
            prev = prev[:1000] + "..."
        return (
            "Your previous visual summary was unusable.\n"
            f"Issue: {issue}\n"
            "Look at the screenshot again and regenerate a better plain-text summary.\n"
            "- No JSON\n"
            "- No markdown\n"
            "- Mention occlusion / workspace or container alignment / immediate blocker / anomaly risk if visible\n"
            "- Use 2-4 short lines\n"
            "\n"
            "Previous response:\n"
            f"{prev}\n"
        )

    def _generate_text(self, *, prompt_text: str, screenshot_path: str) -> tuple[str, Optional[Dict[str, Any]]]:
        if self.cfg.vlm.provider in OPENAI_PROVIDER_TYPES or self.cfg.vlm.provider in ANTHROPIC_PROVIDER_TYPES:
            completion = chat_complete_text(
                cfg=self.cfg.vlm,
                prompt=prompt_text,
                screenshot_path=screenshot_path,
                force_use_vision=True,
            )
            return completion.content, completion.raw

        if self.cfg.vlm.provider == "qwen3vl_http":
            client = Qwen3VlHttpClient(
                base_url=self.cfg.vlm.base_url,
                connect_timeout_s=5.0,
                read_timeout_s=_normalize_timeout(float(self.cfg.vlm.timeout_s)),
            )
            try:
                text = client.chat(
                    prompt=prompt_text,
                    image_path=screenshot_path if bool(self.cfg.vlm.use_vision) else None,
                    max_new_tokens=int(self.cfg.vlm.max_tokens),
                )
                try:
                    write_model_call_trace(
                        trace_dir=default_model_trace_dir(request_metrics_path=self.cfg.vlm.request_metrics_path),
                        call_name="vlm_perception_qwen3vl_http",
                        prompt_text=prompt_text,
                        response_text=text,
                        provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                        model=str(getattr(self.cfg.vlm, "model", "") or ""),
                        screenshot_paths=([screenshot_path] if (screenshot_path and bool(self.cfg.vlm.use_vision)) else []),
                    )
                except Exception:
                    pass
                return text, None
            except Exception as e:
                try:
                    write_model_call_trace(
                        trace_dir=default_model_trace_dir(request_metrics_path=self.cfg.vlm.request_metrics_path),
                        call_name="vlm_perception_qwen3vl_http",
                        prompt_text=prompt_text,
                        response_text="",
                        error_text=repr(e),
                        provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                        model=str(getattr(self.cfg.vlm, "model", "") or ""),
                        screenshot_paths=([screenshot_path] if (screenshot_path and bool(self.cfg.vlm.use_vision)) else []),
                    )
                except Exception:
                    pass
                raise

        raise ValueError(f"unsupported_vlm_provider:{self.cfg.vlm.provider}")

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import ctypes
import os
import re
import threading
from pathlib import Path
import time
import io
from contextlib import redirect_stdout
from typing import Any, Optional
from loguru import logger as logging

from epm.brain.plan_schema import PlanResponse, validate_plan_against_allowlist
from epm.brain.interfaces import Percept, PlannerContext, PlannerModule
from epm.brain.plan_schema import PlanStep
from epm.brain.modules.perception import NoPerception, OraclePerception, OraclePerceptionConfig
from epm.brain.modules.prompt_policy import DefaultPromptPolicy
from epm.brain.modules.task_progress_maintainer import (
    TaskProgressMaintainer,
    TaskProgressMaintainerConfig,
    apply_task_progress_maintenance,
)
from epm.brain.modules.precondition_checker import (
    PreconditionChecker,
    PreconditionCheckerConfig,
    PreconditionCheckResult,
)
from epm.brain.modules.step_success_judge import StepSuccessJudge, StepSuccessJudgeConfig
from epm.brain.modules.visual_anomaly_observer import (
    VisualAnomalyObservation,
    VisualAnomalyObserver,
    VisualAnomalyObserverConfig,
)
from epm.brain.modules.vlm_perception import VlmPerception, VlmPerceptionConfig
from epm.brain.modules.epm import EPMConfig
from epm.brain.planner import (
    OpenAIPlanner,
    OpenAIPlannerConfig,
    Qwen3VlHttpPlanner,
    Qwen3VlHttpPlannerConfig,
    ScriptedPlanner,
    ScriptedPlannerConfig,
    VlmPlanner,
    VlmPlannerConfig,
)
from epm.brain.prompt_builder import PromptBuilder, PromptBuilderConfig
from epm.brain.skills_prompt import list_skill_card_names, select_skill_cards
from epm.brain.task_progress import write_task_progress_epm
from epm.brain.tools_manifest import build_tool_manifest, to_openai_tools
from epm.cerebellum.cookbench_api import CookBenchActionAPI, ActionResult
from epm.cerebellum.gui_actions.feedback_gui import get_latest_recipe_feedback
from epm.cerebellum.action_catalog import ActionCatalog
from epm.cerebellum.action_specs import to_prompt_text as action_specs_to_prompt_text
from epm.cerebellum.interaction_snapshot import read_interaction_snapshot, render_interaction_snapshot
from epm.cerebellum.raw_input_controller import RawInputController, StopRequested
from epm.cerebellum.realtime_products import best_match_by_name, extract_items, item_display_name, read_realtime_products
from epm.cerebellum.skills.registry import list_skills, run_skill, skill_specs_to_prompt_text
from epm.core.epm_types import Decision, Observation, PlanRef
from epm.core.action_feedback import render_action_feedback
from epm.core.http_403_pause import NetworkAbortRequired, NetworkPauseRequired, classify_network_exception
from epm.core.prompt_ablation import (
    RAW_INPUT_ACTIONS,
    disabled_prompt_groups,
    is_prompt_group_enabled,
    normalize_disabled_prompt_groups,
    normalize_prompt_ablation_profile,
    restrict_to_actions_only,
    restrict_to_raw_input_actions,
)
from epm.core.settings import LlmSettings, VlmSettings
from epm.kb.recipes import get_dish_by_id
from epm.memory_store.store import FileBackedMemoryStore
from epm.memory_store.align import diff_planned_executed
from epm.memory_store.progress_updater import (
    NeedReplan,
    PlanRef as UpdaterPlanRef,
    apply_execution_result_to_task_progress,
    append_off_plan_event,
)
from epm.world_adapter.adapter import WorldAdapter, WorldAdapterConfig


def _network_probable_causes(err: BaseException) -> str:
    kind = str(getattr(err, "kind", "") or "").strip().lower()
    http_status = getattr(err, "http_status", None)
    if http_status == 401:
        return "invalid_api_key_or_bearer_token/key_revoked_or_expired/wrong_provider_channel_or_endpoint_auth"
    if kind == "ssl":
        return (
            "network_unstable/tls_handshake_or_stream_interrupted/"
            "server_closed_connection_early/proxy_or_gateway_reset"
        )
    if kind == "timeout":
        return "api_slow/network_latency/provider_overloaded/request_timeout"
    if kind in {"connection_reset", "connection_aborted", "remote_disconnected", "connection_error"}:
        return "network_unstable/server_reset_connection/proxy_interrupted/upstream_disconnected"
    if kind == "http_403" or http_status == 403:
        return "api_key_invalid_or_expired/quota_or_account_restricted/provider_rejected_request"
    if http_status == 429:
        return "rate_limit/quota_exceeded/provider_throttling"
    if isinstance(http_status, int) and http_status >= 500:
        return "provider_server_error/upstream_gateway_error/temporary_overload"
    return "network_path_unstable/provider_temporary_failure/request_chain_interrupted"


def _network_diagnosis(err: BaseException) -> str:
    kind = str(getattr(err, "kind", "") or "").strip().lower()
    http_status = getattr(err, "http_status", None)
    if http_status == 401:
        return "认证失败：当前 Bearer API key 无效、已过期、被撤销，或与当前 provider/base_url 不匹配。"
    if kind == "http_403" or http_status == 403:
        return "请求被 provider 拒绝：当前 key 常见原因是额度不足、账户受限、预扣费失败，或该渠道暂时不允许继续调用。"
    if http_status == 429:
        return "接口被限流：provider 认为当前请求频率过高，或该账户/模型通道触发了速率限制。"
    if kind == "ssl":
        return "TLS/SSL 连接异常：握手过程中断、证书链异常，或中间代理/网关提前关闭了加密连接。"
    if kind == "timeout":
        return "请求超时：provider 返回太慢，或网络链路抖动导致在超时时间内没收到响应。"
    if kind in {"connection_reset", "connection_aborted", "remote_disconnected", "connection_error"}:
        return "连接被对端或中间网关直接断开：请求已发出，但 provider/代理在完整返回前关闭了连接。"
    if kind == "dns":
        return "DNS 解析失败：当前机器无法把 provider 域名解析成 IP。"
    if isinstance(http_status, int) and http_status >= 500:
        return "Provider 服务器内部异常：上游服务、网关或模型后端暂时不可用。"
    return "网络链路或 provider 临时异常：请求没有正常走完整个响应链路。"


def _network_common_triggers(err: BaseException) -> str:
    kind = str(getattr(err, "kind", "") or "").strip().lower()
    http_status = getattr(err, "http_status", None)
    if http_status == 401:
        return "key 填错、key 失效、key 不属于这个第三方渠道、base_url 配错到了别的供应商。"
    if kind == "http_403" or http_status == 403:
        return "余额不足、预扣费失败、账户风控、模型权限未开通、provider 把该 key 暂时封禁。"
    if http_status == 429:
        return "同一 key 请求过密、短时间并发太高、provider 对余额查询或模型接口单独限流。"
    if kind == "ssl":
        return "本地网络不稳、代理转发异常、上游网关 TLS 关闭连接、企业/校园网络中间人代理。"
    if kind == "timeout":
        return "模型响应太慢、请求体太大、provider 高峰拥塞、网络丢包。"
    if kind in {"connection_reset", "connection_aborted", "remote_disconnected", "connection_error"}:
        return "provider 网关重置连接、代理链路中断、远端服务重启、长连接被中间层回收。"
    if kind == "dns":
        return "本机 DNS 配置异常、代理没接管 DNS、当前网络环境无法访问该域名。"
    if isinstance(http_status, int) and http_status >= 500:
        return "provider 后端服务异常、模型服务重启、网关/负载均衡临时故障。"
    return "provider 临时抖动、代理链路异常、网络环境不稳定。"


def _network_suggested_action(err: BaseException) -> str:
    kind = str(getattr(err, "kind", "") or "").strip().lower()
    http_status = getattr(err, "http_status", None)
    if http_status == 401:
        return "检查 api_keys.local.json 中这把 key 是否正确、是否属于当前渠道；若有多把 key，直接换下一把。"
    if kind == "http_403" or http_status == 403:
        return "优先切换到下一把 key；若所有 key 都 403，再检查余额、账户权限和 provider 返回体中的 quota/permission 提示。"
    if http_status == 429:
        return "降低请求频率或稍后重试；如果是余额接口 429，不代表模型 key 本身无效。"
    if kind == "ssl":
        return "稍后重试；若频繁出现，检查代理/VPN、系统时间、证书链和网络环境。"
    if kind == "timeout":
        return "稍后重试；必要时减小 prompt/图像体积，或提高 timeout。"
    if kind in {"connection_reset", "connection_aborted", "remote_disconnected", "connection_error"}:
        return "这通常不是动作逻辑错误；优先视为网络/网关抖动，稍后重试并观察是否持续发生。"
    if kind == "dns":
        return "先确认当前机器能否访问 provider 域名，再检查 DNS、代理和防火墙。"
    if isinstance(http_status, int) and http_status >= 500:
        return "等待 provider 恢复后再试；这类错误通常与本地动作或 prompt 无关。"
    return "先看 http_status/kind/http_body_preview，再判断是 key、限流还是网络链路问题。"


def _format_network_error_detail(err: BaseException, *, episode_step: int | None = None, rolled_back_to: int | None = None) -> str:
    def _clean(value: Any) -> str:
        return str(value or "").replace("\r", " ").replace("\n", " ").strip()

    parts: list[str] = []
    if isinstance(episode_step, int):
        parts.append(f"episode_step={episode_step}")
    if isinstance(rolled_back_to, int):
        parts.append(f"rolled_back_to={rolled_back_to}")
    source = _clean(getattr(err, "source", ""))
    if source:
        parts.append(f"source={source}")
    kind = _clean(getattr(err, "kind", ""))
    if kind:
        parts.append(f"kind={kind}")
    http_status = getattr(err, "http_status", None)
    if isinstance(http_status, int):
        parts.append(f"http_status={http_status}")
    retry_after_s = getattr(err, "retry_after_s", None)
    if isinstance(retry_after_s, int):
        parts.append(f"retry_after_s={retry_after_s}")
    http_reason = _clean(getattr(err, "http_reason", ""))
    if http_reason:
        parts.append(f"http_reason={http_reason}")
    http_body_preview = _clean(getattr(err, "http_body_preview", ""))
    if http_body_preview:
        parts.append(f"http_body_preview={http_body_preview}")
    probable_causes = _clean(_network_probable_causes(err))
    if probable_causes:
        parts.append(f"probable_causes={probable_causes}")
    diagnosis = _clean(_network_diagnosis(err))
    if diagnosis:
        parts.append(f"diagnosis={diagnosis}")
    common_triggers = _clean(_network_common_triggers(err))
    if common_triggers:
        parts.append(f"common_triggers={common_triggers}")
    suggested_action = _clean(_network_suggested_action(err))
    if suggested_action:
        parts.append(f"suggested_action={suggested_action}")
    message = _clean(err)
    if message:
        parts.append(f"message={message}")
    return " | ".join(parts)


@dataclass
class AgentConfig:
    memory_dir: Path
    realtime_products_path: Path
    recipes_path: Path
    dish_id: int
    screenshot_dir: Path
    camera_info_path: Optional[Path] = None
    agent_state_path: Optional[Path] = None
    window_title: str = "CookingSimulator"
    activate_window_each_step: bool = True
    screenshot_format: str = "jpeg"
    screenshot_jpeg_quality: int = 70
    stm_window_size: int = 200
    action_catalog_path: Optional[Path] = None
    exposed_action_categories: Optional[list[str]] = None
    max_actions_per_category: int = -1
    inject_memory: bool = True
    inject_tool_schemas: bool = True
    scripted_plan_path: Optional[Path] = None
    verbose: bool = True
    planner_mode: str = "scripted"
    llm: Optional[LlmSettings] = None
    vlm: Optional[VlmSettings] = None
    pipeline: str = "planner_executor"
    perception_mode: str = "oracle"
    max_visible_items: int = 20
    plan_min_steps: int = 3
    plan_max_steps: int = 8
    allow_incremental_plan: bool = True
    task_progress_maintenance: Optional[dict] = None
    step_success_judge: Optional[dict] = None
    subgoal_done_judge: Optional[dict] = None
    visual_anomaly_observer: Optional[dict] = None
    precondition_checker: Optional[dict] = None
    reflexion: Optional[dict] = None
    prompt_policy: Optional[dict] = None
    prompt_ablation_profile: str = "full"
    prompt_disabled_groups: Optional[frozenset[str]] = None
    prompt_layout_path: Optional[Path] = None
    log_mode: str = "verbose"  # "verbose" | "minimal"
    log_color: bool = True
    resume: bool = False
    force_submit_step_threshold: int = 0
    force_submit_active: bool = False
    force_submit_within_steps: int = 50


class EpmAgent:
    """
    Minimal runnable skeleton agent.

    This is NOT the final Brain (LLM/VLM) implementation. It only wires:
    - WorldAdapter.observe()
    - Cerebellum.call_action()
    - MemoryStore.record_step()
    """

    def __init__(self, cfg: AgentConfig) -> None:
        self.log = logging.bind(component="EPM")
        self.cfg = cfg
        self.prompt_ablation_profile = normalize_prompt_ablation_profile(cfg.prompt_ablation_profile)
        self.prompt_disabled_groups = (
            normalize_disabled_prompt_groups(cfg.prompt_disabled_groups)
            if cfg.prompt_disabled_groups is not None
            else disabled_prompt_groups(self.prompt_ablation_profile)
        )
        self.request_metrics_path = Path(cfg.memory_dir) / "planner_request_metrics.jsonl"
        self._attach_request_metrics_paths()
        self.rgb_input_enabled = is_prompt_group_enabled(
            profile=self.prompt_ablation_profile,
            group="rgb_input",
            disabled_groups=self.prompt_disabled_groups,
        )
        if not self.rgb_input_enabled:
            if self.cfg.llm is not None:
                self.cfg.llm = replace(self.cfg.llm, use_vision=False)
            if self.cfg.vlm is not None:
                self.cfg.vlm = replace(self.cfg.vlm, use_vision=False)
            self.log.info("[EPM] no_rgb=true image_input_disabled_for_all_model_calls")
        self.pipeline_name = (cfg.pipeline or "planner_executor").strip().lower()
        self._action_catalog_text = ""
        self.task_progress_mode = self._resolve_task_progress_mode(pipeline=self.pipeline_name)
        self.task_progress_path = self._resolve_task_progress_path(
            mode=self.task_progress_mode,
            memory_dir=Path(cfg.memory_dir),
        )
        self.memory = FileBackedMemoryStore(
            cfg.memory_dir,
            window_size=cfg.stm_window_size,
            agent_state_path=cfg.agent_state_path,
            task_progress_path=self.task_progress_path,
            enable_reflexion_memory=(self.pipeline_name == "reflexion"),
        )
        self.world = WorldAdapter(
            WorldAdapterConfig(
                realtime_products_path=cfg.realtime_products_path,
                screenshot_dir=cfg.screenshot_dir,
                window_title=cfg.window_title,
                activate_window_each_step=cfg.activate_window_each_step,
                screenshot_format=cfg.screenshot_format,
                screenshot_jpeg_quality=cfg.screenshot_jpeg_quality,
            )
        )
        self.api = CookBenchActionAPI(
            raw_input_only=restrict_to_raw_input_actions(
                self.prompt_ablation_profile,
                disabled_groups=self.prompt_disabled_groups,
            )
        )
        self.step_id = 0
        self.last_completed_step_id = 0
        self.last_completed_step_type = ""
        self.last_completed_step_name = ""
        self.last_physical_step_id = 0
        self.last_physical_step_type = ""
        self.last_physical_step_name = ""
        self._last_action_feedback: str = ""
        # Cross-step cache of recent query_scene_objects results for planner reuse.
        self._query_memory: dict[str, dict[str, Any]] = {}
        self._episode_start_monotonic = time.monotonic()
        self._done_override = False
        self._done_reason = ""
        # Latest planner rationale snapshot (for dashboard/status visibility).
        self._last_plan_goal: str = ""
        self._last_plan_thoughts: str = ""
        self._last_plan_steps: list[str] = []
        self._last_plan_updated_episode_step: int = 0
        self._plan_generation_count: int = 0
        self._mode_entry_retry_signature: str = ""
        self._mode_entry_retry_count: int = 0
        # PE active plan state (continuously maintained execution queue snapshot).
        self._pe_active_plan: dict[str, Any] = {}
        self.dish = get_dish_by_id(self.cfg.recipes_path, int(self.cfg.dish_id))
        pipeline_name = self.pipeline_name
        effective_plan_min_steps = int(cfg.plan_min_steps)
        effective_plan_max_steps = int(cfg.plan_max_steps)

        requested_tool_calling = bool(
            (str(getattr(cfg, "planner_mode", "") or "").strip().lower() == "vlm")
            and (cfg.vlm is not None)
            and bool(getattr(cfg.vlm, "use_tools", False))
        )
        self.prompter = self._build_prompter(
            plan_min_steps=effective_plan_min_steps,
            plan_max_steps=effective_plan_max_steps,
            tool_calling=requested_tool_calling,
        )
        self.allowed_actions = self.api.list_actions()
        self.allowed_skills = list_skills()
        if restrict_to_raw_input_actions(
            self.prompt_ablation_profile,
            disabled_groups=self.prompt_disabled_groups,
        ):
            available_actions = set(self.allowed_actions)
            self.allowed_actions = [name for name in RAW_INPUT_ACTIONS if name in available_actions]
            self.allowed_skills = []
            self.log.info(f"[EPM] no_action=true raw_input_actions={self.allowed_actions}")
        elif restrict_to_actions_only(
            self.prompt_ablation_profile,
            disabled_groups=self.prompt_disabled_groups,
        ):
            self.allowed_skills = []
            self.log.info("[EPM] no_skill=true semantic_actions_only")
        self.perception = self._init_perception()
        self.planner = self._init_planner()
        tool_calling_supported = self._probe_tool_calling_support()
        if tool_calling_supported is not None:
            if requested_tool_calling and not tool_calling_supported:
                self.log.warning(
                    "[EPM] tool_calling_requested=true but probe reports unsupported; falling back to JSON-only planning."
                )
                self._set_planner_tool_calling(False)
                if self.prompter.cfg.tool_calling:
                    self.prompter = self._build_prompter(
                        plan_min_steps=effective_plan_min_steps,
                        plan_max_steps=effective_plan_max_steps,
                        tool_calling=False,
                    )
            elif (not requested_tool_calling) and tool_calling_supported:
                if str(getattr(cfg, "planner_mode", "") or "").strip().lower() == "vlm":
                    hint = "vlm.use_tools"
                else:
                    hint = "llm.use_tools"
                self.log.warning(
                    f"[EPM] tool_calling_supported=true but disabled; set {hint}=true to enable."
                )
        self.pipeline = self._init_pipeline()
        self.maintainer = self._init_task_progress_maintainer()
        self.precondition_checker = self._init_precondition_checker()
        self.success_judge = self._init_step_success_judge()
        self.visual_anomaly_observer = self._init_visual_anomaly_observer()
        self._init_task_plan()
        self._load_pe_active_plan_state()
        self._hydrate_pipeline_plan_from_pe_active_plan_if_resume()
        self._init_action_catalog()
        self._write_tool_manifest()
        if self.cfg.verbose:
            self.log.info(f"[EPM] dish_id={self.dish.id} dish_name={self.dish.dish_name!r}")
            self.log.info(f"[EPM] recipes_path={self._display_path(self.cfg.recipes_path)}")
            self.log.info(f"[EPM] realtime_products_path={self._display_path(self.cfg.realtime_products_path)}")
            self.log.info(f"[EPM] screenshot_dir={self._display_path(self.cfg.screenshot_dir)}")
            self.log.info(f"[EPM] memory_dir={self._display_path(self.cfg.memory_dir)}")
            self.log.info(f"[EPM] allowed_actions_count={len(self.allowed_actions)} (see {self._display_path(self.cfg.memory_dir / 'action_specs.txt')})")
            self.log.info(f"[EPM] action_catalog_source={self._display_path(self.cfg.action_catalog_path)}")
            self.log.info(f"[EPM] skill_cards_catalog_path={self._display_path(self.cfg.memory_dir / 'skills_catalog.json')}")
            self.log.info(f"[EPM] builtin_skill_specs_path={self._display_path(self.cfg.memory_dir / 'skill_specs.txt')}")
            self.log.info(f"[EPM] allowed_skills={self.allowed_skills} (use type=skill)")
            self.log.info(f"[EPM] brain.pipeline={self.cfg.pipeline} brain.perception_mode={self.cfg.perception_mode}")
            mode = self.cfg.planner_mode
            if mode == "scripted":
                self.log.info(
                    f"[EPM] planner_mode=scripted scripted_plan_path={self._display_path(self.cfg.scripted_plan_path or (self.cfg.memory_dir / 'next_plan.json'))}"
                )
            elif mode == "vlm":
                self.log.info(
                    f"[EPM] planner_mode=vlm provider={getattr(self.cfg.vlm, 'provider', '')!r} "
                    f"model={getattr(self.cfg.vlm, 'model', '')!r} base_url={getattr(self.cfg.vlm, 'base_url', '')!r} "
                    f"use_vision={bool(getattr(self.cfg.vlm, 'use_vision', False))}"
                )
            else:
                self.log.info(
                    f"[EPM] planner_mode={mode} model={getattr(self.cfg.llm, 'model', '')!r} base_url={getattr(self.cfg.llm, 'base_url', '')!r}"
                )
            if self.maintainer is not None:
                self.log.info("[EPM] task_progress_maintenance=enabled")
            if self.success_judge is not None:
                self.log.info("[EPM] step_success_judge=enabled")
            if self.visual_anomaly_observer is not None:
                self.log.info("[EPM] visual_anomaly_observer=enabled")

    def _attach_request_metrics_paths(self) -> None:
        if self.cfg.llm is not None:
            self.cfg.llm = replace(self.cfg.llm, request_metrics_path=self.request_metrics_path)
        if self.cfg.vlm is not None:
            self.cfg.vlm = replace(self.cfg.vlm, request_metrics_path=self.request_metrics_path)

    def _model_observation(self, observation: Observation) -> Observation:
        """Keep screenshots available to the executor but never expose them to models in no_rgb."""
        if self.rgb_input_enabled or not getattr(observation, "screenshot_path", None):
            return observation
        return replace(observation, screenshot_path=None)

    @staticmethod
    def _is_fatal_error(errors: str) -> tuple[bool, str]:
        """
        Decide whether an execution error should abort the whole episode.

        We treat missing runtime dependencies (ImportError/ModuleNotFoundError) as fatal,
        because retrying will never succeed without fixing the environment.
        """
        s = (errors or "").strip()
        if not s:
            return False, ""
        sl = s.lower()
        if "feedback_dish_mismatch" in sl:
            return True, "feedback_dish_mismatch_manual_review_required"
        if ("no module named" in sl) or ("modulenotfounderror" in sl) or ("importerror" in sl):
            # Extract missing module name when possible.
            m = re.search(r"No module named '([^']+)'", s)
            mod = m.group(1) if m else ""
            return True, (f"missing_dependency:{mod}" if mod else "missing_dependency")
        return False, ""

    @staticmethod
    def _is_camera_only_adjustment_step(step: PlanStep) -> bool:
        typ = str(getattr(step, "type", "") or "").strip().lower()
        name = str(getattr(step, "name", "") or "").strip().lower()
        return typ == "action" and name in {"look_up", "look_down", "look_left", "look_right"}

    @staticmethod
    def _mode_entry_failure_signature(step: PlanStep, errors: str) -> str:
        name = str(getattr(step, "name", "") or "").strip().lower()
        sl = str(errors or "").strip().lower()
        if "not_in_pouring_mode" in sl and name in {"enter_pouring_mode", "auto_pour"}:
            return "pouring_mode_not_entered"
        return ""

    def _update_mode_entry_retry_guard(self, *, step: PlanStep, final_success: bool, errors: str) -> tuple[bool, str]:
        signature = self._mode_entry_failure_signature(step, errors)

        if signature:
            if self._mode_entry_retry_signature == signature:
                self._mode_entry_retry_count += 1
            else:
                self._mode_entry_retry_signature = signature
                self._mode_entry_retry_count = 1
            if self._mode_entry_retry_count >= 4:
                return (
                    True,
                    f"repeated_{signature}_loop:count={int(self._mode_entry_retry_count)}:last_error={errors}",
                )
            return False, ""

        if final_success and self._is_camera_only_adjustment_step(step):
            return False, ""

        self._mode_entry_retry_signature = ""
        self._mode_entry_retry_count = 0
        return False, ""

    @staticmethod
    def _is_non_physical_step(step: PlanStep) -> bool:
        typ = str(getattr(step, "type", "") or "").strip().lower()
        name = str(getattr(step, "name", "") or "").strip().lower()
        return (typ, name) in {
            ("skill", "query_scene_objects"),
            ("skill", "auto_perception"),
            ("skill", "list_supported_items"),
        }

    def _init_perception(self):
        mode = (self.cfg.perception_mode or "oracle").strip().lower()
        if mode == "none":
            return NoPerception()
        if mode == "oracle":
            return OraclePerception(
                OraclePerceptionConfig(
                    realtime_products_path=self.cfg.realtime_products_path,
                    max_items=int(self.cfg.max_visible_items),
                    only_on_screen=True,
                )
            )
        if mode == "vlm":
            if self.cfg.vlm is None:
                raise ValueError("perception_mode=vlm requires vlm settings")
            return VlmPerception(
                VlmPerceptionConfig(
                    vlm=self.cfg.vlm,
                    save_raw_dir=self.cfg.memory_dir / "perception_raw",
                )
            )
        return NoPerception()

    def _display_path(self, path: Optional[Path | str]) -> str:
        text = str(path or "").strip()
        if not text:
            return ""
        try:
            return os.path.relpath(text, start=str(self.cfg.memory_dir.parent))
        except Exception:
            return text

    def _build_task_execution_context_text(self) -> str:
        active_dish_id = int(getattr(self.dish, "id", self.cfg.dish_id))
        active_dish_name = str(getattr(self.dish, "dish_name", "") or "").strip()
        lines = [
            "episode_mode=single_active_dish",
            f"active_dish_ids_for_this_prompt={json.dumps([active_dish_id], ensure_ascii=False)}",
            f"active_dish_names_for_this_prompt={json.dumps([active_dish_name], ensure_ascii=False)}",
            "final_submission_count_for_this_prompt=1",
            "submission_scope=current_active_dish_only",
            (
                "interpretation_note=The current recipe belongs to the active dish set listed above. "
                "Multiple components or multiple serve hot/cold lines still belong to one dish and one final submission "
                "unless this execution context explicitly lists multiple active dishes."
            ),
        ]
        return "\n".join(lines).strip()

    def _step_screenshot_ext(self) -> str:
        fmt = str(getattr(self.cfg, "screenshot_format", "jpeg") or "jpeg").strip().lower()
        if fmt in {"jpg", "jpeg"}:
            return ".jpg"
        return ".png"

    def _step_screenshot_filename(self, *, step_id: int, post: bool = False) -> str:
        suffix = "_post" if post else ""
        return f"step_{int(step_id):06d}{suffix}{self._step_screenshot_ext()}"

    def _pipeline_requires_planner_round(self) -> bool:
        checker = getattr(self.pipeline, "needs_planner_round", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return True
        getter = getattr(self.pipeline, "current_remaining_plan", None)
        if callable(getter):
            try:
                return len(list(getter() or [])) <= 0
            except Exception:
                return True
        return True

    def _should_run_planner_aux_modules(self, *, force_active: bool) -> bool:
        if force_active:
            return False
        return self._pipeline_requires_planner_round()

    def _should_run_visual_aux_after_step(self, *, force_active: bool, final_success: bool) -> bool:
        if force_active:
            return False
        if not final_success:
            return True
        getter = getattr(self.pipeline, "current_remaining_plan", None)
        if callable(getter):
            try:
                return len(list(getter() or [])) <= 1
            except Exception:
                return True
        return True

    def _init_task_progress_maintainer(self) -> Optional[TaskProgressMaintainer]:
        if not is_prompt_group_enabled(
            profile=self.prompt_ablation_profile,
            group="history",
            disabled_groups=self.prompt_disabled_groups,
        ):
            self.log.info("[EPM] task_progress_maintainer=disabled_by_prompt_ablation")
            return None
        if self.task_progress_path is None:
            return None
        raw = self.cfg.task_progress_maintenance or {}
        if not isinstance(raw, dict):
            raw = {}
        enabled = bool(raw.get("enabled", False))
        if not enabled:
            return None
        if self.cfg.vlm is None:
            raise ValueError("task_progress_maintenance.enabled=true requires vlm settings")
        return TaskProgressMaintainer(
            TaskProgressMaintainerConfig(
                vlm=self.cfg.vlm,
                enabled=True,
                on_new_plan_generated=bool(raw.get("on_new_plan_generated", True)),
                on_need_replan=bool(raw.get("on_need_replan", True)),
                history_window_steps=int(raw.get("history_window_steps", 30) or 30),
                min_step_gap_steps=int(raw.get("min_step_gap_steps", 30) or 30),
                save_prompt_dir=self.cfg.memory_dir / "task_progress_maintenance_prompts",
                save_raw_dir=self.cfg.memory_dir / "task_progress_maintenance_raw",
            )
        )

    def _init_precondition_checker(self) -> Optional[PreconditionChecker]:
        if not is_prompt_group_enabled(
            profile=self.prompt_ablation_profile,
            group="feedback",
            disabled_groups=self.prompt_disabled_groups,
        ):
            self.log.info("[EPM] precondition_checker=disabled_by_prompt_ablation")
            return None
        raw = self.cfg.precondition_checker or {}
        if not isinstance(raw, dict):
            raw = {}
        enabled = bool(raw.get("enabled", False))
        if not enabled:
            return None
        mode = str(raw.get("mode", "key_rules") or "key_rules").strip().lower()
        if mode != "key_rules" and self.cfg.vlm is None:
            raise ValueError("precondition_checker.enabled=true requires vlm settings")
        return PreconditionChecker(
            PreconditionCheckerConfig(
                enabled=True,
                mode=mode,
                vlm=self.cfg.vlm,
                max_retries=int(raw.get("max_retries", 1) or 1),
                max_plan_repair_attempts=int(raw.get("max_plan_repair_attempts", 2) or 2),
                recheck_after_repair=bool(raw.get("recheck_after_repair", False)),
                strip_think_tags=bool(raw.get("strip_think_tags", True)),
                save_prompt_dir=self.cfg.memory_dir / "precondition_check_prompts",
                save_raw_dir=self.cfg.memory_dir / "precondition_check_raw",
            )
        )

    def _init_step_success_judge(self) -> Optional[StepSuccessJudge]:
        raw = self.cfg.step_success_judge or {}
        if not isinstance(raw, dict):
            raw = {}
        enabled = bool(raw.get("enabled", False))
        if not enabled:
            return None
        if self.cfg.vlm is None:
            raise ValueError("step_success_judge.enabled=true requires vlm settings")
        return StepSuccessJudge(
            StepSuccessJudgeConfig(
                enabled=True,
                vlm=self.cfg.vlm,
                max_retries=int(raw.get("max_retries", 1) or 1),
                strip_think_tags=bool(raw.get("strip_think_tags", True)),
                save_prompt_dir=self.cfg.memory_dir / "step_success_prompts",
                save_raw_dir=self.cfg.memory_dir / "step_success_raw",
            )
        )

    def _init_visual_anomaly_observer(self) -> Optional[VisualAnomalyObserver]:
        raw = self.cfg.visual_anomaly_observer or {}
        if not isinstance(raw, dict):
            raw = {}
        enabled = bool(raw.get("enabled", False))
        if not enabled:
            return None
        if self.cfg.vlm is None:
            raise ValueError("visual_anomaly_observer.enabled=true requires vlm settings")
        return VisualAnomalyObserver(
            VisualAnomalyObserverConfig(
                enabled=True,
                vlm=self.cfg.vlm,
                max_retries=int(raw.get("max_retries", 0) or 0),
                strip_think_tags=bool(raw.get("strip_think_tags", True)),
                history_window_steps=int(raw.get("history_window_steps", 4) or 4),
                save_prompt_dir=self.cfg.memory_dir / "visual_anomaly_prompts",
                save_raw_dir=self.cfg.memory_dir / "visual_anomaly_raw",
            )
        )

    def _init_planner(self) -> PlannerModule:
        mode = (self.cfg.planner_mode or "scripted").strip().lower()
        if mode == "llm":
            if self.cfg.llm is None:
                raise ValueError("planner_mode=llm requires llm settings")
            if self.cfg.llm.provider == "qwen3vl_http":
                return Qwen3VlHttpPlanner(Qwen3VlHttpPlannerConfig(llm=self.cfg.llm))
            return OpenAIPlanner(
                OpenAIPlannerConfig(llm=self.cfg.llm, allowed_actions=self.allowed_actions, allowed_skills=self.allowed_skills)
            )
        if mode == "vlm":
            if self.cfg.vlm is None:
                raise ValueError("planner_mode=vlm requires vlm settings")
            return VlmPlanner(
                VlmPlannerConfig(
                    vlm=self.cfg.vlm,
                    allowed_actions=self.allowed_actions,
                    allowed_skills=self.allowed_skills,
                    save_last_raw_path=self.cfg.memory_dir / "vlm_planner_last.txt",
                    save_per_step_dir=self.cfg.memory_dir / "vlm_planner_raw",
                    request_metrics_path=self.cfg.memory_dir / "planner_request_metrics.jsonl",
                    plan_min_steps=int(self.cfg.plan_min_steps),
                    plan_max_steps=int(self.cfg.plan_max_steps),
                    allow_incremental_plan=bool(self.cfg.allow_incremental_plan),
                )
            )
        return ScriptedPlanner(ScriptedPlannerConfig(plan_path=(self.cfg.scripted_plan_path or (self.cfg.memory_dir / "next_plan.json"))))

    def _init_prompt_policy(self, *, pipeline: str) -> DefaultPromptPolicy:
        raw = self.cfg.prompt_policy or {}
        include_task_progress_default: bool
        if isinstance(raw, dict) and "include_task_progress" in raw:
            include_task_progress_default = bool(raw.get("include_task_progress", True))
        elif pipeline in ("react", "reflexion", "cap"):
            include_task_progress_default = False
        else:
            include_task_progress_default = True
        return DefaultPromptPolicy(include_task_progress_default=include_task_progress_default)

    def _init_pipeline(self):
        pipeline = (self.cfg.pipeline or "planner_executor").strip().lower()
        policy = self._init_prompt_policy(pipeline=pipeline)
        if pipeline == "react":
            from epm.brain.pipelines.react import ReActPipeline

            return ReActPipeline(planner=self.planner, prompter=self.prompter, policy=policy)
        if pipeline in ("planner_executor", "planner-executor", "pe"):
            from epm.brain.pipelines.planner_executor import PlannerExecutorPipeline

            return PlannerExecutorPipeline(planner=self.planner, prompter=self.prompter, policy=policy)
        if pipeline in ("open_loop", "open-loop", "openloop", "saycan"):
            from epm.brain.pipelines.open_loop import OpenLoopSequentialPipeline

            return OpenLoopSequentialPipeline(planner=self.planner, prompter=self.prompter, policy=policy)

        # Baselines that require an auxiliary chat model (Reflexion/EPM/CaP).
        mode = (self.cfg.planner_mode or "").strip().lower()
        chat_cfg = None
        if mode == "vlm" and self.cfg.vlm is not None and getattr(self.cfg.vlm, "model", "").strip():
            chat_cfg = self.cfg.vlm
        elif self.cfg.llm is not None and getattr(self.cfg.llm, "model", "").strip():
            chat_cfg = self.cfg.llm
        elif self.cfg.vlm is not None and getattr(self.cfg.vlm, "model", "").strip():
            chat_cfg = self.cfg.vlm
        if chat_cfg is None or not getattr(chat_cfg, "model", "").strip():
            raise ValueError(f"pipeline={pipeline} requires llm/vlm model settings (missing model)")

        if pipeline == "reflexion":
            from epm.brain.pipelines.planner_executor import PlannerExecutorPipeline
            from epm.brain.pipelines.reflexion import ReflexionPipeline
            from epm.brain.modules.reflexion import ReflexionConfig

            base = PlannerExecutorPipeline(planner=self.planner, prompter=self.prompter, policy=policy)
            raw = self.cfg.reflexion or {}
            if not isinstance(raw, dict):
                raw = {}
            return ReflexionPipeline(
                base=base,
                memory_dir=self.cfg.memory_dir,
                chat_cfg=chat_cfg,
                reflexion_cfg=ReflexionConfig(
                    enabled=bool(raw.get("enabled", True)),
                    reflect_on_failure=bool(raw.get("reflect_on_failure", True)),
                    min_failures_to_reflect=int(raw.get("min_failures_to_reflect", 1) or 1),
                    failure_window_steps=int(raw.get("failure_window_steps", 8) or 8),
                    min_reflection_gap_steps=int(raw.get("min_reflection_gap_steps", 8) or 8),
                    reflect_every_n_steps=int(raw.get("reflect_every_n_steps", 0) or 0),
                    reflect_on_loop=bool(raw.get("reflect_on_loop", True)),
                    loop_window_steps=int(raw.get("loop_window_steps", 6) or 6),
                    loop_repeat_threshold=int(raw.get("loop_repeat_threshold", 3) or 3),
                    memory_window_size=int(raw.get("memory_window_size", 10) or 10),
                    max_reflections=int(raw.get("max_reflections", 200) or 200),
                    history_window_steps=int(raw.get("history_window_steps", 12) or 12),
                    prompt_ablation_profile=self.prompt_ablation_profile,
                    prompt_disabled_groups=self.prompt_disabled_groups,
                ),
            )
        if pipeline in ("epm", "epm_agent"):
            from epm.brain.pipelines.planner_executor import PlannerExecutorPipeline
            from epm.brain.pipelines.epm import EPMPipeline
            from epm.brain.modules.epm import EPMConfig

            # EPM hierarchical executor using planner-executor for multi-step subgoal plans.
            # Context isolation is currently disabled for both pipelines.
            base = PlannerExecutorPipeline(planner=self.planner, prompter=self.prompter, policy=policy)
            raw_subgoal_done = (self.cfg.subgoal_done_judge or {})
            if not isinstance(raw_subgoal_done, dict):
                raw_subgoal_done = {}
            return EPMPipeline(
                base_executor=base,
                memory_dir=self.cfg.memory_dir,
                chat_cfg=chat_cfg,
                epm_cfg=EPMConfig(
                    save_raw=True,
                    subgoal_done_judge_enabled=bool(raw_subgoal_done.get("enabled", False)),
                    prompt_ablation_profile=self.prompt_ablation_profile,
                    prompt_disabled_groups=self.prompt_disabled_groups,
                ),
                context_isolation=False,
                persist_goal_tree=False,
            )
        if pipeline == "cap":
            from epm.brain.pipelines.cap import CaPPipeline
            from epm.brain.modules.cap import CaPConfig

            return CaPPipeline(
                memory_dir=self.cfg.memory_dir,
                chat_cfg=chat_cfg,
                cap_cfg=CaPConfig(
                    prompt_ablation_profile=self.prompt_ablation_profile,
                    prompt_disabled_groups=self.prompt_disabled_groups,
                ),
                plan_min_steps=int(self.cfg.plan_min_steps),
                plan_max_steps=int(self.cfg.plan_max_steps),
            )

        # Default fallback
        from epm.brain.pipelines.planner_executor import PlannerExecutorPipeline

        return PlannerExecutorPipeline(planner=self.planner, prompter=self.prompter, policy=policy)

    @staticmethod
    def _resolve_task_progress_mode(*, pipeline: str) -> Optional[str]:
        name = (pipeline or "").strip().lower()
        if name in ("epm", "epm_agent"):
            return "epm"
        return None

    @staticmethod
    def _resolve_task_progress_path(*, mode: Optional[str], memory_dir: Path) -> Optional[Path]:
        if mode == "epm":
            return Path(memory_dir) / "task_progress_epm.txt"
        return None


    def is_done(self, obs) -> bool:
        if bool(getattr(self, "_done_override", False)):
            if not str(getattr(self, "_done_reason", "")).strip():
                self._done_reason = "done_override"
            return True
        # Keep episode termination deterministic: only explicit override from
        # submit-feedback verification can mark done.
        return False

    def get_done_reason(self) -> str:
        return str(getattr(self, "_done_reason", "") or "").strip()

    def _verify_submit_feedback_and_mark_done(
        self,
        *,
        dish_name: str,
        step_wall_time: float,
        submit_raw: dict[str, Any] | None = None,
    ) -> None:
        run_root = os.environ.get("EPM_RUN_ROOT", "").strip()
        if not run_root:
            raise RuntimeError("feedback_verification_failed:run_root_missing")
        root = Path(run_root)
        if not root.exists():
            raise RuntimeError("feedback_verification_failed:run_root_missing")

        want = str(dish_name or "").strip()

        def _norm_dish_name(text: Any) -> str:
            return re.sub(r"\s+", " ", str(text or "").strip()).lower()

        def _mark_done_from_feedback_obj(data: dict[str, Any]) -> bool:
            if not isinstance(data, dict) or not data:
                return False
            got = str(data.get("dishName") or data.get("dish_name") or "").strip()
            if want and got and _norm_dish_name(got) != _norm_dish_name(want):
                self._done_reason = f"feedback_dish_mismatch:want={want!r}:got={got!r}"
                return False
            self._done_override = True
            if got:
                self._done_reason = f"submit_feedback_file_detected:dish={got}"
            else:
                self._done_reason = "submit_feedback_file_detected"
            return True

        direct_feedback = submit_raw.get("feedback") if isinstance(submit_raw, dict) else None
        if isinstance(direct_feedback, dict) and bool(direct_feedback.get("success", True)):
            if _mark_done_from_feedback_obj(direct_feedback):
                return

        def _find_matching_feedback() -> tuple[dict, str]:
            files = sorted(root.glob("recipe_feedback_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not files:
                return {}, "feedback_file_missing"
            fresh_files: list[Path] = []
            stale_seen = False
            for path in files:
                try:
                    if path.stat().st_mtime >= float(step_wall_time) - 2.0:
                        fresh_files.append(path)
                    else:
                        stale_seen = True
                except Exception:
                    fresh_files.append(path)
            candidates = fresh_files or files
            seen_dish_names: list[str] = []
            last_err: Exception | None = None
            want_key = _norm_dish_name(want)
            matched_data: dict[str, Any] = {}
            for path in candidates:
                data = {}
                for enc in ("utf-8-sig", "utf-8", "gbk"):
                    try:
                        raw = path.read_text(encoding=enc, errors="ignore")
                        obj = json.loads(raw)
                        if isinstance(obj, dict):
                            data = obj
                            break
                    except Exception as e:
                        last_err = e
                        continue
                if not data:
                    continue
                got = str(data.get("dishName") or data.get("dish_name") or "").strip()
                if got:
                    seen_dish_names.append(got)
                if not want_key or _norm_dish_name(got) == want_key:
                    matched_data = data
                    break
            if matched_data:
                return matched_data, ""
            if seen_dish_names:
                return {}, f"feedback_dish_mismatch:want={want!r}:seen={seen_dish_names[:3]!r}"
            if last_err is not None:
                return {}, f"feedback_read_error:{last_err!r}"
            if stale_seen and not fresh_files:
                return {}, "feedback_file_stale"
            return {}, "feedback_file_missing"

        def _trigger_altk_dump() -> None:
            try:
                get_latest_recipe_feedback(window_title="CookingSimulator")
            except Exception:
                return

        for attempt in range(3):
            data, err = _find_matching_feedback()
            if err:
                if attempt < 2:
                    _trigger_altk_dump()
                    continue
                self._done_reason = f"submit_feedback_unavailable:{err}"
                return
            if _mark_done_from_feedback_obj(data):
                return
            return

        self._done_reason = f"submit_feedback_unavailable_after_retries:want={want!r}"
        return

    def _init_task_plan(self) -> None:
        path = self.task_progress_path
        if path is None:
            return
        if bool(getattr(self.cfg, "resume", False)) and path.exists():
            if self.cfg.verbose:
                self.log.info(f"[EPM] resume=true skip task_progress init (existing {path.name} found)")
            return
        if self.task_progress_mode == "epm":
            write_task_progress_epm(path=path, dish=self.dish)

    def _append_pe_plan_history(self, plan: PlanResponse) -> None:
        if self.task_progress_mode is not None:
            return
        pipeline = (self.cfg.pipeline or "").strip().lower()
        if pipeline not in ("planner_executor", "planner-executor", "pe"):
            return
        path = self.cfg.memory_dir / "pe_plan_history.jsonl"
        payload = {
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "high_level_id": plan.high_level_id,
            "goal": plan.goal,
            "thoughts": plan.thoughts,
            "steps": [
                {"step_id": s.step_id, "type": s.type, "name": s.name, "args": s.args}
                for s in (plan.action_list or [])
            ],
        }
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            return

    def _init_action_catalog(self) -> None:
        # `action_catalog.txt` is optional. Keep it in-memory instead of writing another
        # per-run file under epm/runs/.../memory.

        self._action_catalog_text = ""
        if self.cfg.action_catalog_path is not None and self.cfg.action_catalog_path.exists():
            catalog = ActionCatalog.from_text(self.cfg.action_catalog_path)
            if self.cfg.exposed_action_categories:
                catalog = catalog.select(category_queries=self.cfg.exposed_action_categories)
            self._action_catalog_text = catalog.to_prompt_text(
                max_actions_per_category=self.cfg.max_actions_per_category
            )
        try:
            (self.cfg.memory_dir / "action_catalog.txt").unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

        # Supported items snapshot (static names for prompting; replaces list_supported_items skill).
        self._write_supported_items_snapshot()

    def _write_supported_items_snapshot(self) -> None:
        """
        Write a static snapshot of supported item/platform/tool names for prompt injection.
        This replaces the list_supported_items skill in the prompt.
        """
        def _sanitize_for_prompt(obj: Any) -> Any:
            if isinstance(obj, dict):
                out: dict[str, Any] = {}
                for k, v in obj.items():
                    key = str(k)
                    if key == "object_id":
                        continue
                    if key.endswith("_path") or key == "path":
                        continue
                    out[key] = _sanitize_for_prompt(v)
                return out
            if isinstance(obj, list):
                return [_sanitize_for_prompt(v) for v in obj]
            return obj

        path = Path(self.memory.paths.get("supported_items") or (self.cfg.memory_dir / "supported_items.txt"))
        try:
            from epm.cerebellum.skills.list_supported_items.skill import ListSupportedItemsArgs, run as run_list_supported_items

            res = run_list_supported_items(args=ListSupportedItemsArgs())
            if not bool(res.success) or not isinstance(res.raw, dict):
                text = f"Supported items snapshot failed: {res.error or 'unknown_error'}"
            else:
                payload = _sanitize_for_prompt(res.raw)
                text = "Supported items (static snapshot from data mapping):\n" + json.dumps(
                    payload, ensure_ascii=False, indent=2
                )
            path.write_text(text.strip() + "\n", encoding="utf-8")
        except Exception as e:
            try:
                path.write_text(f"Supported items snapshot failed: {e!r}\n", encoding="utf-8")
            except Exception:
                pass

    def _read_order_state(self) -> dict:
        path = self.cfg.memory_dir / "order_state.json"
        if not path.exists():
            return {
                "ordered_dish_ids": [],
                "ordered_dish_names": [],
                "last_ordered_dish_id": None,
                "last_ordered_dish_name": "",
            }
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _is_dish_ordered(self) -> bool:
        state = self._read_order_state()
        ordered_ids = state.get("ordered_dish_ids")
        if not isinstance(ordered_ids, list):
            ordered_ids = []
        try:
            if int(self.dish.id) in {int(x) for x in ordered_ids if isinstance(x, (int, str))}:
                return True
        except Exception:
            pass
        ordered_names = state.get("ordered_dish_names")
        if isinstance(ordered_names, list) and self.dish.dish_name:
            return self.dish.dish_name in ordered_names
        return False

    def _mark_dish_ordered(self) -> None:
        state = self._read_order_state()
        ordered_ids = state.get("ordered_dish_ids")
        if not isinstance(ordered_ids, list):
            ordered_ids = []
        ordered_ids = [int(x) for x in ordered_ids if isinstance(x, (int, str))]
        if int(self.dish.id) not in ordered_ids:
            ordered_ids.append(int(self.dish.id))
        ordered_names = state.get("ordered_dish_names")
        if not isinstance(ordered_names, list):
            ordered_names = []
        if self.dish.dish_name and self.dish.dish_name not in ordered_names:
            ordered_names.append(self.dish.dish_name)
        state.update(
            {
                "ordered_dish_ids": ordered_ids,
                "ordered_dish_names": ordered_names,
                "last_ordered_dish_id": int(self.dish.id),
                "last_ordered_dish_name": str(self.dish.dish_name),
            }
        )
        try:
            path = self.cfg.memory_dir / "order_state.json"
            path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _build_skills_catalog_from_local_actions_by_category(*, allowed_actions: list[str]) -> list[dict]:
        """
        Build a category->actions catalog from:
          epm/src/epm/cerebellum/local_actions_by_category.txt

        This file is generated from live code and is a better source of truth than the
        legacy taxonomy file used by ActionCatalog.from_text().
        """
        try:
            epm_dir = Path(__file__).resolve().parents[3]  # <repo>/epm
            src_path = epm_dir / "src" / "epm" / "cerebellum" / "local_actions_by_category.txt"
            if not src_path.exists():
                return []
            text = src_path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            return []

        allow = set(str(x) for x in (allowed_actions or []))
        header_re = re.compile(r"^\s*\[(?P<cat>[^\]]+)\]")
        action_re = re.compile(r"^\s*-\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")

        skills: list[dict] = []
        current_cat: str | None = None
        current_actions: list[str] = []

        def _flush() -> None:
            nonlocal current_cat, current_actions, skills
            if not current_cat:
                current_actions = []
                return
            filtered = [a for a in current_actions if a in allow]
            if filtered:
                skills.append(
                    {
                        "name": current_cat,
                        "category": current_cat,
                        "description": f"Use actions in category [{current_cat}] to solve the current subgoal.",
                        "allowed_actions": filtered,
                    }
                )
            current_actions = []

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            hm = header_re.match(line)
            if hm:
                _flush()
                current_cat = hm.group("cat").strip()
                continue
            am = action_re.match(line)
            if am and current_cat:
                current_actions.append(am.group("name").strip())
                continue

        _flush()
        return skills

    @staticmethod
    def _load_prompt_layout(path: Optional[Path]) -> Optional[dict]:
        if path is None:
            return None
        try:
            p = Path(path)
            if not p.exists():
                return None
            raw = json.loads(p.read_text(encoding="utf-8-sig", errors="replace"))
            return raw if isinstance(raw, dict) else None
        except Exception:
            return None

    def _build_prompter(self, *, plan_min_steps: int, plan_max_steps: int, tool_calling: bool) -> PromptBuilder:
        return PromptBuilder(
            # Prompt layout is optional and should not block execution.
            PromptBuilderConfig(
                inject_memory=bool(self.cfg.inject_memory),
                inject_tool_schemas=bool(self.cfg.inject_tool_schemas),
                save_last_prompt_path=self.cfg.memory_dir / "prompt_last.txt",
                save_per_step_dir=self.cfg.memory_dir / "prompts",
                plan_min_steps=int(plan_min_steps),
                plan_max_steps=int(plan_max_steps),
                force_submit_step_threshold=int(getattr(self.cfg, "force_submit_step_threshold", 0) or 0),
                force_submit_within_steps=int(getattr(self.cfg, "force_submit_within_steps", 50) or 50),
                tool_calling=bool(tool_calling),
                prompt_layout=self._load_prompt_layout(self.cfg.prompt_layout_path),
                prompt_ablation_profile=self.prompt_ablation_profile,
                prompt_disabled_groups=self.prompt_disabled_groups,
                vision_image_max_side=int(getattr(self.cfg.vlm, "image_max_side", 768) or 768),
            )
        )

    def _set_planner_tool_calling(self, enabled: bool) -> None:
        if isinstance(self.planner, VlmPlanner):
            self.planner.set_tool_calling_enabled(bool(enabled))
        elif isinstance(self.planner, OpenAIPlanner):
            self.planner.set_tool_calling_enabled(bool(enabled))

    def _probe_tool_calling_support(self) -> Optional[bool]:
        if isinstance(self.planner, VlmPlanner):
            return self.planner.probe_tool_calling_support()
        if isinstance(self.planner, OpenAIPlanner):
            return self.planner.probe_tool_calling_support()
        return None

    def _write_tool_manifest(self) -> None:
        """
        Write a model-friendly tool definition file (OpenAI `tools=[...]`).

        This is the authoritative, consolidated view for tool calling. The file is stable (no per-run deltas),
        so we only write it to the repo-level `epm/memory/` directory.
        """
        try:
            tools = build_tool_manifest(allowed_actions=self.allowed_actions, allowed_skills=self.allowed_skills)
            openai_tools = to_openai_tools(tools)

            def _write(path: Path, text: str) -> None:
                try:
                    # Atomic write to avoid leaving a truncated/invalid JSON file if the process is interrupted.
                    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
                    tmp.write_text(text, encoding="utf-8")
                    tmp.replace(path)
                except PermissionError:
                    # Windows sync/AV can briefly lock files; fall back to a unique name.
                    pid = os.getpid()
                    path.with_name(path.name + f".fallback.{pid}").write_text(text, encoding="utf-8")

            text_openai = json.dumps(openai_tools, ensure_ascii=False, indent=2) + "\n"

            # Stable repo-level memory dir (epm/memory).
            try:
                epm_dir = Path(__file__).resolve().parents[3]  # .../epm/
                stable_dir = epm_dir / "memory"
                stable_dir.mkdir(parents=True, exist_ok=True)
                out_path = stable_dir / "tools_manifest_openai.json"
                # Do not auto-overwrite a user-curated manifest. Regenerate manually via:
                #   python epm/test/generate_tools_manifest.py
                # To force overwrite, set env var:
                #   EPM_FORCE_WRITE_TOOLS_MANIFEST=1
                if out_path.exists() and os.environ.get("EPM_FORCE_WRITE_TOOLS_MANIFEST", "").strip() != "1":
                    return
                _write(out_path, text_openai)
            except Exception:
                pass
        except Exception:
            # Best-effort only; never block execution.
            return

    def _pick_current_high_level_goal(self) -> tuple[str, str]:
        """
        Best-effort parser for current high-level goal from task_progress file.
        Falls back to H1 + "Follow recipe".
        """
        if str(self.pipeline_name or "").strip().lower() in ("epm", "epm_agent"):
            try:
                goal_tree_path = self.cfg.memory_dir / "goal_tree.json"
                if goal_tree_path.exists():
                    obj = json.loads(goal_tree_path.read_text(encoding="utf-8-sig"))
                    if isinstance(obj, dict):
                        subgoals = obj.get("subgoals")
                        idx = int(obj.get("current_index", 0) or 0)
                        if isinstance(subgoals, list) and 0 <= idx < len(subgoals):
                            subgoal = str(subgoals[idx] or "").strip()
                            if subgoal:
                                return f"H{idx + 1}", subgoal
            except Exception:
                pass
        path = self.task_progress_path
        if path is None or not path.exists():
            return "H1", "Follow recipe"
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(obj, dict):
                goal_state = obj.get("goal_state")
                if isinstance(goal_state, dict):
                    current_id = str(goal_state.get("current_high_level_id") or "").strip()
                    current_goal = str(
                        goal_state.get("current_high_level_goal")
                        or goal_state.get("current_subgoal")
                        or goal_state.get("current_focus")
                        or ""
                    ).strip()
                    if current_id or current_goal:
                        return current_id or "H1", current_goal or "Follow recipe"
        except Exception:
            pass
        return "H1", "Follow recipe"

    def _ensure_atomic_plan_written(self, plan: PlanResponse) -> None:
        """
        Legacy hook kept for compatibility.

        EPM task_progress is now a semantic state summary, not an atomic action queue,
        so there is nothing to append here.
        """
        _ = plan
        return

    def run_one_step(self):
        step_start = time.monotonic()
        step_wall_time = time.time()
        # Global emergency stop (Ctrl+Alt+Q). This needs to be checked here (not only
        # inside mouse/keyboard methods), otherwise non-input skills like query_scene_objects
        # will keep running after stop is requested.
        try:
            RawInputController.check_stop()
        except StopRequested:
            raise
        except Exception:
            pass
        self.step_id += 1
        logical_committed_step = self._logical_committed_step_id()
        self._write_current_step_status(
            phase="step_started",
            step=None,
            result="",
            error="",
        )
        if self.cfg.verbose and self.cfg.log_mode != "minimal":
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            self.log.info(
                f"\n[EPM][{now_iso}] ===== attempt_step {self.step_id} "
                f"(logical_committed_step={logical_committed_step}) ====="
            )
        obs = self.world.observe(
            frame_id=str(self.step_id),
            screenshot_filename=self._step_screenshot_filename(step_id=self.step_id),
        )

        high_level_id, high_level_goal = self._pick_current_high_level_goal()

        # Update agent_state.json as a lightweight "action feedback" snapshot before building the prompt.
        try:
            self._update_agent_state_from_realtime_products()
        except Exception as e:
            if self.cfg.verbose:
                self.log.info(f"[EPM] agent_state update failed: {e!r}")
        try:
            self._tick_timers(step_id=self.step_id)
        except Exception as e:
            if self.cfg.verbose:
                self.log.info(f"[EPM] timer_tick failed: {e!r}")
        try:
            self._update_force_submit_status(step_id=self.step_id)
        except Exception as e:
            if self.cfg.verbose:
                self.log.info(f"[EPM] force_submit update failed: {e!r}")

        bundle = self.memory.bundle_for_brain()
        if restrict_to_raw_input_actions(
            self.prompt_ablation_profile,
            disabled_groups=self.prompt_disabled_groups,
        ) or restrict_to_actions_only(
            self.prompt_ablation_profile,
            disabled_groups=self.prompt_disabled_groups,
        ):
            action_only_manifest = build_tool_manifest(allowed_actions=self.allowed_actions, allowed_skills=[])
            bundle["tools_manifest_openai"] = json.dumps(to_openai_tools(action_only_manifest), ensure_ascii=False, indent=2)
            bundle["action_specs"] = action_specs_to_prompt_text(action_names=self.allowed_actions).strip()
            bundle["skill_specs"] = ""
            bundle["skills_catalog"] = ""
        if self._action_catalog_text:
            bundle["action_catalog"] = self._action_catalog_text
        bundle["recipe_text"] = self.dish.recipe_text
        bundle["task_execution_context"] = self._build_task_execution_context_text()
        bundle["_inject_memory"] = bool(self.cfg.inject_memory)
        bundle["_inject_tool_schemas"] = bool(self.cfg.inject_tool_schemas)
        bundle["_prompt_ablation_profile"] = self.prompt_ablation_profile
        bundle["_prompt_ablation_disabled_groups"] = sorted(self.prompt_disabled_groups)
        camera_info = self._read_camera_info()
        if camera_info:
            bundle["camera_info"] = camera_info
        interaction_info = self._read_alt_j_interaction_info()
        if interaction_info:
            bundle["interaction_info"] = interaction_info

        raw_prompt_policy = self.cfg.prompt_policy if isinstance(self.cfg.prompt_policy, dict) else {}
        bundle["_include_percept"] = bool(raw_prompt_policy.get("include_percept", True))
        bundle["_include_oracle_observation"] = bool(raw_prompt_policy.get("include_oracle_observation", True))
        bundle["_include_task_progress"] = bool(raw_prompt_policy.get("include_task_progress", True))
        bundle["_include_tool_en_catalog"] = bool(raw_prompt_policy.get("include_tool_en_catalog", True))
        bundle["_show_high_level_id"] = str(self.pipeline_name in ("epm", "epm_agent")).lower()
        bundle["_pipeline_name"] = str(self.pipeline_name or "").strip().lower()
        bundle["_episode_step"] = int(self.step_id)
        bundle["query_memory"] = self._render_query_memory(max_entries=-1, max_chars=-1)
        bundle["instance_query_snapshot"] = self._build_instance_query_snapshot(feedback=self._last_action_feedback)
        if self.pipeline_name != "reflexion":
            # Reflexion memory is a method-specific prompt section; keep it out of other pipelines.
            bundle.pop("reflexion_memory", None)
            bundle.pop("reflexion_progress_memory", None)
        # Parameter heuristics are disabled for now (do not inject into prompts).
        bundle.pop("parameter_heuristics", None)

        try:
            force_state = self.memory.read_json("agent_state")
        except Exception:
            force_state = {}
        force_active = bool(force_state.get("force_submit_active", False))
        run_planner_aux = self._should_run_planner_aux_modules(force_active=force_active)
        model_obs = self._model_observation(obs)
        percept = self.perception.run(observation=model_obs) if run_planner_aux else Percept(text="", slots={})

        # Force-submit mode: restrict allowlists and prompt content after threshold triggers.
        effective_allowed_actions = self.allowed_actions
        effective_allowed_skills = self.allowed_skills
        if force_active:
            fa, fs = self._force_submit_allowlists()
            effective_allowed_actions = [a for a in fa if a in (self.allowed_actions or [])]
            effective_allowed_skills = [s for s in fs if s in (self.allowed_skills or [])]
            self._apply_force_submit_bundle_overrides(
                bundle=bundle,
                allowed_actions=effective_allowed_actions,
                allowed_skills=effective_allowed_skills,
            )
            self._apply_force_submit_planner_overrides(
                allowed_actions=effective_allowed_actions,
                allowed_skills=effective_allowed_skills,
            )

        context = PlannerContext(
            high_level_id=high_level_id,
            high_level_goal=high_level_goal,
            observation=model_obs,
            recipe_text=self.dish.recipe_text,
            feedback=self._last_action_feedback,
            memory_bundle=bundle,
            percept=percept,
        )

        step: PlanStep
        planning_start = time.monotonic()
        planning_duration_s = 0.0
        try:
            if force_active:
                step = self._force_submit_decide_step(state=force_state, high_level_id=high_level_id)
                if self.cfg.verbose and self.cfg.log_mode != "minimal":
                    self.log.info(f"[EPM] force_submit_override=true step={step.name} args={step.args}")
            else:
                step = self.pipeline.next_step(context=context)
            planning_duration_s = max(0.0, time.monotonic() - planning_start)
        except NetworkPauseRequired as e:
            planning_duration_s = max(0.0, time.monotonic() - planning_start)
            self.step_id = max(0, int(self.step_id) - 1)
            detail = _format_network_error_detail(
                e,
                episode_step=int(self.step_id) + 1,
                rolled_back_to=int(self.step_id),
            )
            self._write_current_step_status(
                phase="waiting_network",
                step=None,
                result="",
                error=detail,
            )
            if self.cfg.verbose and self.cfg.log_mode != "minimal":
                self.log.warning(f"[EPM] planner_network_pause {detail}")
            raise
        except NetworkAbortRequired as e:
            planning_duration_s = max(0.0, time.monotonic() - planning_start)
            self.step_id = max(0, int(self.step_id) - 1)
            detail = _format_network_error_detail(
                e,
                episode_step=int(self.step_id) + 1,
                rolled_back_to=int(self.step_id),
            )
            self._write_current_step_status(
                phase="network_error",
                step=None,
                result="",
                error=detail,
            )
            if self.cfg.verbose and self.cfg.log_mode != "minimal":
                self.log.error(f"[EPM] planner_network_abort {detail}")
            raise
        except Exception as e:
            planning_duration_s = max(0.0, time.monotonic() - planning_start)
            network_signal = classify_network_exception(e, source="agent.run_one_step.planner_exception")
            if network_signal is not None:
                self.step_id = max(0, int(self.step_id) - 1)
                pause_phase = "waiting_network" if isinstance(network_signal, NetworkPauseRequired) else "network_error"
                detail = _format_network_error_detail(
                    network_signal,
                    episode_step=int(self.step_id) + 1,
                    rolled_back_to=int(self.step_id),
                )
                self._write_current_step_status(
                    phase=pause_phase,
                    step=None,
                    result="",
                    error=detail,
                )
                if self.cfg.verbose and self.cfg.log_mode != "minimal":
                    self.log.warning(f"[EPM] planner_network_reclassified {detail}")
                raise network_signal from e
            if isinstance(e, RuntimeError) and str(e).startswith("fatal_episode_error:"):
                if self.cfg.verbose and self.cfg.log_mode != "minimal":
                    self.log.error(f"[EPM] planner_fatal step={self.step_id} err={e!r}")
                raise
            if type(e).__name__ == "OpenLoopPlanExhausted":
                force_state = self._activate_force_submit_now(
                    step_id=self.step_id,
                    reason="open_loop_plan_exhausted",
                )
                step = self._force_submit_decide_step(state=force_state, high_level_id=high_level_id)
                if self.cfg.verbose and self.cfg.log_mode != "minimal":
                    self.log.info(
                        f"[EPM] strict_open_loop_plan_exhausted step={self.step_id} -> "
                        f"arm_force_submit next={step.name} args={step.args}"
                    )
                planning_duration_s = max(0.0, time.monotonic() - planning_start)
            else:
                # Planner failures (e.g. invalid JSON) should not crash the whole episode.
                error_msg = f"planner_exception:{type(e).__name__}:{str(e)}"
                if self.cfg.verbose and self.cfg.log_mode != "minimal":
                    self.log.info(f"[EPM] planner_exception step={self.step_id} err={error_msg!r}")
                # Feed back into next round to encourage strict JSON/tool-name compliance.
                self._last_action_feedback = (
                    "planner_error=true\n"
                    f"error={error_msg}\n"
                    "instruction=Return ONE valid JSON plan object; keep action_list within the required length; "
                    "use exact action/skill names from schemas.\n"
                )
                self.memory.record_step(
                    step_id=self.step_id,
                    observation_summary="planner_exception",
                    planned=None,
                    executed=Decision(action_or_skill="(planner_exception)", params={}, plan_ref=PlanRef(high_level_id, "P0")),
                    result_summary="failure",
                    screenshot_path=obs.screenshot_path,
                    time_iso=obs.time,
                    duration_s=time.monotonic() - step_start,
                    episode_elapsed_s=time.monotonic() - self._episode_start_monotonic,
                    diff={"type": "planner_exception", "details": {"error": error_msg}},
                    errors=error_msg,
                )
                self._write_current_step_status(
                    phase="planner_exception",
                    step=None,
                    result="failure",
                    error=error_msg,
                )
                return obs

        # Planner sometimes mislabels an action as a skill (or vice versa) because
        # Skill Cards contain category names like "PickupObject(...)". This can
        # cause a "planner_invalid_step: unknown_skill" loop. We coerce the type
        # using allowlists before validation/execution.
        step = self._coerce_step_type_with_allowlist(step, allowed_actions=effective_allowed_actions, allowed_skills=effective_allowed_skills)
        step = self._canonicalize_dish_name_for_gui_steps(step)
        # If a new plan was generated, write atomic steps into task_progress for alignment.
        # EPM precondition checking happens once at plan-generation / repair time, not before every queued execution step.
        new_plan = self._consume_pipeline_new_plan_if_any()
        precondition_result = None
        if new_plan is not None:
            if self._should_run_task_progress_for_new_plan():
                self._maybe_run_task_progress_maintenance(
                    step_id=self.step_id,
                    new_plan_generated=True,
                    need_replan=False,
                )
            step, precondition_result = self._epm_validate_step_before_execution(
                step=step,
                obs=obs,
                bundle=bundle,
                percept=percept,
                high_level_id=high_level_id,
                high_level_goal=high_level_goal,
            )

        # Validate allowlists (early fail for mechanism experiments)
        plan_stub = PlanResponse(
            high_level_id=high_level_id,
            goal=high_level_goal,
            explanation=None,
            thoughts="",
            action_list=[step],
        )
        ok, err = validate_plan_against_allowlist(
            plan_stub,
            allowed_actions=effective_allowed_actions,
            allowed_skills=effective_allowed_skills,
        )
        if not ok:
            # Treat as a planning failure: record it, feed it back, and continue next step.
            error_msg = f"invalid_plan:{err}"
            if self.cfg.verbose and self.cfg.log_mode != "minimal":
                self.log.info(f"[EPM] planner_invalid_step step_id={step.step_id} type={step.type} name={step.name} err={error_msg}")
            self.pipeline.on_step_result(step=step, success=False, error=error_msg)
            if self.cfg.log_mode == "minimal":
                self._log_minimal_step(step=step, result_summary="failure", errors=error_msg)
            self.memory.record_step(
                step_id=self.step_id,
                observation_summary="planner_invalid_step",
                planned=Decision(action_or_skill=step.name, params=dict(step.args), plan_ref=PlanRef(high_level_id, step.step_id)),
                executed=Decision(action_or_skill="(none)", params={}, plan_ref=PlanRef(high_level_id, step.step_id)),
                result_summary="failure",
                screenshot_path=obs.screenshot_path,
                time_iso=obs.time,
                duration_s=time.monotonic() - step_start,
                episode_elapsed_s=time.monotonic() - self._episode_start_monotonic,
                diff={"type": "planner_invalid_step", "details": {"error": error_msg}},
                errors=error_msg,
            )
            self._write_current_step_status(
                phase="planner_invalid_step",
                step=step,
                result="failure",
                error=error_msg,
            )
            return obs

        planned = Decision(action_or_skill=step.name, params=dict(step.args), plan_ref=PlanRef(high_level_id, step.step_id))
        self._write_current_step_status(
            phase="planned",
            step=step,
            result="",
            error="",
        )
        if self.cfg.verbose and self.cfg.log_mode != "minimal":
            self.log.info(f"[EPM] step={self.step_id} decide {step.type}:{step.name} args={step.args}")
            self.log.info(f"[EPM] planned_step={step.step_id} type={step.type} name={step.name} args={step.args}")
            try:
                lay = getattr(getattr(self, "prompter", None), "cfg", None)
                lay = getattr(lay, "prompt_layout", None) if lay is not None else None
                lay = lay if isinstance(lay, dict) else {}
                if "skill_cards" not in (lay.get("system_sections", []) or []):
                    raise RuntimeError("skill_cards_disabled")
                mode = str(lay.get("skill_cards_mode", "selected") or "selected").strip().lower()
                if mode == "all":
                    selected_cards = list_skill_card_names(skills_catalog_json=bundle.get("skills_catalog", ""))
                else:
                    # max_cards=0 means "no cap"
                    max_cards = 0
                    limits = lay.get("limits", None)
                    if isinstance(limits, dict):
                        try:
                            max_cards = int(limits.get("skill_cards_max_cards", 0) or 0)
                        except Exception:
                            max_cards = 0
                    selected_cards = select_skill_cards(
                        skills_catalog_json=bundle.get("skills_catalog", ""),
                        high_level_goal=high_level_goal,
                        recipe_text=bundle.get("recipe_text", ""),
                        feedback=self._last_action_feedback,
                        max_cards=max_cards,
                    )
                if selected_cards:
                    self.log.info(f"[EPM] selected_skill_cards={selected_cards}")
            except Exception:
                pass
        self._append_step_trace(kind="plan", step=step, result="", error="")

        exec_stdout = None
        if self.cfg.log_mode == "minimal":
            exec_stdout = io.StringIO()

        def _exec_step_with_args(args_for_call: dict[str, Any]) -> ActionResult:
            if exec_stdout is not None:
                with redirect_stdout(exec_stdout):
                    if step.type == "skill":
                        return run_skill(
                            self.api,
                            name=step.name,
                            args=dict(args_for_call),
                            realtime_products_path=self.cfg.realtime_products_path,
                        )
                    return self.api.call_action(step.name, **dict(args_for_call))
            if step.type == "skill":
                return run_skill(
                    self.api,
                    name=step.name,
                    args=dict(args_for_call),
                    realtime_products_path=self.cfg.realtime_products_path,
                )
            return self.api.call_action(step.name, **dict(args_for_call))

        execution_start = time.monotonic()
        execution_duration_s = 0.0
        executed_args: dict[str, Any] = dict(step.args or {})
        try:
            executed_args = dict(step.args or {})
            if precondition_result is not None:
                action_result = precondition_result
            elif step.name == "gui_order_dish_via_computer" and self._is_dish_ordered():
                action_result = ActionResult(
                    True,
                    raw={"skipped": "already_ordered", "dish_id": int(self.dish.id), "dish_name": str(self.dish.dish_name)},
                    error="",
                )
            elif step.type == "skill" and step.name == "auto_navigation":
                blocked, mode_raw = self._is_navigation_blocked_by_mode()
                if blocked:
                    mode_l = str(mode_raw or "").strip().lower()
                    mode_class = "interaction_mode"
                    mode_hint = "Exit current interaction mode first, then retry auto_navigation."
                    if ("pour" in mode_l) or (mode_l == "pour"):
                        mode_class = "pour"
                        mode_hint = "Call action:exit_pouring_mode first, then retry auto_navigation."
                    elif ("cut" in mode_l) or (mode_l == "cut"):
                        mode_class = "cut"
                        mode_hint = "Finish current cutting or leave cutting state, then retry auto_navigation."
                    elif ("mix" in mode_l) or (mode_l == "mix"):
                        mode_class = "mix"
                        mode_hint = "Finish current mixing or leave mixing state, then retry auto_navigation."
                    elif ("sprinkle" in mode_l) or (mode_l == "sprinkle"):
                        mode_class = "sprinkle"
                        mode_hint = "Finish sprinkle operation (or exit sprinkle state), then retry auto_navigation."
                    elif ("flip" in mode_l) or (mode_l == "flip"):
                        mode_class = "flip"
                        mode_hint = "Finish flipping operation (or exit flip state), then retry auto_navigation."
                    action_result = ActionResult(
                        False,
                        raw={
                            "blocked_by_mode": mode_raw,
                            "blocked_mode_class": mode_class,
                            "hint": (
                                "Cannot navigate while currently in interaction mode. "
                                "Either finish the current mode action or exit the mode first, then retry auto_navigation."
                            ),
                            "mode_hint": mode_hint,
                        },
                        error="navigation_blocked_by_mode",
                    )
                else:
                    action_result = _exec_step_with_args(executed_args)
            elif step.type == "action" and step.name in {"Horizontal_movement", "horizontal_movement"}:
                allowed, mode_raw = self._horizontal_movement_mode_check()
                if not allowed:
                    action_result = ActionResult(
                        False,
                        raw={
                            "current_mode": mode_raw,
                            "hint": (
                                "Horizontal_movement is only valid in interaction modes "
                                "(pouring/cutting/mixing/flip/sprinkle). Enter the required mode first, "
                                "then retry Horizontal_movement."
                            ),
                        },
                        error="horizontal_movement_requires_interaction_mode",
                    )
                else:
                    action_result = _exec_step_with_args(executed_args)
            else:
                action_result = _exec_step_with_args(executed_args)

            # Inline instance-id repair (same step): let planner choose instance_id, then retry once.
            if (
                not bool(action_result.success)
                and str(action_result.error or "") in {"instance_id_required", "instance_id_not_found"}
                and isinstance(action_result.raw, dict)
            ):
                try:
                    repair_feedback = render_action_feedback(
                        step=step,
                        action_result=action_result,
                        final_success=False,
                        error=str(action_result.error or ""),
                    )
                    repair_bundle = dict(bundle)
                    repair_bundle["instance_query_snapshot"] = self._build_instance_query_snapshot(feedback=repair_feedback)
                    repair_context = PlannerContext(
                        high_level_id=high_level_id,
                        high_level_goal=high_level_goal,
                        observation=self._model_observation(obs),
                        recipe_text=self.dish.recipe_text,
                        feedback=repair_feedback,
                        memory_bundle=repair_bundle,
                        percept=percept,
                    )
                    repair_prompt = self.prompter.build_planner_prompt(
                        observation=repair_context.observation,
                        high_level_id=repair_context.high_level_id,
                        high_level_goal=repair_context.high_level_goal,
                        memory_bundle=repair_context.memory_bundle,
                        feedback=repair_context.feedback,
                        percept_text=(repair_context.percept.text if repair_context.percept else ""),
                        inject_memory=bool(self.cfg.inject_memory),
                        inject_tool_schemas=bool(self.cfg.inject_tool_schemas),
                        include_task_progress=bool((self.cfg.prompt_policy or {}).get("include_task_progress", True)),
                    )
                    repair_plan = self.planner.plan(context=repair_context, prompt=repair_prompt)
                    retry_step = repair_plan.action_list[0] if (repair_plan and repair_plan.action_list) else None
                    if retry_step is not None and str(retry_step.name or "") == str(step.name or ""):
                        retry_args = dict(executed_args)
                        retry_args.update(dict(retry_step.args or {}))
                        if self.cfg.verbose and self.cfg.log_mode != "minimal":
                            self.log.info(
                                f"[EPM] inline_instance_repair step={self.step_id} name={step.name} retry_args={retry_args}"
                            )
                        retry_result = _exec_step_with_args(retry_args)
                        retry_raw = dict(retry_result.raw or {})
                        retry_raw["inline_instance_repair"] = {
                            "applied": True,
                            "repair_step_id": str(getattr(retry_step, "step_id", "") or ""),
                            "repair_args": dict(retry_step.args or {}),
                        }
                        action_result = ActionResult(
                            success=bool(retry_result.success),
                            raw=retry_raw,
                            error=str(retry_result.error or ""),
                        )
                        executed_args = retry_args
                except NetworkPauseRequired:
                    raise
                except NetworkAbortRequired:
                    raise
                except Exception as _repair_e:
                    if self.cfg.verbose and self.cfg.log_mode != "minimal":
                        self.log.warning(f"[EPM] inline_instance_repair_failed step={self.step_id} err={_repair_e!r}")

        finally:
            execution_duration_s = max(0.0, time.monotonic() - execution_start)
            if exec_stdout is not None:
                captured = exec_stdout.getvalue()
                if captured.strip():
                    self._append_exec_stdout(text=captured)

        executed = Decision(action_or_skill=step.name, params=dict(executed_args), plan_ref=PlanRef(high_level_id, step.step_id))
        # Post-action observation (same step, _post screenshot) for realtime feedback/success judgement.
        post_obs = self.world.observe(
            frame_id=str(self.step_id),
            screenshot_filename=self._step_screenshot_filename(step_id=self.step_id, post=True),
        )

        judge_result_box: dict[str, Any] = {}
        judge_thread: Optional[threading.Thread] = None
        if self.success_judge is not None:
            def _judge_worker() -> None:
                try:
                    judge_result_box["result"] = self.success_judge.judge(
                        step=step,
                        pre=self._model_observation(obs),
                        post=self._model_observation(post_obs),
                        api_success=bool(action_result.success),
                        api_error=str(action_result.error or ""),
                    )
                except Exception as e:
                    judge_result_box["error"] = e

            judge_thread = threading.Thread(target=_judge_worker, name=f"step-success-judge-{self.step_id}", daemon=True)
            judge_thread.start()

        anomaly_result_box: dict[str, Any] = {}
        anomaly_thread: Optional[threading.Thread] = None

        # Model-judged success (optional)
        judged_success = None
        judged_reason = ""
        if judge_thread is not None:
            judge_thread.join()
            j = judge_result_box.get("result")
            if j is not None:
                judged_success = bool(j.success)
                judged_reason = (j.reason or "").strip()
                if self.cfg.verbose and self.cfg.log_mode != "minimal":
                    self.log.info(f"[EPM] judge_success={judged_success} reason={judged_reason!r}")
            elif self.cfg.verbose and self.cfg.log_mode != "minimal" and "error" in judge_result_box:
                self.log.warning(f"[EPM] step_success_judge_failed step={self.step_id} err={judge_result_box['error']!r}")

        # Final success for closed-loop control:
        # - if executor crashed, always failure
        # - otherwise, follow judge if enabled; else follow executor
        if not action_result.success:
            final_success = False
        elif judged_success is not None:
            final_success = bool(judged_success)
        else:
            final_success = True

        result_summary = "success" if final_success else "failure"
        errors = ""
        if not action_result.success:
            errors = str(action_result.error or "")
        elif judged_success is not None and not judged_success:
            errors = judged_reason or "judge_marked_failure"

        # Strict post-check: put_down must leave the agent in idle / hands-empty state.
        if step.name == "put_down" and final_success:
            ok_idle, mode_raw, is_held = self._post_check_put_down_idle()
            if not ok_idle:
                final_success = False
                result_summary = "failure"
                errors = (
                    f"post_check_failed:put_down_not_idle mode={mode_raw!r} is_held={is_held!r}; "
                    "likely=occupied_place_or_item_not_placeable_here; "
                    "hint=choose_another_empty_place_point_and_retry_put_down"
                )
        # Pick-up post-check: always include post-state details.
        if step.name == "pick_up":
            pick_state = self._post_check_pick_up_state()
            raw_payload = dict(action_result.raw or {})
            raw_payload["return"] = {
                "is_held": bool(pick_state.get("is_held")),
                "held_item": pick_state.get("held_item"),
                "held_item_instance_id": pick_state.get("held_item_instance_id"),
                "item_kind": pick_state.get("item_kind"),
                "mode": pick_state.get("mode"),
                "api_return": raw_payload.get("return"),
            }
            recovered_by_post_check = (not bool(action_result.success)) and bool(pick_state.get("is_held"))
            if recovered_by_post_check:
                raw_payload["post_check_recovered_success"] = True
                raw_payload["post_check_recovered_reason"] = "pick_up_holding_confirmed_after_action"
            action_result = ActionResult(
                success=bool(action_result.success) or recovered_by_post_check,
                raw=raw_payload,
                error=("" if recovered_by_post_check else str(action_result.error or "")),
            )
            if final_success and not bool(pick_state.get("is_held")):
                final_success = False
                result_summary = "failure"
                errors = (
                    "post_check_failed:pick_up_not_holding_item "
                    f"is_held={pick_state.get('is_held')!r} "
                    f"held_item={pick_state.get('held_item')!r} "
                    f"held_item_instance_id={pick_state.get('held_item_instance_id')!r} "
                    f"mode={pick_state.get('mode')!r}; "
                    "likely=target_not_interactable_or_crosshair_not_aligned_or_too_far; "
                    "hint=align_crosshair_to_target_navigate_closer_and_retry_pick_up"
                )
            elif not final_success:
                if bool(pick_state.get("is_held")):
                    final_success = True
                    result_summary = "success"
                    errors = ""
                else:
                    details = (
                        f"is_held={pick_state.get('is_held')!r} "
                        f"held_item={pick_state.get('held_item')!r} "
                        f"held_item_instance_id={pick_state.get('held_item_instance_id')!r} "
                        f"mode={pick_state.get('mode')!r}"
                    )
                    base = str(errors or "").strip()
                    errors = f"{base}; post_state:{details}" if base else f"post_state:{details}"
        if step.name == "gui_buy_new_item":
            buy_state = self._post_check_pick_up_state()
            raw_payload = dict(action_result.raw or {})
            raw_payload["return"] = {
                "is_held": bool(buy_state.get("is_held")),
                "held_item": buy_state.get("held_item"),
                "held_item_instance_id": buy_state.get("held_item_instance_id"),
                "item_kind": buy_state.get("item_kind"),
                "mode": buy_state.get("mode"),
                "api_return": raw_payload.get("return"),
            }
            action_result = ActionResult(
                success=bool(action_result.success),
                raw=raw_payload,
                error=str(action_result.error or ""),
            )

        if errors:
            errors = self._augment_mode_error_details(
                error=errors,
                raw=(action_result.raw if isinstance(action_result.raw, dict) else {}),
            )

        if self.visual_anomaly_observer is not None and self._should_run_visual_aux_after_step(force_active=force_active, final_success=final_success):
            recent_history_text = self._load_recent_stm_window_text()
            agent_state_text = self._load_agent_state_text()
            try:
                _, current_high_level_goal_for_observer = self._pick_current_high_level_goal()
            except Exception:
                current_high_level_goal_for_observer = str(high_level_goal or "")

            def _anomaly_worker() -> None:
                try:
                    result = self.visual_anomaly_observer.observe(
                        step=step,
                        pre=self._model_observation(obs),
                        post=self._model_observation(post_obs),
                        recent_history_text=recent_history_text,
                        current_high_level_goal=current_high_level_goal_for_observer,
                        last_feedback=str(self._last_action_feedback or ""),
                        agent_state_text=agent_state_text,
                    )
                except Exception as e:
                    result = VisualAnomalyObservation(
                        has_anomaly=False,
                        severity="none",
                        tags=[],
                        feedback="",
                        raw={"error": repr(e)},
                    )
                    anomaly_result_box["error"] = e
                anomaly_result_box["result"] = result
                self._persist_visual_anomaly_feedback(step_id=self.step_id, result=result)

            anomaly_thread = threading.Thread(target=_anomaly_worker, name=f"visual-anomaly-{self.step_id}", daemon=True)
            anomaly_thread.start()

        if self.cfg.verbose and self.cfg.log_mode != "minimal":
            if final_success:
                self.log.info(f"[EPM] action_result=success step={self.step_id} name={step.name}")
            else:
                self.log.warning(f"[EPM] action_result=failure step={self.step_id} name={step.name}")
                self.log.error(f"[EPM] action_error step={self.step_id} name={step.name} error={errors!r}")
            if step.type == "skill" and isinstance(action_result.raw, dict) and action_result.raw:
                if not action_result.success:
                    # Failure diagnostics should be explicit for quick root-cause inspection.
                    self.log.error(f"[EPM] skill_output.raw={action_result.raw}")
                # Print a compact preview of skill outputs (useful for query/perception skills).
                elif "results" in action_result.raw and isinstance(action_result.raw.get("results"), list):
                    results_all = action_result.raw.get("results") or []
                    self.log.info(f"[EPM] skill_output.results(len={len(results_all)})={results_all}")
                elif "visible_items" in action_result.raw and isinstance(action_result.raw.get("visible_items"), list):
                    visible_all = action_result.raw.get("visible_items") or []
                    self.log.info(f"[EPM] skill_output.visible_items(len={len(visible_all)})={visible_all}")
                else:
                    keys = list(action_result.raw.keys())[:10]
                    self.log.info(f"[EPM] skill_output.keys={keys}")

        # Persist last failure raw payload for offline debugging (avoid relying on console logs).
        if not action_result.success:
            try:
                p = self.cfg.memory_dir / "last_failure_raw.json"
                p.write_text(json.dumps(action_result.raw, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
            except Exception:
                pass
        if self.cfg.log_mode == "minimal":
            self._log_minimal_step(step=step, result_summary=result_summary, errors=errors)
        self._append_step_trace(kind="result", step=step, result=result_summary, error=errors)
        self.last_completed_step_id = int(self.step_id)
        self.last_completed_step_type = str(step.type or "")
        self.last_completed_step_name = str(step.name or "")
        if not self._is_non_physical_step(step):
            self.last_physical_step_id = int(self.step_id)
            self.last_physical_step_type = str(step.type or "")
            self.last_physical_step_name = str(step.name or "")
        self._write_current_step_status(
            phase="finished",
            step=step,
            result=result_summary,
            error=errors,
            timing_s={
                "planning": float(planning_duration_s),
                "execution": float(execution_duration_s),
                "total": float(max(0.0, time.monotonic() - step_start)),
            },
        )
        self._enforce_force_submit_guards(step=step, final_success=final_success, errors=errors)

        if step.name == "gui_order_dish_via_computer" and final_success:
            self._mark_dish_ordered()
        if step.name == "gui_submit_dish_via_checkout_stand" and final_success:
            self._verify_submit_feedback_and_mark_done(
                dish_name=str(self.dish.dish_name or ""),
                step_wall_time=step_wall_time,
                submit_raw=(action_result.raw if isinstance(action_result.raw, dict) else None),
            )

        try:
            self._maybe_update_put_place_occupancy_snapshot(step=step, action_result=action_result)
        except Exception:
            pass
        try:
            self._maybe_update_active_container(step=step, action_result=action_result, final_success=final_success)
        except Exception:
            pass
        try:
            self._update_force_submit_stage_after_step(step=step, final_success=final_success)
        except Exception:
            pass
        step_duration_s = time.monotonic() - step_start
        episode_elapsed_s = time.monotonic() - self._episode_start_monotonic
        if self.cfg.verbose and self.cfg.log_mode != "minimal":
            self.log.info(
                f"[EPM] step_timing step={self.step_id} plan_s={planning_duration_s:.3f} "
                f"exec_s={execution_duration_s:.3f} total_s={step_duration_s:.3f}"
            )

        self.pipeline.on_step_result(step=step, success=final_success, error=errors)
        try:
            self._update_pe_plan_after_step(step=step, final_success=final_success, error=errors)
        except Exception:
            pass

        # Rich feedback (includes structured returns) for the next planning round.
        try:
            fb = render_action_feedback(step=step, action_result=action_result, final_success=final_success, error=errors)
            self._last_action_feedback = fb
        except Exception:
            # Never fail the control loop due to feedback formatting.
            self._last_action_feedback = ""
        try:
            self._update_query_memory(step=step, action_result=action_result, final_success=final_success)
        except Exception:
            pass
        try:
            self._maybe_update_reflexion_progress_memory(step=step, action_result=action_result, final_success=final_success)
        except Exception:
            pass

        obs_summary = "percept=" + (percept.text.strip() if percept.text else "")
        obs_summary += f" objects={len(obs.objects)} state_keys={list(obs.state.keys())}"

        diff = None
        if step.type == "skill":
            diff = {"type": "skill_executed", "details": {"name": step.name}}
        else:
            planned_payload = {"action_or_skill": step.name, "params": dict(step.args)}
            executed_payload = {"action_or_skill": planned.action_or_skill, "params": planned.params}
            diff = diff_planned_executed(planned_payload, executed_payload)
        # Attach post-observation pointers for offline analysis (no schema change).
        diff = dict(diff or {})
        diff["post_screenshot_path"] = post_obs.screenshot_path
        diff["pre_on_screen"] = obs.state.get("_on_screen_objects", [])
        diff["post_on_screen"] = post_obs.state.get("_on_screen_objects", [])
        diff["timing_s"] = {
            "planning": float(planning_duration_s),
            "execution": float(execution_duration_s),
            "total": float(step_duration_s),
        }
        if judged_success is not None:
            diff["judged_success"] = judged_success
            diff["judged_reason"] = judged_reason

        self.memory.record_step(
            step_id=self.step_id,
            observation_summary=obs_summary,
            planned=planned,
            executed=executed,
            result_summary=result_summary,
            screenshot_path=obs.screenshot_path,
            time_iso=obs.time,
            duration_s=step_duration_s,
            episode_elapsed_s=episode_elapsed_s,
            diff=diff,
            errors=errors,
        )

        guard_fatal, guard_fatal_kind = self._update_mode_entry_retry_guard(
            step=step,
            final_success=final_success,
            errors=errors,
        )
        if guard_fatal:
            if anomaly_thread is not None:
                anomaly_thread.join()
            raise RuntimeError(f"fatal_episode_error:{guard_fatal_kind}")

        fatal, fatal_kind = self._is_fatal_error(errors)
        if fatal:
            # Abort the whole run: this is not a recoverable environment error.
            if anomaly_thread is not None:
                anomaly_thread.join()
            raise RuntimeError(f"fatal_episode_error:{fatal_kind}:{errors}")

        need_replan_triggered = False
        update_mode = self._task_progress_update_mode()
        # Update task_progress atomic step check-off (best-effort).
        if update_mode != "maintainer_only" and executed.plan_ref is not None and self.task_progress_path is not None:
            try:
                plan_ref = UpdaterPlanRef(
                    high_level_id=executed.plan_ref.high_level_id,
                    atomic_step_id=executed.plan_ref.atomic_step_id,
                )
                apply_execution_result_to_task_progress(
                    task_progress_path=self.task_progress_path,
                    plan_ref=plan_ref,
                    result_summary=result_summary,
                    error=errors,
                )
            except NeedReplan as e:
                # Planner/format mismatch; keep running but log for debugging.
                if self.cfg.verbose:
                    self.log.info(f"[EPM] progress_updater: NeedReplan (task_progress format/plan_ref mismatch) err={e!r}")
                try:
                    append_off_plan_event(
                        task_progress_path=self.task_progress_path,
                        step_id=int(self.step_id),
                        action_or_skill=str(executed.action_or_skill),
                        reason="plan_ref_mismatch",
                        plan_ref=UpdaterPlanRef(
                            high_level_id=executed.plan_ref.high_level_id,
                            atomic_step_id=executed.plan_ref.atomic_step_id,
                        ),
                        time_iso=str(obs.time),
                    )
                except Exception:
                    pass
                try:
                    self._force_replan_next_step()
                except Exception:
                    pass
                need_replan_triggered = True
            except Exception as e:
                if self.cfg.verbose:
                    self.log.info(f"[EPM] progress_updater: error={e!r}")

        maintenance_thread: Optional[threading.Thread] = None
        if need_replan_triggered:
            maintenance_thread = threading.Thread(
                target=self._maybe_run_task_progress_maintenance,
                kwargs={
                    "step_id": self.step_id,
                    "new_plan_generated": False,
                    "need_replan": True,
                },
                name=f"task-progress-maintainer-{self.step_id}",
                daemon=True,
            )
            maintenance_thread.start()

        if anomaly_thread is not None:
            anomaly_thread.join()
            anomaly_result = anomaly_result_box.get("result")
            if (
                anomaly_result is not None
                and bool(getattr(anomaly_result, "has_anomaly", False))
                and self.cfg.verbose
                and self.cfg.log_mode != "minimal"
            ):
                self.log.warning(
                    f"[EPM] visual_anomaly severity={getattr(anomaly_result, 'severity', 'none')!r} "
                    f"tags={list(getattr(anomaly_result, 'tags', []) or [])} "
                    f"feedback={str(getattr(anomaly_result, 'feedback', '') or '').strip()!r}"
                )
            elif self.cfg.verbose and self.cfg.log_mode != "minimal" and "error" in anomaly_result_box:
                self.log.warning(f"[EPM] visual_anomaly_observer_failed step={self.step_id} err={anomaly_result_box['error']!r}")
        if maintenance_thread is not None:
            maintenance_thread.join()
        return obs

    def _colorize(self, text: str, color: str) -> str:
        if not bool(getattr(self.cfg, "log_color", True)):
            return text
        if os.environ.get("NO_COLOR"):
            return text
        colors = {
            "red": "\x1b[31m",
            "green": "\x1b[32m",
            "yellow": "\x1b[33m",
            "cyan": "\x1b[36m",
            "gray": "\x1b[90m",
        }
        reset = "\x1b[0m"
        return f"{colors.get(color,'')}{text}{reset}" if color in colors else text

    def _format_args_preview(self, args: dict) -> str:
        if not isinstance(args, dict) or not args:
            return ""
        # Keep a stable, small preview.
        keys_priority = ["target", "query", "instance_id", "target_instance_id", "duration", "direction"]
        picked: list[str] = []
        used = set()
        for k in keys_priority:
            if k in args and k not in used:
                picked.append(k)
                used.add(k)
            if len(picked) >= 3:
                break
        if len(picked) < 3:
            for k in sorted(args.keys()):
                if k in used:
                    continue
                picked.append(k)
                used.add(k)
                if len(picked) >= 3:
                    break
        parts: list[str] = []
        for k in picked:
            try:
                v = args.get(k)
                parts.append(f"{k}={v!r}")
            except Exception:
                continue
        more = len(args) - len(picked)
        if more > 0:
            parts.append(f"...(+{more})")
        return " " + " ".join(parts) if parts else ""

    def _log_minimal_step(self, *, step: PlanStep, result_summary: str, errors: str) -> None:
        rs = (result_summary or "").strip().lower()
        ok = rs == "success"
        tag = self._colorize("SUCCESS", "green") if ok else self._colorize("FAILURE", "red")
        args_preview = self._format_args_preview(dict(step.args))
        # Example:
        # step=12 H1.A3 action PickupObject target='Olive Oil' -> SUCCESS
        self.log.info(f"[EPM] step={self.step_id} {step.step_id} {step.type} {step.name}{args_preview} -> {tag}")
        if (not ok) and errors:
            self.log.info(f"[EPM] step={self.step_id} error={errors!r}")

    def _append_step_trace(self, *, kind: str, step: PlanStep, result: str, error: str) -> None:
        """
        Write a compact per-step trace that can be tailed from another terminal:
        `Get-Content <run>/memory/step_trace.log -Wait`
        """
        try:
            path = Path(self.cfg.memory_dir) / "step_trace.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            line = (
                f"{ts} | step={self.step_id} kind={kind} step_id={step.step_id} "
                f"type={step.type} name={step.name} args={json.dumps(step.args, ensure_ascii=False, separators=(',', ':'))} "
                f"result={result!r} error={error!r}"
            )
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            return

    def _append_planner_thoughts(self, *, plan: PlanResponse) -> None:
        """
        Persist planner thought trace for offline analysis.
        """
        path = self.cfg.memory_dir / "planner_thoughts.jsonl"
        payload = {
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "episode_step": int(self.step_id),
            "high_level_id": str(plan.high_level_id or ""),
            "goal": str(plan.goal or ""),
            "thoughts": str(plan.thoughts or ""),
            "steps": [str(s.name or "") for s in (plan.action_list or [])],
        }
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            return

    def _sync_new_plan(self, *, new_plan: PlanResponse) -> None:
        self._last_plan_goal = str(new_plan.goal or "").strip()
        self._last_plan_thoughts = str(new_plan.thoughts or "").strip()
        self._last_plan_steps = [str(s.name or "").strip() for s in (new_plan.action_list or []) if str(s.name or "").strip()]
        self._last_plan_updated_episode_step = int(self.step_id)
        if self.cfg.verbose:
            step_names = [s.name for s in (new_plan.action_list or [])]
            self.log.info(f"[EPM] plan.goal={new_plan.goal!r}")
            if (new_plan.thoughts or "").strip():
                self.log.info(f"[EPM] plan.thoughts={new_plan.thoughts.strip()!r}")
            self.log.info(f"[EPM] plan.steps={step_names}")
        self._ensure_atomic_plan_written(new_plan)
        self._append_pe_plan_history(new_plan)
        self._append_planner_thoughts(plan=new_plan)
        self._on_new_pe_plan(plan=new_plan)

    def _consume_pipeline_new_plan_if_any(self):
        try:
            consume = getattr(self.pipeline, "consume_new_plan", None)
            if not callable(consume):
                return None
            new_plan = consume()
            if new_plan is None:
                return None
            self._sync_new_plan(new_plan=new_plan)
            return new_plan
        except Exception as e:
            if self.cfg.verbose:
                self.log.info(f"[EPM] task_progress semantic_sync hook failed: {e!r}")
        return None

    def _should_run_task_progress_for_new_plan(self) -> bool:
        self._plan_generation_count += 1
        if str(self.pipeline_name or "").strip().lower() != "reflexion":
            return True
        raw = self.cfg.task_progress_maintenance or {}
        if not isinstance(raw, dict):
            return True
        every_n_plans = int(raw.get("every_n_plans", 0) or 0)
        if every_n_plans <= 0:
            return True
        return (self._plan_generation_count % every_n_plans) == 0

    def _is_pe_pipeline(self) -> bool:
        pipeline = (self.cfg.pipeline or "").strip().lower()
        return pipeline in ("planner_executor", "planner-executor", "pe", "epm", "epm_agent", "reflexion", "cap")

    def _get_pe_like_pipeline_target(self):
        pipe = getattr(self, "pipeline", None)
        if pipe is None:
            return None
        base = getattr(pipe, "base", None)
        if base is not None and hasattr(getattr(base, "state", None), "current_plan"):
            return base
        return pipe

    def _pe_active_plan_path(self) -> Path:
        return self.cfg.memory_dir / "pe_active_plan.json"

    def _load_pe_active_plan_state(self) -> None:
        if not self._is_pe_pipeline():
            self._pe_active_plan = {}
            return
        p = self._pe_active_plan_path()
        if not p.exists():
            self._pe_active_plan = {
                "cursor_index": 0,
                "plan": [],
            }
            self._write_pe_active_plan_state()
            return
        try:
            obj = json.loads(p.read_text(encoding="utf-8-sig"))
            self._pe_active_plan = obj if isinstance(obj, dict) else {}
            # Backward compatibility: old key `action_list` -> new key `plan`.
            if "plan" not in self._pe_active_plan and isinstance(self._pe_active_plan.get("action_list"), list):
                self._pe_active_plan["plan"] = list(self._pe_active_plan.get("action_list") or [])
            if "cursor_index" not in self._pe_active_plan:
                self._pe_active_plan["cursor_index"] = 0
        except Exception:
            self._pe_active_plan = {}

    def _hydrate_pipeline_plan_from_pe_active_plan_if_resume(self) -> None:
        """
        Resume optimization for PE:
        if we already have an unconsumed active plan queue on disk, hydrate the
        in-memory PlannerExecutor pipeline state so next_step() executes directly
        from remaining queue instead of calling planner again.
        """
        if not self._is_pe_pipeline():
            return
        if not bool(getattr(self.cfg, "resume", False)):
            return
        pipe = self._get_pe_like_pipeline_target()
        state = getattr(pipe, "state", None)
        if state is None:
            return
        if getattr(state, "current_plan", None) is not None:
            return
        plan_rows = self._pe_active_plan.get("plan")
        if not isinstance(plan_rows, list) or not plan_rows:
            return
        try:
            cursor = int(self._pe_active_plan.get("cursor_index", 0) or 0)
        except Exception:
            cursor = 0
        cursor = max(0, min(cursor, len(plan_rows)))

        steps: list[PlanStep] = []
        for row in plan_rows:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("step_id") or "").strip()
            typ = str(row.get("type") or "").strip()
            name = str(row.get("name") or "").strip()
            args = row.get("args") if isinstance(row.get("args"), dict) else {}
            if not sid or typ not in {"action", "skill"} or not name:
                continue
            steps.append(
                PlanStep(
                    step_id=sid,
                    type=typ,
                    name=name,
                    args=dict(args),
                    expectation=str(row.get("expectation") or ""),
                )
            )
        if not steps:
            return
        if cursor >= len(steps):
            return
        pending = 0
        for i in range(cursor, len(plan_rows)):
            r = plan_rows[i] if isinstance(plan_rows[i], dict) else {}
            if str(r.get("status") or "pending") == "pending":
                pending += 1
        if pending <= 0:
            return

        first_sid = str(steps[0].step_id or "")
        high_level_id = first_sid.split(".", 1)[0] if "." in first_sid else "H1"
        plan_obj = PlanResponse(
            high_level_id=high_level_id,
            goal=str(self._last_plan_goal or ""),
            explanation=None,
            thoughts=str(self._last_plan_thoughts or ""),
            action_list=steps,
        )
        try:
            setattr(state, "current_plan", plan_obj)
            setattr(state, "current_index", int(cursor))
            setattr(state, "new_plan", None)
            max_sid = 0
            for s in steps:
                sid = str(getattr(s, "step_id", "") or "")
                m = re.search(r"\.A(\d+)$", sid)
                if m:
                    try:
                        max_sid = max(max_sid, int(m.group(1)))
                    except Exception:
                        pass
            set_counter = getattr(pipe, "set_atomic_counter", None)
            if callable(set_counter):
                set_counter(max_sid or len(steps))
            if self.cfg.verbose and self.cfg.log_mode != "minimal":
                self.log.info(
                    f"[EPM] resume_queue=hydrate_from_pe_active_plan cursor={cursor} "
                    f"total={len(steps)} pending_from_cursor={pending}"
                )
        except Exception:
            return

    def _write_pe_active_plan_state(self) -> None:
        if not self._is_pe_pipeline():
            return
        plan = self._pe_active_plan.get("plan")
        if not isinstance(plan, list):
            plan = []
        try:
            cursor = int(self._pe_active_plan.get("cursor_index", 0) or 0)
        except Exception:
            cursor = 0
        payload = {
            "cursor_index": max(0, min(cursor, len(plan))),
            "plan": plan,
        }
        payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        payload["episode_step"] = int(self.step_id)
        try:
            self._pe_active_plan_path().write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            return

    def _on_new_pe_plan(self, *, plan: PlanResponse) -> None:
        if not self._is_pe_pipeline():
            return
        current_plan = self._pe_active_plan.get("plan")
        if not isinstance(current_plan, list):
            current_plan = []
        try:
            cursor = int(self._pe_active_plan.get("cursor_index", 0) or 0)
        except Exception:
            cursor = 0
        cursor = max(0, min(cursor, len(current_plan)))
        # A replan replaces the remaining queue from current cursor onward.
        # Drop the old pending suffix entirely so the persisted plan stays
        # aligned with what will actually execute next.
        current_plan = list(current_plan[:cursor])
        steps = []
        start_index = len(current_plan)
        for i, s in enumerate(plan.action_list or []):
            steps.append(
                {
                    "index": int(start_index + i),
                    "step_id": str(s.step_id or ""),
                    "type": str(s.type or ""),
                    "name": str(s.name or ""),
                    "args": dict(s.args or {}),
                    "status": "pending",
                    "last_result": "",
                    "last_error": "",
                    "updated_at_episode_step": None,
                }
            )
        self._pe_active_plan = {"cursor_index": cursor, "plan": current_plan + steps}
        self._write_pe_active_plan_state()

    def _update_pe_plan_after_step(self, *, step: PlanStep, final_success: bool, error: str) -> None:
        if not self._is_pe_pipeline():
            return
        if not isinstance(self._pe_active_plan, dict):
            self._pe_active_plan = {}
        action_list = self._pe_active_plan.get("plan")
        if not isinstance(action_list, list) or not action_list:
            return
        sid = str(step.step_id or "")
        idx = None
        for i, it in enumerate(action_list):
            if isinstance(it, dict) and str(it.get("step_id") or "") == sid:
                idx = i
                break
        if idx is None:
            return
        row = action_list[idx]
        if not isinstance(row, dict):
            return
        row["status"] = "success" if final_success else "failed"
        row["last_result"] = "success" if final_success else "failure"
        row["last_error"] = str(error or "")
        row["updated_at_episode_step"] = int(self.step_id)

        if final_success:
            cursor = int(idx) + 1
            self._pe_active_plan["cursor_index"] = int(cursor)
        else:
            self._pe_active_plan["cursor_index"] = int(idx)

        self._write_pe_active_plan_state()

    def _write_current_step_status(
        self,
        *,
        phase: str,
        step: Optional[PlanStep],
        result: str,
        error: str,
        timing_s: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Overwrite a single-file live status for easy "pinned" monitoring in another terminal.
        """
        try:
            path = Path(self.cfg.memory_dir) / "current_step_status.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "episode_step": int(self.step_id),
                "logical_step": int(self._logical_committed_step_id()),
                "tentative_step": bool(str(phase or "").strip() == "step_started"),
                "phase": str(phase or "").strip(),
                "result": str(result or "").strip(),
                "error": str(error or "").strip(),
            }
            if step is not None:
                payload.update(
                    {
                        "step_id": str(step.step_id or ""),
                        "type": str(step.type or ""),
                        "name": str(step.name or ""),
                        "args": dict(step.args or {}),
                    }
                )
            if self._last_plan_goal:
                payload["planner_goal"] = str(self._last_plan_goal)
            if self._last_plan_thoughts:
                payload["planner_thoughts"] = str(self._last_plan_thoughts)
            if self._last_plan_steps:
                payload["planner_steps"] = list(self._last_plan_steps)
            if int(self._last_plan_updated_episode_step) > 0:
                payload["planner_updated_at_episode_step"] = int(self._last_plan_updated_episode_step)
            if isinstance(timing_s, dict) and timing_s:
                payload["timing_s"] = dict(timing_s)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            return
        try:
            self._append_error_event(phase=phase, step=step, result=result, error=error)
        except Exception:
            pass

    def _append_error_event(
        self,
        *,
        phase: str,
        step: Optional[PlanStep],
        result: str,
        error: str,
    ) -> None:
        raw_error = str(error or "").strip()
        if not raw_error:
            return
        step_id = str(step.step_id or "") if step is not None else ""
        signature = (
            int(self.step_id),
            str(phase or "").strip(),
            step_id,
            raw_error,
        )
        if signature == getattr(self, "_last_error_event_signature", None):
            return
        setattr(self, "_last_error_event_signature", signature)
        one_line_error = re.sub(r"\s+", " ", raw_error).strip()
        path = Path(self.cfg.memory_dir) / "error_events.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "episode_step": int(self.step_id),
            "logical_step": int(self._logical_committed_step_id()),
            "phase": str(phase or "").strip(),
            "result": str(result or "").strip(),
            "step_id": step_id,
            "type": (str(step.type or "").strip() if step is not None else ""),
            "name": (str(step.name or "").strip() if step is not None else ""),
            "args": (dict(step.args or {}) if step is not None else {}),
            "error": one_line_error,
        }
        line = (
            f"{event['ts']} | step={event['episode_step']} | logical={event['logical_step']} "
            f"| phase={event['phase']} | result={event['result'] or '-'} | step_id={event['step_id'] or '-'} "
            f"| type={event['type'] or '-'} | name={event['name'] or '-'} | error={event['error']}"
        )
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _logical_committed_step_id(self) -> int:
        vals: list[int] = []
        for name in ("last_completed_step_id", "last_physical_step_id"):
            try:
                v = int(getattr(self, name, 0) or 0)
            except Exception:
                v = 0
            if v > 0:
                vals.append(v)
        return max(vals) if vals else 0

    @staticmethod
    def _mode_string_blocks_navigation(mode_raw: str) -> bool:
        mode = str(mode_raw or "").strip().lower()
        if not mode:
            return False
        blocked_markers = (
            "pouring_mode",
            "pour_mode",
            "cutting_mode",
            "mixing_mode",
            "flip_mode",
            "sprinkle_mode",
            "pouring (",
            "pour (",
            "cutting (",
            "mixing (",
            "flip (",
            "sprinkle (",
        )
        blocked_literals = {"pour", "cutting", "mixing", "flip", "sprinkle"}
        return (mode in blocked_literals) or any(m in mode for m in blocked_markers)

    @staticmethod
    def _realtime_mode_flags(items: list[dict[str, Any]]) -> tuple[bool, str]:
        def _item_text(it: dict[str, Any]) -> str:
            return " ".join(
                str(it.get(k) or "").strip().lower()
                for k in ("name_en", "name_cn", "game_object")
            )

        def _is_knife_like(it: dict[str, Any]) -> bool:
            return "knife" in _item_text(it)

        def _any_true(keys: tuple[str, ...]) -> bool:
            for it in items:
                if not isinstance(it, dict):
                    continue
                for k in keys:
                    if bool(it.get(k, False)):
                        return True
            return False

        for it in items:
            if not isinstance(it, dict):
                continue
            if not _is_knife_like(it):
                continue
            if bool(it.get("is_cut_mode") or it.get("is_cutting") or it.get("is_cutting_mode")):
                return True, "cutting_mode(realtime_knife)"
        if _any_true(("is_cut_mode", "is_cutting", "is_cutting_mode")):
            return True, "cutting_mode(realtime)"
        for it in items:
            if not isinstance(it, dict):
                continue
            if bool(it.get("is_pouring_mode") or it.get("is_pouring")):
                return True, "pouring_mode(realtime)"
        if _any_true(("is_mixing_mode",)):
            return True, "mixing_mode(realtime)"
        if _any_true(("is_filp_mode", "is_flip_mode", "is_flipping_mode")):
            return True, "flip_mode(realtime)"
        if _any_true(("is_sprinkle_mode", "is_sprinkle")):
            return True, "sprinkle_mode(realtime)"
        return False, ""

    def _is_navigation_blocked_by_mode(self) -> tuple[bool, str]:
        try:
            self._update_agent_state_from_realtime_products()
        except Exception:
            pass
        try:
            st = self.memory.read_json("agent_state")
        except Exception:
            return False, ""
        mode_raw = str(st.get("mode") or "").strip()
        if self._mode_string_blocks_navigation(mode_raw):
            return True, mode_raw
        # Fallback: read realtime mode flags directly to reduce stale-state false negatives.
        try:
            data = read_realtime_products(self.cfg.realtime_products_path)
            items = extract_items(data)
            blocked_rt, mode_rt = self._realtime_mode_flags(items)
            if blocked_rt:
                return True, (mode_rt or mode_raw or "interaction_mode_active")
        except Exception:
            pass
        return False, mode_raw

    def _horizontal_movement_mode_check(self) -> tuple[bool, str]:
        """
        Horizontal_movement is intended for interaction/operation modes only.
        """
        try:
            self._update_agent_state_from_realtime_products()
            st = self.memory.read_json("agent_state")
        except Exception:
            return False, ""
        mode_raw = str(st.get("mode") or "").strip()
        mode = mode_raw.lower()
        allow_markers = (
            "pouring_mode",
            "cutting_mode",
            "mixing_mode",
            "flip_mode",
            "sprinkle_mode",
            "pouring (",
        )
        allowed = any(m in mode for m in allow_markers)
        return allowed, mode_raw

    def _is_epm_pipeline(self) -> bool:
        pipeline = (self.cfg.pipeline or "").strip().lower()
        return pipeline in ("epm", "epm_agent")

    def _read_agent_state_snapshot(self) -> dict[str, Any]:
        try:
            self._update_agent_state_from_realtime_products()
        except Exception:
            pass
        try:
            state = self.memory.read_json("agent_state")
            return state if isinstance(state, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _mode_class(mode_raw: str) -> str:
        mode = str(mode_raw or "").strip().lower()
        if ("pour" in mode) or (mode == "pour"):
            return "pour"
        if ("cut" in mode) or (mode == "cutting"):
            return "cut"
        if "mix" in mode:
            return "mix"
        if ("sprinkle" in mode) or (mode == "sprinkle"):
            return "sprinkle"
        if ("flip" in mode) or (mode == "filp"):
            return "flip"
        return ""

    @staticmethod
    def _active_modes_from_hold_snapshot(snap: dict[str, Any]) -> list[str]:
        if not isinstance(snap, dict):
            return []
        out: list[str] = []
        if bool(snap.get("is_pouring_mode")):
            out.append("pour")
        if bool(snap.get("is_sprinkle_mode")):
            out.append("sprinkle")
        if bool(snap.get("is_cutting_mode")):
            out.append("cut")
        if bool(snap.get("is_mixing_mode")):
            out.append("mix")
        if bool(snap.get("is_flip_mode")):
            out.append("flip")
        return out

    def _augment_mode_error_details(self, *, error: str, raw: dict[str, Any] | None) -> str:
        text = str(error or "").strip()
        if not text:
            return text
        text_l = text.lower()
        if not any(
            token in text_l
            for token in (
                "blocked_by_mode",
                "interaction_mode_still_active",
                "cutting_mode_still_active",
                "realtime_products_unavailable",
            )
        ):
            return text

        payload = raw if isinstance(raw, dict) else {}
        parts: list[str] = []

        blocked_mode = str(payload.get("blocked_by_mode") or "").strip()
        if blocked_mode:
            parts.append(f"blocked_mode={blocked_mode!r}")
        blocked_mode_class = str(payload.get("blocked_mode_class") or "").strip()
        if blocked_mode_class:
            parts.append(f"blocked_mode_class={blocked_mode_class!r}")

        for label, snap_key in (("pre_modes", "precheck"), ("post_modes", "postcheck")):
            snap = payload.get(snap_key)
            if isinstance(snap, dict):
                modes = self._active_modes_from_hold_snapshot(snap)
                parts.append(f"{label}={modes if modes else ['none']}")
                if label == "post_modes":
                    parts.append(f"post_is_held={bool(snap.get('is_held'))!r}")
                    held_name = str(snap.get("held_name_en") or snap.get("held_name_cn") or "").strip()
                    if held_name:
                        parts.append(f"post_held={held_name!r}")

        state = self._read_agent_state_snapshot()
        agent_mode = str(state.get("mode") or "").strip()
        if agent_mode:
            parts.append(f"agent_mode={agent_mode!r}")

        if not parts:
            return text
        suffix = " ".join(parts)
        if suffix in text:
            return text
        return f"{text}; {suffix}"

    def _read_task_progress_text(self) -> str:
        path = self.task_progress_path
        if path is None or not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _current_remaining_plan_steps(self, *, step: PlanStep) -> list[PlanStep]:
        getter = getattr(self.pipeline, "current_remaining_plan", None)
        if callable(getter):
            try:
                steps = getter()
                if isinstance(steps, list) and steps:
                    return [s for s in steps if isinstance(s, PlanStep)]
            except Exception:
                pass
        return [step]

    @staticmethod
    def _format_plan_steps_for_feedback(steps: list[PlanStep]) -> str:
        rows: list[str] = []
        for i, s in enumerate(steps, start=1):
            if not isinstance(s, PlanStep):
                continue
            rows.append(
                f"{i}. {str(s.step_id or '').strip()} {str(s.type or '').strip()}:{str(s.name or '').strip()} "
                f"args={json.dumps(dict(s.args or {}), ensure_ascii=False, separators=(',', ':'))}"
            )
        return "\n".join(rows)

    @staticmethod
    def _compact_plan_step_text(step: PlanStep) -> str:
        if not isinstance(step, PlanStep):
            return ""
        label = f"{str(step.step_id or '').strip()} {str(step.type or '').strip()}:{str(step.name or '').strip()}".strip()
        args = dict(step.args or {})
        compact_keys = (
            "query",
            "target",
            "target_instance_id",
            "dish_name",
            "computer_target",
            "container",
            "item",
        )
        parts: list[str] = [label]
        for key in compact_keys:
            if key not in args:
                continue
            val = args.get(key)
            if val is None:
                continue
            txt = str(val).strip()
            if not txt:
                continue
            parts.append(f"{key}={txt!r}")
        return " ".join(p for p in parts if p).strip()

    @classmethod
    def _format_plan_steps_compact_for_feedback(cls, steps: list[PlanStep], *, max_steps: int = 8) -> str:
        rows: list[str] = []
        limit = max(1, int(max_steps))
        for i, s in enumerate(steps[:limit], start=1):
            text = cls._compact_plan_step_text(s)
            if text:
                rows.append(f"{i}. {text}")
        remaining = max(0, len(steps) - limit)
        if remaining:
            rows.append(f"... ({remaining} more steps omitted)")
        return "\n".join(rows)

    @staticmethod
    def _summarize_blocked_steps(blocked_steps: list[dict[str, Any]], *, max_items: int = 4) -> list[str]:
        out: list[str] = []
        limit = max(1, int(max_items))
        for item in (blocked_steps or [])[:limit]:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            name = str(item.get("name") or "").strip()
            reason = str(item.get("reason") or "").strip()
            reason_code = str(item.get("reason_code") or "").strip()
            prefix = f"[{idx}]" if idx is not None else "[?]"
            body = reason or reason_code or name or "blocked"
            if name and reason and name not in reason:
                body = f"{name}: {reason}"
            out.append(f"{prefix} {body}".strip())
        remaining = max(0, len(blocked_steps or []) - limit)
        if remaining:
            out.append(f"... ({remaining} more blocked steps omitted)")
        return out

    @staticmethod
    def _plan_step_to_record(step: PlanStep | None) -> dict[str, Any]:
        if step is None:
            return {}
        return {
            "step_id": str(step.step_id or "").strip(),
            "type": str(step.type or "").strip(),
            "name": str(step.name or "").strip(),
            "args": dict(step.args or {}),
            "expectation": str(step.expectation or "").strip(),
        }

    @staticmethod
    def _normalize_precondition_instruction(text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        s = s.rstrip(".")
        if not s:
            return ""
        return s[:1].lower() + s[1:]

    def _render_precondition_feedback_text(
        self,
        *,
        step: PlanStep,
        status: str,
        reason_code: str,
        reason: str,
        feedback_to_planner: str,
        blocked_steps: list[dict[str, Any]],
        checks: dict[str, Any],
        visual_blockers: list[str],
        relevant_state: list[str],
    ) -> str:
        lines: list[str] = []
        lines.append(f"episode_step={int(self.step_id)}")
        lines.append(f"status={str(status or '').strip()}")
        lines.append(f"step={str(step.type or '').strip()}:{str(step.name or '').strip()}")
        if step.args:
            lines.append(f"step_args={json.dumps(dict(step.args or {}), ensure_ascii=False)}")
        if reason_code:
            lines.append(f"reason_code={reason_code}")
        if reason:
            lines.append(f"reason={reason}")
        blocked_summary = self._summarize_blocked_steps(blocked_steps)
        if blocked_summary:
            lines.append(f"blocked_steps={json.dumps(blocked_summary, ensure_ascii=False)}")
        failed_checks = [k for k, v in (checks or {}).items() if not bool(v)]
        if failed_checks:
            lines.append(f"failed_checks={json.dumps(failed_checks, ensure_ascii=False)}")
        if visual_blockers:
            lines.append(f"visual_blockers={json.dumps(list(visual_blockers)[:3], ensure_ascii=False)}")
        if relevant_state:
            lines.append(f"state_notes={json.dumps(list(relevant_state)[:3], ensure_ascii=False)}")
        if feedback_to_planner:
            lines.append(f"feedback_to_planner={feedback_to_planner}")
        return "\n".join(lines).strip()

    def _deterministic_feedback_to_planner(self, *, step: PlanStep, hint: str, code: str) -> str:
        fix = self._normalize_precondition_instruction(hint)
        reason = str(code or "").strip().replace("_", " ")
        name = str(step.name or "").strip() or "this step"
        if fix and reason:
            return f"before retrying {name}, first {fix} because {reason}."
        if fix:
            return f"before retrying {name}, first {fix} because the current precondition is not satisfied."
        if reason:
            return f"before retrying {name}, first satisfy the required precondition because {reason}."
        return f"before retrying {name}, first satisfy the required precondition because the current state is incompatible."

    def _persist_precondition_check_trace(self, *, payload: dict[str, Any], latest_feedback_text: str) -> None:
        if not self._is_epm_pipeline():
            return
        try:
            self.memory.save_precondition_check(step_id=int(self.step_id), payload=payload)
        except Exception:
            pass
        try:
            self.memory.write_text("latest_precondition_feedback", str(latest_feedback_text or "").strip())
        except Exception:
            pass

    def _render_task_progress_feedback_text(self, *, result: Any) -> str:
        goal_state = dict(getattr(result, "goal_state", {}) or {})
        blocking_conditions = list(getattr(result, "blocking_conditions", []) or [])
        temporal_checkpoints = dict(getattr(result, "temporal_checkpoints", {}) or {})
        lines: list[str] = []
        lines.append(f"episode_step={int(self.step_id)}")
        lines.append(f"current_high_level_id={str(getattr(result, 'current_high_level_id', '') or '').strip()}")
        lines.append(f"current_high_level_goal={str(getattr(result, 'current_high_level_goal', '') or '').strip()}")
        if goal_state.get("current_subgoal"):
            lines.append(f"current_subgoal={goal_state.get('current_subgoal')}")
        if goal_state.get("status"):
            lines.append(f"status={goal_state.get('status')}")
        if goal_state.get("current_activity"):
            lines.append(f"current_activity={goal_state.get('current_activity')}")
        if blocking_conditions:
            compact_blockers: list[dict[str, Any]] = []
            for item in blocking_conditions[:3]:
                if not isinstance(item, dict):
                    continue
                compact_blockers.append(
                    {
                        "type": str(item.get("type") or "").strip(),
                        "detail": str(item.get("detail") or "").strip(),
                    }
                )
            if compact_blockers:
                lines.append(f"blocking_conditions={json.dumps(compact_blockers, ensure_ascii=False)}")
        if "force_submit_remaining_steps" in temporal_checkpoints:
            lines.append(
                f"force_submit_remaining_steps={json.dumps(temporal_checkpoints.get('force_submit_remaining_steps'), ensure_ascii=False)}"
            )
        return "\n".join(lines).strip()

    def _precondition_check_failure_result(
        self,
        *,
        step: PlanStep,
        check: PreconditionCheckResult,
        state: dict[str, Any],
    ) -> ActionResult:
        raw = {
            "precondition_checker": {
                "status": str(check.status or ""),
                "reason_code": str(check.reason_code or ""),
                "reason": str(check.reason or ""),
                "feedback_to_planner": str(check.feedback_to_planner or ""),
                "relevant_state": list(check.relevant_state or []),
                "checks": dict(check.checks or {}),
                "visual_blockers": list(check.visual_blockers or []),
                "all_blocked_steps": list(check.all_blocked_steps or []),
                "checker_raw": dict(check.raw or {}),
                "step_type": str(step.type or ""),
                "step_name": str(step.name or ""),
                "step_args": dict(step.args or {}),
                "state": {
                    "mode": state.get("mode"),
                    "is_held": state.get("is_held"),
                    "held_item": state.get("held_item"),
                    "active_container": state.get("active_container"),
                    "force_submit_remaining_steps": state.get("force_submit_remaining_steps"),
                },
            }
        }
        code = str(check.reason_code or "checker_failed").strip() or "checker_failed"
        return ActionResult(False, raw=raw, error=f"epm_precondition_failed:{code}")

    def _repair_remaining_plan_after_precondition_failure(
        self,
        *,
        current_step: PlanStep,
        obs,
        bundle: dict[str, Any],
        percept,
        high_level_id: str,
        high_level_goal: str,
        checker_feedback: str,
        original_remaining_plan: list[PlanStep],
        attempt_index: int,
    ) -> PlanStep | None:
        pipeline_name = str(self.cfg.pipeline or "").strip().lower()
        if pipeline_name in ("open_loop", "open-loop", "openloop", "saycan"):
            return None
        locally_repaired = self._apply_local_precondition_patch(
            current_step=current_step,
            checker_feedback=checker_feedback,
            original_remaining_plan=original_remaining_plan,
            bundle=bundle,
            high_level_id=high_level_id,
            high_level_goal=high_level_goal,
        )
        if locally_repaired is not None:
            replacer = getattr(self.pipeline, "replace_remaining_plan", None)
            if callable(replacer):
                replaced_plan = replacer(
                    context=PlannerContext(
                        high_level_id=high_level_id,
                        high_level_goal=high_level_goal,
                        observation=self._model_observation(obs),
                        recipe_text=self.dish.recipe_text,
                        feedback="local_precondition_patch=true",
                        memory_bundle=dict(bundle),
                        percept=None,
                    ),
                    plan=locally_repaired,
                )
                self._consume_pipeline_new_plan_if_any()
                if replaced_plan is not None and getattr(replaced_plan, "action_list", None):
                    next_step = replaced_plan.action_list[0]
                    next_step = self._coerce_step_type_with_allowlist(
                        next_step,
                        allowed_actions=self.allowed_actions,
                        allowed_skills=self.allowed_skills,
                    )
                    next_step = self._canonicalize_dish_name_for_gui_steps(next_step)
                    ok, _ = validate_plan_against_allowlist(
                        PlanResponse(high_level_id=high_level_id, goal=high_level_goal, explanation=None, thoughts="", action_list=[next_step]),
                        allowed_actions=self.allowed_actions,
                        allowed_skills=self.allowed_skills,
                    )
                    return next_step if ok else None
            next_step = locally_repaired.action_list[0]
            next_step = self._coerce_step_type_with_allowlist(
                next_step,
                allowed_actions=self.allowed_actions,
                allowed_skills=self.allowed_skills,
            )
            next_step = self._canonicalize_dish_name_for_gui_steps(next_step)
            ok, _ = validate_plan_against_allowlist(
                PlanResponse(high_level_id=high_level_id, goal=high_level_goal, explanation=None, thoughts="", action_list=[next_step]),
                allowed_actions=self.allowed_actions,
                allowed_skills=self.allowed_skills,
            )
            return next_step if ok else None

        remaining_text = self._format_plan_steps_compact_for_feedback(original_remaining_plan)
        feedback = (
            "precondition_check_failed=true\n"
            f"failed_step={current_step.type}:{current_step.name}\n"
            f"failed_step_args={json.dumps(dict(current_step.args or {}), ensure_ascii=False)}\n"
            f"repair_attempt={int(attempt_index)}\n"
            "instruction=Revise the remaining action sequence so the NEXT first step is executable now while preserving completed progress.\n"
            "remaining_plan_outline:\n"
            f"{remaining_text}\n"
            "checker_feedback:\n"
            f"{str(checker_feedback or '').strip()}\n"
        )
        repair_context = PlannerContext(
            high_level_id=high_level_id,
            high_level_goal=high_level_goal,
            observation=self._model_observation(obs),
            recipe_text=self.dish.recipe_text,
            feedback=feedback,
            memory_bundle=dict(bundle),
            percept=percept,
        )
        repair_prompt = self.prompter.build_planner_prompt(
            observation=repair_context.observation,
            high_level_id=repair_context.high_level_id,
            high_level_goal=repair_context.high_level_goal,
            memory_bundle=repair_context.memory_bundle,
            feedback=repair_context.feedback,
            percept_text="",
            inject_memory=bool(self.cfg.inject_memory),
            inject_tool_schemas=bool(self.cfg.inject_tool_schemas),
            include_task_progress=bool((self.cfg.prompt_policy or {}).get("include_task_progress", True)),
        )
        try:
            repaired_plan = self.planner.plan(context=repair_context, prompt=repair_prompt)
        except NetworkPauseRequired:
            raise
        except NetworkAbortRequired:
            raise
        except Exception as e:
            if self.cfg.verbose and self.cfg.log_mode != "minimal":
                self.log.info(
                    "[EPM] precondition_repair_planner_exception "
                    f"step_id={current_step.step_id} name={current_step.name} err={e!r}"
                )
            return None
        if repaired_plan is None or not getattr(repaired_plan, "action_list", None):
            return None
        replacer = getattr(self.pipeline, "replace_remaining_plan", None)
        if callable(replacer):
            replaced_plan = replacer(context=repair_context, plan=repaired_plan)
            self._consume_pipeline_new_plan_if_any()
            if replaced_plan is not None and getattr(replaced_plan, "action_list", None):
                next_step = replaced_plan.action_list[0]
                next_step = self._coerce_step_type_with_allowlist(
                    next_step,
                    allowed_actions=self.allowed_actions,
                    allowed_skills=self.allowed_skills,
                )
                next_step = self._canonicalize_dish_name_for_gui_steps(next_step)
                ok, _ = validate_plan_against_allowlist(
                    PlanResponse(high_level_id=high_level_id, goal=high_level_goal, explanation=None, thoughts="", action_list=[next_step]),
                    allowed_actions=self.allowed_actions,
                    allowed_skills=self.allowed_skills,
                )
                return next_step if ok else None
        next_step = repaired_plan.action_list[0]
        next_step = self._coerce_step_type_with_allowlist(
            next_step,
            allowed_actions=self.allowed_actions,
            allowed_skills=self.allowed_skills,
        )
        next_step = self._canonicalize_dish_name_for_gui_steps(next_step)
        ok, _ = validate_plan_against_allowlist(
            PlanResponse(high_level_id=high_level_id, goal=high_level_goal, explanation=None, thoughts="", action_list=[next_step]),
            allowed_actions=self.allowed_actions,
            allowed_skills=self.allowed_skills,
        )
        return next_step if ok else None

    @staticmethod
    def _is_instance_disambiguation_reason(reason_code: str) -> bool:
        code = str(reason_code or "").strip().lower()
        if not code:
            return False
        if "disambiguation" in code:
            return True
        return "instance" in code and any(tok in code for tok in ("missing", "required", "not_found"))

    @staticmethod
    def _parse_feedback_fields(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = str(k).strip()
            if not key:
                continue
            out[key] = str(v).strip()
        return out

    @staticmethod
    def _best_instance_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None

        def _rank(item: dict[str, Any]) -> tuple[int, float]:
            on_screen = 0 if bool(item.get("is_on_screen")) else 1
            try:
                distance = float(item.get("distance"))
            except Exception:
                distance = 999999.0
            return (on_screen, distance)

        valid = [it for it in candidates if isinstance(it, dict) and it.get("instance_id") is not None]
        if not valid:
            return None
        valid.sort(key=_rank)
        return valid[0]

    @classmethod
    def _parse_instance_query_snapshot_map(cls, snapshot: str) -> dict[str, list[dict[str, Any]]]:
        rows: dict[str, list[dict[str, Any]]] = {}
        in_candidates = False
        for raw in str(snapshot or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line == "candidates_by_query=":
                in_candidates = True
                continue
            if not in_candidates:
                continue
            if ":" not in line:
                continue
            name, payload = line.split(":", 1)
            key = str(name).strip()
            if not key:
                continue
            try:
                obj = json.loads(str(payload).strip())
            except Exception:
                obj = None
            if isinstance(obj, list):
                rows[key.lower()] = [it for it in obj if isinstance(it, dict)]
        return rows

    @staticmethod
    def _clone_step(step: PlanStep, *, args: dict[str, Any] | None = None) -> PlanStep:
        return PlanStep(
            step_id=str(step.step_id or ""),
            type=str(step.type or ""),
            name=str(step.name or ""),
            args=dict(step.args if args is None else args),
            expectation=str(step.expectation or ""),
        )

    def _patch_plan_missing_instance_ids(
        self,
        *,
        steps: list[PlanStep],
        snapshot: str,
    ) -> list[PlanStep] | None:
        cand_map = self._parse_instance_query_snapshot_map(snapshot)
        if not cand_map:
            return None
        patched: list[PlanStep] = []
        changed = False
        bindings = (
            ("target", "target_instance_id"),
            ("container_name", "container_instance_id"),
            ("object_name", "object_instance_id"),
            ("item_name", "item_instance_id"),
            ("name", "instance_id"),
        )
        for step in steps:
            args = dict(step.args or {})
            new_args = dict(args)
            for name_key, iid_key in bindings:
                name_val = args.get(name_key)
                if not isinstance(name_val, str) or not name_val.strip():
                    continue
                if str(new_args.get(iid_key, "")).strip():
                    continue
                cands = cand_map.get(name_val.strip().lower())
                best = self._best_instance_candidate(cands or [])
                if best is None:
                    continue
                new_args[iid_key] = best.get("instance_id")
                changed = True
            patched.append(self._clone_step(step, args=new_args))
        return patched if changed else None

    def _patch_plan_bottle_closed(self, *, steps: list[PlanStep]) -> list[PlanStep] | None:
        if not steps:
            return None
        first = steps[0]
        if str(first.name or "") == "unscrew_bottle":
            return None
        out: list[PlanStep] = []
        inserted = False
        for idx, step in enumerate(steps):
            if not inserted and str(step.name or "") in {"enter_pouring_mode", "auto_pour"}:
                out.append(
                    PlanStep(
                        step_id=str(step.step_id or ""),
                        type="action",
                        name="unscrew_bottle",
                        args={},
                        expectation="Unscrew the held bottle before entering pouring / pouring.",
                    )
                )
                inserted = True
            out.append(self._clone_step(step))
        return out if inserted else None

    def _patch_plan_exit_mode_first(self, *, steps: list[PlanStep], checker_feedback: str) -> list[PlanStep] | None:
        if not steps:
            return None
        first = steps[0]
        fb = str(checker_feedback or "")
        fb_l = fb.lower()
        if "blocked_by_mode" not in fb_l:
            return None
        fields = self._parse_feedback_fields(fb)
        mode_raw = str(fields.get("blocked_mode_class") or fields.get("blocked_by_mode") or "").strip()
        mode_class = self._mode_class(mode_raw)
        if not mode_class:
            if "pour" in fb_l:
                mode_class = "pour"
            elif "sprinkle" in fb_l:
                mode_class = "sprinkle"
            elif "cut" in fb_l:
                mode_class = "cut"
            elif "mix" in fb_l:
                mode_class = "mix"
            elif "flip" in fb_l or "filp" in fb_l:
                mode_class = "flip"
        exit_name_map = {
            "pour": "exit_pouring_mode",
            "sprinkle": "exit_spices_sprinkle_mode",
            "cut": "exit_cutting_mode",
            "mix": "exit_mixing_mode",
            "flip": "exit_the_flipping_mode",
        }
        exit_expectation_map = {
            "pour": "Exit pouring mode before continuing the queued action sequence.",
            "sprinkle": "Exit sprinkle mode before continuing the queued action sequence.",
            "cut": "Exit cutting mode before continuing the queued action sequence.",
            "mix": "Exit mixing mode before continuing the queued action sequence.",
            "flip": "Exit flipping mode before continuing the queued action sequence.",
        }
        exit_name = exit_name_map.get(mode_class)
        if not exit_name:
            return None
        if str(first.name or "") == exit_name:
            return None
        return [
            PlanStep(
                step_id=str(first.step_id or ""),
                type="action",
                name=exit_name,
                args={},
                expectation=exit_expectation_map.get(mode_class, "Exit the current interaction mode before continuing the queued action sequence."),
            )
        ] + [self._clone_step(step) for step in steps]

    def _apply_local_precondition_patch(
        self,
        *,
        current_step: PlanStep,
        checker_feedback: str,
        original_remaining_plan: list[PlanStep],
        bundle: dict[str, Any],
        high_level_id: str,
        high_level_goal: str,
    ) -> PlanResponse | None:
        steps = [self._clone_step(step) for step in (original_remaining_plan or []) if isinstance(step, PlanStep)]
        if not steps:
            return None
        fields = self._parse_feedback_fields(checker_feedback)
        reason_code = str(fields.get("reason_code") or fields.get("error") or "").strip().lower()
        reason = str(fields.get("reason") or "").strip().lower()
        feedback_to_planner = str(fields.get("feedback_to_planner") or "").strip().lower()
        snapshot = str(bundle.get("instance_query_snapshot", "") or "").strip()

        patched_steps: list[PlanStep] | None = None
        if self._is_instance_disambiguation_reason(reason_code):
            patched_steps = self._patch_plan_missing_instance_ids(steps=steps, snapshot=snapshot)
        elif "bottle_closed" in reason_code or "bottle closed" in reason or "unscrew_bottle" in feedback_to_planner:
            patched_steps = self._patch_plan_bottle_closed(steps=steps)
        elif "blocked_by_mode" in reason_code or "exit the current interaction mode" in feedback_to_planner:
            patched_steps = self._patch_plan_exit_mode_first(steps=steps, checker_feedback=checker_feedback)

        if not patched_steps:
            return None
        return PlanResponse(
            high_level_id=high_level_id,
            goal=high_level_goal,
            explanation="local_precondition_patch",
            thoughts=f"Locally patched remaining plan for {reason_code or 'precondition failure'}.",
            action_list=patched_steps,
        )

    @staticmethod
    def _extract_query_targets_from_args(args: dict[str, Any] | None) -> list[str]:
        if not isinstance(args, dict):
            return []
        out: list[str] = []
        seen: set[str] = set()
        preferred_keys = (
            "target",
            "query",
            "object_name",
            "container_name",
            "item_name",
            "name",
        )
        for key in preferred_keys:
            val = args.get(key)
            if not isinstance(val, str):
                continue
            txt = val.strip()
            low = txt.lower()
            if not txt or low in seen:
                continue
            seen.add(low)
            out.append(txt)
        return out

    @staticmethod
    def _extract_args_from_blocked_step_text(text: str) -> dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            return {}
        for pat in (r"\bargs=(\{.*\})", r"\bstep_args=(\{.*\})"):
            m = re.search(pat, raw)
            if not m:
                continue
            try:
                parsed = json.loads(str(m.group(1) or "").strip())
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _collect_precondition_disambiguation_targets(
        self,
        *,
        current_step: PlanStep,
        checker_result: PreconditionCheckResult,
    ) -> list[str]:
        targets: list[str] = []
        seen: set[str] = set()

        def _push(name: str) -> None:
            txt = str(name or "").strip()
            low = txt.lower()
            if not txt or low in seen:
                return
            seen.add(low)
            targets.append(txt)

        if self._is_instance_disambiguation_reason(checker_result.reason_code):
            for name in self._extract_query_targets_from_args(dict(current_step.args or {})):
                _push(name)

        raw_blocked = []
        if isinstance(checker_result.raw, dict) and isinstance(checker_result.raw.get("all_blocked_steps"), list):
            raw_blocked = list(checker_result.raw.get("all_blocked_steps") or [])
        norm_blocked = list(checker_result.all_blocked_steps or [])
        for item in raw_blocked + norm_blocked:
            if not isinstance(item, dict):
                continue
            if not self._is_instance_disambiguation_reason(item.get("reason_code")):
                continue
            for name in self._extract_query_targets_from_args(item.get("args")):
                _push(name)
            parsed_args = self._extract_args_from_blocked_step_text(item.get("name"))
            for name in self._extract_query_targets_from_args(parsed_args):
                _push(name)

        if not targets:
            fb = str(checker_result.feedback_to_planner or "")
            for m in re.finditer(r"['\"]([^'\"]+)['\"]", fb):
                _push(str(m.group(1) or "").strip())
                if len(targets) >= 3:
                    break
        return targets[:3]

    def _build_precondition_instance_query_snapshot(
        self,
        *,
        current_step: PlanStep,
        checker_result: PreconditionCheckResult,
    ) -> str:
        targets = self._collect_precondition_disambiguation_targets(
            current_step=current_step,
            checker_result=checker_result,
        )
        if not targets:
            return ""
        try:
            from epm.cerebellum.skills.query_scene_objects.skill import QuerySceneObjectsArgs, run as run_query_scene_objects

            query_text = json.dumps(targets, ensure_ascii=False) if len(targets) > 1 else targets[0]
            res = run_query_scene_objects(
                realtime_products_path=self.cfg.realtime_products_path,
                args=QuerySceneObjectsArgs(
                    query=query_text,
                    only_on_screen=False,
                    max_items=8,
                    max_distance=0.0,
                ),
            )
            if not bool(res.success) or not isinstance(res.raw, dict):
                return f"source=precondition_checker queries={json.dumps(targets, ensure_ascii=False)} error={res.error!r}"

            keep_keys = ("name_en", "name_cn", "kind", "instance_id", "distance", "is_on_screen", "container")
            compact_rows: list[str] = []
            results_by_query = res.raw.get("results_by_query")
            if isinstance(results_by_query, dict):
                for target in targets:
                    cur = results_by_query.get(target)
                    compact: list[dict[str, Any]] = []
                    if isinstance(cur, list):
                        for it in cur[:6]:
                            if not isinstance(it, dict):
                                continue
                            compact.append({k: it.get(k) for k in keep_keys if k in it})
                    compact_rows.append(f"{target}: " + json.dumps(compact, ensure_ascii=False, separators=(",", ":")))
            if not compact_rows:
                return f"source=precondition_checker queries={json.dumps(targets, ensure_ascii=False)} results=[]"
            return (
                "source=precondition_checker\n"
                f"queries={json.dumps(targets, ensure_ascii=False)}\n"
                "candidates_by_query=\n"
                + "\n".join(compact_rows)
            )
        except Exception as e:
            return f"source=precondition_checker queries={json.dumps(targets, ensure_ascii=False)} error={e!r}"

    def _epm_validate_step_before_execution(
        self,
        *,
        step: PlanStep,
        obs,
        bundle: dict[str, Any],
        percept,
        high_level_id: str,
        high_level_goal: str,
    ) -> tuple[PlanStep, ActionResult | None]:
        current_step = step
        max_repairs = 0
        if self.precondition_checker is not None:
            max_repairs = max(0, int(self.precondition_checker.cfg.max_plan_repair_attempts))
        recheck_after_repair = bool(getattr(getattr(self.precondition_checker, "cfg", None), "recheck_after_repair", False))
        trace_payload: dict[str, Any] = {
            "episode_step": int(self.step_id),
            "high_level_id": str(high_level_id or "").strip(),
            "high_level_goal": str(high_level_goal or "").strip(),
            "initial_step": self._plan_step_to_record(step),
            "attempts": [],
            "final_status": "",
            "final_step": {},
        }
        latest_feedback_text = ""
        for attempt in range(max_repairs + 1):
            state = self._read_agent_state_snapshot()
            remaining_steps = self._current_remaining_plan_steps(step=current_step)
            attempt_record: dict[str, Any] = {
                "attempt_index": int(attempt + 1),
                "candidate_step": self._plan_step_to_record(current_step),
                "remaining_plan": [self._plan_step_to_record(s) for s in remaining_steps],
                "agent_state": {
                    "mode": state.get("mode"),
                    "is_held": state.get("is_held"),
                    "held_item": state.get("held_item"),
                    "active_container": state.get("active_container"),
                    "force_submit_remaining_steps": state.get("force_submit_remaining_steps"),
                    "timers": state.get("timers"),
                    "timer_events": state.get("timer_events"),
                },
            }
            checker_result: PreconditionCheckResult | None = None
            if self.precondition_checker is not None:
                checker_result = self.precondition_checker.check(
                    observation=self._model_observation(obs),
                    step=current_step,
                    remaining_plan_text=self._format_plan_steps_for_feedback(remaining_steps),
                    recipe_text=self.dish.recipe_text,
                    high_level_goal=high_level_goal,
                    task_progress_text=self._read_task_progress_text(),
                    recent_history_text=self._load_recent_stm_window_text(),
                    latest_visual_feedback_text=str(bundle.get("latest_visual_anomaly_feedback", "") or ""),
                    agent_state_text=json.dumps(state, ensure_ascii=False, indent=2),
                    body_rules_text=str(bundle.get("body_rules", "") or ""),
                    strategy_notes_text=str(bundle.get("strategy_notes", "") or ""),
                    tool_manifest_text=str(bundle.get("tools_manifest_openai", "") or ""),
                    last_feedback=str(self._last_action_feedback or ""),
                )
                attempt_record["checker_result"] = {
                    "status": str(checker_result.status or ""),
                    "reason_code": str(checker_result.reason_code or ""),
                    "reason": str(checker_result.reason or ""),
                    "feedback_to_planner": str(checker_result.feedback_to_planner or ""),
                    "relevant_state": list(checker_result.relevant_state or []),
                    "checks": dict(checker_result.checks or {}),
                    "visual_blockers": list(checker_result.visual_blockers or []),
                    "all_blocked_steps": list(checker_result.all_blocked_steps or []),
                    "raw": dict(checker_result.raw or {}),
                }
                if not checker_result.passed:
                    latest_feedback_text = self._render_precondition_feedback_text(
                        step=current_step,
                        status="fail",
                        reason_code=str(checker_result.reason_code or ""),
                        reason=str(checker_result.reason or ""),
                        feedback_to_planner=str(checker_result.feedback_to_planner or ""),
                        blocked_steps=list(checker_result.all_blocked_steps or []),
                        checks=dict(checker_result.checks or {}),
                        visual_blockers=list(checker_result.visual_blockers or []),
                        relevant_state=list(checker_result.relevant_state or []),
                    )
                    bundle["latest_precondition_feedback"] = latest_feedback_text
                    precondition_query_snapshot = self._build_precondition_instance_query_snapshot(
                        current_step=current_step,
                        checker_result=checker_result,
                    )
                    if precondition_query_snapshot:
                        bundle["instance_query_snapshot"] = precondition_query_snapshot
                    failure = self._precondition_check_failure_result(step=current_step, check=checker_result, state=state)
                    if attempt >= max_repairs:
                        attempt_record["repair"] = {
                            "attempted": False,
                            "applied": False,
                            "reason": "max_plan_repair_attempts_exhausted",
                        }
                        trace_payload["attempts"].append(attempt_record)
                        trace_payload["final_status"] = "failed"
                        trace_payload["final_step"] = self._plan_step_to_record(current_step)
                        trace_payload["final_error"] = str(failure.error or "")
                        self._persist_precondition_check_trace(
                            payload=trace_payload,
                            latest_feedback_text=latest_feedback_text,
                        )
                        return current_step, failure
                    repaired = self._repair_remaining_plan_after_precondition_failure(
                        current_step=current_step,
                        obs=obs,
                        bundle=bundle,
                        percept=percept,
                        high_level_id=high_level_id,
                        high_level_goal=high_level_goal,
                        checker_feedback=(
                            f"reason_code={checker_result.reason_code}\n"
                            f"reason={checker_result.reason}\n"
                            f"blocked_steps={json.dumps(self._summarize_blocked_steps(list(checker_result.all_blocked_steps or [])), ensure_ascii=False)}\n"
                            f"feedback_to_planner={checker_result.feedback_to_planner}\n"
                            f"failed_checks={json.dumps([k for k, v in dict(checker_result.checks or {}).items() if not bool(v)], ensure_ascii=False)}\n"
                            f"state_notes={json.dumps(list(checker_result.relevant_state or [])[:3], ensure_ascii=False)}\n"
                            f"auto_query_attached={json.dumps(bool(precondition_query_snapshot), ensure_ascii=False)}"
                        ),
                        original_remaining_plan=remaining_steps,
                        attempt_index=attempt + 1,
                    )
                    if repaired is None:
                        attempt_record["repair"] = {
                            "attempted": True,
                            "applied": False,
                            "reason": "planner_failed_to_repair_remaining_plan",
                        }
                        trace_payload["attempts"].append(attempt_record)
                        trace_payload["final_status"] = "failed"
                        trace_payload["final_step"] = self._plan_step_to_record(current_step)
                        trace_payload["final_error"] = str(failure.error or "")
                        self._persist_precondition_check_trace(
                            payload=trace_payload,
                            latest_feedback_text=latest_feedback_text,
                        )
                        return current_step, failure
                    attempt_record["repair"] = {
                        "attempted": True,
                        "applied": True,
                        "repaired_first_step": self._plan_step_to_record(repaired),
                    }
                    trace_payload["attempts"].append(attempt_record)
                    if not recheck_after_repair:
                        trace_payload["final_status"] = "repaired"
                        trace_payload["final_step"] = self._plan_step_to_record(repaired)
                        self._persist_precondition_check_trace(
                            payload=trace_payload,
                            latest_feedback_text=latest_feedback_text,
                        )
                        return repaired, None
                    current_step = repaired
                    continue
            rule_result = self._epm_precondition_check(step=current_step)
            if rule_result is not None:
                raw_pf = {}
                if isinstance(rule_result.raw, dict) and isinstance(rule_result.raw.get("precondition_failure"), dict):
                    raw_pf = dict(rule_result.raw.get("precondition_failure") or {})
                reason_code = str(raw_pf.get("code") or str(rule_result.error or "")).strip()
                reason = str(raw_pf.get("hint") or str(rule_result.error or "")).strip()
                feedback_to_planner = self._deterministic_feedback_to_planner(
                    step=current_step,
                    hint=reason,
                    code=reason_code,
                )
                latest_feedback_text = self._render_precondition_feedback_text(
                    step=current_step,
                    status="fail",
                    reason_code=reason_code,
                    reason=reason,
                    feedback_to_planner=feedback_to_planner,
                    blocked_steps=[
                        {
                            "index": 1,
                            "name": str(current_step.name or "").strip(),
                            "reason_code": reason_code,
                            "reason": reason,
                        }
                    ],
                    checks={},
                    visual_blockers=[],
                    relevant_state=[],
                )
                bundle["latest_precondition_feedback"] = latest_feedback_text
                attempt_record["deterministic_result"] = {
                    "error": str(rule_result.error or ""),
                    "raw": dict(rule_result.raw or {}),
                    "feedback_to_planner": feedback_to_planner,
                }
                if attempt >= max_repairs:
                    attempt_record["repair"] = {
                        "attempted": False,
                        "applied": False,
                        "reason": "max_plan_repair_attempts_exhausted",
                    }
                    trace_payload["attempts"].append(attempt_record)
                    trace_payload["final_status"] = "failed"
                    trace_payload["final_step"] = self._plan_step_to_record(current_step)
                    trace_payload["final_error"] = str(rule_result.error or "")
                    self._persist_precondition_check_trace(
                        payload=trace_payload,
                        latest_feedback_text=latest_feedback_text,
                    )
                    return current_step, rule_result
                repaired = self._repair_remaining_plan_after_precondition_failure(
                    current_step=current_step,
                    obs=obs,
                    bundle=bundle,
                    percept=percept,
                    high_level_id=high_level_id,
                    high_level_goal=high_level_goal,
                    checker_feedback=(
                        "deterministic_precondition_failure=true\n"
                        f"error={str(rule_result.error or '')}\n"
                        f"feedback_to_planner={feedback_to_planner}\n"
                        f"state_notes={json.dumps(list(raw_pf.get('relevant_state') or [])[:3], ensure_ascii=False)}"
                    ),
                    original_remaining_plan=remaining_steps,
                    attempt_index=attempt + 1,
                )
                if repaired is None:
                    attempt_record["repair"] = {
                        "attempted": True,
                        "applied": False,
                        "reason": "planner_failed_to_repair_remaining_plan",
                    }
                    trace_payload["attempts"].append(attempt_record)
                    trace_payload["final_status"] = "failed"
                    trace_payload["final_step"] = self._plan_step_to_record(current_step)
                    trace_payload["final_error"] = str(rule_result.error or "")
                    self._persist_precondition_check_trace(
                        payload=trace_payload,
                        latest_feedback_text=latest_feedback_text,
                    )
                    return current_step, rule_result
                attempt_record["repair"] = {
                    "attempted": True,
                    "applied": True,
                    "repaired_first_step": self._plan_step_to_record(repaired),
                }
                trace_payload["attempts"].append(attempt_record)
                if not recheck_after_repair:
                    trace_payload["final_status"] = "repaired"
                    trace_payload["final_step"] = self._plan_step_to_record(repaired)
                    self._persist_precondition_check_trace(
                        payload=trace_payload,
                        latest_feedback_text=latest_feedback_text,
                    )
                    return repaired, None
                current_step = repaired
                continue
            trace_payload["attempts"].append(attempt_record)
            trace_payload["final_status"] = "pass" if not latest_feedback_text else "repaired_and_passed"
            trace_payload["final_step"] = self._plan_step_to_record(current_step)
            self._persist_precondition_check_trace(
                payload=trace_payload,
                latest_feedback_text=latest_feedback_text,
            )
            return current_step, None
        trace_payload["final_status"] = "pass" if not latest_feedback_text else "repaired_and_passed"
        trace_payload["final_step"] = self._plan_step_to_record(current_step)
        self._persist_precondition_check_trace(
            payload=trace_payload,
            latest_feedback_text=latest_feedback_text,
        )
        return current_step, None

    def _epm_precondition_failure(
        self,
        *,
        code: str,
        step: PlanStep,
        state: dict[str, Any],
        hint: str,
        details: dict[str, Any] | None = None,
    ) -> ActionResult:
        raw = {
            "precondition_failure": {
                "code": str(code),
                "step_type": str(step.type or ""),
                "step_name": str(step.name or ""),
                "step_args": dict(step.args or {}),
                "hint": str(hint or ""),
                "state": {
                    "mode": state.get("mode"),
                    "is_held": state.get("is_held"),
                    "held_item": state.get("held_item"),
                    "active_container": state.get("active_container"),
                    "force_submit_remaining_steps": state.get("force_submit_remaining_steps"),
                    "timers": state.get("timers"),
                    "timer_events": state.get("timer_events"),
                },
                "details": dict(details or {}),
            }
        }
        return ActionResult(False, raw=raw, error=f"epm_precondition_failed:{code}")

    def _epm_precondition_check(self, *, step: PlanStep) -> ActionResult | None:
        if not self._is_epm_pipeline():
            return None
        state = self._read_agent_state_snapshot()
        mode_raw = str(state.get("mode") or "").strip()
        mode_class = self._mode_class(mode_raw)
        is_held = bool(state.get("is_held"))

        if step.type == "skill" and step.name == "auto_navigation" and self._mode_string_blocks_navigation(mode_raw):
            return self._epm_precondition_failure(
                code="navigation_blocked_by_mode",
                step=step,
                state=state,
                hint="Exit the current interaction mode before auto_navigation.",
                details={"mode_class": mode_class},
            )

        if step.name in {"pick_up", "pick_up_into_the_container"}:
            if is_held:
                return self._epm_precondition_failure(
                    code="pick_up_requires_empty_hands",
                    step=step,
                    state=state,
                    hint="Put the currently held item down before attempting another pick-up action.",
                )
            if self._mode_string_blocks_navigation(mode_raw):
                return self._epm_precondition_failure(
                    code="pick_up_blocked_by_mode",
                    step=step,
                    state=state,
                    hint="Exit the current interaction mode before pick_up.",
                    details={"mode_class": mode_class},
                )

        if step.name == "put_down":
            if not is_held:
                return self._epm_precondition_failure(
                    code="put_down_requires_holding_item",
                    step=step,
                    state=state,
                    hint="put_down is only valid when the agent is currently holding an item.",
                )
            if self._mode_string_blocks_navigation(mode_raw):
                return self._epm_precondition_failure(
                    code="put_down_blocked_by_mode",
                    step=step,
                    state=state,
                    hint="Exit the current interaction mode before put_down.",
                    details={"mode_class": mode_class},
                )

        if step.name == "repair":
            held_item = str(state.get("held_item") or "").strip().lower()
            is_repair_phone = (
                "repair phone" in held_item
                or ("repair" in held_item and "phone" in held_item)
            )
            if not is_held or not is_repair_phone:
                return self._epm_precondition_failure(
                    code="repair_requires_repair_phone",
                    step=step,
                    state=state,
                    hint="Hold the repair phone before executing repair.",
                )
            if self._mode_string_blocks_navigation(mode_raw):
                return self._epm_precondition_failure(
                    code="repair_blocked_by_mode",
                    step=step,
                    state=state,
                    hint="Exit the current interaction mode before repair.",
                    details={"mode_class": mode_class},
                )

        enter_rules = {
            "enter_pouring_mode": "pour",
            "enter_cutting_mode": "cut",
            "enter_mixing_mode": "mix",
            "enter_the_flipping_mode": "flip",
            "enter_spices_sprinkle_mode": "sprinkle",
        }
        exit_rules = {
            "exit_pouring_mode": "pour",
            "exit_cutting_mode": "cut",
            "exit_mixing_mode": "mix",
            "exit_the_flipping_mode": "flip",
            "exit_spices_sprinkle_mode": "sprinkle",
        }

        expected_enter = enter_rules.get(str(step.name or ""))
        if expected_enter:
            if not is_held:
                return self._epm_precondition_failure(
                    code=f"{expected_enter}_enter_requires_holding_item",
                    step=step,
                    state=state,
                    hint="Hold the required tool or container before entering this interaction mode.",
                )
            if mode_class == expected_enter:
                return self._epm_precondition_failure(
                    code=f"{expected_enter}_mode_already_active",
                    step=step,
                    state=state,
                    hint="The requested interaction mode is already active.",
                )
            if mode_class and mode_class != expected_enter:
                return self._epm_precondition_failure(
                    code=f"{expected_enter}_enter_blocked_by_other_mode",
                    step=step,
                    state=state,
                    hint="Exit the current interaction mode before entering a different one.",
                    details={"active_mode_class": mode_class, "requested_mode_class": expected_enter},
                )

        expected_exit = exit_rules.get(str(step.name or ""))
        if expected_exit and mode_class != expected_exit:
            return self._epm_precondition_failure(
                code=f"{expected_exit}_exit_requires_active_mode",
                step=step,
                state=state,
                hint="This exit action is only valid when the matching interaction mode is active.",
                details={"active_mode_class": mode_class, "requested_mode_class": expected_exit},
            )

        if step.type == "skill" and step.name == "wait_for_timer":
            timers = state.get("timers")
            if not isinstance(timers, list):
                timers = []
            if not timers:
                return self._epm_precondition_failure(
                    code="wait_for_timer_requires_active_timer",
                    step=step,
                    state=state,
                    hint="No active timer is currently tracked. Start or register a timer before waiting for it.",
                )

        return None

    def _post_check_put_down_idle(self, *, timeout_s: float = 1.8, poll_s: float = 0.1) -> tuple[bool, str, bool]:
        deadline = time.time() + max(0.0, float(timeout_s))
        last_mode = ""
        last_is_held = True
        while True:
            try:
                # Refresh from latest realtime feed first.
                self._update_agent_state_from_realtime_products()
                st = self.memory.read_json("agent_state")
            except Exception:
                st = {}
            mode_raw = str(st.get("mode") or "").strip()
            is_held = bool(st.get("is_held"))
            idle = mode_raw.lower().startswith("idle")
            mode_busy_state = self._mode_string_blocks_navigation(mode_raw)

            mode_busy_rt = False
            mode_rt = ""
            try:
                data = read_realtime_products(self.cfg.realtime_products_path)
                items = extract_items(data)
                mode_busy_rt, mode_rt = self._realtime_mode_flags(items)
            except Exception:
                mode_busy_rt = False
                mode_rt = ""

            mode_busy = bool(mode_busy_state or mode_busy_rt)
            if mode_rt:
                last_mode = f"{mode_raw} [rt={mode_rt}]".strip()
            else:
                last_mode = mode_raw
            last_is_held = is_held

            if (not is_held) and idle and (not mode_busy):
                return True, last_mode, last_is_held
            if time.time() >= deadline:
                return False, last_mode, last_is_held
            time.sleep(max(0.01, float(poll_s)))

    def _post_check_pick_up_state(self, *, timeout_s: float = 2.0, poll_s: float = 0.1) -> dict[str, Any]:
        """
        Refresh and return current holding state after pick_up.
        """
        deadline = time.time() + max(0.0, float(timeout_s))
        last: dict[str, Any] = {}
        while True:
            try:
                self._update_agent_state_from_realtime_products()
                st = self.memory.read_json("agent_state")
            except Exception:
                st = {}
            held_item = st.get("held_item")
            if not isinstance(held_item, str) or not held_item.strip():
                held_item = None
            last = {
                "is_held": bool(st.get("is_held")),
                "held_item": held_item,
                "held_item_instance_id": st.get("held_item_instance_id"),
                "item_kind": str(st.get("item_kind") or ""),
                "mode": str(st.get("mode") or ""),
            }
            if bool(last.get("is_held")):
                return last
            if time.time() >= deadline:
                return last
            time.sleep(max(0.01, float(poll_s)))

    def _append_exec_stdout(self, *, text: str) -> None:
        try:
            path = Path(self.cfg.memory_dir) / "exec_stdout.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            header = f"\n# step={self.step_id} time={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())}\n"
            with path.open("a", encoding="utf-8") as f:
                f.write(header)
                f.write(text.rstrip() + "\n")
        except Exception:
            return

    def _maybe_update_active_container(self, *, step: PlanStep, action_result: Any, final_success: bool) -> None:
        if not final_success:
            return

        args = dict(step.args) if isinstance(getattr(step, "args", None), dict) else {}
        name: str | None = None
        instance_id: int | None = None

        # Prefer explicit container fields from step args.
        for key in ("container_name", "container", "target_container"):
            v = args.get(key)
            if isinstance(v, str) and v.strip():
                name = v.strip()
                break

        for key in ("container_instance_id", "target_instance_id"):
            v = args.get(key)
            if v is None:
                continue
            try:
                instance_id = int(v)
                break
            except Exception:
                continue

        # Fall back to returned payloads if present.
        raw = getattr(action_result, "raw", None)
        ret = raw.get("return") if isinstance(raw, dict) else None
        if name is None and isinstance(ret, dict):
            for key in ("container_name", "container"):
                v = ret.get(key)
                if isinstance(v, str) and v.strip():
                    name = v.strip()
                    break

        if instance_id is None and isinstance(ret, dict):
            v = ret.get("container_instance_id")
            if v is not None:
                try:
                    instance_id = int(v)
                except Exception:
                    pass

        state = {}
        try:
            state = self.memory.read_json("agent_state")
        except Exception:
            state = {}

        # Some actions imply the current active container even when they do not carry
        # explicit container fields in args/return (e.g. ladle transfer or pick-up-into-container).
        step_name = str(step.name or "").strip()
        if not name and step_name in {"pick_up_into_the_container", "get_liquid_in_Ladle", "get_liquid_out_Ladle"}:
            active_name = str(state.get("active_container") or "").strip()
            if active_name:
                name = active_name
                active_id = state.get("active_container_instance_id")
                if instance_id is None and active_id is not None and str(active_id).strip():
                    try:
                        instance_id = int(active_id)
                    except Exception:
                        pass

        active_list = state.get("active_containers")
        if not isinstance(active_list, list):
            active_list = []

        evidence_list = state.get("active_container_evidence")
        if not isinstance(evidence_list, list):
            evidence_list = []

        score_delta = 0.0
        add_delta = 0
        remove_delta = 0
        if step_name == "auto_pour":
            score_delta = 5.0
            add_delta = 1
        elif step_name == "pick_up_into_the_container":
            score_delta = 4.0
            add_delta = 1
        elif step_name == "get_liquid_out_Ladle":
            score_delta = 3.0
            add_delta = 1
        elif step_name == "get_liquid_in_Ladle":
            score_delta = -3.0
            remove_delta = 1
        elif step_name == "auto_mix":
            score_delta = 2.0

        active_name_update = str(state.get("active_container") or "").strip() or None
        active_id_update = state.get("active_container_instance_id")
        try:
            active_id_update = int(active_id_update) if active_id_update is not None and str(active_id_update).strip() else None
        except Exception:
            active_id_update = None

        def _same_ref(entry: Any, cand_name: str, cand_id: Optional[int]) -> bool:
            if not isinstance(entry, dict):
                return False
            return entry.get("name") == cand_name and entry.get("instance_id") == cand_id

        def _record_container_reference(
            cand_name: str | None,
            cand_id: Optional[int],
            *,
            score_delta_local: float,
            add_delta_local: int,
            remove_delta_local: int,
            set_active: bool,
        ) -> None:
            nonlocal active_list, evidence_list, active_name_update, active_id_update
            raw_name = str(cand_name or "").strip()
            if not raw_name:
                return

            active_list = [e for e in active_list if not _same_ref(e, raw_name, cand_id)]
            active_list.append({"name": raw_name, "instance_id": cand_id})
            if len(active_list) > 8:
                active_list = active_list[-8:]

            existing: dict[str, Any] | None = None
            kept: list[dict[str, Any]] = []
            for entry in evidence_list:
                if not isinstance(entry, dict):
                    continue
                if _same_ref(entry, raw_name, cand_id) and existing is None:
                    existing = dict(entry)
                    continue
                kept.append(entry)
            entry = existing or {"name": raw_name, "instance_id": cand_id}
            try:
                prev_score = float(entry.get("score", 0.0) or 0.0)
            except Exception:
                prev_score = 0.0
            try:
                prev_add = int(entry.get("add_count", 0) or 0)
            except Exception:
                prev_add = 0
            try:
                prev_remove = int(entry.get("remove_count", 0) or 0)
            except Exception:
                prev_remove = 0
            try:
                prev_touch = int(entry.get("touch_count", 0) or 0)
            except Exception:
                prev_touch = 0
            entry["name"] = raw_name
            entry["instance_id"] = cand_id
            entry["score"] = round(prev_score + float(score_delta_local), 3)
            entry["add_count"] = prev_add + int(add_delta_local)
            entry["remove_count"] = prev_remove + int(remove_delta_local)
            entry["touch_count"] = prev_touch + 1
            entry["last_touch_step"] = int(self.step_id)
            if int(add_delta_local) > 0:
                entry["last_add_step"] = int(self.step_id)
            if int(remove_delta_local) > 0:
                entry["last_remove_step"] = int(self.step_id)
            kept.append(entry)
            if len(kept) > 12:
                kept = kept[-12:]
            evidence_list = kept

            if set_active:
                active_name_update = raw_name
                active_id_update = cand_id

        if name:
            _record_container_reference(
                name,
                instance_id,
                score_delta_local=score_delta,
                add_delta_local=add_delta,
                remove_delta_local=remove_delta,
                set_active=True,
            )

        held_name: str | None = None
        held_instance_id: int | None = None
        held_kind = ""
        held_is_held = False
        if isinstance(ret, dict):
            held_is_held = bool(ret.get("is_held"))
            v = ret.get("held_item")
            if isinstance(v, str) and v.strip():
                held_name = v.strip()
            v = ret.get("held_item_instance_id")
            if v is not None and str(v).strip():
                try:
                    held_instance_id = int(v)
                except Exception:
                    held_instance_id = None
            held_kind = str(ret.get("item_kind") or "").strip().lower()

        if not held_name:
            v = state.get("held_item")
            if isinstance(v, str) and v.strip():
                held_name = v.strip()
        if held_instance_id is None:
            v = state.get("held_item_instance_id")
            if v is not None and str(v).strip():
                try:
                    held_instance_id = int(v)
                except Exception:
                    held_instance_id = None
        if not held_kind:
            held_kind = str(state.get("item_kind") or "").strip().lower()
        if not held_is_held:
            held_is_held = bool(state.get("is_held"))

        held_is_container = bool(held_name) and (
            held_kind == "container" or self._is_force_submit_probe_container(held_name)
        )
        if held_is_held and held_is_container:
            same_as_explicit = bool(name) and str(name).strip() == str(held_name).strip() and instance_id == held_instance_id
            if not same_as_explicit:
                held_score_delta = 1.5 if step_name in {"pick_up", "gui_buy_new_item"} else 0.75
                _record_container_reference(
                    held_name,
                    held_instance_id,
                    score_delta_local=held_score_delta,
                    add_delta_local=0,
                    remove_delta_local=0,
                    set_active=not bool(name),
                )

        if not active_list and not evidence_list and not active_name_update:
            return

        updates = {
            "active_containers": active_list,
            "active_container_evidence": evidence_list,
        }
        if active_name_update:
            updates["active_container"] = active_name_update
            updates["active_container_instance_id"] = active_id_update
        self.memory.update_agent_state(**updates)

    def _coerce_step_type(self, step: PlanStep) -> PlanStep:
        """
        Coerce step.type when the name clearly belongs to the other allowlist.

        This is a robustness feature to prevent invalid-plan loops:
        - if step.type=="skill" but name is an allowed action, treat it as action
        - if step.type=="action" but name is an allowed skill, treat it as skill
        """
        st = (step.type or "").strip().lower()
        name = (step.name or "").strip()
        if not name:
            return step

        # Robustness: tolerate tool-style prefixes in JSON action_list names.
        # Example: "skill__auto_navigation" / "action__pick_up".
        normalized = False
        if name.startswith("action__"):
            name = name[len("action__") :]
            st = "action"
            normalized = True
        elif name.startswith("skill__"):
            name = name[len("skill__") :]
            st = "skill"
            normalized = True
        if normalized and self.cfg.verbose:
            self.log.info(f"[EPM] normalize_step_name: raw={step.name!r} -> name={name!r} type={st!r}")

        action_set = set(self.allowed_actions or [])
        skill_set = set(self.allowed_skills or [])

        # Prefer explicit membership over type.
        if st == "skill" and (name in action_set) and (name not in skill_set):
            if self.cfg.verbose:
                self.log.info(f"[EPM] coerce_step_type: skill->{ 'action' } name={name!r}")
            return PlanStep(step_id=step.step_id, type="action", name=name, args=dict(step.args), expectation=step.expectation)
        if st == "action" and (name in skill_set) and (name not in action_set):
            if self.cfg.verbose:
                self.log.info(f"[EPM] coerce_step_type: action->{ 'skill' } name={name!r}")
            return PlanStep(step_id=step.step_id, type="skill", name=name, args=dict(step.args), expectation=step.expectation)
        if normalized:
            return PlanStep(step_id=step.step_id, type=(st or step.type), name=name, args=dict(step.args), expectation=step.expectation)
        return step

    def _coerce_step_type_with_allowlist(self, step: PlanStep, *, allowed_actions: list[str], allowed_skills: list[str]) -> PlanStep:
        """
        Same as _coerce_step_type, but uses the provided allowlists.
        """
        st = (step.type or "").strip().lower()
        name = (step.name or "").strip()
        if not name:
            return step
        normalized = False
        if name.startswith("action__"):
            name = name[len("action__") :]
            st = "action"
            normalized = True
        elif name.startswith("skill__"):
            name = name[len("skill__") :]
            st = "skill"
            normalized = True
        if normalized and self.cfg.verbose:
            self.log.info(f"[EPM] normalize_step_name: raw={step.name!r} -> name={name!r} type={st!r}")
        action_set = set(allowed_actions or [])
        skill_set = set(allowed_skills or [])
        if st == "skill" and (name in action_set) and (name not in skill_set):
            if self.cfg.verbose:
                self.log.info(f"[EPM] coerce_step_type: skill->{ 'action' } name={name!r}")
            return PlanStep(step_id=step.step_id, type="action", name=name, args=dict(step.args), expectation=step.expectation)
        if st == "action" and (name in skill_set) and (name not in action_set):
            if self.cfg.verbose:
                self.log.info(f"[EPM] coerce_step_type: action->{ 'skill' } name={name!r}")
            return PlanStep(step_id=step.step_id, type="skill", name=name, args=dict(step.args), expectation=step.expectation)
        if normalized:
            return PlanStep(step_id=step.step_id, type=(st or step.type), name=name, args=dict(step.args), expectation=step.expectation)
        return step

    def _canonicalize_dish_name_for_gui_steps(self, step: PlanStep) -> PlanStep:
        """
        Enforce canonical recipe dish_name for GUI order/submit steps.

        Planner may output aliases like "Chicken with Lemon", while UI list entry is
        "Lemon Chicken Breasts". This causes dish_text_not_found on submit.
        """
        name = str(step.name or "").strip()
        if name not in {"gui_order_dish_via_computer", "gui_submit_dish_via_checkout_stand"}:
            return step
        canonical = str(getattr(self.dish, "dish_name", "") or "").strip()
        if not canonical:
            return step
        args = dict(step.args or {})
        cur = str(args.get("dish_name") or "").strip()
        if cur == canonical:
            return step
        args["dish_name"] = canonical
        if self.cfg.verbose and self.cfg.log_mode != "minimal":
            self.log.info(
                f"[EPM] canonicalize_dish_name step={step.step_id} name={name} "
                f"raw={cur!r} -> canonical={canonical!r}"
            )
        return PlanStep(
            step_id=step.step_id,
            type=step.type,
            name=step.name,
            args=args,
            expectation=step.expectation,
        )

    def _update_agent_state_from_realtime_products(self) -> None:
        """
        Update agent_state.json from realtime_products.json.

        Tracks:
        - held item (name_en, kind, is_held)
        - weight only when kind == "products"
        - interaction mode flags (if present): is_pouring_mode / is_pouring / is_filp_mode / is_flip_mode / is_cut_mode / is_sprinkle_mode

        Writes into the memory store's agent_state.json (typically `epm/memory/agent_state.json`).
        """
        path = self.cfg.realtime_products_path
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
        except Exception:
            return

        items: list[dict[str, Any]] = []
        if isinstance(data, dict):
            if isinstance(data.get("products"), list):
                items = [x for x in data["products"] if isinstance(x, dict)]
            elif isinstance(data.get("objects"), list):
                items = [x for x in data["objects"] if isinstance(x, dict)]
        elif isinstance(data, list):
            items = [x for x in data if isinstance(x, dict)]

        held: dict[str, Any] | None = None
        for it in items:
            if bool(it.get("is_held", False)):
                held = it
                break

        is_held = held is not None
        name_en = (held.get("name_en") if isinstance(held, dict) else None) if held else None
        if not isinstance(name_en, str) or not name_en.strip():
            name_en = None
        held_instance_id = held.get("instance_id") if isinstance(held, dict) else None
        kind = (held.get("kind") if isinstance(held, dict) else None) if held else None
        if not isinstance(kind, str) or not kind.strip():
            kind = "unknown"
        else:
            kind = kind.strip()

        weight: float | None = None
        if held is not None and kind == "products":
            w = held.get("weight")
            if isinstance(w, (int, float)):
                weight = float(w)

        def _any_true(keys: tuple[str, ...]) -> bool:
            for it in items:
                for k in keys:
                    if bool(it.get(k, False)):
                        return True
            return False

        def _item_text(it: dict[str, Any]) -> str:
            return " ".join(
                str(it.get(k) or "").strip().lower()
                for k in ("name_en", "name_cn", "game_object")
            )

        def _is_knife_like(it: dict[str, Any]) -> bool:
            return "knife" in _item_text(it)

        def _any_true(keys: tuple[str, ...]) -> bool:
            for it in items:
                if not isinstance(it, dict):
                    continue
                for k in keys:
                    if bool(it.get(k, False)):
                        return True
            return False

        knife_cutting_active = any(
            isinstance(it, dict)
            and _is_knife_like(it)
            and bool(it.get("is_cut_mode") or it.get("is_cutting") or it.get("is_cutting_mode"))
            for it in items
        )

        held_mode_flags = {
            "is_pouring": bool(held.get("is_pouring")) if held else False,
            "is_pouring_mode": bool(held.get("is_pouring_mode")) if held else False,
            "is_cutting_mode": bool(held.get("is_cut_mode") or held.get("is_cutting") or held.get("is_cutting_mode")) if held else False,
            "is_mixing_mode": bool(held.get("is_mixing_mode")) if held else False,
            "is_flip_mode": bool(held.get("is_flip_mode") or held.get("is_filp_mode") or held.get("is_flipping_mode")) if held else False,
            "is_sprinkle_mode": bool(held.get("is_sprinkle_mode") or held.get("is_sprinkle")) if held else False,
        }

        # Agent-state mode should describe the player's current actionable interaction state.
        # Prefer knife cutting mode first; in cutting mode the knife may report is_held=false.
        if knife_cutting_active:
            mode = "cutting_mode (knife realtime)"
        elif held_mode_flags["is_pouring"]:
            mode = "pouring (liquid flowing)"
        elif held_mode_flags["is_pouring_mode"]:
            mode = "pouring_mode (ready to pour)"
        elif held_mode_flags["is_cutting_mode"]:
            mode = "cutting_mode (cutting ingredients)"
        elif held_mode_flags["is_mixing_mode"]:
            mode = "mixing_mode (mixing ingredients)"
        elif held_mode_flags["is_flip_mode"]:
            mode = "flip_mode (flipping food)"
        elif held_mode_flags["is_sprinkle_mode"]:
            mode = "sprinkle_mode (sprinkling spices)"
        elif not is_held:
            global_modes = {
                "cut": _any_true(("is_cut_mode", "is_cutting", "is_cutting_mode")),
                "pour": _any_true(("is_pouring_mode", "is_pouring")),
                "mix": _any_true(("is_mixing_mode",)),
                "sprinkle": _any_true(("is_sprinkle_mode", "is_sprinkle")),
                "flip": _any_true(("is_filp_mode", "is_flip_mode", "is_flipping_mode")),
            }
            mode = ""
            for key in ("cut", "pour", "mix", "sprinkle", "flip"):
                if global_modes.get(key):
                    mode = f"{key}_mode (global_realtime_fallback)"
                    break
            if not mode:
                mode = "idle (hands empty)"
        else:
            global_modes = {
                "cut": _any_true(("is_cut_mode", "is_cutting", "is_cutting_mode")),
                "pour": _any_true(("is_pouring_mode", "is_pouring")),
                "mix": _any_true(("is_mixing_mode",)),
                "sprinkle": _any_true(("is_sprinkle_mode", "is_sprinkle")),
                "flip": _any_true(("is_filp_mode", "is_flip_mode", "is_flipping_mode")),
            }
            mode = ""
            for key in ("cut", "pour", "mix", "sprinkle", "flip"):
                if global_modes.get(key):
                    mode = f"{key}_mode (global_realtime_fallback)"
                    break
            if not mode:
                mode = "normal (holding item)"

        posture = "unknown"
        try:
            ctrl_down = bool(ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000)
            posture = "crouching" if ctrl_down else "standing"
        except Exception:
            posture = "unknown"

        update: dict[str, Any] = {
            "is_held": bool(is_held),
            "held_item": name_en,
            "held_item_name": name_en,
            "held_item_instance_id": held_instance_id,
            "item_kind": kind,
            "mode": mode,
            "posture": posture,
            "held_item_weight": weight,
        }
        self.memory.update_agent_state(**update)

    def _tick_timers(self, *, step_id: int) -> None:
        """
        Update timer events based on agent_state timers.

        - If a timer's deadline has passed, emit a "done" event and remove it from timers.
        - Keep timer_events for 50 steps, then prune.
        """
        try:
            state = self.memory.read_json("agent_state")
        except Exception:
            return

        timers = state.get("timers")
        if not isinstance(timers, list):
            timers = []
        events = state.get("timer_events")
        if not isinstance(events, list):
            events = []

        now = time.time()
        changed = False

        # Normalize events (keep newest per id+status)
        seen: set[tuple[str, str]] = set()
        event_map: dict[tuple[str, str], dict[str, Any]] = {}
        for ev in events:
            if not isinstance(ev, dict):
                changed = True
                continue
            eid = str(ev.get("id") or "").strip()
            status = str(ev.get("status") or "").strip()
            if not eid or not status:
                changed = True
                continue
            key = (eid, status)
            prev = event_map.get(key)
            if prev is not None:
                changed = True
                try:
                    prev_step = int(prev.get("step_id", -1))
                except Exception:
                    prev_step = -1
                try:
                    cur_step = int(ev.get("step_id", -1))
                except Exception:
                    cur_step = -1
                if cur_step >= prev_step:
                    event_map[key] = ev
            else:
                event_map[key] = ev
        norm_events: list[dict[str, Any]] = []
        for key, ev in event_map.items():
            seen.add(key)
            if not isinstance(ev.get("step_id"), int):
                ev["step_id"] = int(step_id)
                changed = True
            if not isinstance(ev.get("at_ts"), (int, float)):
                ev["at_ts"] = float(now)
                changed = True
            norm_events.append(ev)

        # Prune old events (older than 50 steps).
        keep_events: list[dict[str, Any]] = []
        for ev in norm_events:
            try:
                ev_step = int(ev.get("step_id", step_id))
            except Exception:
                ev_step = int(step_id)
                ev["step_id"] = ev_step
                changed = True
            if (int(step_id) - ev_step) <= 50:
                keep_events.append(ev)
            else:
                changed = True

        # Process timers
        keep_timers: list[dict[str, Any]] = []
        for t in timers:
            if not isinstance(t, dict):
                changed = True
                continue
            tid = str(t.get("id") or "").strip()
            if not tid:
                changed = True
                continue
            deadline = t.get("deadline_ts")
            if deadline is None:
                try:
                    deadline = float(t.get("start_ts", 0.0)) + float(t.get("duration", 0.0))
                    t["deadline_ts"] = float(deadline)
                    changed = True
                except Exception:
                    deadline = None
            if deadline is not None and now >= float(deadline):
                key = (tid, "done")
                if key not in seen:
                    keep_events.append({"id": tid, "status": "done", "at_ts": float(now), "step_id": int(step_id)})
                    seen.add(key)
                changed = True
                continue
            keep_timers.append(t)

        if changed:
            self.memory.update_agent_state(timers=keep_timers, timer_events=keep_events)

    def _update_force_submit_status(self, *, step_id: int) -> None:
        """
        Update agent_state with force-submit status once step_id >= threshold.
        """
        try:
            threshold = int(getattr(self.cfg, "force_submit_step_threshold", 0) or 0)
            force_on = bool(getattr(self.cfg, "force_submit_active", False))
            within = int(getattr(self.cfg, "force_submit_within_steps", 50) or 50)
            failure_limit = self._force_submit_failure_limit()
        except Exception:
            return

        if force_on:
            threshold = 1
        if threshold <= 0 or within <= 0:
            return

        try:
            state = self.memory.read_json("agent_state")
        except Exception:
            state = {}

        active = bool(state.get("force_submit_active", False))
        triggered_step = state.get("force_submit_triggered_step")
        deadline_step = state.get("force_submit_deadline_step")

        if not active and int(step_id) >= int(threshold):
            active = True
            triggered_step = int(step_id)
            deadline_step = int(step_id) + int(within)

        if not active:
            return

        if not isinstance(triggered_step, int) or triggered_step <= 0:
            triggered_step = int(step_id)
        if not isinstance(deadline_step, int) or deadline_step <= 0:
            deadline_step = int(triggered_step) + int(within)

        remaining = int(deadline_step) - int(step_id)
        overdue = remaining < 0

        self.memory.update_agent_state(
            force_submit_active=True,
            force_submit_triggered_step=int(triggered_step),
            force_submit_deadline_step=int(deadline_step),
            force_submit_remaining_steps=int(remaining),
            force_submit_overdue=bool(overdue),
            force_submit_step_threshold=int(threshold),
            force_submit_within_steps=int(within),
            force_submit_failure_limit=int(failure_limit),
        )
        self._log_force_submit(
            "status",
            step_id=int(step_id),
            triggered_step=int(triggered_step),
            deadline_step=int(deadline_step),
            remaining_steps=int(remaining),
            overdue=bool(overdue),
            threshold=int(threshold),
            within=int(within),
            failure_limit=int(failure_limit),
        )

    def _activate_force_submit_now(self, *, step_id: int, reason: str = "") -> dict[str, Any]:
        """
        Immediately arm force-submit regardless of the configured threshold.
        Useful for strict open-loop baselines when the one-shot plan is exhausted
        before reaching the normal threshold.
        """
        try:
            within = int(getattr(self.cfg, "force_submit_within_steps", 50) or 50)
        except Exception:
            within = 50
        if within <= 0:
            within = 50
        threshold = int(getattr(self.cfg, "force_submit_step_threshold", 0) or 0)
        failure_limit = self._force_submit_failure_limit()
        triggered_step = int(step_id)
        deadline_step = int(step_id) + int(within)
        payload: dict[str, Any] = {
            "force_submit_active": True,
            "force_submit_triggered_step": triggered_step,
            "force_submit_deadline_step": deadline_step,
            "force_submit_remaining_steps": int(within),
            "force_submit_overdue": False,
            "force_submit_step_threshold": int(threshold),
            "force_submit_within_steps": int(within),
            "force_submit_failure_limit": int(failure_limit),
            "force_submit_consecutive_failures": 0,
        }
        if str(reason or "").strip():
            payload["force_submit_reason"] = str(reason).strip()
        self.memory.update_agent_state(**payload)
        self._log_force_submit(
            "activated",
            step_id=int(step_id),
            reason=str(reason or "").strip(),
            deadline_step=int(deadline_step),
            within_steps=int(within),
            failure_limit=int(failure_limit),
        )
        try:
            return self.memory.read_json("agent_state")
        except Exception:
            return payload

    def _force_submit_allowlists(self) -> tuple[list[str], list[str]]:
        """
        Restricted allowlists after force-submit triggers.
        """
        actions = ["pick_up", "put_down", "throw_away", "exit_current_interaction_mode", "kneel_down", "stand_up", "enter_pouring_mode"]
        skills = ["auto_navigation", "auto_pour", "gui_submit_dish_via_checkout_stand"]
        return actions, skills

    def _force_submit_failure_limit(self) -> int:
        try:
            limit = int(getattr(self.cfg, "force_submit_failure_limit", 3) or 3)
        except Exception:
            limit = 3
        return max(1, int(limit))

    def _log_force_submit(self, event: str, **fields: Any) -> None:
        try:
            parts = [f"event={event}"]
            for key, value in fields.items():
                if value is None:
                    continue
                if isinstance(value, float):
                    parts.append(f"{key}={value:.3f}")
                elif isinstance(value, (dict, list, tuple)):
                    parts.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
                else:
                    parts.append(f"{key}={value!r}")
            self.log.info("[force_submit] " + " ".join(parts))
        except Exception:
            pass

    @staticmethod
    def _summarize_force_submit_checked_containers(entries: Any) -> str:
        if not isinstance(entries, list):
            return ""
        parts: list[str] = []
        for raw in entries:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or raw.get("container_name") or "").strip() or "?"
            instance_id = raw.get("instance_id")
            try:
                instance_id = int(instance_id) if instance_id is not None and str(instance_id).strip() else None
            except Exception:
                instance_id = None
            success = bool(raw.get("success", False))
            assumed_nonempty = bool(raw.get("assumed_nonempty", False))
            item_count_raw = raw.get("item_count", 0)
            try:
                item_count = int(item_count_raw or 0)
            except Exception:
                item_count = 0
            error = str(raw.get("error") or "").strip()
            label = name if instance_id is None else f"{name}#{instance_id}"
            items_text = "unknown" if assumed_nonempty and (not success) else str(item_count)
            detail = f"{label}(success={str(success).lower()},items={items_text}"
            if assumed_nonempty:
                detail += ",assumed_nonempty=true"
            if error:
                detail += f",error={error}"
            detail += ")"
            parts.append(detail)
        return "; ".join(parts)

    @staticmethod
    def _is_submit_checkout_list_failure(*, step: PlanStep, errors: str) -> bool:
        if str(step.name or "").strip() != "gui_submit_dish_via_checkout_stand":
            return False
        return "checkout_dish_list_not_found" in str(errors or "")

    def _enforce_force_submit_guards(self, *, step: PlanStep, final_success: bool, errors: str) -> None:
        try:
            state = self.memory.read_json("agent_state")
        except Exception:
            state = {}

        if bool(state.get("force_submit_active", False)):
            limit = int(self._force_submit_failure_limit())
            updates: dict[str, Any] = {"force_submit_failure_limit": limit}
            if final_success:
                updates["force_submit_consecutive_failures"] = 0
                updates["force_submit_last_error"] = ""
                updates["force_submit_last_failed_step_name"] = ""
                self.memory.update_agent_state(**updates)
                self._log_force_submit(
                    "guard_success",
                    step=str(step.name or ""),
                    consecutive_failures=0,
                )
            else:
                try:
                    prev_failures = int(state.get("force_submit_consecutive_failures", 0) or 0)
                except Exception:
                    prev_failures = 0
                failures = int(prev_failures) + 1
                updates["force_submit_consecutive_failures"] = failures
                updates["force_submit_last_error"] = str(errors or "")
                updates["force_submit_last_failed_step_name"] = str(step.name or "")
                self.memory.update_agent_state(**updates)
                self._log_force_submit(
                    "guard_failure",
                    step=str(step.name or ""),
                    consecutive_failures=int(failures),
                    failure_limit=int(limit),
                    overdue=bool(state.get("force_submit_overdue", False)),
                    error=str(errors or ""),
                )
                if bool(state.get("force_submit_overdue", False)) and failures >= limit:
                    raise RuntimeError(
                        "fatal_episode_error:force_submit_guard:"
                        f"overdue_consecutive_failures={failures}/{limit}:"
                        f"last_step={str(step.name or '')}:"
                        f"last_error={str(errors or '')}"
                    )

        if (not final_success) and self._is_submit_checkout_list_failure(step=step, errors=errors):
            raise RuntimeError(
                "fatal_episode_error:submit_checkout_ui_unrecoverable:"
                f"{str(errors or '')}"
            )

    @classmethod
    def _force_submit_mode_needs_exit(cls, *, state: dict) -> tuple[bool, str]:
        mode_raw = str(state.get("mode") or "").strip()
        mode_class = cls._mode_class(mode_raw)
        stage = str(state.get("force_submit_stage") or "").strip().lower()
        # Some force-submit stages intentionally require staying in the current
        # interaction mode; do not immediately cancel the mode we just entered.
        if stage == "transfer_pour_ready" and mode_class == "pour":
            return False, mode_raw or mode_class
        if mode_class:
            return True, mode_raw or mode_class
        return False, mode_raw

    def _get_force_submit_side_table_indices(self) -> list[int]:
        cached = getattr(self, "_force_submit_side_table_indices", None)
        if isinstance(cached, list) and cached:
            return cached
        indices: list[int] = []
        try:
            epm_dir = Path(__file__).resolve().parents[3]
            path = epm_dir / "data" / "put_place_list.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                platforms = data.get("platforms") if isinstance(data, dict) else None
                if isinstance(platforms, list):
                    for p in platforms:
                        if not isinstance(p, dict):
                            continue
                        if str(p.get("category") or "") != "Side Table":
                            continue
                        ranges = p.get("index_ranges")
                        if isinstance(ranges, list):
                            for r in ranges:
                                if isinstance(r, list) and len(r) >= 2:
                                    start = int(r[0])
                                    end = int(r[1])
                                    if end >= start:
                                        indices.extend(list(range(start, end + 1)))
                        break
        except Exception:
            indices = []
        if not indices:
            indices = [1, 2]
        setattr(self, "_force_submit_side_table_indices", indices)
        return indices

    def _choose_force_submit_place_index(self, *, state: dict) -> Optional[int]:
        indices = self._get_force_submit_side_table_indices()
        used_raw = state.get("force_submit_used_place_indices", [])
        used: set[int] = set()
        if isinstance(used_raw, list):
            for x in used_raw:
                try:
                    used.add(int(x))
                except Exception:
                    continue
        for idx in indices:
            if int(idx) not in used:
                return int(idx)
        return None

    def _resolve_force_submit_active_container(self, *, state: dict) -> tuple[str, Optional[int]]:
        active_name = str(state.get("active_container") or "Big Pot").strip() or "Big Pot"
        active_id = state.get("active_container_instance_id")
        resolved_id: Optional[int] = None
        if active_id is not None and str(active_id).strip():
            try:
                resolved_id = int(active_id)
            except Exception:
                resolved_id = None

        try:
            data = read_realtime_products(self.cfg.realtime_products_path)
            items = extract_items(data)
        except Exception:
            return active_name, resolved_id

        def _name_matches(target: str, item: dict[str, Any]) -> bool:
            want = str(target or "").strip().lower()
            if not want:
                return False
            for k in ("name_en", "name_cn", "game_object", "name"):
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    cand = v.strip().lower()
                    if cand == want or want in cand or cand in want:
                        return True
            return False

        def _same_ref(name_a: str, id_a: Optional[int], name_b: str, id_b: Optional[int]) -> bool:
            if id_a is not None and id_b is not None:
                return int(id_a) == int(id_b)
            return str(name_a or "").strip().lower() == str(name_b or "").strip().lower()

        evidence_list = state.get("active_container_evidence")
        if not isinstance(evidence_list, list):
            evidence_list = []
        active_list = state.get("active_containers")
        if not isinstance(active_list, list):
            active_list = []

        active_rank: dict[tuple[str, Optional[int]], int] = {}
        for idx, entry in enumerate(active_list):
            if not isinstance(entry, dict):
                continue
            n = str(entry.get("name") or "").strip()
            if not n:
                continue
            raw_id = entry.get("instance_id")
            cand_id: Optional[int] = None
            try:
                cand_id = int(raw_id) if raw_id is not None else None
            except Exception:
                cand_id = None
            active_rank[(n.lower(), cand_id)] = idx + 1

        probe_results = state.get("force_submit_probe_results")
        if not isinstance(probe_results, list):
            probe_results = []

        def _candidate_sort_key(name: str, instance_id: Optional[int]) -> tuple[int, float, int, int, int, int, int]:
            best_score = 0.0
            best_add = 0
            best_remove = 0
            best_touch = -1
            best_add_step = -1
            for entry in evidence_list:
                if not isinstance(entry, dict):
                    continue
                e_name = str(entry.get("name") or "").strip()
                if not e_name:
                    continue
                e_raw_id = entry.get("instance_id")
                e_id: Optional[int] = None
                try:
                    e_id = int(e_raw_id) if e_raw_id is not None else None
                except Exception:
                    e_id = None
                if not _same_ref(name, instance_id, e_name, e_id):
                    continue
                try:
                    score = float(entry.get("score", 0.0) or 0.0)
                except Exception:
                    score = 0.0
                try:
                    add_count = int(entry.get("add_count", 0) or 0)
                except Exception:
                    add_count = 0
                try:
                    remove_count = int(entry.get("remove_count", 0) or 0)
                except Exception:
                    remove_count = 0
                try:
                    touch_step = int(entry.get("last_touch_step", -1) or -1)
                except Exception:
                    touch_step = -1
                try:
                    add_step = int(entry.get("last_add_step", -1) or -1)
                except Exception:
                    add_step = -1
                metric = score + (add_count * 4.0) - (remove_count * 3.0)
                if (metric, add_step, touch_step) > (
                    best_score + (best_add * 4.0) - (best_remove * 3.0),
                    best_add_step,
                    best_touch,
                ):
                    best_score = score
                    best_add = add_count
                    best_remove = remove_count
                    best_touch = touch_step
                    best_add_step = add_step
            probe_priority = 0
            probe_item_count = -1
            for entry in probe_results:
                if not isinstance(entry, dict):
                    continue
                e_name = str(entry.get("name") or "").strip()
                if not e_name:
                    continue
                e_raw_id = entry.get("instance_id")
                e_id: Optional[int] = None
                try:
                    e_id = int(e_raw_id) if e_raw_id is not None else None
                except Exception:
                    e_id = None
                if not _same_ref(name, instance_id, e_name, e_id):
                    continue
                if self._force_submit_probe_entry_submitworthy_nonempty(entry):
                    probe_priority = max(probe_priority, 3)
                    try:
                        probe_item_count = max(probe_item_count, int(entry.get("item_count", 0) or 0))
                    except Exception:
                        pass
                elif bool(entry.get("valid_candidate", True)) and bool(entry.get("success", False)):
                    probe_priority = max(probe_priority, 1)
            rank = active_rank.get((str(name or "").strip().lower(), instance_id), 0)
            current_bonus = 1 if _same_ref(name, instance_id, active_name, resolved_id) else 0
            return (
                probe_priority,
                float(probe_item_count),
                best_score + (best_add * 4.0) - (best_remove * 3.0),
                best_add_step,
                best_touch,
                rank,
                current_bonus,
            )

        candidates: list[tuple[str, Optional[int]]] = []

        def _push_candidate(name: str | None, instance_id: Optional[int], *, item: dict[str, Any] | None = None) -> None:
            cand_name = self._normalize_force_submit_candidate_name(name)
            matched_item = item if isinstance(item, dict) else self._force_submit_find_item(items, candidate_name=cand_name, candidate_instance_id=instance_id)
            if not self._force_submit_is_valid_candidate(name=cand_name, instance_id=instance_id, item=matched_item):
                return
            if self._is_force_submit_candidate_banned(state=state, name=cand_name, instance_id=instance_id):
                return
            for old_name, old_id in candidates:
                if _same_ref(cand_name, instance_id, old_name, old_id):
                    return
            candidates.append((cand_name, instance_id))

        best: Optional[dict[str, Any]] = None
        if resolved_id is not None:
            for it in items:
                try:
                    if int(it.get("instance_id")) == resolved_id:
                        best = it
                        break
                except Exception:
                    continue
            if best is not None and _name_matches(active_name, best):
                disp = item_display_name(best)
                new_name = disp.strip() if isinstance(disp, str) and disp.strip() and disp.strip().lower() != "unknown" else active_name
                new_id: Optional[int] = None
                try:
                    if best.get("instance_id") is not None:
                        new_id = int(best.get("instance_id"))
                except Exception:
                    new_id = None
                _push_candidate(new_name, new_id, item=best)

        if best is None:
            matched = best_match_by_name(items, active_name)
            if matched is not None:
                disp = item_display_name(matched)
                new_name = disp.strip() if isinstance(disp, str) and disp.strip() and disp.strip().lower() != "unknown" else active_name
                new_id: Optional[int] = None
                try:
                    if matched.get("instance_id") is not None:
                        new_id = int(matched.get("instance_id"))
                except Exception:
                    new_id = None
                _push_candidate(new_name, new_id, item=matched)

        for entry in active_list:
            if not isinstance(entry, dict):
                continue
            cand_name = str(entry.get("name") or "").strip()
            cand_id_raw = entry.get("instance_id")
            cand_id: Optional[int] = None
            try:
                cand_id = int(cand_id_raw) if cand_id_raw is not None else None
            except Exception:
                cand_id = None
            matched_item = self._force_submit_find_item(items, candidate_name=cand_name, candidate_instance_id=cand_id)
            _push_candidate(cand_name, cand_id, item=matched_item)

        for entry in evidence_list:
            if not isinstance(entry, dict):
                continue
            cand_name = str(entry.get("name") or "").strip()
            cand_id_raw = entry.get("instance_id")
            cand_id: Optional[int] = None
            try:
                cand_id = int(cand_id_raw) if cand_id_raw is not None else None
            except Exception:
                cand_id = None
            matched_item = self._force_submit_find_item(items, candidate_name=cand_name, candidate_instance_id=cand_id)
            _push_candidate(cand_name, cand_id, item=matched_item)

        scene_bake_tray = self._find_force_submit_scene_container("Bake Tray")
        if scene_bake_tray is not None:
            matched_item = self._force_submit_find_item(items, candidate_name=scene_bake_tray[0], candidate_instance_id=scene_bake_tray[1])
            _push_candidate(scene_bake_tray[0], scene_bake_tray[1], item=matched_item)

        if candidates:
            picked_name, picked_id = max(candidates, key=lambda it: _candidate_sort_key(it[0], it[1]))
            try:
                self.memory.update_agent_state(active_container=picked_name, active_container_instance_id=picked_id)
            except Exception:
                pass
            self._log_force_submit(
                "resolve_active_container",
                picked_name=picked_name,
                picked_instance_id=picked_id,
                active_name=active_name,
                active_instance_id=resolved_id,
                candidate_count=len(candidates),
                candidates=[{"name": n, "instance_id": i} for n, i in candidates],
            )
            return picked_name, picked_id

        self._log_force_submit(
            "resolve_active_container_fallback",
            active_name=active_name,
            active_instance_id=resolved_id,
        )
        return active_name, resolved_id

    @staticmethod
    def _force_submit_drop_shelf_target() -> str:
        return "Top Shelf 2 (2-Shelf)-13"

    @staticmethod
    def _force_submit_parse_contents(contents: str) -> tuple[int, int]:
        raw = str(contents or "").strip()
        if not raw:
            return 0, 0
        parts = [p.strip() for p in re.split(r"[,;]+", raw) if str(p).strip()]
        return len(parts), len(raw)

    def _force_submit_probe_candidates(self, *, state: dict, active_name: str, active_id: Optional[int]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        try:
            data = read_realtime_products(self.cfg.realtime_products_path)
            items = extract_items(data)
        except Exception:
            items = []

        def _push(name: str | None, instance_id: Optional[int], *, source_priority: int, item: dict[str, Any] | None = None) -> None:
            cand_name = self._normalize_force_submit_candidate_name(name)
            matched_item = item if isinstance(item, dict) else self._force_submit_find_item(items, candidate_name=cand_name, candidate_instance_id=instance_id)
            if not self._force_submit_is_valid_candidate(name=cand_name, instance_id=instance_id, item=matched_item):
                return
            if self._is_force_submit_candidate_banned(state=state, name=cand_name, instance_id=instance_id):
                return
            for old in candidates:
                if str(old.get("name") or "").strip().lower() == cand_name.lower() and old.get("instance_id") == instance_id:
                    prev_source_priority = old.get("source_priority")
                    try:
                        prev_source_priority = int(prev_source_priority)
                    except Exception:
                        prev_source_priority = 99
                    if int(source_priority) < prev_source_priority:
                        old["source_priority"] = int(source_priority)
                    return
            candidates.append({
                "name": cand_name,
                "instance_id": instance_id,
                "source_priority": int(source_priority),
            })

        _push(active_name, active_id, source_priority=0)
        active_list = state.get("active_containers")
        if isinstance(active_list, list):
            for entry in reversed(active_list):
                if not isinstance(entry, dict):
                    continue
                cand_id = entry.get("instance_id")
                try:
                    cand_id = int(cand_id) if cand_id is not None else None
                except Exception:
                    cand_id = None
                matched_item = self._force_submit_find_item(items, candidate_name=entry.get("name"), candidate_instance_id=cand_id)
                _push(entry.get("name"), cand_id, source_priority=0, item=matched_item)
        evidence_list = state.get("active_container_evidence")
        if isinstance(evidence_list, list):
            for entry in reversed(evidence_list):
                if not isinstance(entry, dict):
                    continue
                cand_id = entry.get("instance_id")
                try:
                    cand_id = int(cand_id) if cand_id is not None else None
                except Exception:
                    cand_id = None
                matched_item = self._force_submit_find_item(items, candidate_name=entry.get("name"), candidate_instance_id=cand_id)
                _push(entry.get("name"), cand_id, source_priority=0, item=matched_item)
        scene_big_pot = self._find_force_submit_scene_container("Big Pot")
        if scene_big_pot is not None:
            matched_item = self._force_submit_find_item(items, candidate_name=scene_big_pot[0], candidate_instance_id=scene_big_pot[1])
            _push(scene_big_pot[0], scene_big_pot[1], source_priority=1, item=matched_item)
        scene_bake_tray = self._find_force_submit_scene_container("Bake Tray")
        if scene_bake_tray is not None:
            matched_item = self._force_submit_find_item(items, candidate_name=scene_bake_tray[0], candidate_instance_id=scene_bake_tray[1])
            _push(scene_bake_tray[0], scene_bake_tray[1], source_priority=1, item=matched_item)
        for item in items:
            if not isinstance(item, dict):
                continue
            disp = item_display_name(item)
            cand_name = disp.strip() if isinstance(disp, str) and disp.strip() else ""
            cand_id = item.get("instance_id")
            try:
                cand_id = int(cand_id) if cand_id is not None and str(cand_id).strip() else None
            except Exception:
                cand_id = None
            if not self._force_submit_is_valid_candidate(name=cand_name, instance_id=cand_id, item=item):
                continue
            _push(cand_name, cand_id, source_priority=2, item=item)
        candidates.sort(
            key=lambda it: (
                int(it.get("source_priority", 99) or 99),
                self._force_submit_probe_candidate_sort_key(it.get("name"), it.get("instance_id")),
            )
        )
        return candidates

    def _read_force_submit_interaction_raw(self) -> dict[str, Any]:
        path = self.cfg.realtime_products_path.parent / "realtime_interaction_info.txt"
        try:
            exists = path.exists()
        except Exception:
            exists = False
        if not exists:
            return {"path": str(path), "exists": False, "age_s": None, "raw_text": ""}
        try:
            mtime = path.stat().st_mtime
            age_s = max(0.0, time.time() - float(mtime))
        except Exception:
            age_s = None
        try:
            raw_text = path.read_text(encoding="utf-8-sig", errors="ignore")
        except Exception as e:
            raw_text = f"<read_failed:{e}>"
        return {
            "path": str(path),
            "exists": True,
            "age_s": (round(float(age_s), 3) if isinstance(age_s, (int, float)) else None),
            "raw_text": str(raw_text or "").strip(),
        }

    @staticmethod
    def _force_submit_probe_offsets(
        distance_px: int = 50,
        *,
        include_center: bool = True,
    ) -> list[dict[str, int | str]]:
        distance = max(1, int(distance_px or 50))
        suffix = "" if distance == 50 else str(distance)
        offsets: list[dict[str, int | str]] = []
        if include_center:
            offsets.append({"label": "center", "dx": 0, "dy": 0})
        offsets.extend([
            {"label": f"up{suffix}", "dx": 0, "dy": -distance},
            {"label": f"down{suffix}", "dx": 0, "dy": distance},
            {"label": f"left{suffix}", "dx": -distance, "dy": 0},
            {"label": f"right{suffix}", "dx": distance, "dy": 0},
        ])
        return offsets

    @staticmethod
    def _force_submit_probe_sample_has_live_alt_j(sample: dict[str, Any]) -> bool:
        if not isinstance(sample, dict):
            return False
        for key in ("has_target",):
            if bool(sample.get(key)):
                return True
        for key in ("item_name", "action", "weight", "container_name", "contents"):
            if str(sample.get(key) or "").strip():
                return True
        raw_info = sample.get("raw_alt_j")
        if not isinstance(raw_info, dict):
            return False
        if not bool(raw_info.get("exists", False)):
            return False
        age_s = raw_info.get("age_s")
        raw_text = str(raw_info.get("raw_text") or "").strip()
        return isinstance(age_s, (int, float)) and float(age_s) <= 2.5 and bool(raw_text)

    @classmethod
    def _force_submit_probe_observation_sort_key(cls, entry: Any) -> tuple[int, int, int, int, int, int, int]:
        if not isinstance(entry, dict):
            return (-1, -1, -1, -1, -1, -1, -1)
        success = 1 if bool(entry.get("success", False)) else 0
        submitworthy_nonempty = 1 if cls._force_submit_probe_entry_submitworthy_nonempty(entry) else 0
        live_alt_j = 1 if cls._force_submit_probe_sample_has_live_alt_j(entry) else 0
        try:
            item_count = int(entry.get("item_count", 0) or 0)
        except Exception:
            item_count = 0
        try:
            char_count = int(entry.get("char_count", 0) or 0)
        except Exception:
            char_count = 0
        contents_len = len(str(entry.get("contents") or ""))
        item_name_len = len(str(entry.get("item_name") or ""))
        return (
            success,
            submitworthy_nonempty,
            item_count,
            char_count,
            contents_len,
            item_name_len,
            live_alt_j,
        )

    @staticmethod
    def _force_submit_probe_result_sort_key(entry: Any) -> tuple[int, int, int, int]:
        if not isinstance(entry, dict):
            return (-1, -1, -1, -1)
        valid_candidate = bool(entry.get("valid_candidate", True))
        try:
            item_count = int(entry.get("item_count", 0) or 0)
        except Exception:
            item_count = 0
        try:
            char_count = int(entry.get("char_count", 0) or 0)
        except Exception:
            char_count = 0
        contents_len = len(str(entry.get("contents") or ""))
        if not valid_candidate:
            return (-1, -1, -1, -1)
        if EpmAgent._force_submit_probe_entry_submitworthy_nonempty(entry):
            return (2, max(1, item_count), char_count, contents_len)
        if bool(entry.get("assumed_nonempty", False)):
            return (1, 0, 0, contents_len)
        return (-1, -1, -1, -1)

    @classmethod
    def _force_submit_probe_entry_confirmed_nonempty(cls, entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        if not bool(entry.get("valid_candidate", True)):
            return False
        if not bool(entry.get("success", False)):
            return False
        try:
            return int(entry.get("item_count", 0) or 0) > 0
        except Exception:
            return False

    @staticmethod
    def _force_submit_probe_error_suggests_nonempty(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        error = str(entry.get("error") or "").strip().lower()
        return error.startswith("alt_j_expected_container_not_visible_after_nav:")

    @staticmethod
    def _force_submit_probe_has_explicit_pickup_signal(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        if str(entry.get("container_name") or "").strip():
            return True
        if str(entry.get("item_name") or "").strip():
            return True
        if str(entry.get("contents") or "").strip():
            return True
        try:
            if int(entry.get("item_count", 0) or 0) > 0:
                return True
        except Exception:
            return False
        return False

    @classmethod
    def _force_submit_probe_entry_submitworthy_nonempty(cls, entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        if not bool(entry.get("valid_candidate", True)):
            return False
        if cls._force_submit_probe_entry_confirmed_nonempty(entry):
            return True
        return cls._force_submit_probe_error_suggests_nonempty(entry)

    @classmethod
    def _force_submit_probe_name_matches(cls, expected_name: str, got_name: str) -> bool:
        expected = cls._normalize_force_submit_candidate_name(expected_name).lower()
        got = cls._normalize_force_submit_candidate_name(got_name).lower()
        if not expected or not got:
            return False
        return (expected == got) or (expected in got) or (got in expected)

    @classmethod
    def _force_submit_altj_item_signal(cls, sample: dict[str, Any], *, expected_name: str) -> bool:
        if not isinstance(sample, dict):
            return False
        if not cls._is_force_submit_probe_container(expected_name):
            return False
        item_name = str(sample.get("item_name") or "").strip()
        if not item_name:
            return False
        item_norm = cls._normalize_force_submit_candidate_name(item_name)
        expected_norm = cls._normalize_force_submit_candidate_name(expected_name)
        if not item_norm or cls._force_submit_probe_name_matches(expected_norm, item_norm):
            return False
        if cls._is_force_submit_probe_container(item_norm):
            return False
        return True

    @classmethod
    def _force_submit_altj_food_signal(cls, sample: dict[str, Any], *, expected_name: str) -> bool:
        if not isinstance(sample, dict):
            return False
        if not cls._is_force_submit_transfer_source_vessel(expected_name):
            return False
        return cls._force_submit_altj_item_signal(sample, expected_name=expected_name)

    def _find_force_submit_scene_container(self, target_name: str) -> tuple[str, Optional[int]] | None:
        raw_target = self._normalize_force_submit_candidate_name(target_name)
        if not raw_target:
            return None
        try:
            data = read_realtime_products(self.cfg.realtime_products_path)
            items = extract_items(data)
        except Exception:
            return None
        matched = best_match_by_name(items, raw_target)
        if not isinstance(matched, dict):
            return None
        disp = self._normalize_force_submit_candidate_name(item_display_name(matched) or raw_target)
        if not disp:
            disp = raw_target
        raw_id = matched.get("instance_id")
        instance_id: Optional[int] = None
        try:
            instance_id = int(raw_id) if raw_id is not None and str(raw_id).strip() else None
        except Exception:
            instance_id = None
        return disp, instance_id

    def _read_force_submit_container_probe_once(self, *, expected_name: str) -> dict[str, Any]:
        raw_info = self._read_force_submit_interaction_raw()
        try:
            userdata_root = self.cfg.realtime_products_path.parent
            snap = read_interaction_snapshot(userdata_root=userdata_root, max_age_s=2.5)
        except Exception as e:
            probe = {
                "success": False,
                "error": f"alt_j_snapshot_exception:{e}",
                "container_name": "",
                "contents": "",
                "item_count": 0,
                "char_count": 0,
                "raw_alt_j": raw_info,
            }
            self._log_force_submit("probe_read", expected_name=expected_name, **probe)
            return probe
        if snap is None:
            probe = {
                "success": False,
                "error": "alt_j_snapshot_missing_or_stale",
                "container_name": "",
                "contents": "",
                "item_count": 0,
                "char_count": 0,
                "raw_alt_j": raw_info,
            }
            self._log_force_submit("probe_read", expected_name=expected_name, **probe)
            return probe
        has_target = bool(getattr(snap, "has_target", False))
        item_name = str(getattr(snap, "item_name", "") or "").strip()
        action = str(getattr(snap, "action", "") or "").strip()
        weight = str(getattr(snap, "weight", "") or "").strip()
        got_name = str(getattr(snap, "container_name", "") or "").strip()
        contents = str(getattr(snap, "container_contents", "") or "").strip()
        item_count, char_count = self._force_submit_parse_contents(contents)
        name_matches = bool(got_name) and self._force_submit_probe_name_matches(str(expected_name or ""), got_name)
        if not got_name:
            error = f"alt_j_container_name_missing:expected={expected_name!r}"
        elif name_matches:
            error = ""
        else:
            error = f"alt_j_container_mismatch:expected={expected_name!r}:got={got_name!r}"
        probe = {
            "success": bool(name_matches),
            "error": error,
            "has_target": has_target,
            "item_name": item_name,
            "action": action,
            "weight": weight,
            "container_name": got_name,
            "contents": contents,
            "item_count": int(item_count),
            "char_count": int(char_count),
            "raw_alt_j": raw_info,
        }
        self._log_force_submit("probe_read", expected_name=expected_name, **probe)
        return probe

    def _read_force_submit_container_probe_at_offset(
        self,
        *,
        expected_name: str,
        offset_label: str,
        offset_dx: int,
        offset_dy: int,
        ) -> dict[str, Any]:
        move_error = ""
        io: Optional[RawInputController] = None
        moved = bool(int(offset_dx) or int(offset_dy))
        try:
            if moved:
                io = RawInputController()
                io.mouse_move_relative(dx=int(offset_dx), dy=int(offset_dy))
            probe = dict(
                self._read_force_submit_container_probe_with_dwell(
                    expected_name=expected_name,
                    settle_s=(0.3 if moved else 0.0),
                    dwell_reads=1,
                    dwell_interval_s=0.08,
                )
            )
        except Exception as e:
            move_error = str(e)
            probe = {
                "success": False,
                "error": f"alt_j_probe_offset_exception:{e}",
                "container_name": "",
                "contents": "",
                "item_count": 0,
                "char_count": 0,
                "raw_alt_j": self._read_force_submit_interaction_raw(),
            }
        finally:
            if moved and io is not None:
                try:
                    io.mouse_move_relative(dx=-int(offset_dx), dy=-int(offset_dy))
                    time.sleep(0.05)
                except Exception as e:
                    move_error = move_error or f"restore_failed:{e}"
        probe["probe_offset_label"] = str(offset_label or "").strip() or "center"
        probe["probe_offset_dx"] = int(offset_dx)
        probe["probe_offset_dy"] = int(offset_dy)
        if move_error:
            probe["probe_offset_error"] = move_error
        return probe

    def _read_force_submit_container_probe_with_dwell(
        self,
        *,
        expected_name: str,
        settle_s: float = 0.0,
        dwell_reads: int = 1,
        dwell_interval_s: float = 0.08,
    ) -> dict[str, Any]:
        settle = max(0.0, float(settle_s or 0.0))
        reads = max(1, int(dwell_reads or 1))
        interval = max(0.0, float(dwell_interval_s or 0.0))
        if settle > 0.0:
            time.sleep(settle)
        dwell_samples: list[dict[str, Any]] = []
        for idx in range(reads):
            probe = dict(self._read_force_submit_container_probe_once(expected_name=expected_name))
            probe["dwell_read_index"] = int(idx + 1)
            dwell_samples.append(probe)
            if idx + 1 < reads:
                time.sleep(interval)
        best = max(dwell_samples, key=self._force_submit_probe_observation_sort_key)
        final_probe = dict(best)
        final_probe["dwell_reads"] = int(reads)
        final_probe["dwell_settle_s"] = float(settle)
        final_probe["dwell_interval_s"] = float(interval)
        final_probe["dwell_samples"] = dwell_samples
        return final_probe

    def _read_force_submit_container_probe(self, *, expected_name: str) -> dict[str, Any]:
        samples: list[dict[str, Any]] = []

        def _run_probe_offsets(offset_specs: list[dict[str, int | str]]) -> None:
            for idx, offset in enumerate(offset_specs):
                sample = dict(
                    self._read_force_submit_container_probe_at_offset(
                        expected_name=expected_name,
                        offset_label=str(offset.get("label") or "center"),
                        offset_dx=int(offset.get("dx", 0) or 0),
                        offset_dy=int(offset.get("dy", 0) or 0),
                    )
                )
                sample["sample_index"] = int(len(samples) + 1)
                samples.append(sample)
                if idx + 1 < len(offset_specs):
                    time.sleep(0.06)

        _run_probe_offsets(self._force_submit_probe_offsets(distance_px=50, include_center=True))
        first_pass_successes = [s for s in samples if bool(s.get("success", False))]
        first_pass_item_signals = [
            s for s in samples if self._force_submit_altj_item_signal(s, expected_name=expected_name)
        ]
        first_pass_content_signals = [
            s
            for s in samples
            if int(s.get("item_count", 0) or 0) > 0 or bool(str(s.get("contents") or "").strip())
        ]
        wide_probe_attempted = False
        if not first_pass_successes and not first_pass_item_signals and not first_pass_content_signals:
            wide_probe_attempted = True
            time.sleep(0.08)
            _run_probe_offsets(self._force_submit_probe_offsets(distance_px=100, include_center=False))

        success_samples = [s for s in samples if bool(s.get("success", False))]
        live_alt_j_samples = [s for s in samples if self._force_submit_probe_sample_has_live_alt_j(s)]
        item_signal_samples = [
            s for s in live_alt_j_samples if self._force_submit_altj_item_signal(s, expected_name=expected_name)
        ]
        transfer_food_samples = [
            s for s in item_signal_samples if self._force_submit_altj_food_signal(s, expected_name=expected_name)
        ]
        matched_offsets = [str(s.get("probe_offset_label") or "") for s in success_samples if str(s.get("probe_offset_label") or "").strip()]
        if success_samples:
            best = max(
                success_samples,
                key=lambda s: (
                    int(s.get("item_count", 0) or 0),
                    int(s.get("char_count", 0) or 0),
                    len(str(s.get("contents") or "")),
                ),
            )
            final_probe = dict(best)
            final_probe["success"] = True
            final_probe["error"] = ""
            final_probe["assumed_nonempty"] = False
            final_probe["occlusion_suspected"] = False
        else:
            best = live_alt_j_samples[-1] if live_alt_j_samples else (samples[-1] if samples else {})
            final_probe = dict(best)
            if transfer_food_samples:
                best = max(
                    transfer_food_samples,
                    key=lambda s: (
                        int(s.get("item_count", 0) or 0),
                        int(s.get("char_count", 0) or 0),
                        len(str(s.get("contents") or "")),
                        len(str(s.get("item_name") or "")),
                    ),
                )
                final_probe = dict(best)
                final_probe["success"] = True
                final_probe["assumed_nonempty"] = False
                final_probe["occlusion_suspected"] = False
                final_probe["container_name"] = str(expected_name or "")
                if int(final_probe.get("item_count", 0) or 0) <= 0:
                    inferred_item = str(final_probe.get("item_name") or "").strip()
                    final_probe["item_count"] = 1 if inferred_item else 0
                    inferred_contents = str(final_probe.get("contents") or "").strip()
                    if inferred_item and not inferred_contents:
                        final_probe["contents"] = inferred_item
                final_probe["char_count"] = len(str(final_probe.get("contents") or ""))
                final_probe["error"] = ""
                final_probe["inferred_from_alt_j_food_signal"] = True
            elif item_signal_samples:
                best = max(
                    item_signal_samples,
                    key=lambda s: (
                        int(s.get("item_count", 0) or 0),
                        int(s.get("char_count", 0) or 0),
                        len(str(s.get("contents") or "")),
                        len(str(s.get("item_name") or "")),
                    ),
                )
                final_probe = dict(best)
                final_probe["success"] = True
                final_probe["assumed_nonempty"] = False
                final_probe["occlusion_suspected"] = False
                final_probe["container_name"] = str(expected_name or "")
                inferred_item = str(final_probe.get("item_name") or "").strip()
                if int(final_probe.get("item_count", 0) or 0) <= 0:
                    final_probe["item_count"] = 1 if inferred_item else 0
                inferred_contents = str(final_probe.get("contents") or "").strip()
                if inferred_item and not inferred_contents:
                    final_probe["contents"] = inferred_item
                final_probe["char_count"] = len(str(final_probe.get("contents") or ""))
                final_probe["error"] = ""
                final_probe["inferred_from_alt_j_item_signal"] = True
            else:
                final_probe["success"] = False
                final_probe["assumed_nonempty"] = bool(live_alt_j_samples)
                final_probe["occlusion_suspected"] = bool(live_alt_j_samples)
                if live_alt_j_samples:
                    final_probe["error"] = (
                        f"alt_j_expected_container_not_visible_after_nav:expected={expected_name!r}:"
                        f"offsets_checked={len(samples)}"
                    )
                else:
                    final_probe["error"] = (
                        f"alt_j_probe_unavailable_or_stale:expected={expected_name!r}:"
                        f"offsets_checked={len(samples)}"
                    )

        final_probe["samples"] = samples
        final_probe["wide_probe_attempted"] = bool(wide_probe_attempted)
        final_probe["probe_offsets_attempted"] = [str(s.get("probe_offset_label") or "") for s in samples]
        final_probe["live_alt_j_offsets"] = [str(s.get("probe_offset_label") or "") for s in live_alt_j_samples]
        final_probe["matched_offsets"] = matched_offsets
        final_probe["matched_offset"] = str(final_probe.get("probe_offset_label") or "")
        self._log_force_submit(
            "probe_read_confirmed",
            expected_name=expected_name,
            total_reads=int(len(samples)),
            interval_s=0.0,
            tail_reads=int(len(samples)),
            tail_successes=int(len(success_samples)),
            tail_required_successes=1,
            final_success=bool(final_probe.get("success", False)),
            assumed_nonempty=bool(final_probe.get("assumed_nonempty", False)),
            final_container_name=str(final_probe.get("container_name") or ""),
            final_item_count=int(final_probe.get("item_count", 0) or 0),
            matched_offset=str(final_probe.get("matched_offset") or ""),
            final_error=str(final_probe.get("error") or ""),
        )
        return final_probe

    def _force_submit_align_cursor_to_container(self, *, expected_name: str, max_passes: int = 2) -> dict[str, Any]:
        expected = self._normalize_force_submit_candidate_name(expected_name)
        if not expected:
            return {
                "success": False,
                "aligned": False,
                "error": "expected_name_empty",
                "expected_name": "",
            }
        result: dict[str, Any] = {
            "success": False,
            "aligned": False,
            "expected_name": expected,
            "chosen_offset": "center",
            "chosen_dx": 0,
            "chosen_dy": 0,
            "passes": 0,
        }
        for attempt in range(max(1, int(max_passes))):
            result["passes"] = int(attempt + 1)
            center_probe = self._read_force_submit_container_probe_once(expected_name=expected)
            if bool(center_probe.get("success", False)):
                result.update(
                    {
                        "success": True,
                        "aligned": True,
                        "chosen_offset": "center",
                        "chosen_dx": 0,
                        "chosen_dy": 0,
                        "probe": center_probe,
                    }
                )
                self._log_force_submit(
                    "pickup_align_confirmed",
                    expected_name=expected,
                    pass_index=int(attempt + 1),
                    chosen_offset="center",
                    chosen_dx=0,
                    chosen_dy=0,
                )
                return result

            sweep_probe = self._read_force_submit_container_probe(expected_name=expected)
            samples = sweep_probe.get("samples")
            if not isinstance(samples, list):
                samples = []
            success_samples = [dict(s) for s in samples if isinstance(s, dict) and bool(s.get("success", False))]
            if not success_samples:
                result["probe"] = sweep_probe
                continue

            chosen = max(
                success_samples,
                key=lambda s: (
                    1 if str(s.get("probe_offset_label") or "").strip().lower() == "center" else 0,
                    int(s.get("item_count", 0) or 0),
                    int(s.get("char_count", 0) or 0),
                    -abs(int(s.get("probe_offset_dx", 0) or 0)) - abs(int(s.get("probe_offset_dy", 0) or 0)),
                ),
            )
            move_dx = int(chosen.get("probe_offset_dx", 0) or 0)
            move_dy = int(chosen.get("probe_offset_dy", 0) or 0)
            chosen_offset = str(chosen.get("probe_offset_label") or "").strip() or "center"
            if move_dx or move_dy:
                try:
                    io = RawInputController()
                    io.mouse_move_relative(dx=move_dx, dy=move_dy)
                except Exception as e:
                    result.update(
                        {
                            "success": False,
                            "aligned": False,
                            "error": f"align_move_exception:{e}",
                            "probe": sweep_probe,
                            "chosen_offset": chosen_offset,
                            "chosen_dx": move_dx,
                            "chosen_dy": move_dy,
                        }
                    )
                    self._log_force_submit(
                        "pickup_align_failed",
                        expected_name=expected,
                        pass_index=int(attempt + 1),
                        chosen_offset=chosen_offset,
                        chosen_dx=move_dx,
                        chosen_dy=move_dy,
                        error=str(e),
                    )
                    return result
            confirm_probe = self._read_force_submit_container_probe_with_dwell(
                expected_name=expected,
                settle_s=(0.15 if (move_dx or move_dy) else 0.05),
                dwell_reads=1,
                dwell_interval_s=0.08,
            )
            result.update(
                {
                    "probe": confirm_probe,
                    "chosen_offset": chosen_offset,
                    "chosen_dx": move_dx,
                    "chosen_dy": move_dy,
                }
            )
            if bool(confirm_probe.get("success", False)):
                result["success"] = True
                result["aligned"] = True
                self._log_force_submit(
                    "pickup_align_confirmed",
                    expected_name=expected,
                    pass_index=int(attempt + 1),
                    chosen_offset=chosen_offset,
                    chosen_dx=move_dx,
                    chosen_dy=move_dy,
                )
                return result

        result["error"] = str(result.get("error") or "alt_j_alignment_unconfirmed")
        self._log_force_submit(
            "pickup_align_unconfirmed",
            expected_name=expected,
            passes=int(result.get("passes", 0) or 0),
            chosen_offset=str(result.get("chosen_offset") or "center"),
            chosen_dx=int(result.get("chosen_dx", 0) or 0),
            chosen_dy=int(result.get("chosen_dy", 0) or 0),
            error=str(result.get("error") or ""),
        )
        return result

    @staticmethod
    def _force_submit_contains_english_term(name: str | None, term: str) -> bool:
        raw = str(name or "").strip()
        token = str(term or "").strip().lower()
        if not raw or not token:
            return False
        return re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", raw.lower()) is not None

    @staticmethod
    def _force_submit_norm_text(value: Any) -> str:
        return str(value or "").strip().lower()

    @classmethod
    def _force_submit_submission_vessel_allowlist(cls) -> set[str]:
        return {
            "big pot",
            "small pot",
            "bowl",
            "plastic bowl",
            "plate",
            "large plate",
            "small plate",
            "deep plate",
            "square plate",
            "casserole",
            "food processor container",
            "paella pan",
            "大锅",
            "小锅",
            "碗",
            "塑料碗",
            "盘",
            "大盘",
            "小盘",
            "深盘",
            "方盘",
            "砂锅",
            "料理机容器",
            "食物处理机容器",
            "双耳锅",
        }

    @classmethod
    def _force_submit_transfer_source_allowlist(cls) -> set[str]:
        return {
            "bake tray",
            "baking tray",
            "pan",
            "grill pan",
            "烤盘",
            "煎锅",
            "烤纹锅",
        }

    @classmethod
    def _force_submit_direct_submit_food_allowlist(cls) -> set[str]:
        return {
            "tart",
            "挞",
        }

    @classmethod
    def _force_submit_item_matches_candidate(
        cls,
        item: dict[str, Any],
        *,
        candidate_name: str | None,
        candidate_instance_id: Any,
    ) -> bool:
        if not isinstance(item, dict):
            return False
        try:
            item_id = int(item.get("instance_id")) if item.get("instance_id") is not None and str(item.get("instance_id")).strip() else None
        except Exception:
            item_id = None
        try:
            want_id = int(candidate_instance_id) if candidate_instance_id is not None and str(candidate_instance_id).strip() else None
        except Exception:
            want_id = None
        if item_id is not None and want_id is not None:
            return item_id == want_id
        want = cls._normalize_force_submit_candidate_name(candidate_name).strip().lower()
        if not want:
            return False
        for key in ("name_en", "name_cn", "game_object", "name", "label"):
            cand = cls._normalize_force_submit_candidate_name(item.get(key)).strip().lower()
            if cand and (cand == want or cand in want or want in cand):
                return True
        return False

    @classmethod
    def _force_submit_find_item(
        cls,
        items: list[dict[str, Any]],
        *,
        candidate_name: str | None,
        candidate_instance_id: Any,
    ) -> Optional[dict[str, Any]]:
        for item in items:
            if cls._force_submit_item_matches_candidate(
                item,
                candidate_name=candidate_name,
                candidate_instance_id=candidate_instance_id,
            ):
                return item
        return None

    @classmethod
    def _force_submit_is_valid_candidate(
        cls,
        *,
        name: str | None,
        instance_id: Any,
        item: dict[str, Any] | None = None,
    ) -> bool:
        cand_name = cls._normalize_force_submit_candidate_name(name)
        if isinstance(item, dict):
            cand_name = cls._normalize_force_submit_candidate_name(item_display_name(item) or cand_name)
        if not cand_name:
            return False
        return cls._is_force_submit_probe_container(cand_name)

    @classmethod
    def _is_force_submit_bake_tray(cls, name: str | None) -> bool:
        if not name:
            return False
        raw = str(name).strip()
        if not raw:
            return False
        if cls._force_submit_contains_english_term(raw, "bake tray") or cls._force_submit_contains_english_term(raw, "baking tray"):
            return True
        return raw == "鐑ょ洏"

    @classmethod
    def _is_force_submit_direct_submit_food(cls, name: str | None) -> bool:
        if not name:
            return False
        raw = str(name).strip()
        if not raw:
            return False
        return cls._force_submit_norm_text(raw) in cls._force_submit_direct_submit_food_allowlist()

    @classmethod
    def _is_force_submit_big_pot(cls, name: str | None) -> bool:
        if not name:
            return False
        raw = str(name).strip()
        if not raw:
            return False
        return cls._force_submit_norm_text(raw) in {"big pot", "大锅"}

    @classmethod
    def _is_force_submit_food_processor_container(cls, name: str | None) -> bool:
        if not name:
            return False
        raw = str(name).strip()
        if not raw:
            return False
        return cls._force_submit_norm_text(raw) in {"food processor container", "料理机容器", "食物处理机容器"}

    @classmethod
    def _is_force_submit_ignored_probe_name(cls, name: str | None) -> bool:
        if not name:
            return False
        raw = str(name).strip()
        if not raw:
            return False
        low = raw.lower()
        return any(token in low for token in ("source", "spawn", "generator"))

    @classmethod
    def _is_force_submit_allowed_pan_vessel(cls, name: str | None) -> bool:
        if not name:
            return False
        raw = str(name).strip()
        if not raw:
            return False
        return cls._force_submit_norm_text(raw) in {"paella pan", "paella_pan", "双耳锅"}

    @classmethod
    def _is_force_submit_transfer_source_vessel(cls, name: str | None) -> bool:
        if not name:
            return False
        raw = str(name).strip()
        if not raw or cls._is_force_submit_allowed_pan_vessel(raw):
            return False
        return cls._force_submit_norm_text(raw) in cls._force_submit_transfer_source_allowlist()

    @classmethod
    def _is_force_submit_submission_vessel(cls, name: str | None) -> bool:
        if not name:
            return False
        raw = str(name).strip()
        if not raw:
            return False
        if cls._is_force_submit_ignored_probe_name(raw):
            return False
        if cls._is_force_submit_bake_tray(raw):
            return False
        if cls._is_force_submit_direct_submit_food(raw):
            return True
        return cls._force_submit_norm_text(raw) in cls._force_submit_submission_vessel_allowlist()

    @classmethod
    def _is_force_submit_probe_container(cls, name: str | None) -> bool:
        if cls._is_force_submit_ignored_probe_name(name):
            return False
        if cls._is_force_submit_direct_submit_food(name):
            return False
        return cls._is_force_submit_submission_vessel(name) or cls._is_force_submit_transfer_source_vessel(name)

    @classmethod
    def _normalize_force_submit_candidate_name(cls, name: str | None) -> str:
        raw = str(name or "").strip()
        if not raw:
            return ""
        if "," in raw:
            parts = [p.strip() for p in raw.split(",") if str(p).strip()]
            if parts:
                primary = parts[0]
                # Scene/perception names may include variant suffixes like "Bowl, Basic"
                # while navigation works better with the canonical vessel family name.
                if cls._is_force_submit_probe_container(primary):
                    return primary
        return raw

    @classmethod
    def _force_submit_probe_candidate_sort_key(cls, name: str | None, instance_id: Any) -> tuple[int, int, str]:
        raw = cls._normalize_force_submit_candidate_name(name)
        low = raw.lower()
        if cls._is_force_submit_big_pot(raw):
            priority = 0
        elif cls._is_force_submit_food_processor_container(raw):
            priority = 1
        elif cls._is_force_submit_transfer_source_vessel(raw):
            priority = 2
        elif cls._is_force_submit_submission_vessel(raw):
            priority = 3
        else:
            priority = 9
        has_id = 0 if instance_id is not None else 1
        return priority, has_id, low

    @classmethod
    def _force_submit_same_candidate(
        cls,
        name_a: str | None,
        id_a: Any,
        name_b: str | None,
        id_b: Any,
    ) -> bool:
        try:
            a_id = int(id_a) if id_a is not None and str(id_a).strip() else None
        except Exception:
            a_id = None
        try:
            b_id = int(id_b) if id_b is not None and str(id_b).strip() else None
        except Exception:
            b_id = None
        if a_id is not None and b_id is not None:
            return a_id == b_id
        a_name = cls._normalize_force_submit_candidate_name(name_a).strip().lower()
        b_name = cls._normalize_force_submit_candidate_name(name_b).strip().lower()
        return bool(a_name) and a_name == b_name

    def _is_force_submit_candidate_banned(self, *, state: dict, name: str | None, instance_id: Any) -> bool:
        banned = state.get("force_submit_banned_candidates")
        if not isinstance(banned, list):
            return False
        for entry in banned:
            if not isinstance(entry, dict):
                continue
            if self._force_submit_same_candidate(
                name,
                instance_id,
                entry.get("name"),
                entry.get("instance_id"),
            ):
                return True
        return False

    def _invalidate_force_submit_candidate(
        self,
        *,
        state: dict,
        bad_name: str | None,
        bad_id: Any,
        reason: str,
        failed_stage: str,
    ) -> dict[str, Any]:
        norm_name = self._normalize_force_submit_candidate_name(bad_name)
        banned = state.get("force_submit_banned_candidates")
        banned_entries: list[dict[str, Any]] = []
        if isinstance(banned, list):
            for entry in banned:
                if isinstance(entry, dict):
                    banned_entries.append(dict(entry))
        if norm_name and not self._is_force_submit_candidate_banned(state={"force_submit_banned_candidates": banned_entries}, name=norm_name, instance_id=bad_id):
            banned_entries.append(
                {
                    "name": norm_name,
                    "instance_id": bad_id,
                    "reason": str(reason or "").strip(),
                    "stage": str(failed_stage or "").strip(),
                    "step_id": int(self.step_id),
                }
            )

        def _filter_dict_entries(items: Any) -> list[dict[str, Any]]:
            kept: list[dict[str, Any]] = []
            if not isinstance(items, list):
                return kept
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                if self._force_submit_same_candidate(
                    norm_name,
                    bad_id,
                    entry.get("name"),
                    entry.get("instance_id"),
                ):
                    continue
                kept.append(dict(entry))
            return kept

        active_name = str(state.get("active_container") or "").strip()
        active_id = state.get("active_container_instance_id")
        clear_active = self._force_submit_same_candidate(norm_name, bad_id, active_name, active_id)

        updates: dict[str, Any] = {
            "force_submit_banned_candidates": banned_entries[-20:],
            "force_submit_stage": "",
            "force_submit_probe_candidates": _filter_dict_entries(state.get("force_submit_probe_candidates")),
            "force_submit_probe_results": _filter_dict_entries(state.get("force_submit_probe_results")),
            "force_submit_probe_best": None,
            "force_submit_probe_current_name": "",
            "force_submit_probe_current_instance_id": None,
            "force_submit_probe_last_probe": None,
            "force_submit_probe_index": 0,
            "force_submit_transfer_source_name": "",
            "force_submit_transfer_source_instance_id": None,
            "force_submit_transfer_dest_name": "",
            "force_submit_transfer_dest_instance_id": None,
            "force_submit_consecutive_failures": 0,
            "force_submit_last_error": "",
            "force_submit_last_failed_step_name": "",
        }
        if clear_active:
            updates["active_container"] = ""
            updates["active_container_instance_id"] = None
        self.memory.update_agent_state(**updates)
        self._log_force_submit(
            "invalidate_candidate",
            bad_name=norm_name,
            bad_id=bad_id,
            reason=reason,
            failed_stage=failed_stage,
            banned_count=len(banned_entries),
        )
        new_state = dict(state)
        new_state.update(updates)
        return new_state

    def _recover_force_submit_from_repeated_failure(self, *, state: dict) -> dict[str, Any]:
        try:
            failures = int(state.get("force_submit_consecutive_failures", 0) or 0)
        except Exception:
            failures = 0
        if failures < 2:
            return state

        stage = str(state.get("force_submit_stage") or "").strip().lower()
        last_step = str(state.get("force_submit_last_failed_step_name") or "").strip().lower()
        last_error = str(state.get("force_submit_last_error") or "").strip().lower()
        if not last_step or not last_error:
            return state

        if last_step == "gui_submit_dish_via_checkout_stand" and "held_item_not_submit_holdable" in last_error:
            bad_name = str(state.get("held_item") or state.get("active_container") or "").strip()
            bad_id = state.get("held_item_instance_id")
            if bad_id is None:
                bad_id = state.get("active_container_instance_id")
            return self._invalidate_force_submit_candidate(
                state=state,
                bad_name=bad_name,
                bad_id=bad_id,
                reason="submit_target_not_holdable",
                failed_stage=stage,
            )

        if last_step == "pick_up" and "not_holding_after_click" in last_error:
            if stage == "transfer_source_ready":
                bad_name = str(state.get("force_submit_transfer_source_name") or "").strip()
                bad_id = state.get("force_submit_transfer_source_instance_id")
            elif stage == "transfer_selected_ready":
                bad_name = str(state.get("force_submit_transfer_dest_name") or "").strip()
                bad_id = state.get("force_submit_transfer_dest_instance_id")
            else:
                bad_name = str(state.get("active_container") or state.get("force_submit_probe_current_name") or "").strip()
                bad_id = state.get("active_container_instance_id")
                if bad_id is None:
                    bad_id = state.get("force_submit_probe_current_instance_id")
            return self._invalidate_force_submit_candidate(
                state=state,
                bad_name=bad_name,
                bad_id=bad_id,
                reason="pickup_not_holding_after_click",
                failed_stage=stage,
            )

        if last_step == "auto_navigation" and "navigation_failed" in last_error:
            if stage == "probe_nav":
                bad_name = str(state.get("force_submit_probe_current_name") or "").strip()
                bad_id = state.get("force_submit_probe_current_instance_id")
            elif stage == "transfer_nav_source":
                bad_name = str(state.get("force_submit_transfer_source_name") or "").strip()
                bad_id = state.get("force_submit_transfer_source_instance_id")
            else:
                bad_name = str(state.get("active_container") or "").strip()
                bad_id = state.get("active_container_instance_id")
            return self._invalidate_force_submit_candidate(
                state=state,
                bad_name=bad_name,
                bad_id=bad_id,
                reason="navigation_failed_repeatedly",
                failed_stage=stage,
            )

        return state

    def _force_submit_recheck_submit_target_before_pickup(
        self,
        *,
        state: dict,
        candidate_name: str,
        candidate_id: Any,
        stage: str,
        reason_prefix: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        check_name = self._normalize_force_submit_candidate_name(candidate_name)
        try:
            check_id = int(candidate_id) if candidate_id is not None and str(candidate_id).strip() else None
        except Exception:
            check_id = None
        if not check_name:
            return state, None

        probe = self._read_force_submit_container_probe(expected_name=check_name)
        current_item: dict[str, Any] | None = None
        try:
            data = read_realtime_products(self.cfg.realtime_products_path)
            current_item = self._force_submit_find_item(
                extract_items(data),
                candidate_name=check_name,
                candidate_instance_id=check_id,
            )
        except Exception:
            current_item = None
        valid_candidate = self._force_submit_is_valid_candidate(
            name=check_name,
            instance_id=check_id,
            item=current_item,
        )
        recheck_entry = {
            "name": check_name,
            "instance_id": check_id,
            "success": bool(probe.get("success", False)),
            "assumed_nonempty": bool(probe.get("assumed_nonempty", False)),
            "item_count": int(probe.get("item_count", 0) or 0),
            "char_count": int(probe.get("char_count", 0) or 0),
            "contents": str(probe.get("contents") or ""),
            "error": str(probe.get("error") or ""),
            "matched_offset": str(probe.get("matched_offset") or ""),
            "valid_candidate": bool(valid_candidate),
            "probe_name_matches": bool(
                self._force_submit_probe_name_matches(
                    check_name,
                    self._normalize_force_submit_candidate_name(probe.get("container_name")),
                )
            ),
        }
        self._log_force_submit(
            f"{reason_prefix}_recheck_submit_target",
            stage=stage,
            candidate_name=check_name,
            candidate_id=check_id,
            valid_candidate=bool(valid_candidate),
            probe=recheck_entry,
        )
        explicit_pickup_signal = self._force_submit_probe_has_explicit_pickup_signal(probe)
        self._log_force_submit(
            f"{reason_prefix}_recheck_pickup_gate",
            stage=stage,
            candidate_name=check_name,
            candidate_id=check_id,
            explicit_pickup_signal=bool(explicit_pickup_signal),
            container_name=str(probe.get("container_name") or ""),
            item_name=str(probe.get("item_name") or ""),
            contents=str(probe.get("contents") or ""),
            item_count=int(probe.get("item_count", 0) or 0),
            error=str(probe.get("error") or ""),
        )
        if self._force_submit_probe_entry_submitworthy_nonempty(recheck_entry) and explicit_pickup_signal:
            return state, probe

        self._log_force_submit(
            f"{reason_prefix}_recheck_pickup_gate_invalidate",
            stage=stage,
            candidate_name=check_name,
            candidate_id=check_id,
            probe=recheck_entry,
            explicit_pickup_signal=bool(explicit_pickup_signal),
        )
        new_state = self._invalidate_force_submit_candidate(
            state=state,
            bad_name=check_name,
            bad_id=check_id,
            reason=(
                f"{reason_prefix}_missing_explicit_alt_j_signal_before_pickup"
                if self._force_submit_probe_entry_submitworthy_nonempty(recheck_entry) and not explicit_pickup_signal
                else f"{reason_prefix}_empty_before_pickup"
            ),
            failed_stage=stage,
        )
        return new_state, None

    def _filter_skill_specs(self, *, allowed_skills: list[str]) -> str:
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

    def _apply_force_submit_bundle_overrides(
        self,
        *,
        bundle: dict,
        allowed_actions: list[str],
        allowed_skills: list[str],
    ) -> None:
        # Restrict action specs to the limited set.
        bundle["action_specs"] = action_specs_to_prompt_text(action_names=allowed_actions).strip()
        # Reduce skill specs to allowed skills only.
        bundle["skill_specs"] = self._filter_skill_specs(allowed_skills=allowed_skills)
        # Clear skill cards to avoid unrelated action groups.
        bundle["skills_catalog"] = ""
        # Restrict tools manifest for tool-calling planners.
        try:
            manifest = build_tool_manifest(allowed_actions=allowed_actions, allowed_skills=allowed_skills)
            bundle["tools_manifest_openai"] = json.dumps(to_openai_tools(manifest), ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _apply_force_submit_planner_overrides(self, *, allowed_actions: list[str], allowed_skills: list[str]) -> None:
        """
        For tool-calling VLM planners, rebuild the tool list to the restricted allowlists.
        """
        try:
            planner = self.planner
            if planner is None:
                return
            if not hasattr(planner, "_openai_tools"):
                return
            if not bool(getattr(getattr(self.cfg, "vlm", None), "use_tools", False)):
                return
            manifest = build_tool_manifest(allowed_actions=allowed_actions, allowed_skills=allowed_skills)
            setattr(planner, "_openai_tools", to_openai_tools(manifest))
        except Exception:
            pass

    def _force_submit_decide_step(self, *, state: dict, high_level_id: str) -> PlanStep:
        """
        Decide the next step under force-submit mode.
        """
        try:
            latest_state = self._read_agent_state_snapshot()
            if isinstance(latest_state, dict) and latest_state:
                merged_state = dict(state or {})
                merged_state.update(latest_state)
                state = merged_state
        except Exception:
            pass

        state = self._recover_force_submit_from_repeated_failure(state=state)
        stage = str(state.get("force_submit_stage") or "").strip().lower()
        should_exit_mode, blocked_mode = self._force_submit_mode_needs_exit(state=state)
        if should_exit_mode:
            self._log_force_submit(
                "decide_exit_mode",
                stage=stage,
                mode=blocked_mode,
                held_item=str(state.get("held_item") or ""),
            )
            return PlanStep(
                step_id=f"{high_level_id}.FORCE.EXIT_MODE",
                type="action",
                name="exit_current_interaction_mode",
                args={},
                expectation=f"Right-click to exit the current interaction mode before force-submit continues ({blocked_mode or 'interaction_mode'}).",
            )

        is_held = bool(state.get("is_held", False))
        held_item = str(state.get("held_item") or "")
        held_item_kind = str(state.get("item_kind") or "").strip().lower()
        is_bake_tray = self._is_force_submit_bake_tray(held_item)
        is_direct_submit_food = self._is_force_submit_direct_submit_food(held_item)
        is_container = self._force_submit_is_valid_candidate(name=held_item, instance_id=state.get("held_item_instance_id"))
        active_name, active_id = self._resolve_force_submit_active_container(state=state)
        probe_candidates = state.get("force_submit_probe_candidates")
        if not isinstance(probe_candidates, list):
            probe_candidates = []
        try:
            probe_index = int(state.get("force_submit_probe_index", 0) or 0)
        except Exception:
            probe_index = 0
        probe_best = state.get("force_submit_probe_best") if isinstance(state.get("force_submit_probe_best"), dict) else None
        transfer_source_name = str(state.get("force_submit_transfer_source_name") or "").strip()
        transfer_source_id = state.get("force_submit_transfer_source_instance_id")
        try:
            transfer_source_id = int(transfer_source_id) if transfer_source_id is not None and str(transfer_source_id).strip() else None
        except Exception:
            transfer_source_id = None
        transfer_dest_name = str(state.get("force_submit_transfer_dest_name") or active_name).strip() or active_name
        transfer_dest_id = state.get("force_submit_transfer_dest_instance_id", active_id)
        try:
            transfer_dest_id = int(transfer_dest_id) if transfer_dest_id is not None and str(transfer_dest_id).strip() else None
        except Exception:
            transfer_dest_id = None

        if is_held and (not is_container) and stage in {"selected_picked", "transfer_selected_picked", "transfer_source_picked"}:
            if stage == "transfer_source_picked":
                bad_name = transfer_source_name or held_item
                bad_id = transfer_source_id if transfer_source_id is not None else state.get("held_item_instance_id")
                reason = "picked_non_container_transfer_source"
            elif stage == "transfer_selected_picked":
                bad_name = transfer_dest_name or held_item
                bad_id = transfer_dest_id if transfer_dest_id is not None else state.get("held_item_instance_id")
                reason = "picked_non_container_transfer_dest"
            else:
                bad_name = active_name or held_item
                bad_id = active_id if active_id is not None else state.get("held_item_instance_id")
                reason = "picked_non_container_submit_candidate"
            self._log_force_submit(
                "picked_non_container_after_pickup",
                stage=stage,
                held_item=held_item,
                held_item_kind=held_item_kind,
                bad_name=bad_name,
                bad_id=bad_id,
                reason=reason,
            )
            state = self._invalidate_force_submit_candidate(
                state=state,
                bad_name=bad_name,
                bad_id=bad_id,
                reason=reason,
                failed_stage=stage,
            )
            stage = str(state.get("force_submit_stage") or "").strip().lower()
            is_held = bool(state.get("is_held", False))
            held_item = str(state.get("held_item") or "")
            held_item_kind = str(state.get("item_kind") or "").strip().lower()
            is_container = self._force_submit_is_valid_candidate(name=held_item, instance_id=state.get("held_item_instance_id"))
            active_name, active_id = self._resolve_force_submit_active_container(state=state)

        if is_held and is_direct_submit_food:
            self._log_force_submit("decide_submit_direct_food", stage=stage, held_item=held_item, dish_name=str(self.dish.dish_name))
            return PlanStep(
                step_id=f"{high_level_id}.FORCE.SUBMIT",
                type="skill",
                name="gui_submit_dish_via_checkout_stand",
                args={"dish_name": str(self.dish.dish_name)},
            )

        if stage == "drop_shelf_put_done":
            self._log_force_submit("decide_stand_after_drop", stage=stage, held_item=held_item)
            return PlanStep(step_id=f"{high_level_id}.FORCE.STAND", type="action", name="stand_up", args={})

        if stage == "transfer_nav_source":
            if not transfer_source_name:
                raise RuntimeError("fatal_episode_error:force_submit_transfer_source_missing")
            self._log_force_submit("decide_transfer_nav_source", stage=stage, source_name=transfer_source_name, source_id=transfer_source_id)
            nav_args: dict[str, Any] = {"target": transfer_source_name}
            if transfer_source_id is not None:
                nav_args["target_instance_id"] = str(transfer_source_id)
            return PlanStep(step_id=f"{high_level_id}.FORCE.TRANSFER_NAV_SOURCE", type="skill", name="auto_navigation", args=nav_args)

        if stage == "transfer_source_ready":
            align_info = self._force_submit_align_cursor_to_container(expected_name=transfer_source_name)
            self._log_force_submit(
                "decide_transfer_pick_source_alignment",
                stage=stage,
                source_name=transfer_source_name,
                source_id=transfer_source_id,
                aligned=bool(align_info.get("aligned", False)),
                chosen_offset=str(align_info.get("chosen_offset") or "center"),
                chosen_dx=int(align_info.get("chosen_dx", 0) or 0),
                chosen_dy=int(align_info.get("chosen_dy", 0) or 0),
                error=str(align_info.get("error") or ""),
            )
            self._log_force_submit("decide_transfer_pick_source", stage=stage, source_name=transfer_source_name, source_id=transfer_source_id)
            return PlanStep(step_id=f"{high_level_id}.FORCE.TRANSFER_PICK_SOURCE", type="action", name="pick_up", args={})

        if stage == "transfer_nav_dest":
            if not transfer_dest_name:
                raise RuntimeError("fatal_episode_error:force_submit_transfer_dest_missing")
            self._log_force_submit("decide_transfer_nav_dest", stage=stage, dest_name=transfer_dest_name, dest_id=transfer_dest_id)
            nav_args = {"target": transfer_dest_name}
            if transfer_dest_id is not None:
                nav_args["target_instance_id"] = str(transfer_dest_id)
            return PlanStep(step_id=f"{high_level_id}.FORCE.TRANSFER_NAV_DEST", type="skill", name="auto_navigation", args=nav_args)

        if stage == "transfer_dest_ready":
            self._log_force_submit("decide_transfer_enter_pour_mode", stage=stage, dest_name=transfer_dest_name, dest_id=transfer_dest_id)
            return PlanStep(step_id=f"{high_level_id}.FORCE.TRANSFER_ENTER_POUR", type="action", name="enter_pouring_mode", args={})

        if stage == "transfer_pour_ready":
            if not transfer_dest_name:
                raise RuntimeError("fatal_episode_error:force_submit_transfer_dest_missing")
            self._log_force_submit("decide_transfer_auto_pour", stage=stage, dest_name=transfer_dest_name, dest_id=transfer_dest_id)
            pour_args: dict[str, Any] = {"container_name": transfer_dest_name, "target_ml": 50.0}
            if transfer_dest_id is not None:
                pour_args["container_instance_id"] = str(transfer_dest_id)
            return PlanStep(step_id=f"{high_level_id}.FORCE.TRANSFER_AUTO_POUR", type="skill", name="auto_pour", args=pour_args)

        if stage == "transfer_drop_shelf_put_done":
            self._log_force_submit("decide_transfer_stand_after_drop", stage=stage)
            return PlanStep(step_id=f"{high_level_id}.FORCE.TRANSFER_STAND", type="action", name="stand_up", args={})

        if stage == "transfer_nav_selected":
            if not transfer_dest_name:
                raise RuntimeError("fatal_episode_error:force_submit_transfer_dest_missing")
            self._log_force_submit("decide_transfer_nav_selected", stage=stage, dest_name=transfer_dest_name, dest_id=transfer_dest_id)
            nav_args = {"target": transfer_dest_name}
            if transfer_dest_id is not None:
                nav_args["target_instance_id"] = str(transfer_dest_id)
            return PlanStep(step_id=f"{high_level_id}.FORCE.TRANSFER_NAV_SELECTED", type="skill", name="auto_navigation", args=nav_args)

        if stage == "transfer_selected_ready":
            state, recheck_probe = self._force_submit_recheck_submit_target_before_pickup(
                state=state,
                candidate_name=transfer_dest_name,
                candidate_id=transfer_dest_id,
                stage=stage,
                reason_prefix="transfer_selected",
            )
            if recheck_probe is None:
                return self._force_submit_decide_step(state=state, high_level_id=high_level_id)
            align_info = self._force_submit_align_cursor_to_container(expected_name=transfer_dest_name)
            self._log_force_submit(
                "decide_transfer_pick_selected_alignment",
                stage=stage,
                dest_name=transfer_dest_name,
                dest_id=transfer_dest_id,
                aligned=bool(align_info.get("aligned", False)),
                chosen_offset=str(align_info.get("chosen_offset") or "center"),
                chosen_dx=int(align_info.get("chosen_dx", 0) or 0),
                chosen_dy=int(align_info.get("chosen_dy", 0) or 0),
                error=str(align_info.get("error") or ""),
            )
            self._log_force_submit("decide_transfer_pick_selected", stage=stage, dest_name=transfer_dest_name, dest_id=transfer_dest_id)
            return PlanStep(step_id=f"{high_level_id}.FORCE.TRANSFER_PICK_SELECTED", type="action", name="pick_up", args={})

        if is_held and is_container:
            if stage == "selected_picked":
                self._log_force_submit("decide_submit_selected_container", stage=stage, held_item=held_item)
                return PlanStep(
                    step_id=f"{high_level_id}.FORCE.SUBMIT",
                    type="skill",
                    name="gui_submit_dish_via_checkout_stand",
                    args={"dish_name": str(self.dish.dish_name)},
                )
            if stage == "transfer_selected_picked":
                self._log_force_submit("decide_submit_transfer_dest_container", stage=stage, held_item=held_item, dest_name=transfer_dest_name)
                return PlanStep(
                    step_id=f"{high_level_id}.FORCE.SUBMIT",
                    type="skill",
                    name="gui_submit_dish_via_checkout_stand",
                    args={"dish_name": str(self.dish.dish_name)},
                )
            if stage == "transfer_source_picked":
                self.memory.update_agent_state(force_submit_stage="transfer_nav_dest")
                self._log_force_submit("decide_transfer_after_pick_nav_dest", stage=stage, held_item=held_item, dest_name=transfer_dest_name)
                nav_args = {"target": transfer_dest_name}
                if transfer_dest_id is not None:
                    nav_args["target_instance_id"] = str(transfer_dest_id)
                return PlanStep(step_id=f"{high_level_id}.FORCE.TRANSFER_NAV_DEST", type="skill", name="auto_navigation", args=nav_args)
            if stage == "transfer_at_drop_shelf":
                self._log_force_submit("decide_transfer_kneel_for_drop", stage=stage, held_item=held_item)
                return PlanStep(step_id=f"{high_level_id}.FORCE.TRANSFER_KNEEL", type="action", name="kneel_down", args={})
            if stage == "transfer_drop_shelf_crouched":
                self._log_force_submit("decide_transfer_put_down_source", stage=stage, held_item=held_item)
                return PlanStep(step_id=f"{high_level_id}.FORCE.TRANSFER_PUT_DOWN", type="action", name="put_down", args={})
            if stage == "transfer_nav_drop_shelf":
                self._log_force_submit("decide_transfer_nav_drop_shelf", stage=stage, held_item=held_item, target=self._force_submit_drop_shelf_target())
                return PlanStep(
                    step_id=f"{high_level_id}.FORCE.TRANSFER_NAV_DROP_SHELF",
                    type="skill",
                    name="auto_navigation",
                    args={"target": self._force_submit_drop_shelf_target()},
                )
            if stage == "at_drop_shelf":
                self._log_force_submit("decide_kneel_for_drop", stage=stage, held_item=held_item)
                return PlanStep(step_id=f"{high_level_id}.FORCE.KNEEL", type="action", name="kneel_down", args={})
            if stage == "drop_shelf_crouched":
                self._log_force_submit("decide_put_down_held_container", stage=stage, held_item=held_item)
                return PlanStep(step_id=f"{high_level_id}.FORCE.PUT_DOWN", type="action", name="put_down", args={})
            self.memory.update_agent_state(force_submit_stage="nav_drop_shelf")
            self._log_force_submit(
                "decide_nav_drop_shelf",
                stage=stage,
                held_item=held_item,
                target=self._force_submit_drop_shelf_target(),
            )
            return PlanStep(
                step_id=f"{high_level_id}.FORCE.NAV_DROP_SHELF",
                type="skill",
                name="auto_navigation",
                args={"target": self._force_submit_drop_shelf_target()},
            )

        if is_held and (not is_container):
            self._log_force_submit("decide_throw_non_container", stage=stage, held_item=held_item)
            return PlanStep(step_id=f"{high_level_id}.FORCE.THROW_AWAY", type="action", name="throw_away", args={})

        if stage == "selected_ready":
            state, recheck_probe = self._force_submit_recheck_submit_target_before_pickup(
                state=state,
                candidate_name=active_name,
                candidate_id=active_id,
                stage=stage,
                reason_prefix="selected",
            )
            if recheck_probe is None:
                return self._force_submit_decide_step(state=state, high_level_id=high_level_id)
            align_info = self._force_submit_align_cursor_to_container(expected_name=active_name)
            self._log_force_submit(
                "decide_pick_selected_container_alignment",
                stage=stage,
                active_name=active_name,
                active_id=active_id,
                aligned=bool(align_info.get("aligned", False)),
                chosen_offset=str(align_info.get("chosen_offset") or "center"),
                chosen_dx=int(align_info.get("chosen_dx", 0) or 0),
                chosen_dy=int(align_info.get("chosen_dy", 0) or 0),
                error=str(align_info.get("error") or ""),
            )
            self._log_force_submit("decide_pick_selected_container", stage=stage, active_name=active_name, active_id=active_id)
            return PlanStep(step_id=f"{high_level_id}.FORCE.PICK_UP", type="action", name="pick_up", args={})

        if stage == "probe_read":
            current_name = self._normalize_force_submit_candidate_name(
                state.get("force_submit_probe_current_name") or active_name
            )
            current_id = state.get("force_submit_probe_current_instance_id")
            try:
                current_id = int(current_id) if current_id is not None and str(current_id).strip() else None
            except Exception:
                current_id = None
            probe = self._read_force_submit_container_probe(expected_name=current_name)
            current_item: dict[str, Any] | None = None
            try:
                data = read_realtime_products(self.cfg.realtime_products_path)
                current_item = self._force_submit_find_item(
                    extract_items(data),
                    candidate_name=current_name,
                    candidate_instance_id=current_id,
                )
            except Exception:
                current_item = None
            valid_candidate = self._force_submit_is_valid_candidate(
                name=current_name,
                instance_id=current_id,
                item=current_item,
            )
            probe_container_name = self._normalize_force_submit_candidate_name(probe.get("container_name"))
            probe_name_matches = bool(probe_container_name) and self._force_submit_probe_name_matches(current_name, probe_container_name)
            safe_assumed_nonempty = (
                valid_candidate
                and bool(probe.get("assumed_nonempty", False))
                and (
                    bool(probe.get("success", False))
                    or probe_name_matches
                    or bool(probe.get("inferred_from_alt_j_item_signal", False))
                    or bool(probe.get("inferred_from_alt_j_food_signal", False))
                    or self._force_submit_probe_error_suggests_nonempty(probe)
                )
            )
            probe_results = state.get("force_submit_probe_results")
            if not isinstance(probe_results, list):
                probe_results = []
            merged_probe_results: list[dict[str, Any]] = []
            for entry in probe_results:
                if not isinstance(entry, dict):
                    continue
                same_name = str(entry.get("name") or "").strip().lower() == current_name.lower()
                same_id = entry.get("instance_id") == current_id
                if same_name and same_id:
                    continue
                merged_probe_results.append(dict(entry))
            merged_probe_results.append({
                "name": current_name,
                "instance_id": current_id,
                "success": bool(probe.get("success", False)),
                "assumed_nonempty": bool(safe_assumed_nonempty),
                "occlusion_suspected": bool(probe.get("occlusion_suspected", False)),
                "error": str(probe.get("error") or ""),
                "container_name": str(probe.get("container_name") or ""),
                "contents": str(probe.get("contents") or ""),
                "item_count": int(probe.get("item_count", 0) or 0),
                "char_count": int(probe.get("char_count", 0) or 0),
                "matched_offset": str(probe.get("matched_offset") or ""),
                "probe_offsets_attempted": probe.get("probe_offsets_attempted") if isinstance(probe.get("probe_offsets_attempted"), list) else [],
                "valid_candidate": bool(valid_candidate),
                "probe_name_matches": bool(probe_name_matches),
            })
            updates: dict[str, Any] = {
                "force_submit_probe_last_probe": probe,
                "force_submit_probe_results": merged_probe_results,
            }
            current_probe_entry = {
                "name": current_name,
                "instance_id": current_id,
                "success": bool(probe.get("success", False)),
                "assumed_nonempty": bool(safe_assumed_nonempty),
                "item_count": int(probe.get("item_count", 0) or 0),
                "char_count": int(probe.get("char_count", 0) or 0),
                "contents": str(probe.get("contents") or ""),
                "error": str(probe.get("error") or ""),
                "matched_offset": str(probe.get("matched_offset") or ""),
                "valid_candidate": bool(valid_candidate),
                "probe_name_matches": bool(probe_name_matches),
            }
            if valid_candidate and self._is_force_submit_submission_vessel(current_name):
                current_metric = self._force_submit_probe_result_sort_key(current_probe_entry)
                prev_metric = self._force_submit_probe_result_sort_key(probe_best)
                if current_metric > prev_metric:
                    updates["force_submit_probe_best"] = dict(current_probe_entry)
                    probe_best = updates["force_submit_probe_best"]
            probe_index = int(probe_index) + 1
            if probe_index < len(probe_candidates):
                nxt = probe_candidates[probe_index] if isinstance(probe_candidates[probe_index], dict) else {}
                nxt_name = self._normalize_force_submit_candidate_name(nxt.get("name"))
                nxt_id = nxt.get("instance_id")
                try:
                    nxt_id = int(nxt_id) if nxt_id is not None and str(nxt_id).strip() else None
                except Exception:
                    nxt_id = None
                updates.update({
                    "force_submit_stage": "probe_nav",
                    "force_submit_probe_index": int(probe_index),
                    "force_submit_probe_current_name": nxt_name,
                    "force_submit_probe_current_instance_id": nxt_id,
                })
                self.memory.update_agent_state(**updates)
                self._log_force_submit(
                    "decide_probe_next_candidate",
                    stage=stage,
                    current_name=current_name,
                    current_id=current_id,
                    next_name=nxt_name,
                    next_id=nxt_id,
                    probe_index=int(probe_index),
                    probe_best=probe_best,
                )
                nav_args: dict[str, Any] = {"target": nxt_name}
                if nxt_id is not None:
                    nav_args["target_instance_id"] = str(nxt_id)
                return PlanStep(step_id=f"{high_level_id}.FORCE.PROBE_NAV_{probe_index}", type="skill", name="auto_navigation", args=nav_args)
            big_pot_probe: dict[str, Any] | None = None
            best_transfer_source: dict[str, Any] | None = None
            best_transfer_metric = (-1, -1)
            best_confirmed_submission: dict[str, Any] | None = None
            best_confirmed_submission_metric = (-1, -1, -1, -1)
            for entry in merged_probe_results:
                if not isinstance(entry, dict):
                    continue
                entry_name = str(entry.get("name") or "").strip()
                if self._is_force_submit_big_pot(entry_name):
                    if bool(entry.get("success", False)):
                        big_pot_probe = dict(entry)
                if (
                    bool(entry.get("success", False))
                    and int(entry.get("item_count", 0) or 0) > 0
                    and self._is_force_submit_transfer_source_vessel(entry_name)
                ):
                    metric = (
                        int(entry.get("item_count", 0) or 0),
                        int(entry.get("char_count", 0) or 0),
                    )
                    if metric > best_transfer_metric:
                        best_transfer_metric = metric
                        best_transfer_source = dict(entry)
                if (
                    self._force_submit_probe_entry_submitworthy_nonempty(entry)
                    and self._is_force_submit_submission_vessel(entry_name)
                ):
                    metric = self._force_submit_probe_result_sort_key(entry)
                    if metric > best_confirmed_submission_metric:
                        best_confirmed_submission_metric = metric
                        best_confirmed_submission = dict(entry)
            big_pot_empty = False
            if isinstance(big_pot_probe, dict):
                big_pot_empty = bool(big_pot_probe.get("success", False)) and int(big_pot_probe.get("item_count", 0) or 0) <= 0
            scene_big_pot = self._find_force_submit_scene_container("Big Pot")
            if (not isinstance(big_pot_probe, dict)) and scene_big_pot is not None:
                big_pot_probe = {
                    "name": scene_big_pot[0],
                    "instance_id": scene_big_pot[1],
                    "success": True,
                    "item_count": 0,
                    "char_count": 0,
                    "contents": "",
                    "assumed_nonempty": False,
                    "error": "scene_big_pot_fallback",
                }
                big_pot_empty = True
            if isinstance(best_confirmed_submission, dict) and str(best_confirmed_submission.get("name") or "").strip():
                best_name = self._normalize_force_submit_candidate_name(best_confirmed_submission.get("name"))
                best_id = best_confirmed_submission.get("instance_id")
                try:
                    best_id = int(best_id) if best_id is not None and str(best_id).strip() else None
                except Exception:
                    best_id = None
                updates.update({
                    "force_submit_stage": "nav_selected",
                    "active_container": best_name,
                    "active_container_instance_id": best_id,
                    "force_submit_probe_best": dict(best_confirmed_submission),
                })
                self.memory.update_agent_state(**updates)
                self._log_force_submit(
                    "decide_probe_selected_confirmed_nonempty",
                    stage=stage,
                    best_name=best_name,
                    best_id=best_id,
                    best_confirmed_submission=best_confirmed_submission,
                )
                nav_args: dict[str, Any] = {"target": best_name}
                if best_id is not None:
                    nav_args["target_instance_id"] = str(best_id)
                return PlanStep(step_id=f"{high_level_id}.FORCE.NAV_SELECTED", type="skill", name="auto_navigation", args=nav_args)
            if big_pot_empty and isinstance(best_transfer_source, dict):
                dest_name = self._normalize_force_submit_candidate_name(
                    big_pot_probe.get("name") or active_name or "Big Pot"
                ) or "Big Pot"
                dest_id = big_pot_probe.get("instance_id", active_id)
                try:
                    dest_id = int(dest_id) if dest_id is not None and str(dest_id).strip() else None
                except Exception:
                    dest_id = None
                source_name = self._normalize_force_submit_candidate_name(best_transfer_source.get("name"))
                source_id = best_transfer_source.get("instance_id")
                try:
                    source_id = int(source_id) if source_id is not None and str(source_id).strip() else None
                except Exception:
                    source_id = None
                updates.update({
                    "force_submit_stage": "transfer_nav_source",
                    "force_submit_transfer_source_name": source_name,
                    "force_submit_transfer_source_instance_id": source_id,
                    "force_submit_transfer_dest_name": dest_name,
                    "force_submit_transfer_dest_instance_id": dest_id,
                    "active_container": dest_name,
                    "active_container_instance_id": dest_id,
                })
                self.memory.update_agent_state(**updates)
                self._log_force_submit(
                    "decide_probe_transfer_to_big_pot",
                    stage=stage,
                    source_name=source_name,
                    source_id=source_id,
                    dest_name=dest_name,
                    dest_id=dest_id,
                    big_pot_probe=big_pot_probe,
                    transfer_source=best_transfer_source,
                )
                nav_args = {"target": source_name}
                if source_id is not None:
                    nav_args["target_instance_id"] = str(source_id)
                return PlanStep(step_id=f"{high_level_id}.FORCE.TRANSFER_NAV_SOURCE", type="skill", name="auto_navigation", args=nav_args)
            if isinstance(probe_best, dict) and str(probe_best.get("name") or "").strip():
                best_name = self._normalize_force_submit_candidate_name(probe_best.get("name"))
                best_id = probe_best.get("instance_id")
                try:
                    best_id = int(best_id) if best_id is not None and str(best_id).strip() else None
                except Exception:
                    best_id = None
                updates.update({
                    "force_submit_stage": "nav_selected",
                    "active_container": best_name,
                    "active_container_instance_id": best_id,
                })
                self.memory.update_agent_state(**updates)
                self._log_force_submit(
                    "decide_probe_selected_best",
                    stage=stage,
                    best_name=best_name,
                    best_id=best_id,
                    probe_best=probe_best,
                )
                nav_args: dict[str, Any] = {"target": best_name}
                if best_id is not None:
                    nav_args["target_instance_id"] = str(best_id)
                return PlanStep(step_id=f"{high_level_id}.FORCE.NAV_SELECTED", type="skill", name="auto_navigation", args=nav_args)
            self._log_force_submit(
                "decide_probe_failed_no_nonempty",
                stage=stage,
                probe_candidates=probe_candidates,
                probe_last=probe,
                checked_containers=self._summarize_force_submit_checked_containers(merged_probe_results),
            )
            checked_summary = self._summarize_force_submit_checked_containers(merged_probe_results)
            checked_suffix = f" checked_containers=[{checked_summary}]" if checked_summary else ""
            raise RuntimeError(
                "fatal_episode_error:manual_intervention_required:"
                "force_submit failed: all candidate containers are empty, manual inspection required."
                + checked_suffix
            )

        if stage == "probe_nav":
            current_name = self._normalize_force_submit_candidate_name(
                state.get("force_submit_probe_current_name")
            )
            current_id = state.get("force_submit_probe_current_instance_id")
            try:
                current_id = int(current_id) if current_id is not None and str(current_id).strip() else None
            except Exception:
                current_id = None
            if not current_name:
                raise RuntimeError("fatal_episode_error:force_submit_probe_candidate_missing")
            self._log_force_submit(
                "decide_probe_nav",
                stage=stage,
                current_name=current_name,
                current_id=current_id,
            )
            nav_args: dict[str, Any] = {"target": current_name}
            if current_id is not None:
                nav_args["target_instance_id"] = str(current_id)
            return PlanStep(step_id=f"{high_level_id}.FORCE.PROBE_NAV", type="skill", name="auto_navigation", args=nav_args)

        if stage == "nav_selected":
            best = probe_best or {}
            best_name = self._normalize_force_submit_candidate_name(best.get("name") or active_name)
            best_id = best.get("instance_id", active_id)
            try:
                best_id = int(best_id) if best_id is not None and str(best_id).strip() else None
            except Exception:
                best_id = None
            self._log_force_submit(
                "decide_nav_selected",
                stage=stage,
                best_name=best_name,
                best_id=best_id,
                probe_best=probe_best,
            )
            nav_args: dict[str, Any] = {"target": best_name}
            if best_id is not None:
                nav_args["target_instance_id"] = str(best_id)
            return PlanStep(step_id=f"{high_level_id}.FORCE.NAV_SELECTED", type="skill", name="auto_navigation", args=nav_args)

        candidates = self._force_submit_probe_candidates(state=state, active_name=active_name, active_id=active_id)
        if not candidates:
            self._log_force_submit(
                "decide_no_probe_candidates",
                stage=stage,
                active_name=active_name,
                active_id=active_id,
            )
            raise RuntimeError("fatal_episode_error:force_submit_no_container_candidates")
        first = candidates[0]
        first_name = str(first.get("name") or "").strip()
        first_id = first.get("instance_id")
        try:
            first_id = int(first_id) if first_id is not None and str(first_id).strip() else None
        except Exception:
            first_id = None
        self.memory.update_agent_state(
            force_submit_stage="probe_nav",
            force_submit_probe_candidates=candidates,
            force_submit_probe_index=0,
            force_submit_probe_current_name=first_name,
            force_submit_probe_current_instance_id=first_id,
            force_submit_probe_best=None,
            force_submit_probe_results=[],
            force_submit_probe_last_probe=None,
            force_submit_transfer_source_name="",
            force_submit_transfer_source_instance_id=None,
            force_submit_transfer_dest_name="",
            force_submit_transfer_dest_instance_id=None,
        )
        self._log_force_submit(
            "decide_start_probe",
            stage=stage,
            active_name=active_name,
            active_id=active_id,
            first_name=first_name,
            first_id=first_id,
            candidates=candidates,
        )
        nav_args: dict[str, Any] = {"target": first_name}
        if first_id is not None:
            nav_args["target_instance_id"] = str(first_id)
        return PlanStep(step_id=f"{high_level_id}.FORCE.PROBE_NAV_0", type="skill", name="auto_navigation", args=nav_args)

    def _update_force_submit_stage_after_step(self, *, step: PlanStep, final_success: bool) -> None:
        if not final_success:
            return
        try:
            state = self.memory.read_json("agent_state")
        except Exception:
            state = {}
        if not bool(state.get("force_submit_active", False)):
            return
        stage = str(state.get("force_submit_stage") or "").strip().lower()

        def _set_stage(new_stage: str, **extra: Any) -> None:
            payload = {"force_submit_stage": new_stage}
            payload.update(extra)
            self.memory.update_agent_state(**payload)
            self._log_force_submit(
                "stage_transition",
                step=str(step.name or ""),
                from_stage=stage,
                to_stage=new_stage,
                extra=extra or None,
            )

        if step.name == "auto_navigation":
            target = str(step.args.get("target") or "").strip().lower()
            if stage == "nav_drop_shelf":
                _set_stage("at_drop_shelf")
            elif stage == "transfer_nav_source":
                _set_stage("transfer_source_ready")
            elif stage == "transfer_nav_dest":
                _set_stage("transfer_dest_ready")
            elif stage == "transfer_nav_drop_shelf":
                _set_stage("transfer_at_drop_shelf")
            elif stage == "transfer_nav_selected":
                _set_stage("transfer_selected_ready")
            elif stage == "probe_nav":
                _set_stage("probe_read")
            elif stage == "nav_selected":
                _set_stage("selected_ready")
            elif "side table" in target:
                _set_stage("at_place")
            else:
                _set_stage("at_container")
        elif step.name == "enter_pouring_mode":
            if stage == "transfer_dest_ready":
                _set_stage("transfer_pour_ready")
        elif step.name == "kneel_down":
            if stage == "at_drop_shelf":
                _set_stage("drop_shelf_crouched")
            elif stage == "transfer_at_drop_shelf":
                _set_stage("transfer_drop_shelf_crouched")
        elif step.name == "put_down":
            if stage == "drop_shelf_crouched":
                _set_stage("drop_shelf_put_done")
            elif stage == "transfer_drop_shelf_crouched":
                _set_stage("transfer_drop_shelf_put_done")
            else:
                used_raw = state.get("force_submit_used_place_indices", [])
                used = []
                if isinstance(used_raw, list):
                    used = [int(x) for x in used_raw if isinstance(x, (int, str))]
                place_index = state.get("force_submit_place_index")
                if isinstance(place_index, int):
                    if place_index not in used:
                        used.append(place_index)
                _set_stage("", force_submit_place_index=None, force_submit_used_place_indices=used)
        elif step.name == "stand_up":
            if stage == "drop_shelf_put_done":
                _set_stage("")
            elif stage == "transfer_drop_shelf_put_done":
                _set_stage("transfer_nav_selected")
        elif step.name == "pick_up":
            if stage == "selected_ready":
                _set_stage("selected_picked")
            elif stage == "transfer_source_ready":
                _set_stage("transfer_source_picked")
            elif stage == "transfer_selected_ready":
                _set_stage("transfer_selected_picked")
            else:
                _set_stage("")
        elif step.name == "auto_pour":
            if stage == "transfer_pour_ready":
                _set_stage("transfer_nav_drop_shelf")
        elif step.name == "throw_away":
            _set_stage("", force_submit_place_index=None)
        elif step.name == "gui_submit_dish_via_checkout_stand":
            _set_stage(
                "submitted",
                force_submit_probe_candidates=[],
                force_submit_probe_best=None,
                force_submit_probe_results=[],
                force_submit_probe_current_name="",
                force_submit_probe_current_instance_id=None,
                force_submit_probe_last_probe=None,
                force_submit_transfer_source_name="",
                force_submit_transfer_source_instance_id=None,
                force_submit_transfer_dest_name="",
                force_submit_transfer_dest_instance_id=None,
            )

    def _build_instance_query_snapshot(self, *, feedback: str) -> str:
        """
        When in instance disambiguation mode, run an automatic query for the target name
        and inject compact candidates into the next prompt.
        """
        fb = str(feedback or "")
        if "blocking=instance_disambiguation" not in fb:
            return ""
        m = re.search(r"(?:^|\n)instance_meta\.name=(?P<name>[^\n]+)", fb)
        if not m:
            return ""
        target_name = str(m.group("name") or "").strip()
        if not target_name:
            return ""
        try:
            from epm.cerebellum.skills.query_scene_objects.skill import QuerySceneObjectsArgs, run as run_query_scene_objects

            res = run_query_scene_objects(
                realtime_products_path=self.cfg.realtime_products_path,
                args=QuerySceneObjectsArgs(
                    query=target_name,
                    only_on_screen=False,
                    max_items=-1,
                    max_distance=0.0,
                ),
            )
            if not bool(res.success) or not isinstance(res.raw, dict):
                return f"auto_query target={target_name!r} failed error={res.error!r}"
            results = res.raw.get("results")
            if not isinstance(results, list):
                return f"auto_query target={target_name!r} results=[]"
            keep_keys = ("name_en", "name_cn", "kind", "instance_id", "distance", "is_on_screen", "container")
            compact: list[dict[str, Any]] = []
            for it in results:
                if not isinstance(it, dict):
                    continue
                compact.append({k: it.get(k) for k in keep_keys if k in it})
            return (
                f"auto_query target={target_name!r} results_len={len(results)}\n"
                + "auto_query_results="
                + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
            )
        except Exception as e:
            return f"auto_query target={target_name!r} failed error={e!r}"

    def _maybe_update_put_place_occupancy_snapshot(self, *, step: PlanStep, action_result: Any) -> None:
        if step.name != "auto_navigation":
            return
        raw = getattr(action_result, "raw", None)
        if not isinstance(raw, dict):
            return
        snap = raw.get("put_place_occupancy_snapshot")
        if not isinstance(snap, dict):
            return
        try:
            path = self.cfg.memory_dir / "put_place_occupancy.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            return


    def _task_progress_update_mode(self) -> str:
        if self.task_progress_path is None:
            return "maintainer_only"
        pipeline_name = (self.cfg.pipeline or "").strip().lower()
        if pipeline_name in ("react", "reflexion", "cap"):
            return "maintainer_only"
        raw = self.cfg.task_progress_maintenance or {}
        if not isinstance(raw, dict):
            return "strict"
        mode = str(raw.get("update_mode", "strict") or "strict").strip().lower()
        if mode not in ("strict", "maintainer_only", "both"):
            mode = "strict"
        return mode

    def _force_replan_next_step(self) -> None:
        """
        Best-effort: clear current plan so the next step will replan.
        """
        pipeline = self.pipeline
        # PlannerExecutor/OpenLoop
        if hasattr(pipeline, "state"):
            state = getattr(pipeline, "state")
            if hasattr(state, "current_plan"):
                setattr(state, "current_plan", None)
            if hasattr(state, "current_index"):
                setattr(state, "current_index", 0)
        # EPMPipeline wraps a base executor
        if hasattr(pipeline, "base"):
            base = getattr(pipeline, "base")
            if hasattr(base, "state"):
                state = getattr(base, "state")
                if hasattr(state, "current_plan"):
                    setattr(state, "current_plan", None)
                if hasattr(state, "current_index"):
                    setattr(state, "current_index", 0)

    def _read_camera_info(self) -> str:
        """
        Best-effort read of realtime camera pose info to inject into prompts.

        Expected format (example):
          Position: (-3.512, 1.750, -0.258)
          Rotation: (357.97, 135.43, 0.00)
          Forward: (0.701, 0.035, -0.712)

        Returns a short multi-line string for prompt injection.
        """
        path = self.cfg.camera_info_path
        if path is None:
            return ""
        try:
            if not path.exists():
                return ""
        except Exception:
            return ""

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                raw = path.read_text(encoding="utf-8-sig", errors="replace").strip()
                if not raw:
                    return ""
                # Normalize to 3 lines if possible.
                pos = ""
                rot = ""
                fwd = ""
                for line in raw.splitlines():
                    s = line.strip()
                    if s.lower().startswith("position:"):
                        pos = s
                    elif s.lower().startswith("rotation:"):
                        rot = s
                    elif s.lower().startswith("forward:"):
                        fwd = s
                lines: list[str] = []
                if pos or rot or fwd:
                    lines.append("Camera pose:")
                    if pos:
                        lines.append(pos)
                    if rot:
                        lines.append(rot)
                    if fwd:
                        lines.append(fwd)
                    return "\n".join(lines).strip()
                return ("Camera pose:\n" + raw).strip()
            except Exception as e:
                last_err = e
                time.sleep(0.05 * (attempt + 1))
        if self.cfg.verbose and last_err is not None:
            self.log.info(f"[EPM] camera_info read failed: {last_err!r}")
        return ""

    def _read_alt_j_interaction_info(self) -> str:
        """
        Best-effort read of the latest Alt+J interaction snapshot.

        This is a lightweight observation signal (what the crosshair is pointing at,
        plus optional PourAmount/Weight fields).
        """
        try:
            userdata_root = self.cfg.realtime_products_path.parent
            snap = read_interaction_snapshot(userdata_root=userdata_root, max_age_s=2.0)
            if snap is None:
                return ""
            return render_interaction_snapshot(snap)
        except Exception:
            return ""

    def _update_query_memory(self, *, step: PlanStep, action_result: ActionResult, final_success: bool) -> None:
        if not final_success:
            return
        if step.type != "skill" or step.name != "query_scene_objects":
            return
        raw = action_result.raw if isinstance(action_result.raw, dict) else {}
        results = raw.get("results")
        if not isinstance(results, list):
            return

        keep_keys = (
            "name_en",
            "name_cn",
            "name",
            "kind",
            "instance_id",
            "distance",
            "is_on_screen",
            "container",
            "position",
            "is_open",
            "open_angle",
        )
        query_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

        def _store_query(q_text: str, q_results: list[dict[str, Any]]) -> None:
            q = str(q_text or "").strip()
            if not q:
                return
            compact: list[dict[str, Any]] = []
            for it in q_results:
                if not isinstance(it, dict):
                    continue
                compact.append({k: it.get(k) for k in keep_keys if k in it})
            key = q.lower()
            if key in self._query_memory:
                self._query_memory.pop(key, None)
            self._query_memory[key] = {
                "query": q,
                "results_len": int(len(q_results)),
                "top_results": compact,
                "step_id": int(self.step_id),
                "query_time": query_time,
            }

        grouped = raw.get("results_by_query")
        if isinstance(grouped, dict) and grouped:
            for q_text, q_results in grouped.items():
                if isinstance(q_results, list):
                    _store_query(str(q_text), list(q_results))
        else:
            q = str(step.args.get("query", "") if isinstance(step.args, dict) else "").strip()
            _store_query(q, list(results))

        # Sliding window: keep at most 20 distinct queries (LRU by insertion/update order).
        while len(self._query_memory) > 20:
            try:
                oldest = next(iter(self._query_memory.keys()))
            except Exception:
                break
            self._query_memory.pop(oldest, None)
    def _render_query_memory(self, *, max_entries: int = -1, max_chars: int = -1) -> str:
        if not self._query_memory:
            return ""
        entries_all = list(self._query_memory.values())
        if int(max_entries) > 0:
            entries = entries_all[-max(1, int(max_entries)) :]
        else:
            entries = entries_all
        lines: list[str] = []
        lines.append("Recent query cache (cross-step):")
        lines.append("Note: each entry includes `source_step` and `query_time`; older entries may be stale after scene changes.")
        for e in reversed(entries):
            q = str(e.get("query") or "")
            n = int(e.get("results_len") or 0)
            source_step = int(e.get("step_id") or 0)
            query_time = str(e.get("query_time") or "")
            top = e.get("top_results") if isinstance(e.get("top_results"), list) else []
            s = json.dumps(top, ensure_ascii=False, separators=(",", ":"))
            lines.append(f"- source_step={source_step} query_time={query_time!r} query={q!r} results_len={n} top={s}")
        return "\n".join(lines).strip()

    def _load_agent_state_text(self) -> str:
        try:
            return json.dumps(self.memory.read_json("agent_state"), ensure_ascii=False, indent=2)
        except Exception:
            return ""

    def _load_recent_stm_window_text(self) -> str:
        try:
            path = self.cfg.memory_dir / "stm_window.txt"
            return path.read_text(encoding="utf-8") if path.exists() else ""
        except Exception:
            return ""

    def _read_reflexion_progress_entries(self) -> list[str]:
        try:
            text = self.memory.read_text("reflexion_progress_memory")
        except Exception:
            return []
        raw = str(text or "").strip()
        if not raw:
            return []
        return [ln.strip() for ln in raw.splitlines() if ln.strip().startswith("- ")]

    def _render_reflexion_progress_text(self, *, entries: list[str]) -> str:
        header = [
            "# Reflexion Progress Memory",
            "# Recent stable progress facts for the current episode.",
            "",
        ]
        body = list(entries or [])
        text = "\n".join(header + body).rstrip()
        return text + "\n"

    def _build_reflexion_progress_line(self, *, step: PlanStep, action_result: ActionResult, final_success: bool) -> str:
        if not final_success:
            return ""
        raw = dict(action_result.raw or {}) if isinstance(action_result.raw, dict) else {}
        args = dict(step.args or {})
        name = str(step.name or "").strip()
        if name == "gui_order_dish_via_computer":
            dish_name = str(args.get("dish_name") or self.dish.dish_name or "").strip()
            return f"- Ordered dish: {dish_name}." if dish_name else "- Ordered the dish."
        if name == "gui_buy_new_item":
            item_name = str(args.get("item") or "").strip()
            return f"- Bought {item_name}." if item_name else "- Bought a new item."
        if name == "auto_pour":
            container_name = str(args.get("container_name") or args.get("container") or "").strip()
            target_ml = args.get("target_ml", args.get("pour_ml"))
            if container_name and target_ml not in (None, ""):
                return f"- Poured {target_ml} ml into {container_name}."
            if container_name:
                return f"- Poured into {container_name}."
        if name == "gui_submit_dish_via_checkout_stand":
            dish_name = str(args.get("dish_name") or self.dish.dish_name or "").strip()
            return f"- Submitted dish: {dish_name}." if dish_name else "- Submitted the dish."
        return ""

    def _maybe_update_reflexion_progress_memory(self, *, step: PlanStep, action_result: ActionResult, final_success: bool) -> None:
        if str(self.pipeline_name or "").strip().lower() != "reflexion":
            return
        new_line = self._build_reflexion_progress_line(step=step, action_result=action_result, final_success=final_success)
        if not new_line:
            return
        entries = self._read_reflexion_progress_entries()
        if entries and entries[-1] == new_line:
            return
        entries.append(new_line)
        entries = entries[-10:]
        self.memory.write_text("reflexion_progress_memory", self._render_reflexion_progress_text(entries=entries))

    def _render_visual_anomaly_feedback_text(self, *, step_id: int, result: VisualAnomalyObservation) -> str:
        lines = [
            f"step={int(step_id)}",
            f"severity={str(result.severity or 'none').strip()}",
            f"tags={json.dumps(list(result.tags or []), ensure_ascii=False)}",
        ]
        if str(result.expected_visible or "").strip():
            lines.append(f"expected_visible={str(result.expected_visible).strip()}")
        if str(result.observed_issue or "").strip():
            lines.append(f"observed_issue={str(result.observed_issue).strip()}")
        if str(result.feedback or "").strip():
            lines.append(f"feedback={str(result.feedback).strip()}")
        return "\n".join(lines).strip()

    def _persist_visual_anomaly_feedback(self, *, step_id: int, result: VisualAnomalyObservation) -> None:
        try:
            payload = {
                "episode_step": int(step_id),
                "has_anomaly": bool(result.has_anomaly),
                "severity": str(result.severity or "none"),
                "tags": list(result.tags or []),
                "feedback": str(result.feedback or ""),
                "expected_visible": str(result.expected_visible or ""),
                "observed_issue": str(result.observed_issue or ""),
                "confidence": result.confidence,
                "raw": (dict(result.raw or {}) if isinstance(result.raw, dict) else result.raw),
            }
            path = self.cfg.memory_dir / "visual_anomaly_feedback" / f"step_{int(step_id):06d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        except Exception:
            pass
        try:
            if result.has_anomaly and str(result.feedback or "").strip():
                self.memory.write_text(
                    "latest_visual_anomaly_feedback",
                    self._render_visual_anomaly_feedback_text(step_id=step_id, result=result),
                )
            else:
                self.memory.write_text("latest_visual_anomaly_feedback", "")
        except Exception:
            pass

    def _maybe_run_task_progress_maintenance(self, *, step_id: int, new_plan_generated: bool, need_replan: bool) -> None:
        if self.maintainer is None:
            return
        if not self.maintainer.should_run(
            step_id=int(step_id),
            new_plan_generated=new_plan_generated,
            need_replan=need_replan,
        ):
            return
        last_feedback = self._last_action_feedback
        try:
            hl_id, hl_goal = self._pick_current_high_level_goal()
        except Exception:
            hl_id, hl_goal = ("", "")
        agent_state_text = self._load_agent_state_text()
        try:
            res = self.maintainer.run(
                step_id=int(step_id),
                task_progress_path=self.task_progress_path,
                stm_window_path=self.cfg.memory_dir / "stm_window.txt",
                recipe_text=self.dish.recipe_text,
                last_feedback=last_feedback,
                current_high_level_id=hl_id,
                current_high_level_goal=hl_goal,
                agent_state_text=agent_state_text,
            )
            apply_task_progress_maintenance(task_progress_path=self.task_progress_path, result=res)
            try:
                self.memory.save_task_progress_update(
                    step_id=int(step_id),
                    payload={
                        "episode_step": int(step_id),
                        "trigger": {
                            "new_plan_generated": bool(new_plan_generated),
                            "need_replan": bool(need_replan),
                        },
                        "current_high_level_id": str(res.current_high_level_id or ""),
                        "current_high_level_goal": str(res.current_high_level_goal or ""),
                        "goal_state": dict(res.goal_state or {}),
                        "semantic_progress": dict(res.semantic_progress or {}),
                        "commitments": list(res.commitments or []),
                        "resource_bindings": dict(res.resource_bindings or {}),
                        "blocking_conditions": list(res.blocking_conditions or []),
                        "temporal_checkpoints": dict(res.temporal_checkpoints or {}),
                        "strategy_notes": dict(res.strategy_notes or {}),
                        "evidence": dict(res.evidence or {}),
                    },
                )
                try:
                    path = self.memory.paths.get("latest_task_progress_feedback")
                    if path is not None:
                        Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception:
                pass
            if self.cfg.verbose:
                self.log.info(
                    f"[EPM] task_progress_maintenance applied current_high_level_id={res.current_high_level_id!r} "
                    f"goal={res.current_high_level_goal!r}"
                )
        except Exception as e:
            if self.cfg.verbose:
                self.log.info(f"[EPM] task_progress_maintenance failed: {e!r}")

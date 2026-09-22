from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EpmPaths:
    """
    Convenience paths derived from an EPM root.

    Kept here to avoid a separate `config.py` module with overlapping meaning.
    """

    epm_root: Path
    data_dir: Path
    memory_dir: Path

    @staticmethod
    def from_epm_root(epm_root: str | Path) -> "EpmPaths":
        root = Path(epm_root).resolve()
        return EpmPaths(epm_root=root, data_dir=root / "data", memory_dir=root / "memory")


def find_epm_root(start: Optional[str | Path] = None) -> Path:
    """
    Find the `epm/` root folder by walking parents until `pyproject.toml` exists.
    """

    current = Path(start).resolve() if start else Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not locate EPM root (pyproject.toml not found).")


@dataclass(frozen=True)
class CaptureSettings:
    window_title: str = "CookingSimulator"
    activate_window_each_step: bool = True
    screenshot_format: str = "jpeg"
    screenshot_jpeg_quality: int = 70


@dataclass(frozen=True)
class PathSettings:
    recipes_path: Path
    memory_dir: Path
    game_userdata_root: Optional[Path]
    realtime_products_path: Path
    screenshot_dir: Path
    camera_info_path: Optional[Path] = None
    hotkeys_status_path: Optional[Path] = None
    realtime_interaction_info_path: Optional[Path] = None
    realtime_ui_info_path: Optional[Path] = None
    recorder_script_path: Optional[Path] = None
    recording_output_root: Optional[Path] = None


@dataclass(frozen=True)
class RuntimeSettings:
    stm_window_size: int = 200
    verbose: bool = True
    # Logging mode:
    # - "verbose": keep current detailed logs (planner/percept/skill previews)
    # - "minimal": print one-line per step (executed action + success/failure); capture noisy stdout to file
    log_mode: str = "verbose"
    log_color: bool = True
    log_level: str = "INFO"
    max_steps: int = 2000
    # If >0, trigger force-submit mode once step_id >= threshold.
    force_submit_step_threshold: int = 0
    # If true, force-submit mode starts immediately from step 1.
    force_submit_active: bool = False
    # Once triggered, the agent must submit within this many steps.
    force_submit_within_steps: int = 50
    # Abort once force-submit is overdue and consecutive failures reach this limit.
    force_submit_failure_limit: int = 3
    # HTTP 403 should pause outside the step loop, so the wait does not consume steps.
    http_403_pause_enabled: bool = True
    http_403_wait_s: int = 60
    http_403_max_wait_s: int = 900
    api_key_rotation_enabled: bool = True
    api_key_rotation_after_http_403_wait_s: int = 60
    api_key_rotation_quarantine_s: int = 1800
    # General retryable network errors should also pause outside the step loop.
    network_pause_enabled: bool = True
    network_wait_s: int = 15
    network_max_wait_s: int = 150
    network_retry_attempts: int = 10
    # Whether to restart the environment before running.
    restart_env: bool = False
    # "api" -> use llm/vlm.base_url
    # "hpc" -> override base_url with hpc_config.service_urls.{llm_service,vlm_service}
    endpoint_profile: str = "api"
    notify_email_enabled: bool = False
    notify_email_smtp_host: str = ""
    notify_email_smtp_port: int = 587
    notify_email_username: str = ""
    notify_email_from: str = ""
    notify_email_password_env: str = ""
    notify_email_to: str = ""
    notify_email_use_tls: bool = True
    balance_check_every_steps: int = 10
    notify_email_low_balance_usd_threshold: float = 10.0


@dataclass(frozen=True)
class BrainSettings:
    exposed_action_categories: list[str]
    max_actions_per_category: int = -1
    inject_memory: bool = True
    inject_tool_schemas: bool = True
    action_catalog_path: Optional[Path] = None
    prompt_policy: Dict[str, Any] = None  # raw dict, parsed in agent/policy
    prompt_ablation_profile: str = "full"
    prompt_layout_path: Optional[Path] = None
    scripted_plan_path: Optional[Path] = None
    planner_mode: str = "scripted"  # "scripted" | "llm" | "vlm"
    use_vlm: bool = True
    pipeline: str = "planner_executor"  # "planner_executor" | "react" | ...
    perception_mode: str = "oracle"  # "oracle" | "vlm" | "none"
    max_visible_items: int = 20
    plan_min_steps: int = 3
    plan_max_steps: int = 8
    # If false, planner outputs below plan_min_steps are rejected (no incremental fallback).
    allow_incremental_plan: bool = True
    task_progress_maintenance: Dict[str, Any] = None  # raw dict, parsed in agent
    step_success_judge: Dict[str, Any] = None  # raw dict, parsed in agent
    subgoal_done_judge: Dict[str, Any] = None  # raw dict, parsed in agent
    visual_anomaly_observer: Dict[str, Any] = None  # raw dict, parsed in agent
    precondition_checker: Dict[str, Any] = None  # raw dict, parsed in agent
    reflexion: Dict[str, Any] = None  # raw dict, parsed in agent


@dataclass(frozen=True)
class LlmSettings:
    provider: str = "openai_compatible"
    base_url: str = "http://localhost:8000"
    request_path: str = ""
    chat_completions_path: str = "/v1/chat/completions"
    messages_path: str = "/v1/messages"
    models_path: str = "/v1/models"
    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    api_key_pool_env: str = ""
    api_key: str = ""
    auth_header_name: str = ""
    auth_header_prefix: str = ""
    extra_headers: Optional[Dict[str, str]] = None
    timeout_s: float = 60.0
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 1200
    use_vision: bool = False
    use_tools: bool = False
    image_max_side: int = 768
    image_format: str = "jpeg"
    jpeg_quality: int = 70
    prompt_cache_enabled: bool = True
    prompt_cache_force: bool = True
    prompt_cache_min_chars: int = 4096
    prompt_cache_ttl_s: int = 3600
    prompt_cache_dir: str = ""
    gemini_cache_api_base_url: str = ""
    request_metrics_path: Optional[Path] = None


@dataclass(frozen=True)
class VlmSettings:
    provider: str = "openai_compatible"
    base_url: str = "http://localhost:8001"
    request_path: str = ""
    chat_completions_path: str = "/v1/chat/completions"
    messages_path: str = "/v1/messages"
    models_path: str = "/v1/models"
    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    api_key_pool_env: str = ""
    api_key: str = ""
    auth_header_name: str = ""
    auth_header_prefix: str = ""
    extra_headers: Optional[Dict[str, str]] = None
    timeout_s: float = 60.0
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 800
    use_vision: bool = True
    strip_think_tags: bool = True
    use_tools: bool = False
    image_max_side: int = 768
    image_format: str = "jpeg"
    jpeg_quality: int = 70
    prompt_cache_enabled: bool = True
    prompt_cache_force: bool = True
    prompt_cache_min_chars: int = 4096
    prompt_cache_ttl_s: int = 3600
    prompt_cache_dir: str = ""
    gemini_cache_api_base_url: str = ""
    request_metrics_path: Optional[Path] = None


@dataclass(frozen=True)
class ApiProviderSpec:
    provider_type: str
    model_name: str
    base_url: str
    request_path: str = ""
    chat_completions_path: str = "/v1/chat/completions"
    messages_path: str = "/v1/messages"
    models_path: str = "/v1/models"
    api_key_env: str = "OPENAI_API_KEY"
    api_key_pool_env: str = ""
    api_key: str = ""
    auth_header_name: str = ""
    auth_header_prefix: str = ""
    extra_headers: Optional[Dict[str, str]] = None
    send_auth: bool = True
    # Optional: model catalog for humans; loader may use default_model/models[0] if model_name empty.
    models: list[str] = None  # filled in load_settings when provided


@dataclass(frozen=True)
class ServiceUrls:
    llm_service: str = ""
    vlm_service: str = ""
    detection_service: str = ""


@dataclass(frozen=True)
class HpcSettings:
    service_urls: ServiceUrls


@dataclass(frozen=True)
class EpmSettings:
    dish_id: int
    paths: PathSettings
    capture: CaptureSettings
    runtime: RuntimeSettings
    brain: BrainSettings
    llm: LlmSettings
    vlm: VlmSettings
    hpc: HpcSettings


def _require_int(obj: Dict[str, Any], key: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected int for '{key}', got: {type(value).__name__}")
    return value


def _require_str(obj: Dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string for '{key}', got: {value!r}")
    return value


def _normalize_provider_type(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return "openai_compatible"
    aliases = {
        "openai": "openai",
        "openai_compatible": "openai_compatible",
        "chat_completions": "openai_compatible",
        "anthropic_messages": "anthropic_messages",
        "anthropic": "anthropic_messages",
        "messages": "anthropic_messages",
    }
    return aliases.get(text, text)


def _require_path(obj: Dict[str, Any], key: str, *, base_dir: Path) -> Path:
    raw = _require_str(obj, key)
    p = Path(raw)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _optional_path(obj: Dict[str, Any], key: str, *, base_dir: Path) -> Optional[Path]:
    if key not in obj:
        return None
    raw = obj.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def resolve_settings_path(path: str | Path) -> Path:
    cfg_path = Path(path).resolve()
    if cfg_path.is_dir():
        cfg_path = (cfg_path / "epm_config.json").resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return cfg_path


def _configs_root_from_path(cfg_path: Path) -> Path:
    parent = cfg_path.parent
    if parent.name in {"pipelines", "examples", "docs", "local"}:
        return parent.parent
    return parent


def _read_json_object(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config JSON must be an object: {path}")
    return raw


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _absolutize_known_config_paths(raw: Dict[str, Any], *, base_dir: Path) -> Dict[str, Any]:
    data = deepcopy(raw)

    def _resolve_in(section: dict[str, Any], key: str) -> None:
        value = section.get(key)
        if not isinstance(value, str):
            return
        text = value.strip()
        if not text:
            return
        p = Path(text)
        if p.is_absolute():
            section[key] = str(p)
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
            "hotkeys_status_path",
            "camera_info_path",
            "realtime_interaction_info_path",
            "realtime_ui_info_path",
            "recorder_script_path",
            "recording_output_root",
            "realtime_radar_scan_path",
        ):
            _resolve_in(paths, key)

    brain = data.get("brain")
    if isinstance(brain, dict):
        for key in ("action_catalog_path", "prompt_layout_path", "scripted_plan_path"):
            _resolve_in(brain, key)

    for role_key in ("llm", "vlm"):
        role = data.get(role_key)
        if isinstance(role, dict):
            for key in ("request_metrics_path", "prompt_cache_dir"):
                _resolve_in(role, key)

    return data


def _load_api_keys_local_settings_override(configs_root: Path) -> Dict[str, Any]:
    path = configs_root / "api_keys.local.json"
    if not path.exists():
        return {}
    raw = _read_json_object(path)
    override = raw.get("settings_override")
    if override is None:
        return {}
    if not isinstance(override, dict):
        raise ValueError(f"'settings_override' in {path} must be an object")
    return _absolutize_known_config_paths(override, base_dir=path.parent)


def _strip_common_local_provider_selection(raw: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = deepcopy(raw)
    # Provider/model selection is intentionally centralized in
    # api_keys.local.json -> settings_override to avoid split-brain local config.
    cleaned.pop("api_model_assignments", None)
    return cleaned


def load_raw_settings(path: str | Path) -> Dict[str, Any]:
    cfg_path = resolve_settings_path(path)
    merged = _load_raw_settings_recursive(cfg_path, seen=[])
    configs_root = _configs_root_from_path(cfg_path)
    local_override_path = configs_root / "common.local.json"
    if local_override_path.exists() and local_override_path.resolve() != cfg_path:
        local_raw = _read_json_object(local_override_path)
        local_raw = _strip_common_local_provider_selection(local_raw)
        local_current = _absolutize_known_config_paths(local_raw, base_dir=local_override_path.parent)
        merged = _deep_merge_dicts(merged, local_current)
    api_keys_local_override = _load_api_keys_local_settings_override(configs_root)
    if api_keys_local_override:
        merged = _deep_merge_dicts(merged, api_keys_local_override)
    return merged


def _load_raw_settings_recursive(cfg_path: Path, *, seen: list[Path]) -> Dict[str, Any]:
    cfg_path = cfg_path.resolve()
    if cfg_path in seen:
        chain = " -> ".join(str(p) for p in [*seen, cfg_path])
        raise ValueError(f"Config extends cycle detected: {chain}")

    raw = _read_json_object(cfg_path)
    extends_raw = raw.get("extends")
    parent_refs: list[str] = []
    if isinstance(extends_raw, str) and extends_raw.strip():
        parent_refs = [extends_raw.strip()]
    elif isinstance(extends_raw, list):
        for item in extends_raw:
            if isinstance(item, str) and item.strip():
                parent_refs.append(item.strip())
            else:
                raise ValueError(f"Invalid extends entry in {cfg_path}: {item!r}")
    elif extends_raw is not None:
        raise ValueError(f"'extends' must be a string or list of strings: {cfg_path}")

    merged: Dict[str, Any] = {}
    next_seen = [*seen, cfg_path]
    for parent_ref in parent_refs:
        parent_path = Path(parent_ref)
        if not parent_path.is_absolute():
            parent_path = (cfg_path.parent / parent_path).resolve()
        parent_raw = _load_raw_settings_recursive(parent_path, seen=next_seen)
        merged = _deep_merge_dicts(merged, parent_raw)

    current = _absolutize_known_config_paths(raw, base_dir=cfg_path.parent)
    merged = _deep_merge_dicts(merged, current)
    return merged


def load_settings(path: str | Path) -> EpmSettings:
    cfg_path = resolve_settings_path(path)
    base_dir = cfg_path.parent
    raw = load_raw_settings(cfg_path)

    dish_id = _require_int(raw, "dish_id")

    raw_paths = raw.get("paths") or {}
    if not isinstance(raw_paths, dict):
        raise ValueError("'paths' must be an object")
    game_userdata_root = _optional_path(raw_paths, "game_userdata_root", base_dir=base_dir)
    if game_userdata_root is None:
        # Backward compatibility: old key name.
        game_userdata_root = _optional_path(raw_paths, "userdata_root", base_dir=base_dir)
    paths = PathSettings(
        recipes_path=_require_path(raw_paths, "recipes_path", base_dir=base_dir),
        memory_dir=_require_path(raw_paths, "memory_dir", base_dir=base_dir),
        game_userdata_root=game_userdata_root,
        realtime_products_path=(
            _require_path(raw_paths, "realtime_products_path", base_dir=base_dir)
            if "realtime_products_path" in raw_paths
            else (
                (game_userdata_root / "realtime_products.json")
                if game_userdata_root is not None
                else None
            )
        ),
        screenshot_dir=_require_path(raw_paths, "screenshot_dir", base_dir=base_dir),
        camera_info_path=(
            _require_path(raw_paths, "camera_info_path", base_dir=base_dir)
            if "camera_info_path" in raw_paths
            else (
                (game_userdata_root / "realtime_camera_info.txt")
                if game_userdata_root is not None
                else None
            )
        ),
        hotkeys_status_path=(
            _require_path(raw_paths, "hotkeys_status_path", base_dir=base_dir)
            if "hotkeys_status_path" in raw_paths
            else (
                (game_userdata_root / "realtime_interaction_status.txt")
                if game_userdata_root is not None
                else None
            )
        ),
        realtime_interaction_info_path=(
            _require_path(raw_paths, "realtime_interaction_info_path", base_dir=base_dir)
            if "realtime_interaction_info_path" in raw_paths
            else (
                (game_userdata_root / "realtime_interaction_info.txt")
                if game_userdata_root is not None
                else None
            )
        ),
        realtime_ui_info_path=(
            _require_path(raw_paths, "realtime_ui_info_path", base_dir=base_dir)
            if "realtime_ui_info_path" in raw_paths
            else (
                (game_userdata_root / "realtime_ui_info.json")
                if game_userdata_root is not None
                else None
            )
        ),
        recorder_script_path=_optional_path(raw_paths, "recorder_script_path", base_dir=base_dir),
        recording_output_root=_optional_path(raw_paths, "recording_output_root", base_dir=base_dir),
    )
    if paths.realtime_products_path is None:
        raise ValueError("paths.realtime_products_path is required, or provide paths.game_userdata_root to derive it.")

    raw_capture = raw.get("capture") or {}
    if not isinstance(raw_capture, dict):
        raise ValueError("'capture' must be an object")
    capture = CaptureSettings(
        window_title=_require_str(raw_capture, "window_title") if "window_title" in raw_capture else "CookingSimulator",
        activate_window_each_step=bool(raw_capture.get("activate_window_each_step", True)),
        screenshot_format=str(raw_capture.get("screenshot_format", "jpeg") or "jpeg"),
        screenshot_jpeg_quality=int(raw_capture.get("screenshot_jpeg_quality", 70) or 70),
    )

    raw_runtime = raw.get("runtime") or {}
    if not isinstance(raw_runtime, dict):
        raise ValueError("'runtime' must be an object")
    runtime = RuntimeSettings(
        stm_window_size=int(raw_runtime.get("stm_window_size", raw_runtime.get("window_size", 200))),
        verbose=bool(raw_runtime.get("verbose", True)),
        log_mode=str(raw_runtime.get("log_mode", "verbose") or "verbose"),
        log_color=bool(raw_runtime.get("log_color", True)),
        log_level=str(raw_runtime.get("log_level", "INFO") or "INFO"),
        max_steps=int(raw_runtime.get("max_steps", 2000)),
        force_submit_step_threshold=int(raw_runtime.get("force_submit_step_threshold", 0) or 0),
        force_submit_active=bool(raw_runtime.get("force_submit_active", False)),
        force_submit_within_steps=int(raw_runtime.get("force_submit_within_steps", 50) or 50),
        force_submit_failure_limit=int(raw_runtime.get("force_submit_failure_limit", 3) or 3),
        http_403_pause_enabled=bool(raw_runtime.get("http_403_pause_enabled", True)),
        http_403_wait_s=int(raw_runtime.get("http_403_wait_s", 60) or 60),
        http_403_max_wait_s=int(raw_runtime.get("http_403_max_wait_s", 900) or 900),
        api_key_rotation_enabled=bool(raw_runtime.get("api_key_rotation_enabled", True)),
        api_key_rotation_after_http_403_wait_s=int(raw_runtime.get("api_key_rotation_after_http_403_wait_s", 60) or 60),
        api_key_rotation_quarantine_s=int(raw_runtime.get("api_key_rotation_quarantine_s", 1800) or 1800),
        network_pause_enabled=bool(raw_runtime.get("network_pause_enabled", raw_runtime.get("http_403_pause_enabled", True))),
        network_wait_s=int(raw_runtime.get("network_wait_s", 15) or 15),
        network_max_wait_s=int(raw_runtime.get("network_max_wait_s", 150) or 150),
        network_retry_attempts=int(raw_runtime.get("network_retry_attempts", 10) or 10),
        restart_env=bool(raw_runtime.get("restart_env", False)),
        endpoint_profile=str(raw_runtime.get("endpoint_profile", "api")),
        notify_email_enabled=bool(raw_runtime.get("notify_email_enabled", False)),
        notify_email_smtp_host=str(raw_runtime.get("notify_email_smtp_host", "") or ""),
        notify_email_smtp_port=int(raw_runtime.get("notify_email_smtp_port", 587) or 587),
        notify_email_username=str(raw_runtime.get("notify_email_username", "") or ""),
        notify_email_from=str(raw_runtime.get("notify_email_from", "") or ""),
        notify_email_password_env=str(raw_runtime.get("notify_email_password_env", "") or ""),
        notify_email_to=str(raw_runtime.get("notify_email_to", "") or ""),
        notify_email_use_tls=bool(raw_runtime.get("notify_email_use_tls", True)),
        balance_check_every_steps=int(raw_runtime.get("balance_check_every_steps", 10) or 10),
        notify_email_low_balance_usd_threshold=float(raw_runtime.get("notify_email_low_balance_usd_threshold", 10.0) or 10.0),
    )

    raw_brain = raw.get("brain") or {}
    if not isinstance(raw_brain, dict):
        raise ValueError("'brain' must be an object")
    exposed = raw_brain.get("exposed_action_categories", [])
    if exposed is None:
        exposed = []
    if not isinstance(exposed, list) or not all(isinstance(x, str) for x in exposed):
        raise ValueError("'brain.exposed_action_categories' must be a list of strings")
    raw_prompt_policy = raw_brain.get("prompt_policy") or {}
    if not isinstance(raw_prompt_policy, dict):
        raw_prompt_policy = {}
    raw_prompt_ablation = raw_brain.get("prompt_ablation") or {}
    if not isinstance(raw_prompt_ablation, dict):
        raise ValueError("'brain.prompt_ablation' must be an object")
    from epm.core.prompt_ablation import normalize_prompt_ablation_profile
    prompt_ablation_profile = normalize_prompt_ablation_profile(raw_prompt_ablation.get("profile", "full"))
    plan_steps_raw = raw_brain.get("plan_steps", None)
    if plan_steps_raw is not None:
        try:
            plan_steps = int(plan_steps_raw)
        except Exception:
            raise ValueError("'brain.plan_steps' must be an integer")
        if plan_steps == 0:
            raise ValueError("'brain.plan_steps' must be positive or -1 for unlimited")
        if plan_steps < 0:
            plan_min = 1
            plan_max = -1
        else:
            plan_min = plan_steps
            plan_max = plan_steps
    else:
        plan_min = int(raw_brain.get("plan_min_steps", 3))
        plan_max = int(raw_brain.get("plan_max_steps", 8))

    planner_mode_raw = str(raw_brain.get("planner_mode", "scripted") or "scripted").strip().lower()
    use_vlm_present = "use_vlm" in raw_brain
    use_vlm_value = bool(raw_brain.get("use_vlm", True))
    if planner_mode_raw == "scripted":
        planner_mode_value = "scripted"
    elif use_vlm_present:
        planner_mode_value = "vlm" if use_vlm_value else "llm"
    else:
        planner_mode_value = planner_mode_raw

    brain = BrainSettings(
        exposed_action_categories=[str(x) for x in exposed],
        max_actions_per_category=int(raw_brain.get("max_actions_per_category", -1)),
        inject_memory=bool(raw_brain.get("inject_memory", True)),
        inject_tool_schemas=bool(raw_brain.get("inject_tool_schemas", True)),
        action_catalog_path=_optional_path(raw_brain, "action_catalog_path", base_dir=base_dir),
        prompt_policy=raw_prompt_policy,
        prompt_ablation_profile=prompt_ablation_profile,
        prompt_layout_path=(
            _require_path(raw_brain, "prompt_layout_path", base_dir=base_dir)
            if "prompt_layout_path" in raw_brain
            else (base_dir / "memory" / "prompt_layout.json").resolve()
        ),
        scripted_plan_path=(
            _require_path(raw_brain, "scripted_plan_path", base_dir=base_dir)
            if "scripted_plan_path" in raw_brain
            else None
        ),
        planner_mode=planner_mode_value,
        use_vlm=use_vlm_value,
        pipeline=str(raw_brain.get("pipeline", "planner_executor")),
        perception_mode=str(raw_brain.get("perception_mode", "oracle")),
        max_visible_items=int(raw_brain.get("max_visible_items", 20)),
        plan_min_steps=plan_min,
        plan_max_steps=plan_max,
        allow_incremental_plan=bool(raw_brain.get("allow_incremental_plan", True)),
        task_progress_maintenance=(raw_brain.get("task_progress_maintenance") or {}),
        step_success_judge=(raw_brain.get("step_success_judge") or {}),
        subgoal_done_judge=(raw_brain.get("subgoal_done_judge") or {}),
        visual_anomaly_observer=(raw_brain.get("visual_anomaly_observer") or {}),
        precondition_checker=(raw_brain.get("precondition_checker") or {}),
        reflexion=(raw_brain.get("reflexion") or {}),
    )

    # Optional HPC service URLs (for multi-node deployments).
    raw_hpc = raw.get("hpc_config") or {}
    if not isinstance(raw_hpc, dict):
        raise ValueError("'hpc_config' must be an object")
    raw_urls = raw_hpc.get("service_urls") or {}
    if not isinstance(raw_urls, dict):
        raw_urls = {}
    hpc = HpcSettings(
        service_urls=ServiceUrls(
            llm_service=str(raw_urls.get("llm_service", "")),
            vlm_service=str(raw_urls.get("vlm_service", "")),
            detection_service=str(raw_urls.get("detection_service", "")),
        )
    )

    # Config style (deploy_agent-like):
    # Use `api_providers` + `api_model_assignments` for endpoint/model selection.
    # The `llm` / `vlm` blocks are treated as *role parameters* only (timeout/tokens/temperature/vision),
    # not as endpoint/model selectors.
    llm: LlmSettings
    vlm: VlmSettings
    raw_api_providers = raw.get("api_providers")
    raw_assignments = raw.get("api_model_assignments")
    if not isinstance(raw_api_providers, dict) or not isinstance(raw_assignments, dict):
        raise ValueError("This EPM config requires `api_providers` and `api_model_assignments` (B-style).")

    def _read_assignment(*names: str) -> dict[str, str]:
        for name in names:
            v = raw_assignments.get(name)
            if isinstance(v, str) and v.strip():
                return {"provider": v.strip()}
            if isinstance(v, dict):
                out: dict[str, str] = {}
                for k in ("provider", "model", "provider_type", "api", "base_url"):
                    if k in v and isinstance(v.get(k), str) and str(v.get(k)).strip():
                        out[k] = str(v.get(k)).strip()
                if out:
                    return out
        return {}

    def _resolve_provider(assign: dict[str, str]) -> ApiProviderSpec:
        provider_key = (assign.get("provider") or "").strip()
        model_override = (assign.get("model") or "").strip()
        provider_type_override = (assign.get("provider_type") or assign.get("api") or "").strip()
        base_url_override = (assign.get("base_url") or "").strip()
        if not provider_key:
            raise ValueError(f"api_model_assignments.{provider_key} is missing/invalid")
        spec_raw = raw_api_providers.get(provider_key)
        if not isinstance(spec_raw, dict):
            raise ValueError(f"api_providers missing provider: {provider_key}")

        provider_type = _normalize_provider_type(
            provider_type_override
            or str(spec_raw.get("provider_type", spec_raw.get("api", "openai_compatible"))).strip()
            or "openai_compatible"
        )

        base_url = base_url_override or str(spec_raw.get("base_url", "")).strip()
        request_path = str(spec_raw.get("request_path", "") or "").strip()
        chat_completions_path = str(spec_raw.get("chat_completions_path", "/v1/chat/completions") or "").strip()
        messages_path = str(spec_raw.get("messages_path", "/v1/messages") or "").strip()
        models_path = str(spec_raw.get("models_path", "/v1/models") or "").strip()

        send_auth = bool(spec_raw.get("send_auth", True))
        api_key_env = ""
        api_key_pool_env = ""
        api_key = ""
        auth_header_name = str(spec_raw.get("auth_header_name", "") or "").strip()
        auth_header_prefix = str(spec_raw.get("auth_header_prefix", "") or "")
        extra_headers = spec_raw.get("extra_headers") if isinstance(spec_raw.get("extra_headers"), dict) else None
        if send_auth:
            api_key_env = str(spec_raw.get("api_key_env", spec_raw.get("api_key_env_name", "OPENAI_API_KEY"))).strip()
            api_key_pool_env = str(spec_raw.get("api_key_pool_env", "")).strip()
            api_key = str(spec_raw.get("api_key", "")).strip()

        models_raw = spec_raw.get("models")
        models: list[str] = []
        if isinstance(models_raw, list):
            for x in models_raw:
                if isinstance(x, str) and x.strip():
                    models.append(x.strip())

        default_model = str(spec_raw.get("default_model", "")).strip()
        model_name = (
            model_override
            or str(spec_raw.get("model_name", "")).strip()
            or default_model
            or (models[0] if models else "")
        )

        return ApiProviderSpec(
            provider_type=provider_type,
            model_name=model_name,
            base_url=base_url.rstrip("/"),
            request_path=request_path,
            chat_completions_path=chat_completions_path,
            messages_path=messages_path,
            models_path=models_path,
            api_key_env=(api_key_env or "OPENAI_API_KEY") if send_auth else "",
            api_key_pool_env=api_key_pool_env if send_auth else "",
            api_key=api_key if send_auth else "",
            auth_header_name=auth_header_name,
            auth_header_prefix=auth_header_prefix,
            extra_headers=extra_headers,
            send_auth=send_auth,
            models=models,
        )

    llm_params = raw.get("llm") if isinstance(raw.get("llm"), dict) else {}
    vlm_params = raw.get("vlm") if isinstance(raw.get("vlm"), dict) else {}

    llm_assign = _read_assignment("llm", "planning")
    if not llm_assign.get("provider"):
        raise ValueError("api_model_assignments.llm (or planning) is required")
    planning_spec = _resolve_provider(llm_assign)

    llm = LlmSettings(
        provider=str(planning_spec.provider_type),
        base_url=planning_spec.base_url
        or (hpc.service_urls.llm_service.rstrip("/") if hpc.service_urls.llm_service else "http://localhost:8000"),
        request_path=planning_spec.request_path,
        chat_completions_path=planning_spec.chat_completions_path or "/v1/chat/completions",
        messages_path=planning_spec.messages_path or "/v1/messages",
        models_path=planning_spec.models_path,
        model=planning_spec.model_name,
        api_key_env=planning_spec.api_key_env or "OPENAI_API_KEY",
        api_key_pool_env=planning_spec.api_key_pool_env,
        api_key=planning_spec.api_key,
        auth_header_name=planning_spec.auth_header_name,
        auth_header_prefix=planning_spec.auth_header_prefix,
        extra_headers=planning_spec.extra_headers,
        timeout_s=float(llm_params.get("timeout_s", 60.0)),
        max_retries=int(llm_params.get("max_retries", 2)),
        temperature=float(llm_params.get("temperature", 0.0)),
        max_tokens=int(llm_params.get("max_tokens", 1200)),
        use_vision=bool(llm_params.get("use_vision", False)),
        use_tools=bool(llm_params.get("use_tools", False)),
        image_max_side=int(llm_params.get("image_max_side", 768)),
        image_format=str(llm_params.get("image_format", "jpeg") or "jpeg"),
        jpeg_quality=int(llm_params.get("jpeg_quality", 70)),
        prompt_cache_enabled=bool(llm_params.get("prompt_cache_enabled", True)),
        prompt_cache_force=bool(llm_params.get("prompt_cache_force", True)),
        prompt_cache_min_chars=int(llm_params.get("prompt_cache_min_chars", 4096)),
        prompt_cache_ttl_s=int(llm_params.get("prompt_cache_ttl_s", 3600)),
        prompt_cache_dir=str(llm_params.get("prompt_cache_dir", "") or ""),
        gemini_cache_api_base_url=str(llm_params.get("gemini_cache_api_base_url", "") or ""),
        request_metrics_path=(paths.memory_dir / "planner_request_metrics.jsonl"),
    )

    scene_assign = _read_assignment("vlm", "scene_understanding")
    if scene_assign.get("provider"):
        scene_spec = _resolve_provider(scene_assign)
        vlm = VlmSettings(
            provider=str(scene_spec.provider_type),
            base_url=scene_spec.base_url
            or (hpc.service_urls.vlm_service.rstrip("/") if hpc.service_urls.vlm_service else llm.base_url),
            request_path=scene_spec.request_path,
            chat_completions_path=scene_spec.chat_completions_path or "/v1/chat/completions",
            messages_path=scene_spec.messages_path or "/v1/messages",
            models_path=scene_spec.models_path,
            model=scene_spec.model_name,
            api_key_env=scene_spec.api_key_env or "OPENAI_API_KEY",
            api_key_pool_env=scene_spec.api_key_pool_env,
            api_key=scene_spec.api_key,
            auth_header_name=scene_spec.auth_header_name,
            auth_header_prefix=scene_spec.auth_header_prefix,
            extra_headers=scene_spec.extra_headers,
            timeout_s=float(vlm_params.get("timeout_s", 60.0)),
            max_retries=int(vlm_params.get("max_retries", 2)),
            temperature=float(vlm_params.get("temperature", 0.0)),
            max_tokens=int(vlm_params.get("max_tokens", 800)),
            use_vision=bool(vlm_params.get("use_vision", True)),
            strip_think_tags=bool(vlm_params.get("strip_think_tags", True)),
            use_tools=bool(vlm_params.get("use_tools", False)),
            image_max_side=int(vlm_params.get("image_max_side", 768)),
            image_format=str(vlm_params.get("image_format", "jpeg") or "jpeg"),
            jpeg_quality=int(vlm_params.get("jpeg_quality", 70)),
            prompt_cache_enabled=bool(vlm_params.get("prompt_cache_enabled", True)),
            prompt_cache_force=bool(vlm_params.get("prompt_cache_force", True)),
            prompt_cache_min_chars=int(vlm_params.get("prompt_cache_min_chars", 4096)),
            prompt_cache_ttl_s=int(vlm_params.get("prompt_cache_ttl_s", 3600)),
            prompt_cache_dir=str(vlm_params.get("prompt_cache_dir", "") or ""),
            gemini_cache_api_base_url=str(vlm_params.get("gemini_cache_api_base_url", "") or ""),
            request_metrics_path=(paths.memory_dir / "planner_request_metrics.jsonl"),
        )
    else:
        vlm = VlmSettings(
            provider=llm.provider,
            base_url=(hpc.service_urls.vlm_service.rstrip("/") if hpc.service_urls.vlm_service else llm.base_url),
            request_path=llm.request_path,
            chat_completions_path=llm.chat_completions_path,
            messages_path=llm.messages_path,
            models_path=llm.models_path,
            model=llm.model,
            api_key_env=llm.api_key_env,
            api_key_pool_env=llm.api_key_pool_env,
            api_key=llm.api_key,
            auth_header_name=llm.auth_header_name,
            auth_header_prefix=llm.auth_header_prefix,
            extra_headers=llm.extra_headers,
            timeout_s=float(vlm_params.get("timeout_s", 60.0)),
            max_retries=int(vlm_params.get("max_retries", 2)),
            temperature=float(vlm_params.get("temperature", 0.0)),
            max_tokens=int(vlm_params.get("max_tokens", 800)),
            use_vision=bool(vlm_params.get("use_vision", True)),
            strip_think_tags=bool(vlm_params.get("strip_think_tags", True)),
            use_tools=bool(vlm_params.get("use_tools", False)),
            image_max_side=int(vlm_params.get("image_max_side", 768)),
            image_format=str(vlm_params.get("image_format", "jpeg") or "jpeg"),
            jpeg_quality=int(vlm_params.get("jpeg_quality", 70)),
            prompt_cache_enabled=bool(vlm_params.get("prompt_cache_enabled", True)),
            prompt_cache_force=bool(vlm_params.get("prompt_cache_force", True)),
            prompt_cache_min_chars=int(vlm_params.get("prompt_cache_min_chars", 4096)),
            prompt_cache_ttl_s=int(vlm_params.get("prompt_cache_ttl_s", 3600)),
            prompt_cache_dir=str(vlm_params.get("prompt_cache_dir", "") or ""),
            gemini_cache_api_base_url=str(vlm_params.get("gemini_cache_api_base_url", "") or ""),
            request_metrics_path=(paths.memory_dir / "planner_request_metrics.jsonl"),
        )

    profile = (runtime.endpoint_profile or "api").strip().lower()
    if profile not in ("api", "hpc"):
        raise ValueError(f"runtime.endpoint_profile must be 'api' or 'hpc', got: {runtime.endpoint_profile!r}")
    if profile == "hpc":
        llm_url = (hpc.service_urls.llm_service or "").strip().rstrip("/")
        vlm_url = (hpc.service_urls.vlm_service or "").strip().rstrip("/")
        if llm_url:
            llm = replace(llm, base_url=llm_url)
        if vlm_url:
            vlm = replace(vlm, base_url=vlm_url)

    return EpmSettings(dish_id=dish_id, paths=paths, capture=capture, runtime=runtime, brain=brain, llm=llm, vlm=vlm, hpc=hpc)

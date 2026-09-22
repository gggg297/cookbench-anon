# Config Description

This file summarizes the most common config fields and how to override them.

## How overrides work

Two styles are supported:

1) Write-back (updates the JSON file before running):
   - `--runtime.stm_window_size 20`
   - `--runtime.stm_window_size=20`

2) Run-only (does not write back):
   - `--stm_window_size 20`

Write-back overrides accept dot-path keys and values are parsed as JSON when
possible (number/bool/array/object), otherwise treated as string.

## Top-level fields

- `dish_id`: numeric id of the dish in the recipes file.
- `paths.*`: file paths used by the run.
- `capture.*`: window title and activation behavior.
- `runtime.*`: run loop behavior (stm_window_size, max_steps, etc).
- `brain.*`: planner/agent behavior (pipeline, perception, plan_steps, etc).
- `llm.*` / `vlm.*`: model settings.
- `api_providers` / `api_model_assignments`: model routing.

## Common overrides

- `--runtime.stm_window_size 20`
- `--runtime.max_steps 4000`
- `--brain.plan_steps 10`
- `--brain.perception_mode oracle`
- `--vlm.use_tools true`

## Notes

- The run uses the file you pass to `--config`.
- If `configs/common.local.json` exists, it is automatically merged last as a local machine override.
- Recommended workflow: keep shared defaults in `common.json`, and put per-machine paths / email settings in `common.local.json`.
- A starter template is provided at `configs/common.local.example.json`.
- Local API key priority can be defined in `configs/api_keys.local.json`; see `configs/api_keys.local.example.json`.
- If you see invalid JSON errors, check trailing commas and quotes.

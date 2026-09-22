# CookBench: Embodied Cooking Agents in Cooking Simulator

Reference implementation and experiment harness for **CookBench**, a benchmark for
long-horizon embodied cooking agents. The agent observes a live *Cooking Simulator*
session, plans with an LLM/VLM, and acts through a semantic action + skill layer.

This repository is released for **double-blind review**; author and institution
information is intentionally omitted.

> **Not included: in-game template assets.** The perception and GUI layers match
> screenshots against small icon crops cropped from the game. Those crops are
> third-party game assets and are **not redistributed here**; the code regenerates
> them from your own installation. See
> [Template assets](#template-assets-not-redistributed).

## Layout

```text
.
  pyproject.toml          # package metadata (src layout)
  README.md
  configs/                # experiment configuration (JSON)
    common.json           # base config, extended by the pipeline configs
    pipelines/*.json      # one config per baseline mechanism
    cookbench_game_lock.json
    docs/config_desc.md   # config field reference and CLI override syntax
  data/                   # knowledge base (recipes, items, tools, nav map)
  memory/                 # prompt assets and documented-memory templates
  docs/                   # environment version-locking and setup notes
  scripts/                # CLI entry points
  src/epm/                # importable package
    brain/                # planner, prompt building, pipelines, modules
    cerebellum/           # action API, GUI flows, skills
    core/                 # agent loop, config loading, ablation control
    kb/                   # recipe access
    memory_store/         # documented memory (STM window, long-horizon history)
    vision/               # screen capture, template matching
    world_adapter/        # environment observation
```

## Installation

Requires Python >= 3.9 and a local, licensed copy of *Cooking Simulator*.

```powershell
cd <path-to-this-repo>
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -e .
```

Optional extras: `pip install -e ".[fast]"` (SciPy, speeds up template matching),
`.[viz]` (matplotlib, live navigation visualisation), `.[win32]` (pywin32, optional
annotation capture tool).

## Configuration

Configuration is plain JSON. A pipeline config uses `"extends"` to inherit from
`configs/common.json`; relative paths resolve against the file that declares them.
`configs/docs/config_desc.md` lists every field and the `--key value` /
`--runtime.key value` CLI override forms.

**1. Point the harness at your game data.** Set `paths.game_userdata_root` in the
config you intend to run (e.g. `configs/pipelines/planner_executor.json`):

```json
{ "paths": { "game_userdata_root": "<steam-library>/steamapps/common/CookingSimulator/UserData" } }
```

The live-state files (`realtime_products.json`, `realtime_radar_scan.txt`,
`realtime_camera_info.txt`, `realtime_interaction_status.txt`) are derived from this
root. You can also set it without editing files, via the environment variable
`COOKGAME_USERDATA_ROOT`.

**2. Choose a model provider.** `configs/common.json` ships with neutral
`template_openai_compatible`, `template_anthropic_messages`, and `local` provider
stubs. Copy `configs/api_keys.local.example.json` to `configs/api_keys.local.json`
(git-ignored), fill in your own endpoint, and select it in `api_model_assignments`.
Prefer environment variables (`api_key_env` / `api_key_pool_env`) over inline keys.

> **Never commit real credentials.** `configs/*.local.json`, `.env`, and `runs/` are
> git-ignored. Verify with `git status --porcelain` before pushing.

**3. Start the game.** Launch Cooking Simulator, enter a stable, repeatable kitchen
scene (e.g. sandbox mode), and ensure the window title matches `capture.window_title`
(default `CookingSimulator`). Keep the window visible and unminimised: capture uses
`mss` + `pygetwindow`, and the harness re-activates the window each step so input
stays focused.

## Running an episode

```powershell
# run until the environment reports completion, or up to a step cap
python scripts/run_episode.py --config configs/pipelines/planner_executor.json --dish-id 1 --max-steps 200

# resume an interrupted run
python scripts/run_episode.py --config configs/pipelines/planner_executor.json --dish-id 1 --resume --refresh-memory

# live dashboard view
python scripts/run_episode_dashboard.py --config configs/pipelines/epm.json --dish-id 1 --run-name demo
```

Global emergency stop hotkey: `Ctrl+Alt+Q`.

Recipe IDs come from `data/classics1.json`, a list of objects with
`id` / `dish_name` / `ingredients` / `recipe`:

```bash
python -c "import json; d=json.load(open('data/classics1.json',encoding='utf-8')); print([(x['id'],x['dish_name']) for x in d[:20]])"
```

### Outputs

Each run writes to `runs/<run_name>/`:

| Path | Contents |
| --- | --- |
| `memory/task_progress.txt` | task progress / milestones (optional atomic plan) |
| `memory/stm_window.txt` | short-term memory window (rolling) |
| `memory/long_horizon_history.txt` | full per-step record (JSONL) |
| `memory/prompt_last.txt` | last step's prompt (useful for debugging) |
| `memory/effective_prompt_ablation.json` | resolved ablation profile for the run |
| `screenshots/step_*.png` | per-step screenshots |

`runs/` is git-ignored.

## Baselines

`brain.pipeline` selects the control-flow mechanism. All baselines share the same
environment, action API, and memory store, so only the Brain changes.

| `brain.pipeline` | Config | Mechanism |
| --- | --- | --- |
| `planner_executor` | `pipelines/planner_executor.json` | rolling-horizon plan-and-execute; stops and replans on failure |
| `open_loop` | `pipelines/open_loop.json` | plan once, execute all steps even after a failure |
| `react` | `pipelines/react.json` | ReAct, one action/skill per step |
| `reflexion` | `pipelines/reflexion.json` | ReAct + Reflexion, reflections written back to memory |
| `epm` (alias `epm_agent`) | `pipelines/epm.json` | **ours** — rolling planner + precondition gating + semantic task-progress maintenance |
| `cap` | `pipelines/cap.json` | Code-as-Policies: generate a constrained Python policy that yields actions |

Related switches: `brain.perception_mode` (`oracle` / `vlm` / `none`),
`brain.planner_mode` (`scripted` / `llm` / `vlm`), and `brain.plan_steps`
(rolling-horizon plan length).

## Main experiment: prompt ablation

The main experiment has eight ablated profiles plus a raw-control baseline, selected
by the single `--ablation-reduce` argument:

```powershell
python scripts/run_episode.py --config configs/pipelines/epm.json --ablation-reduce no_body,no_strategy
```

| Label | Prompt group removed |
| --- | --- |
| `no_body` | Body rules, including common-sense interaction constraints |
| `no_perception_sup` | Oracle object list, placement occupancy, tool interaction locations |
| `no_strategy` | Static planning strategy and experience notes |
| `no_history` | Short-term window, task progress, resume state, cached cross-step queries |
| `no_feedback` | Last-action execution feedback and precondition feedback |
| `no_rgb` | RGB screenshots are not attached to any model request |
| `no_skill` | All high-level skills; semantic action APIs remain |
| `no_action` | Semantic action APIs and skills; replaced by nine raw control primitives |

Omitting the argument runs `full`. `prompt_layout.json` controls rendering order
only. Each run records its resolved profile in
`memory/effective_prompt_ablation.json`.

## Documented memory

Planning and execution state are recorded separately and aligned via `plan_ref`, so
planned steps can be compared against what actually executed:

- task progress (todo list + atomic plan) → `memory/task_progress.txt`
- execution (short window + full JSONL) → `memory/stm_window.txt`, `memory/long_horizon_history.txt`

`memory/` holds shared templates and stable artifacts — `tools_manifest_openai.json`,
`agent_state.template.json`, `body_rules.txt`, `strategy_notes.txt`, the
`prompt_layout.json` assembly template, and per-module prompt assets under
`memory/epm/`. See [`memory/README.md`](memory/README.md).

## Skills

Skills live under `src/epm/cerebellum/skills/`: `auto_navigation/`, `auto_cutting/`,
`auto_pouring/`, `auto_sprinkling/`, `auto_flipping/`, `auto_mix/`,
`auto_perception/`, plus query skills (`query_scene_objects`,
`query_product_catalog`, `query_tool_manual`, `query_tool_en`,
`list_supported_items`).

## Reproducing the environment

Exact game/depot versions used for the experiments, and the procedure for
reconstructing that environment, are in
[`docs/game_environment_lock.md`](docs/game_environment_lock.md).
`scripts/manage_game_environment.py` automates depot download / import / verify.

## Template assets (not redistributed)

The perception layer and several GUI flows match the screen against small reference
icons cropped from the game (item icons, store buttons, checkout UI, etc.). These
are third-party game assets, so they are **not** shipped in this repository. The
code expects them under `data/figure/`:

```text
data/figure/
  computer/            # game UI buttons and panels (submit, order, perks, ...)
  computer/order/
  store/               # store shelves and product icons
  object-icon/         # per-item icons used for matching
  object-icon-transparent/
```

Everything else in the harness — the agent loop, planner, prompt assembly, semantic
action API, memory model, and the non-visual skills — runs without these files.

**Regenerating them.** The crops are produced by screenshotting the running game at
a fixed resolution and cutting the UI regions the code documents in
`src/epm/cerebellum/gui_actions/` and `src/epm/cerebellum/figure_path_mappings.py`
(which lists every expected filename). Place the results in the layout above and the
template-matching paths work unchanged. Adjust the paths in
`figure_path_mappings.py` if you prefer to keep them elsewhere.

The harness drives the game through standard OS input and screen-capture APIs. It
ships no game code and contacts no network endpoint other than the model providers
you configure.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

# EPM Memory (文档化记忆)

本目录用于存放 EPM 运行时维护的“文档化记忆”，用于：
- 作为 prompt 片段提供给大脑（LLM/VLM）
- 作为可读、可复现、可对比的实验记录

重要说明：
- 在默认运行脚本 `scripts/run_episode.py` 中，每次运行的 memory 目录位于：`runs/<run_name>/memory/`
- `memory/` 更偏向“稳定工件/模板/跨 run 共享”的位置（例如 `tools_manifest_openai.json`、`agent_state.json`）
- 下文描述的文件名与结构，对任意 `memory_dir` 都成立（无论它在 `runs/...` 还是 `memory/`）

## 文件类型约定

- `*_rules.*`：人工维护的静态规则（不自动更新）
- 其余文件：运行时动态维护（每 step 覆盖或增量追加，按文件内说明）

## 常见文件（memory_dir 下）

- `stm_window.txt`：短期记忆（最近 N 步交互窗口，滚动覆盖）
- `task_progress.txt`：任务进度/里程碑（可包含 atomic_plan 对齐执行）
- `long_horizon_history.txt`：全量交互记录（JSONL，每 step 追加）
- `resume_state.json`：断点续跑状态（最后一步、状态、更新时间）
- `prompt_last.txt`：最近一次发给 planner 的完整 prompt（便于 debug；通常位于 `runs/<run_name>/memory/`，repo 级 `memory/prompt_last.txt` 可能是历史遗留，不保证是最新）。
- `body_rules.txt`：全局规则（注入 planner system prompt）
- `strategy_notes.txt`：策略备注/经验知识（注入 planner system prompt）
- `reflexion_memory.txt`：Reflexion 反思记忆（Reflexion baseline 失败后追加）
- `agent_state.json`：智能体状态快照（动态覆盖；也可作为跨 run 共享）
- `parameter_heuristics.json`：参数经验（稳定默认值/范围；可人工维护或低频更新）
- `resume_state.json`：断点续跑状态（最后一步、状态、更新时间）

## Prompt Layout（prompt_layout.json）

`memory/prompt_layout.json` 用来控制 prompt 里“注入哪些模块”以及它们的顺序。该文件是 repo 级稳定配置，运行时会通过 `brain.prompt_layout_path` 读取（默认指向 `memory/prompt_layout.json`）。

### 关键字段

- `observation_mode`
  - `image_only`：Observation 只声明“有截图图片作为输入”，不再输出 `time/frame_id/screenshot_path/state_keys/num_objects` 等元信息。 
  - 其它值：会回退到输出这些元信息（用于 debug）。
- `skill_cards_mode`
  - `all`：注入全部 Skill Cards 类别（信息更全，prompt 更长）。
  - `selected`：只注入与当前目标/配方/反馈相关的少量 Skill Cards（默认路由规则在 `src/epm/brain/skills_prompt.py`）。
- `system_sections`：SYSTEM 段落注入模块顺序。
- `user_sections`：USER 段落注入模块顺序。
- `limits.tools_manifest_max_chars`：工具清单注入的最大字符数（防止 prompt 过长）。
- `limits.skill_cards_max_chars`：Skill Cards 注入的最大字符数（防止 prompt 过长，超出会截断并打标记）。
- `limits.skill_cards_max_cards`：`skill_cards_mode=selected` 时的最多卡片数量；`0` 表示不限制。

### 可用模块（sections）

SYSTEM 可用：
- `body_rules`：注入 `body_rules.txt`（会自动移除 legacy 的 `[Capabilities - cookbench_api list]` 块）。
- `strategy_notes`：注入 `strategy_notes.txt`（规划指导、常识与经验类提示）。
- `reflexion_memory`：注入 `reflexion_memory.txt`（短反思/经验）。
- `tools_manifest_openai`：注入 `tools_manifest_openai.json`（以可读文本列出 actions/skills 与参数）。
- `skill_cards`：注入“技能卡片”视图（从 `skills_catalog.json` 里选取与当前目标最相关的少量类别，并展开其允许的 actions），用于降低工具搜索空间。
- `skill_specs`：注入内置 skills 的签名说明。

USER 可用：
- `goal`：高层目标 id + 目标文本。
- `recipe_text`：当前菜品 recipe 原文。
- `observation`：观测输入（受 `observation_mode` 控制）。
- `agent_state`：注入 `agent_state.json`（比如手上拿的物体、模式标记等）。
- `feedback`：上一步执行反馈（success/failure + 结构化提示）。
- `stm_window`：短期记忆窗口（最近 N 步）。
- `percept_text`：感知摘要文本（只有在 `prompt_policy.include_percept=true` 时才会在调用处提供）。
：参考计划/范例（如果启用注入）。
- `task_progress`：任务进度快照（只有在 `prompt_policy.include_task_progress=true` 时才会注入）。

## 实验运行记录（开始前清空 + 结束拷贝）

目标：在不依赖外部 agent 框架的情况下，保证每次实验可复现、可对比、可追踪。

### 1) 建议的实验产出目录

建议把每次实验的 run 输出固定落在：
- `runs/<run_name>/`

其中：
- `memory/`：运行时记忆与日志
- `screenshots/`：逐步截图

`run_name` 建议包含时间戳与 dish_id（你也可以手动通过 CLI 的 `--run-name` 指定）。

### 2) 实验结束时建议拷贝归档的文件

至少归档以下文件（从本次 run 的 `memory_dir` 拷贝到你的归档目录）：
- `long_horizon_history.txt`
- `task_progress.txt`
- `stm_window.txt`

另外建议归档最新菜品反馈（如果存在）：
- `<game_userdata_root>/recipe_feedback_latest.json`

### 3) 下一次实验开始前清空（保留框架/提示文本）

只清空“本次 run 的记录内容”，不要删掉模板/提示字段，否则解析或写入可能出错。

#### `task_progress.txt`
- `last_updated: ""`
- `task.*` 清空为 `""`
- `planning.plan_version: 1`（固定默认值）
- `planning.*` 其余字段清空为 `""`
- `high_level_todos: []`
- `blockers: []`

#### `stm_window.txt`
- `last_updated: ""`
- `window_size: <runtime.stm_window_size>`
- `steps: []`
- `loop_detection` 重置：
  - `loop_detected: false`
  - `consecutive_failures: 0`
  - `last_error_types: []`

#### `long_horizon_history.txt`
- 保留所有以 `#` 开头的说明行
- 清空所有 JSONL 记录行

### 4) 与 memory_store 的更新机制对齐

这些文件主要由 `src/epm/memory_store/store.py` 更新：
- `stm_window.txt`: 每步覆盖写入（窗口大小由 `runtime.stm_window_size` 控制）
- `long_horizon_history.txt`: 每步追加 JSONL
- `task_progress.txt`: 运行时会写入/更新（包含 atomic_plan 对齐与执行结果标注）

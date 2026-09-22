# local_actions.py
from typing import Tuple, Literal, Optional
import time
import inspect
import sys
import json
import os
import re
from pathlib import Path
import pygetwindow as gw
import mss
from PIL import Image
from epm.cerebellum.raw_input_controller import RawInputController
from epm.vision.screen_capture import activate_window as _activate_window
from epm.vision.screen_capture import get_window_rect
import ctypes
from epm.cerebellum.figure_path_mappings import AdvancedFunctionPaths  # 导入图片路径映射 

# 初始化一个全局的输入控制器
io_controller = RawInputController()

FAUCET_FILLING_UI_TIMEOUT_S = 6.0
FAUCET_FILLING_UI_POLL_S = 0.05
FAUCET_FILLING_MODE_CONTEXT_CHECK_TIMEOUT_S = 0.6
FAUCET_FILLING_ALIGN_THRESHOLD_M = 0.05
FAUCET_FILLING_ALIGN_STEP_PX = 20
FAUCET_FILLING_ALIGN_MAX_STEPS = 40
FAUCET_PIPE_CENTER_MAX_DIST_PX = 220.0
FAUCET_DIRECTION_CHANGE_TIMEOUT_S = 3.0
FAUCET_DIRECTION_CHANGE_POLL_S = 0.05
FAUCET_DIRECTION_CHANGE_MIN_DELTA_M = 0.03
FAUCET_FILLING_TARGET_TIMEOUT_S = 20.0
FAUCET_FILLING_TARGET_POLL_S = 0.05
FAUCET_FILLING_PROGRESS_DETECT_TIMEOUT_S = 3.0
FAUCET_FILLING_TARGET_TOLERANCE_ML = 1.0
FAUCET_FILLING_STOP_SETTLE_S = 0.15
DRAWER_STATE_TIMEOUT_S = 1.5
DRAWER_STATE_POLL_S = 0.05
DRAWER_CENTER_MAX_DIST_PX = 260.0
DOOR_CENTER_MAX_DIST_PX = 260.0
INTERACTION_TARGET_MAX_DISTANCE_M = 2.0
DOOR_OPEN_ANGLE_TRACE_TIMEOUT_S = 3.0

##########################################################
##                                                      ##
##              内部辅助函数 (不对AI暴露)                ##
##                                                      ##
##########################################################


def _coerce_mouse_delta(value) -> int:
    """Normalize LLM/runtime-provided mouse deltas like '220' or 220.0."""
    if isinstance(value, bool):
        raise TypeError(f"mouse delta must be numeric, got bool: {value!r}")
    try:
        return int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"mouse delta must be numeric, got {value!r}") from exc


def _coerce_nonnegative_duration(value) -> float:
    """Normalize string/float durations passed from runtime or planner."""
    if isinstance(value, bool):
        raise TypeError(f"duration must be numeric, got bool: {value!r}")
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"duration must be numeric, got {value!r}") from exc
    return max(0.0, duration)


def _coerce_scroll_delta(value) -> float:
    """Normalize wheel deltas while preserving fractional values."""
    if isinstance(value, bool):
        raise TypeError(f"scroll delta must be numeric, got bool: {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"scroll delta must be numeric, got {value!r}") from exc


# # 【新增】获取当前命令行窗口的函数
# def _get_console_window():
#     """获取当前Python脚本所在的命令行窗口。"""
#     # 获取与当前进程关联的控制台窗口的句柄
#     hwnd = ctypes.windll.kernel32.GetConsoleWindow()
#     if hwnd == 0:
#         return None
#     try:
#         # 使用pygetwindow通过句柄找到窗口对象
#         return gw.getWindow(hwnd)
#     except Exception:
#         # 如果gw找不到，尝试用标题（可能不准）
#         # 注意：这部分可能需要根据您的终端（cmd, powershell, vscode terminal）调整
#         titles = ["Cursor"]
#         for title in titles:
#             wins = gw.getWindowsWithTitle(title)
#             if wins: return wins[0]
#     return None

# 【新增】激活命令行窗口的函数
def focus_terminal():
    """将焦点切换到运行此脚本的命令行窗口。"""
    print("[*] 正在切换到命令行窗口等待输入...")
    try:
        console_win = _get_console_window()
        if console_win:
            if console_win.isMinimized:
                console_win.restore()
            console_win.activate()
            time.sleep(0.1) # 稍作等待确保焦点切换成功
        else:
            print("[!] 警告：未能自动找到命令行窗口。请手动点击。")
    except Exception as e:
        print(f"[!] 切换到终端时出错: {e}")



# 定义Windows API常量
SW_RESTORE = 9



    
def _get_window_rect(window_title='CookingSimulator'):
    """使用 pygetwindow 获取指定窗口的位置和大小。"""
    try:
        game_window = gw.getWindowsWithTitle(window_title)[0]
        return {'left': game_window.left, 'top': game_window.top, 'width': game_window.width, 'height': game_window.height}
    except IndexError:
        print(f"获取窗口 '{window_title}' 位置时出错，请确保游戏已运行。")
        # 返回一个默认值或抛出异常
        raise Exception("Game window not found.")

def _capture_screenshot_mss(region=None):
    """使用mss对指定区域截图，返回PIL Image对象。"""
    with mss.mss() as sct:
        monitor = region if region else sct.monitors[1]
        sct_img = sct.grab(monitor)
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

def _capture_screenshot_mss_numpy(region=None):
    """使用mss对指定区域截图，返回numpy数组并确保BGR格式。"""
    import numpy as np
    import cv2

    with mss.mss() as sct:
        # 处理扩展显示器问题
        if region is None:
            # 尝试自动检测游戏窗口在哪个显示器
            print(f"[DEBUG] 检测到 {len(sct.monitors)} 个监视器 (包括全屏)")
            for i, monitor in enumerate(sct.monitors):
                print(f"[DEBUG] Monitor {i}: {monitor}")

            # 默认使用第一个显示器
            monitor_to_use = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            print(f"[DEBUG] 使用显示器: {monitor_to_use}")
            screenshot = sct.grab(monitor_to_use)
        else:
            # 智能检测窗口region是否超出了主显示器边界
            print(f"[DEBUG] 使用自定义区域: {region}")

            # 检查region是否需要调整到正确的显示器坐标系
            # 某些情况下，窗口在扩展显示器上的坐标可能为负数或很大的正数
            adjusted_region = region.copy() if isinstance(region, dict) else region

            # 如果left坐标为负数或很大，可能在扩展显示器上
            if 'left' in adjusted_region and adjusted_region['left'] < 0:
                print(f"[DEBUG] 检测到负left坐标，可能在扩展显示器上")
            elif 'left' in adjusted_region and adjusted_region['left'] > 3000:
                print(f"[DEBUG] 检测到大left坐标，可能在扩展显示器上")

            screenshot = sct.grab(adjusted_region)

        screenshot = np.array(screenshot)  # 转换为 NumPy 数组
        print(f"[DEBUG] 截图尺寸: {screenshot.shape}")

        # mss返回的是BGRA格式，但原始代码期望BGR，所以我们转换一下
        if screenshot.shape[2] == 4:  # BGRA
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            print(f"[DEBUG] 转换后尺寸: {screenshot.shape}")

        return screenshot


def get_window_center(window_name: str='CookingSimulator'):
    """
    获取窗口的中心点（屏幕绝对坐标）。
    Args:
      window_name: 窗口标题（默认 CookingSimulator）。

    """
    try:
        window_rect = _get_window_rect(window_name)
        center_x = window_rect['left'] + window_rect['width'] // 2
        center_y = window_rect['top'] + window_rect['height'] // 2
        return {'x': center_x, 'y': center_y}
    except Exception as e:
        print(f"获取窗口中心时出错: {e}. 返回默认中心点 (640, 360).")
        return {'x': 640, 'y': 360}

## 之后再解开，只是简单的移动相对鼠标位置无法实现根据物品所在的像素位置来直接对准物品
# def smooth_move_to_position(
#     x: int,
#     y: int,
#     click: Optional[str] = None,
#     *,
#     steps: int = 1,
#     scale: float = 1.0,
#     sleep_s: float = 0.01,
# ) -> None:
#     """
#     将“窗口内像素坐标 (x,y)”转换为“准星中心的相对鼠标移动”，并可选点击。

#     说明：CookingSimulator 常见模式会锁定鼠标（准星固定在屏幕中心），此时绝对坐标移动鼠标并不能把光标移到屏幕某点。
#     因此这里采用“目标点相对窗口中心的偏移”作为相对移动量：
#       dx = x - center_x
#       dy = y - center_y

#     参数：
#     - steps：把移动拆成 N 次，减少大幅跳变（默认 1）
#     - scale：缩放因子（不同鼠标灵敏度下可调，默认 1.0）
#     - sleep_s：每小步之间等待（默认 0.01）
#     """
#     try:
#         window_rect = _get_window_rect()
#         center_x = int(window_rect["width"] // 2)
#         center_y = int(window_rect["height"] // 2)

#         dx = int((x - center_x) * float(scale))
#         dy = int((y - center_y) * float(scale))

#         n = int(steps) if int(steps) > 0 else 1
#         step_dx = int(dx / n)
#         step_dy = int(dy / n)
#         rem_dx = dx - step_dx * n
#         rem_dy = dy - step_dy * n

#         for i in range(n):
#             cur_dx = step_dx + (1 if i < abs(rem_dx) and rem_dx > 0 else 0) + (-1 if i < abs(rem_dx) and rem_dx < 0 else 0)
#             cur_dy = step_dy + (1 if i < abs(rem_dy) and rem_dy > 0 else 0) + (-1 if i < abs(rem_dy) and rem_dy < 0 else 0)
#             if cur_dx != 0 or cur_dy != 0:
#                 io_controller.mouse_move_relative(dx=cur_dx, dy=cur_dy)
#             if float(sleep_s) > 0:
#                 time.sleep(float(sleep_s))

#         if click:
#             io_controller.click(click)
#     except Exception as e:
#         print(f"平滑移动鼠标时出错: {e}")
        
##########################################################
##                                                      ##
##                 1. 基础键盘操作                      ##
##                                                      ##
##########################################################

def press_keyboard(key: Literal['w', 'a', 's', 'd', 'e', 'q', 'r', 'shift', 'ctrl']):
    """
    点击按键
    Args:
      key: 按键/鼠标键（例如 'left'/'right'/'mid' 或 'e'/'q' 等）。

    """
    io_controller.key_press(key)

def hold_keyboard(key: Literal['w', 'a', 's', 'd', 'shift', 'ctrl']):
    """
    按住按键
    Args:
      key: 按键/鼠标键（例如 'left'/'right'/'mid' 或 'e'/'q' 等）。

    """
    io_controller.key_down(key)

def leave_keyboard(key: Literal['w', 'a', 's', 'd', 'shift', 'ctrl']):
    """
    松开按键
    Args:
      key: 按键/鼠标键（例如 'left'/'right'/'mid' 或 'e'/'q' 等）。

    """
    io_controller.key_up(key)
    
def _agent_state_path() -> Path:
    env = os.environ.get("EPM_AGENT_STATE_PATH", "").strip()
    if env:
        return Path(env)
    # <repo>/epm/src/epm/cerebellum/local_actions.py -> parents[3] == <repo>/epm
    return Path(__file__).resolve().parents[3] / "memory" / "agent_state.json"


def _read_agent_state() -> dict:
    path = _agent_state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            path.write_text(content, encoding="utf-8")
        except Exception:
            return


def _write_agent_state(state: dict) -> None:
    path = _agent_state_path()
    _atomic_write_text(path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def _append_timer_event(events: list, *, timer_id: str, status: str, at_ts: float) -> list:
    key = (str(timer_id), str(status))
    for ev in events:
        if isinstance(ev, dict) and str(ev.get("id")) == key[0] and str(ev.get("status")) == key[1]:
            return events
    events.append({"id": key[0], "status": key[1], "at_ts": float(at_ts)})
    return events


def countdown_time(duration: float = 5.0):
    """
    倒计时操作（非阻塞，仅登记计时器）
    Args:
      duration: 持续时间（秒）。

    """
    try:
        duration_s = float(duration)
    except Exception:
        duration_s = 0.0
    if duration_s < 0:
        duration_s = 0.0

    now = time.time()
    timer_id = f"timer_{time.time_ns()}"
    deadline_ts = now + duration_s

    state = _read_agent_state()
    timers = state.get("timers")
    if not isinstance(timers, list):
        timers = []
    timers.append(
        {
            "id": timer_id,
            "start_ts": float(now),
            "duration": float(duration_s),
            "deadline_ts": float(deadline_ts),
            "status": "running",
        }
    )
    state["timers"] = timers
    _write_agent_state(state)

    return {
        "success": True,
        "timer_id": timer_id,
        "start_ts": float(now),
        "duration": float(duration_s),
        "deadline_ts": float(deadline_ts),
    }


def wait_for_timer(timer_id: Optional[str] = None, timeout_s: float = 60.0):
    """
    等待计时器完成（可选阻塞）。
    Args:
      timer_id: 计时器ID。为空时仅在存在唯一运行计时器时自动使用。
      timeout_s: 最大等待时间（秒）。

    """
    try:
        timeout_s = float(timeout_s)
    except Exception:
        timeout_s = 0.0
    if timeout_s < 0:
        timeout_s = 0.0

    start = time.time()

    while True:
        state = _read_agent_state()
        timers = state.get("timers")
        if not isinstance(timers, list):
            timers = []
        events = state.get("timer_events")
        if not isinstance(events, list):
            events = []

        running = [t for t in timers if isinstance(t, dict) and str(t.get("status", "running")) == "running"]
        if timer_id is None:
            if len(running) == 1:
                timer_id = str(running[0].get("id", "")).strip()
            elif len(running) == 0:
                return {"success": False, "error": "timer_not_found"}
            else:
                return {
                    "success": False,
                    "error": "timer_id_required",
                    "candidates": [str(t.get("id")) for t in running if str(t.get("id", "")).strip()],
                }

        # Check existing events first.
        for ev in events:
            if not isinstance(ev, dict):
                continue
            if str(ev.get("id")) != str(timer_id):
                continue
            status = str(ev.get("status") or "")
            if status == "done":
                return {"success": True, "timer_id": str(timer_id), "status": "done", "waited_s": time.time() - start}
            if status == "timeout":
                return {"success": False, "error": "timer_timeout", "timer_id": str(timer_id), "status": "timeout"}

        timer = None
        for t in timers:
            if isinstance(t, dict) and str(t.get("id")) == str(timer_id):
                timer = t
                break
        if timer is None:
            return {"success": False, "error": "timer_not_found", "timer_id": str(timer_id)}

        now = time.time()
        deadline_ts = timer.get("deadline_ts")
        if deadline_ts is None:
            try:
                deadline_ts = float(timer.get("start_ts", 0.0)) + float(timer.get("duration", 0.0))
                timer["deadline_ts"] = float(deadline_ts)
            except Exception:
                deadline_ts = None

        if deadline_ts is not None and now >= float(deadline_ts):
            events = _append_timer_event(events, timer_id=str(timer_id), status="done", at_ts=now)
            timers = [t for t in timers if not (isinstance(t, dict) and str(t.get("id")) == str(timer_id))]
            state["timers"] = timers
            state["timer_events"] = events
            _write_agent_state(state)
            return {"success": True, "timer_id": str(timer_id), "status": "done", "waited_s": now - start}

        if now - start >= timeout_s:
            events = _append_timer_event(events, timer_id=str(timer_id), status="timeout", at_ts=now)
            state["timer_events"] = events
            _write_agent_state(state)
            return {
                "success": False,
                "error": "timer_timeout",
                "timer_id": str(timer_id),
                "status": "timeout",
                "waited_s": now - start,
            }

        time.sleep(0.2)

# def end_record_time(start_time):
#     """结束计时操作, 返回经过的时间"""
#     end_time = time.time()
#     return end_time - start_time

##########################################################
##                                                      ##
##                 1.A 基础导航操作                     ##
##                                                      ##
##########################################################

def move_forward(duration=0.3):
    """
    向前移动一小段距离
    Args:
      duration: 持续时间（秒）。

    """
    duration = _coerce_nonnegative_duration(duration)
    io_controller.key_down('w'); time.sleep(duration); io_controller.key_up('w')
def move_backward(duration=0.3):
    """
    向后移动一小段距离
    Args:
      duration: 持续时间（秒）。

    """
    duration = _coerce_nonnegative_duration(duration)
    io_controller.key_down('s'); time.sleep(duration); io_controller.key_up('s')
def move_left(duration=0.3):
    """
    向左移动一小段距离
    Args:
      duration: 持续时间（秒）。

    """
    duration = _coerce_nonnegative_duration(duration)
    io_controller.key_down('a'); time.sleep(duration); io_controller.key_up('a')
def move_right(duration=0.3):
    """
    向右移动一小段距离
    Args:
      duration: 持续时间（秒）。

    """
    duration = _coerce_nonnegative_duration(duration)
    io_controller.key_down('d'); time.sleep(duration); io_controller.key_up('d')

# 组合移动（8方向）
def move_forward_left(duration=0.3):
    """
    向左前方移动一小段距离
    Args:
      duration: 持续时间（秒）。

    """
    duration = _coerce_nonnegative_duration(duration)
    io_controller.key_down('w')
    io_controller.key_down('a')
    time.sleep(duration)
    io_controller.key_up('w')
    io_controller.key_up('a')

def move_forward_right(duration=0.3):
    """
    向右前方移动一小段距离
    Args:
      duration: 持续时间（秒）。

    """
    duration = _coerce_nonnegative_duration(duration)
    print("move_forward_right okok")
    io_controller.key_down('w')
    io_controller.key_down('d')
    time.sleep(duration)
    io_controller.key_up('w')
    io_controller.key_up('d')

def move_backward_left(duration=0.3):
    """
    向左后方移动一小段距离
    Args:
      duration: 持续时间（秒）。

    """
    duration = _coerce_nonnegative_duration(duration)
    io_controller.key_down('s')
    io_controller.key_down('a')
    time.sleep(duration)
    io_controller.key_up('s')
    io_controller.key_up('a')

def move_backward_right(duration=0.3):
    """
    向右后方移动一小段距离
    Args:
      duration: 持续时间（秒）。

    """
    duration = _coerce_nonnegative_duration(duration)
    io_controller.key_down('s')
    io_controller.key_down('d')
    time.sleep(duration)
    io_controller.key_up('s')
    io_controller.key_up('d')


##########################################################
##                                                      ##
##                 2. 基础鼠标操作                      ##
##                                                      ##
##########################################################

def click_mouse(key: Literal['left', 'right', 'mid', 'middle']):
    """
    点击鼠标
    Args:
      key: 按键/鼠标键（例如 'left'/'right'/'mid' 或 'e'/'q' 等）。

    """
    btn = str(key).strip().lower()
    if btn == "mid":
        btn = "middle"
    io_controller.click(btn)

def hold_mouse(key: Literal['left', 'right']):
    """
    按住鼠标
    Args:
      key: 按键/鼠标键（例如 'left'/'right'/'mid' 或 'e'/'q' 等）。

    """
    io_controller.mouse_down(key)

def leave_mouse(key: Literal['left', 'right']):
    """
    松开鼠标
    Args:
      key: 按键/鼠标键（例如 'left'/'right'/'mid' 或 'e'/'q' 等）。

    """
    io_controller.mouse_up(key)

# def move_mouse(x: int, y: int):
#     """移动鼠标到屏幕绝对坐标"""
#     io_controller.mouse_move_absolute(x, y)

def move_related_mouse(x: int, y: int):
    """
    相对移动鼠标（用于非视角转动）
    Args:
      x: 屏幕坐标（相对游戏窗口/画面）。
      y: 屏幕坐标（相对游戏窗口/画面）。

    """
    io_controller.mouse_move_relative(_coerce_mouse_delta(x), _coerce_mouse_delta(y))

def scroll_down_the_wheel(distance: int = 0.5):
    """
    向下滑动滚轮
    Args:
      distance: 参数 `distance`（用于 scroll_down_the_wheel）。

    """
    distance = _coerce_scroll_delta(distance)
    io_controller.scroll_wheel(-distance) # 向下滚动是负值

def scroll_up_the_wheel(distance: int = 0.5):
    """
    向上滑动滚轮
    Args:
      distance: 参数 `distance`（用于 scroll_up_the_wheel）。

    """
    distance = _coerce_scroll_delta(distance)
    io_controller.scroll_wheel(distance) # 向上滚动是正值

##########################################################
##                                                      ##
##                 3.1 角色姿势与移动                   ##
##                                                      ##
##########################################################

def kneel_down():
    """
    角色姿势调整动作。蹲下去
    Args:
    """
    key="ctrl"
    hold_keyboard(key)

def stand_up():
    """
    角色姿势调整动作。站起来
    Args:

    """
    key="ctrl"
    leave_keyboard(key)

def look_right(pixels=100):
    """
    角色视野调整动作。视野向右转动一段距离
    Args:
      pixels: 鼠标移动像素（用于视角调整）。

    """
    pixels = _coerce_mouse_delta(pixels)
    io_controller.mouse_move_relative(dx=pixels)

def look_left(pixels=100):
    """
    角色视野调整动作。视野向左转动一段距离
    Args:
      pixels: 鼠标移动像素（用于视角调整）。

    """
    pixels = _coerce_mouse_delta(pixels)
    io_controller.mouse_move_relative(dx=-pixels)

def look_up(pixels=100):
    """
    角色视野调整动作。视野向上转动一段距离
    Args:
      pixels: 鼠标移动像素（用于视角调整）。

    """
    pixels = _coerce_mouse_delta(pixels)
    io_controller.mouse_move_relative(dy=-pixels) # y轴向上通常是负值

def look_down(pixels=100):
    """
    角色视野调整动作。视野向下转动一段距离
    Args:
      pixels: 鼠标移动像素（用于视角调整）。

    """
    pixels = _coerce_mouse_delta(pixels)
    io_controller.mouse_move_relative(dy=pixels)

########################################################
#                                                      #
#              3.2 基础物体操作                        #
#                 (共10个函数)                         #
#                                                      #
########################################################


def pick_less(steps: int = 1):
    """
    Container multi-pick quantity adjustment action. Use this only while holding a container and aiming at a real on-screen item pile that already supports multi-selection. Scroll the mouse wheel upward to reduce how many items will be picked in one pickup. The currently selected items are highlighted by a yellow outline on screen. Do not use this on ingredient Source/spawn points; Source/spawn points do not support pick-more/pick-less quantity adjustment.
    Args:
      steps: Number of upward wheel steps used to reduce the current selection quantity.

    """
    steps = _coerce_scroll_delta(steps)
    scroll_up_the_wheel(steps)


def pick_more(steps: int = 1):
    """
    Container multi-pick quantity adjustment action. Use this only while holding a container and aiming at a real on-screen item pile that already supports multi-selection. Scroll the mouse wheel downward to increase how many items will be picked in one pickup. The currently selected items are highlighted by a yellow outline on screen. Do not use this on ingredient Source/spawn points; Source/spawn points do not support pick-more/pick-less quantity adjustment.
    Args:
      steps: Number of downward wheel steps used to increase the current selection quantity.

    """
    steps = _coerce_scroll_delta(steps)
    scroll_down_the_wheel(steps)

def pick_up():
    """
    拿起物品类动作。手没拿东西且视野对准物品的情况下，对物品执行拿起动作
    Args:

    """
    def _is_container_like(*, held_name_en: str, held_name_cn: str, held_kind: str) -> bool:
        k = str(held_kind or "").strip().lower()
        if k in {"container"}:
            return True
        name = (str(held_name_en or "") + " " + str(held_name_cn or "")).lower()
        tokens = (
            "pot",
            "pan",
            "bowl",
            "plate",
            "casserole",
            "tray",
            "bucket",
            "wok",
        )
        return any(t in name for t in tokens)

    pre_available, pre = _hold_snapshot()
    if pre_available and bool(pre.get("is_held", False)):
        held_name_en = str(pre.get("held_name_en") or "")
        held_name_cn = str(pre.get("held_name_cn") or "")
        held_kind = str(pre.get("held_kind") or "")
        if not _is_container_like(held_name_en=held_name_en, held_name_cn=held_name_cn, held_kind=held_kind):
            return {
                "success": False,
                "error": "pick_up_precheck_failed:already_holding_item",
                "hint": "put_down_current_item_first_or_use_container_pickup",
                "precheck": pre,
            }
        # Holding a container: allow pickup attempt into container flow.
        key = "left"
        click_mouse(key)
        ok2, post_available2, post2 = _wait_hold_state(expected_held=True, timeout_s=3.5, poll_s=0.06)
        if ok2:
            return {
                "success": True,
                "error": "",
                "is_held": True,
                "held_item": post2.get("held_name_en") or post2.get("held_name_cn") or "",
                "pickup_into_container": True,
                "postcheck": post2,
            }
        return {
            "success": False,
            "error": (
                "pick_up_post_check_failed:realtime_products_unavailable"
                if not post_available2
                else "pick_up_post_check_failed:not_holding_after_click"
            ),
            "hint": "aim_target_and_retry_or_use_pick_up_into_the_container",
            "precheck": pre,
            "postcheck": post2,
        }

    key = "left"
    click_mouse(key)

    ok, post_available, post = _wait_hold_state(expected_held=True, timeout_s=3.5, poll_s=0.06)
    if ok:
        return {
            "success": True,
            "error": "",
            "is_held": True,
            "held_item": post.get("held_name_en") or post.get("held_name_cn") or "",
            "postcheck": post,
        }
    return {
        "success": False,
        "error": (
            "pick_up_post_check_failed:realtime_products_unavailable"
            if not post_available
            else "pick_up_post_check_failed:not_holding_after_click"
        ),
        "hint": "align_crosshair_navigate_closer_and_retry_pick_up",
        "precheck": pre,
        "postcheck": post,
    }

def put_down():
    """
    拿起物品类动作。手拿了东西且对准放置位置的情况下，对物品执行放下动作
    Args:

    """
    pre_available, pre = _hold_snapshot()
    if pre_available and (not bool(pre.get("is_held", False))):
        return {
            "success": False,
            "error": "put_down_precheck_failed:hands_empty",
            "hint": "pick_up_item_before_put_down",
            "precheck": pre,
        }

    key = "left"
    retry_attempts = []

    def _attempt_put_down(*, look_down_pixels: int = 0):
        if int(look_down_pixels) > 0:
            look_down(look_down_pixels)
            time.sleep(0.08)
        click_mouse(key)
        ok, post_available, post = _wait_hold_state(expected_held=False, timeout_s=1.8, poll_s=0.06)
        mode_busy = _interaction_mode_active(post)
        attempt_error = (
            ""
            if ok and (not mode_busy)
            else (
                "put_down_post_check_failed:realtime_products_unavailable"
                if not post_available
                else (
                    "put_down_post_check_failed:interaction_mode_still_active"
                    if mode_busy
                    else "put_down_post_check_failed:still_holding_item"
                )
            )
        )
        return ok, post_available, post, mode_busy, attempt_error

    ok, post_available, post, mode_busy, error = _attempt_put_down()
    if ok and (not mode_busy):
        return {
            "success": True,
            "error": "",
            "is_held": False,
            "postcheck": post,
            "retry_attempts": retry_attempts,
        }

    if error == "put_down_post_check_failed:still_holding_item":
        cumulative_pixels = 0
        for delta_pixels in (50, 50):
            cumulative_pixels += delta_pixels
            retry_attempts.append(
                {
                    "type": "look_down_then_retry_put_down",
                    "delta_pixels": int(delta_pixels),
                    "cumulative_pixels": int(cumulative_pixels),
                }
            )
            ok, post_available, post, mode_busy, error = _attempt_put_down(look_down_pixels=delta_pixels)
            retry_attempts[-1]["success"] = bool(ok and (not mode_busy))
            retry_attempts[-1]["error"] = error
            if ok and (not mode_busy):
                return {
                    "success": True,
                    "error": "",
                    "is_held": False,
                    "postcheck": post,
                    "retry_attempts": retry_attempts,
                }

    hint = "choose_another_empty_place_point_and_retry_put_down"
    feedback_to_planner = None
    if (
        error == "put_down_post_check_failed:still_holding_item"
        and len(retry_attempts) >= 2
        and int(retry_attempts[-1].get("cumulative_pixels", 0)) >= 100
    ):
        hint = (
            "put_down_may_be_blocked_by_held_item_occlusion:"
            "choose_another_place_point_or_free_your_hands_first"
        )
        feedback_to_planner = (
            "Executing put_down may fail because the held item blocks the view of the place point. "
            "Please choose another place point, or use a discard/throw-away style action to free your hands first."
        )

    result = {
        "success": False,
        "error": error,
        "hint": hint,
        "precheck": pre,
        "postcheck": post,
        "retry_attempts": retry_attempts,
    }
    if feedback_to_planner:
        result["feedback_to_planner"] = feedback_to_planner
    return result

def repair():
    """
    维修动作。手持 Repair Phone 且视野对准损坏厨具时，对目标执行维修操作
    Args:

    """
    pre_available, pre = _hold_snapshot()
    if pre_available:
        held_name_en = str(pre.get("held_name_en") or "")
        held_name_cn = str(pre.get("held_name_cn") or "")
        held_name = f"{held_name_en} {held_name_cn}".strip().lower()
        is_repair_phone = (
            "repair phone" in held_name
            or "repairphone" in held_name
            or ("repair" in held_name and "phone" in held_name)
            or ("维修" in held_name and "电话" in held_name)
        )
        if not is_repair_phone:
            return {
                "success": False,
                "error": "repair_precheck_failed:not_holding_repair_phone",
                "hint": "pick_up_repair_phone_before_repair",
                "precheck": pre,
            }

    key = "left"
    click_mouse(key)
    return {
        "success": True,
        "error": "",
        "precheck": pre,
        "held_item": pre.get("held_name_en") or pre.get("held_name_cn") or "",
    }

def throw_away():
    """
    拿起物品类动作。手持物品情况下，将物品向视角前方扔出
    Args:

    """
    pre_available, pre = _hold_snapshot()
    if pre_available and (not bool(pre.get("is_held", False))):
        return {
            "success": False,
            "error": "throw_away_precheck_failed:hands_empty",
            "hint": "pick_up_item_before_throw_away",
            "precheck": pre,
        }

    key = "mid"
    click_mouse(key)

    ok, post_available, post = _wait_hold_state(expected_held=False, timeout_s=1.8, poll_s=0.06)
    if ok:
        return {
            "success": True,
            "error": "",
            "is_held": False,
            "postcheck": post,
        }
    return {
        "success": False,
        "error": (
            "throw_away_post_check_failed:realtime_products_unavailable"
            if not post_available
            else "throw_away_post_check_failed:still_holding_item"
        ),
        "hint": "retry_throw_away_or_try_put_down_at_valid_place",
        "precheck": pre,
        "postcheck": post,
    }

def discard_into_garbage_bin():
    """
    拿起物品类动作。手持物品且视野对准垃圾桶时，左键将手中之后不再需要的物品丢入垃圾桶。该操作不可恢复，需谨慎使用。前置检查要求手中持有物品；执行后会轮询等待最多3秒，后置检查要求恢复为空手。
    Args:

    """
    pre_available, pre = _hold_snapshot()
    if pre_available and (not bool(pre.get("is_held", False))):
        return {
            "success": False,
            "error": "discard_into_garbage_bin_precheck_failed:hands_empty",
            "hint": "pick_up_item_before_discard_into_garbage_bin",
            "precheck": pre,
        }

    key = "left"
    click_mouse(key)

    ok, post_available, post = _wait_hold_state(expected_held=False, timeout_s=3.0, poll_s=0.05)
    if ok:
        return {
            "success": True,
            "error": "",
            "is_held": False,
            "precheck": pre,
            "postcheck": post,
        }
    return {
        "success": False,
        "error": (
            "discard_into_garbage_bin_post_check_failed:realtime_products_unavailable"
            if not post_available
            else "discard_into_garbage_bin_post_check_failed:still_holding_item"
        ),
        "hint": "align_to_garbage_bin_and_retry_discard_into_garbage_bin",
        "precheck": pre,
        "postcheck": post,
    }

def pick_up_into_the_container():
    """
    拿起物品类动作。手持容器时，视野对准物品的前提下，向容器内拾取物体
    Args:

    """
    key="left"
    click_mouse(key)

def skewer_hang_on_string():
    """
    签子交互动作。手拿签子且对准物品的情况下，串上食材
    Args:

    """
    key="left"
    click_mouse(key)

def skewer_remove_the_ingredients():
    """
    签子交互动作。当签子上有食物且对准放置位置的情况下，取下食材
    Args:

    """
    key="e"
    press_keyboard(key)

# # 函数 3.2.6 [总函数 22]: 工具位置回位
# def Tool_reposition(key="mid"):
#     """工具位置回位"""
#     click_mouse(key)

# # 函数 3.2.7 [总函数 23]: 垂直下扔
# def Vertically_throw(key="mid"):
#     """垂直下扔"""
#     click_mouse(key)
#     # key = mid

########################################################
#                                                      #
#              3.3 容器与倾倒操作                      #
#                 (共10个函数)                         #
#                                                      #
########################################################

def _read_realtime_products_quick(retries: int = 6, sleep_s: float = 0.05) -> dict | None:
    """Best-effort realtime_products reader for lightweight mode post-check."""
    try:
        from epm.cerebellum.skills._shared_paths import realtime_products_json
    except Exception:
        return None

    path = realtime_products_json()
    if not path.exists():
        return None

    last_err: Exception | None = None
    for _ in range(max(1, int(retries))):
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                raw = path.read_text(encoding=enc, errors="replace")
                obj = json.loads(raw) if raw.strip() else {}
                if isinstance(obj, dict):
                    return obj
            except Exception as e:
                last_err = e
                continue
        time.sleep(float(sleep_s))
    return None


def _mode_snapshot(data: dict | None) -> dict:
    products = []
    if isinstance(data, dict) and isinstance(data.get("products"), list):
        products = [x for x in data.get("products", []) if isinstance(x, dict)]
    elif isinstance(data, dict) and isinstance(data.get("objects"), list):
        products = [x for x in data.get("objects", []) if isinstance(x, dict)]

    held = None
    for p in products:
        if bool(p.get("is_held", False)):
            held = p
            break

    def _item_text(p: dict) -> str:
        return " ".join(
            str(p.get(k) or "").strip().lower()
            for k in ("name_en", "name_cn", "game_object")
        )

    def _is_cutting_board_like(p: dict) -> bool:
        text = _item_text(p)
        return any(token in text for token in ("cutting board", "chopping board", "菜板", "砧板", "cutting_board"))

    def _is_knife_like(p: dict) -> bool:
        return "knife" in _item_text(p)

    def _display_name(p: dict | None) -> str:
        if not isinstance(p, dict):
            return ""
        return str(p.get("name_en") or p.get("name_cn") or p.get("name") or "").strip()

    def _position_y(p: dict | None) -> float:
        if not isinstance(p, dict):
            return float("-inf")
        pos = p.get("position")
        if isinstance(pos, dict):
            try:
                return float(pos.get("y"))
            except Exception:
                return float("-inf")
        return float("-inf")

    def _pick_highest_y(items: list[dict]) -> dict | None:
        if not items:
            return None
        return max(
            items,
            key=lambda p: (
                _position_y(p),
                1 if bool(p.get("is_on_screen", False)) else 0,
                1 if bool(p.get("is_held", False)) else 0,
            ),
        )

    def _any_true(keys: tuple[str, ...]) -> bool:
        for p in products:
            for k in keys:
                if bool(p.get(k, False)):
                    return True
        return False

    cutting_entry = next(
        (
            p
            for p in products
            if _is_knife_like(p) and bool(p.get("is_cut_mode") or p.get("is_cutting") or p.get("is_cutting_mode"))
        ),
        None,
    )
    pouring_entry = _pick_highest_y(
        [
            p
            for p in products
            if (not _is_cutting_board_like(p)) and bool(p.get("is_pouring_mode"))
        ]
    )
    sprinkle_entry = next((p for p in products if bool(p.get("is_sprinkle_mode") or p.get("is_sprinkle"))), None)
    mixing_entry = next((p for p in products if bool(p.get("is_mixing_mode"))), None)
    flip_entry = next((p for p in products if bool(p.get("is_filp_mode") or p.get("is_flip_mode") or p.get("is_flipping_mode"))), None)

    knife_cutting_active = cutting_entry is not None
    pouring_active = (pouring_entry is not None) and (not knife_cutting_active)
    sprinkle_active = sprinkle_entry is not None
    mixing_active = mixing_entry is not None
    flip_active = flip_entry is not None

    dominant_mode = ""
    dominant_mode_item_name = ""
    if knife_cutting_active:
        dominant_mode = "cut"
        dominant_mode_item_name = _display_name(cutting_entry)
    elif pouring_active:
        dominant_mode = "pour"
        dominant_mode_item_name = _display_name(pouring_entry)
    elif sprinkle_active:
        dominant_mode = "sprinkle"
        dominant_mode_item_name = _display_name(sprinkle_entry)
    elif mixing_active:
        dominant_mode = "mix"
        dominant_mode_item_name = _display_name(mixing_entry)
    elif flip_active:
        dominant_mode = "flip"
        dominant_mode_item_name = _display_name(flip_entry)

    return {
        "held_name_en": (str(held.get("name_en")) if isinstance(held, dict) and held.get("name_en") is not None else ""),
        "held_name_cn": (str(held.get("name_cn")) if isinstance(held, dict) and held.get("name_cn") is not None else ""),
        "held_kind": (str(held.get("kind")) if isinstance(held, dict) and held.get("kind") is not None else ""),
        "is_held": bool(held is not None),
        "is_pouring_mode": bool(pouring_active),
        "is_sprinkle_mode": bool(sprinkle_active),
        "is_cutting_mode": bool(knife_cutting_active),
        "is_mixing_mode": bool(mixing_active),
        "is_flip_mode": bool(flip_active),
        "pouring_mode_item_name": _display_name(pouring_entry),
        "cutting_mode_item_name": _display_name(cutting_entry),
        "sprinkle_mode_item_name": _display_name(sprinkle_entry),
        "mixing_mode_item_name": _display_name(mixing_entry),
        "flip_mode_item_name": _display_name(flip_entry),
        "dominant_mode": dominant_mode,
        "dominant_mode_item_name": dominant_mode_item_name,
    }


def _hold_snapshot() -> tuple[bool, dict]:
    """
    Return (available, hold_snapshot) from realtime_products.
    `available=False` means post-check cannot be trusted.
    """
    data = _read_realtime_products_quick(retries=2, sleep_s=0.02)
    snap = _mode_snapshot(data)
    available = bool(
        isinstance(data, dict)
        and (
            isinstance(data.get("products"), list)
            or isinstance(data.get("objects"), list)
        )
    )
    return available, snap


def _wait_hold_state(expected_held: bool, timeout_s: float = 1.5, poll_s: float = 0.05) -> tuple[bool, bool, dict]:
    """
    Wait until hold state becomes expected_held.
    Returns (matched, realtime_available, last_snapshot).
    """
    deadline = time.time() + max(0.0, float(timeout_s))
    saw_available = False
    last: dict = {}
    while True:
        available, snap = _hold_snapshot()
        saw_available = bool(saw_available or available)
        last = snap
        if available and (bool(snap.get("is_held", False)) == bool(expected_held)):
            return True, True, snap
        if time.time() >= deadline:
            return False, bool(saw_available), last
        time.sleep(max(0.01, float(poll_s)))


def _interaction_mode_active(snap: dict) -> bool:
    return bool(
        snap.get("is_pouring_mode")
        or snap.get("is_sprinkle_mode")
        or snap.get("is_cutting_mode")
        or snap.get("is_mixing_mode")
        or snap.get("is_flip_mode")
    )


def _snapshot_active_modes(snap: dict) -> list[str]:
    modes: list[str] = []
    if not isinstance(snap, dict):
        return modes
    if bool(snap.get("is_cutting_mode")):
        modes.append("cut")
    if bool(snap.get("is_pouring_mode")):
        modes.append("pour")
    if bool(snap.get("is_sprinkle_mode")):
        modes.append("sprinkle")
    if bool(snap.get("is_mixing_mode")):
        modes.append("mix")
    if bool(snap.get("is_flip_mode")):
        modes.append("flip")
    return modes


def _held_item_allows_pouring(snap: dict) -> tuple[bool, str, str, str]:
    if not isinstance(snap, dict):
        return False, "held_item_unknown", "", ""
    held_name = str(snap.get("held_name_en") or snap.get("held_name_cn") or "").strip()
    held_kind = str(snap.get("held_kind") or "").strip().lower()
    text = f"{held_name} {snap.get('held_name_cn', '')}".strip().lower()
    if not bool(snap.get("is_held", False)):
        return False, "no_held_item", held_name, held_kind
    if any(token in text for token in ("cutting board", "chopping board", "菜板", "砧板", "cutting_board")):
        return False, "cutting_board_not_supported_for_pouring_mode", held_name, held_kind
    if held_kind in {"liquid", "liquids", "container", "containers"}:
        return True, "", held_name, held_kind
    return False, "unsupported_held_item_for_pouring_mode", held_name, held_kind


def _wait_interaction_mode_cleared(
    timeout_s: float = 1.5,
    poll_s: float = 0.05,
    settle_s: float = 0.0,
) -> tuple[bool, bool, dict]:
    """
    Wait until no known interaction-mode flag remains active.
    Returns (cleared, realtime_available, last_snapshot).
    """
    if float(settle_s) > 0:
        time.sleep(max(0.0, float(settle_s)))
    deadline = time.time() + max(0.0, float(timeout_s))
    saw_available = False
    last: dict = {}
    while True:
        available, snap = _hold_snapshot()
        saw_available = bool(saw_available or available)
        last = snap
        if available and (not _interaction_mode_active(snap)):
            return True, True, snap
        if time.time() >= deadline:
            return False, bool(saw_available), last
        time.sleep(max(0.01, float(poll_s)))


def _wait_mode_enabled(
    mode_key: str,
    timeout_s: float = 1.5,
    poll_s: float = 0.05,
    settle_s: float = 0.0,
) -> tuple[bool, dict]:
    """Wait until expected mode flag appears in realtime_products."""
    poll = max(0.01, float(poll_s))
    settle_deadline = time.time() + max(0.0, float(settle_s))
    last = {}
    # Settle phase: start polling immediately, but allow a short max settle window.
    while time.time() < settle_deadline:
        snap = _mode_snapshot(_read_realtime_products_quick(retries=2, sleep_s=0.02))
        last = snap
        if bool(snap.get(mode_key, False)):
            return True, snap
        time.sleep(poll)

    deadline = time.time() + max(0.0, float(timeout_s))
    while time.time() < deadline:
        snap = _mode_snapshot(_read_realtime_products_quick(retries=2, sleep_s=0.02))
        last = snap
        if bool(snap.get(mode_key, False)):
            return True, snap
        time.sleep(poll)
    return False, last


def _wait_mode_disabled(
    mode_key: str,
    timeout_s: float = 1.5,
    poll_s: float = 0.05,
    settle_s: float = 0.0,
) -> tuple[bool, bool, dict]:
    """Wait until expected mode flag disappears from realtime_products."""
    if float(settle_s) > 0:
        time.sleep(max(0.0, float(settle_s)))
    deadline = time.time() + max(0.0, float(timeout_s))
    saw_available = False
    last: dict = {}
    while True:
        available, snap = _hold_snapshot()
        saw_available = bool(saw_available or available)
        last = snap
        if available and (not bool(snap.get(mode_key, False))):
            return True, True, snap
        if time.time() >= deadline:
            return False, bool(saw_available), last
        time.sleep(max(0.01, float(poll_s)))


def _wait_hold_snapshot_condition_stable(
    predicate,
    timeout_s: float = 2.0,
    poll_s: float = 0.05,
    stable_s: float = 0.0,
) -> tuple[bool, bool, dict]:
    """
    Wait until a hold-snapshot predicate stays true for `stable_s` seconds.
    Returns (matched, realtime_available, last_snapshot).
    """
    deadline = time.time() + max(0.0, float(timeout_s))
    stable_needed = max(0.0, float(stable_s))
    stable_since: float | None = None
    saw_available = False
    last: dict = {}

    while True:
        available, snap = _hold_snapshot()
        saw_available = bool(saw_available or available)
        last = snap

        if available and bool(predicate(snap)):
            now = time.time()
            if stable_needed <= 0.0:
                return True, True, snap
            if stable_since is None:
                stable_since = now
            elif (now - stable_since) >= stable_needed:
                return True, True, snap
        else:
            stable_since = None

        if time.time() >= deadline:
            return False, bool(saw_available), last
        time.sleep(max(0.01, float(poll_s)))


def _wait_mode_condition_stable(
    mode_key: str,
    *,
    expected: bool = True,
    timeout_s: float = 2.4,
    poll_s: float = 0.05,
    stable_s: float = 2.0,
) -> tuple[bool, bool, dict]:
    """
    Wait until `mode_key` remains at `expected` for `stable_s` seconds.
    Returns (matched, realtime_available, last_snapshot).
    """
    return _wait_hold_snapshot_condition_stable(
        lambda state: bool(state.get(mode_key, False)) is bool(expected),
        timeout_s=timeout_s,
        poll_s=poll_s,
        stable_s=stable_s,
    )


def _read_ui_targets_dump_quick(retries: int = 4, sleep_s: float = 0.05) -> tuple[Path | None, dict | None]:
    try:
        from epm.cerebellum.skills._shared_paths import userdata_root
    except Exception:
        return None, None

    path = userdata_root() / "ui_targets_dump_latest.json"
    if not path.exists():
        return path, None

    for _ in range(max(1, int(retries))):
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                raw = path.read_text(encoding=enc, errors="replace")
                obj = json.loads(raw) if raw.strip() else {}
                if isinstance(obj, dict):
                    return path, obj
            except Exception:
                continue
        time.sleep(max(0.01, float(sleep_s)))
    return path, None


def _collect_ui_dump_texts(obj) -> list[str]:
    texts: list[str] = []

    def _walk(node) -> None:
        if isinstance(node, dict):
            value = node.get("text")
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
            for child in node.values():
                _walk(child)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(obj)
    return texts


def _trigger_ui_targets_dump() -> float | None:
    path, _ = _read_ui_targets_dump_quick(retries=1, sleep_s=0.01)
    prev_mtime = None
    try:
        if path is not None and path.exists():
            prev_mtime = path.stat().st_mtime
    except Exception:
        prev_mtime = None

    io_controller.key_down("alt")
    try:
        io_controller.key_press("y")
    finally:
        io_controller.key_up("alt")
    return prev_mtime


def _wait_ui_dump_contains_texts(
    required_texts: tuple[str, ...],
    *,
    previous_mtime: float | None = None,
    timeout_s: float = FAUCET_FILLING_UI_TIMEOUT_S,
    poll_s: float = FAUCET_FILLING_UI_POLL_S,
) -> tuple[bool, list[str], dict]:
    required = {str(t).strip().upper() for t in required_texts if str(t).strip()}
    deadline = time.time() + max(0.0, float(timeout_s))
    last_data: dict = {}
    last_texts: list[str] = []

    while True:
        path, data = _read_ui_targets_dump_quick(retries=1, sleep_s=0.01)
        last_data = data if isinstance(data, dict) else {}
        last_texts = _collect_ui_dump_texts(last_data)
        text_set = {t.upper() for t in last_texts}

        mtime_ok = True
        if previous_mtime is not None and path is not None and path.exists():
            try:
                mtime_ok = path.stat().st_mtime > previous_mtime
            except Exception:
                mtime_ok = True

        if required.issubset(text_set) and mtime_ok:
            return True, last_texts, last_data
        if time.time() >= deadline:
            return False, last_texts, last_data
        time.sleep(max(0.01, float(poll_s)))


def _ensure_faucet_fill_ui_ready(
    *,
    timeout_s: float = FAUCET_FILLING_UI_TIMEOUT_S,
    poll_s: float = FAUCET_FILLING_UI_POLL_S,
) -> tuple[bool, dict]:
    mode_ok, mode_snap = _wait_mode_enabled(
        "is_pouring_mode",
        timeout_s=timeout_s,
        poll_s=poll_s,
        settle_s=0.0,
    )
    if not mode_ok:
        return False, {
            "error": "faucet_fill_mode_not_ready:is_pouring_mode_not_detected",
            "mode_snap": mode_snap,
        }

    prev_ui_mtime = _trigger_ui_targets_dump()
    ui_ok, ui_texts, ui_data = _wait_ui_dump_contains_texts(
        ("RIGHT HANDLE", "LEFT HANDLE"),
        previous_mtime=prev_ui_mtime,
        timeout_s=timeout_s,
        poll_s=poll_s,
    )
    if not ui_ok:
        return False, {
            "error": "faucet_fill_mode_not_ready:faucet_handles_ui_not_detected",
            "mode_snap": mode_snap,
            "ui_texts_sample": ui_texts[:20],
            "ui_dump_keys": list(ui_data.keys())[:20] if isinstance(ui_data, dict) else [],
        }

    return True, {
        "mode_snap": mode_snap,
        "ui_handles_detected": ["RIGHT HANDLE", "LEFT HANDLE"],
    }


def _check_faucet_fill_mode_context(
    *,
    timeout_s: float = FAUCET_FILLING_MODE_CONTEXT_CHECK_TIMEOUT_S,
    poll_s: float = FAUCET_FILLING_UI_POLL_S,
) -> tuple[bool, dict]:
    mode_snap = _mode_snapshot(_read_realtime_products_quick(retries=2, sleep_s=0.02))
    if not bool(mode_snap.get("is_pouring_mode", False)):
        return False, {
            "error": "faucet_fill_mode_context_not_ready:is_pouring_mode_not_detected",
            "mode_snap": mode_snap,
        }

    prev_ui_mtime = _trigger_ui_targets_dump()
    ui_ok, ui_texts, ui_data = _wait_ui_dump_contains_texts(
        ("RIGHT HANDLE", "LEFT HANDLE"),
        previous_mtime=prev_ui_mtime,
        timeout_s=timeout_s,
        poll_s=poll_s,
    )
    if not ui_ok:
        return False, {
            "error": "faucet_fill_mode_context_not_ready:faucet_handles_ui_not_detected",
            "mode_snap": mode_snap,
            "ui_texts_sample": ui_texts[:20],
            "ui_dump_keys": list(ui_data.keys())[:20] if isinstance(ui_data, dict) else [],
        }

    return True, {
        "mode_snap": mode_snap,
        "ui_handles_detected": ["RIGHT HANDLE", "LEFT HANDLE"],
    }


def _realtime_entries(data: dict | None) -> list[dict]:
    if isinstance(data, dict) and isinstance(data.get("products"), list):
        return [x for x in data.get("products", []) if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("objects"), list):
        return [x for x in data.get("objects", []) if isinstance(x, dict)]
    return []


def _entry_text(entry: dict) -> str:
    return " ".join(
        str(entry.get(k) or "").strip().lower()
        for k in ("name_en", "name_cn", "name", "game_object")
    ).strip()


def _find_active_pouring_entry(data: dict | None) -> dict | None:
    for entry in _realtime_entries(data):
        if bool(entry.get("is_pouring_mode")):
            return entry
    return None


def _find_faucet_pipe_entry(data: dict | None) -> dict | None:
    for entry in _realtime_entries(data):
        text = _entry_text(entry)
        if any(token in text for token in ("faucet pipe", "水龙头出水管")):
            return entry
    return None


def _find_sink_place_point_entry(data: dict | None, *, side: Literal["left", "right"]) -> dict | None:
    side_tokens = ("left sink place point", "水槽左放置点") if side == "left" else ("right sink place point", "水槽右放置点")
    for entry in _realtime_entries(data):
        text = _entry_text(entry)
        if any(token in text for token in side_tokens):
            return entry
    return None


def _find_matching_faucet_pipe_entry(data: dict | None, reference_entry: dict | None) -> dict | None:
    if not isinstance(reference_entry, dict):
        return _find_faucet_pipe_entry(data)
    ref_instance_id = reference_entry.get("instance_id")
    if ref_instance_id not in (None, ""):
        for entry in _realtime_entries(data):
            if entry.get("instance_id") == ref_instance_id:
                return entry
    return _find_faucet_pipe_entry(data)


def _entry_position_xz(entry: dict | None) -> tuple[float, float] | None:
    if not isinstance(entry, dict):
        return None
    pos = entry.get("position")
    if not isinstance(pos, dict):
        return None
    try:
        return float(pos.get("x", 0.0)), float(pos.get("z", 0.0))
    except Exception:
        return None


def _entry_screen_xy(entry: dict | None) -> tuple[float, float] | None:
    if not isinstance(entry, dict):
        return None
    try:
        sx = float(entry.get("screen_x"))
        sy = float(entry.get("screen_y"))
        return sx, sy
    except Exception:
        return None


def _distance_xz(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    try:
        dx = float(a[0]) - float(b[0])
        dz = float(a[1]) - float(b[1])
        return (dx * dx + dz * dz) ** 0.5
    except Exception:
        return None


def _screen_center_xy(data: dict | None) -> tuple[float, float]:
    try:
        width = float((data or {}).get("_screen_width") or 1600.0)
        height = float((data or {}).get("_screen_height") or 900.0)
    except Exception:
        width, height = 1600.0, 900.0
    if width <= 0:
        width = 1600.0
    if height <= 0:
        height = 900.0
    return width / 2.0, height / 2.0


def _find_centered_faucet_pipe_entry(
    data: dict | None,
    *,
    max_center_dist_px: float = FAUCET_PIPE_CENTER_MAX_DIST_PX,
    max_distance_m: float = INTERACTION_TARGET_MAX_DISTANCE_M,
) -> tuple[dict | None, str]:
    center_x, center_y = _screen_center_xy(data)
    candidates: list[tuple[float, float, dict]] = []

    for entry in _realtime_entries(data):
        text = _entry_text(entry)
        if not any(token in text for token in ("faucet pipe", "水龙头出水管")):
            continue
        if entry.get("is_on_screen") is False:
            continue

        try:
            dist = float(entry.get("distance", 9999.0))
        except Exception:
            dist = 9999.0
        if dist > float(max_distance_m):
            continue

        xy = _entry_screen_xy(entry)
        if xy is None:
            continue
        center_dist = ((xy[0] - center_x) ** 2 + (xy[1] - center_y) ** 2) ** 0.5
        candidates.append((center_dist, dist, entry))

    if not candidates:
        return None, "no_centered_faucet_pipe_within_2m"

    candidates.sort(key=lambda item: (item[0], item[1]))
    best_center_dist, _, best = candidates[0]
    if best_center_dist > float(max_center_dist_px):
        return None, "no_centered_faucet_pipe_within_2m"
    return best, ""


def _faucet_pipe_side_analysis(data: dict | None, pipe_entry: dict | None) -> dict:
    pipe_xz = _entry_position_xz(pipe_entry)
    left_entry = _find_sink_place_point_entry(data, side="left")
    right_entry = _find_sink_place_point_entry(data, side="right")
    left_xz = _entry_position_xz(left_entry)
    right_xz = _entry_position_xz(right_entry)
    left_distance = _distance_xz(pipe_xz, left_xz)
    right_distance = _distance_xz(pipe_xz, right_xz)

    closer_side = "unknown"
    if left_distance is not None and right_distance is not None:
        if abs(float(left_distance) - float(right_distance)) <= 1e-6:
            closer_side = "equal"
        elif float(left_distance) < float(right_distance):
            closer_side = "left"
        else:
            closer_side = "right"

    return {
        "pipe_position_xz": (
            {"x": round(float(pipe_xz[0]), 4), "z": round(float(pipe_xz[1]), 4)}
            if pipe_xz is not None
            else None
        ),
        "left_sink_place_point_position_xz": (
            {"x": round(float(left_xz[0]), 4), "z": round(float(left_xz[1]), 4)}
            if left_xz is not None
            else None
        ),
        "right_sink_place_point_position_xz": (
            {"x": round(float(right_xz[0]), 4), "z": round(float(right_xz[1]), 4)}
            if right_xz is not None
            else None
        ),
        "distance_to_left_sink_place_point_m": (round(float(left_distance), 4) if left_distance is not None else None),
        "distance_to_right_sink_place_point_m": (round(float(right_distance), 4) if right_distance is not None else None),
        "closer_sink_place_point": closer_side,
    }


def _is_drawer_entry(entry: dict | None) -> bool:
    text = _entry_text(entry or {})
    return any(token in text for token in ("drawer", "drawers", "抽屉"))


def _is_door_entry(entry: dict | None) -> bool:
    text = _entry_text(entry or {})
    return any(token in text for token in ("door", "doors", "门"))


def _entry_display_name(entry: dict | None) -> str:
    if not isinstance(entry, dict):
        return ""
    for key in ("name_en", "name_cn", "name", "game_object"):
        value = str(entry.get(key) or "").strip()
        if value:
            return value
    return ""


def _entry_open_angle(entry: dict | None) -> float | None:
    if not isinstance(entry, dict):
        return None
    try:
        value = entry.get("open_angle")
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _drawer_entry_snapshot(entry: dict | None) -> dict:
    if not isinstance(entry, dict):
        return {}
    snap = {
        "name": _entry_display_name(entry),
        "instance_id": entry.get("instance_id"),
        "is_open": bool(entry.get("is_open", False)),
        "open_angle": _entry_open_angle(entry),
        "is_on_screen": bool(entry.get("is_on_screen", False)),
        "distance": entry.get("distance"),
        "screen_x": entry.get("screen_x"),
        "screen_y": entry.get("screen_y"),
    }
    pos = entry.get("position")
    if isinstance(pos, dict):
        snap["position"] = {
            "x": pos.get("x"),
            "y": pos.get("y"),
            "z": pos.get("z"),
        }
    return snap


def _interaction_target_snapshot(entry: dict | None, *, kind: str) -> dict:
    snap = _drawer_entry_snapshot(entry)
    if isinstance(snap, dict):
        snap["target_kind"] = str(kind)
    return snap


def _find_centered_interaction_entry(
    data: dict | None,
    *,
    kind: str,
    max_center_dist_px: float,
    max_distance_m: float = INTERACTION_TARGET_MAX_DISTANCE_M,
) -> tuple[dict | None, str]:
    center_x, center_y = _screen_center_xy(data)
    candidates: list[tuple[float, float, dict]] = []

    for entry in _realtime_entries(data):
        if kind == "drawer":
            if not _is_drawer_entry(entry):
                continue
        elif kind == "door":
            if not _is_door_entry(entry) or _is_drawer_entry(entry):
                continue
        else:
            continue

        if entry.get("is_on_screen") is False:
            continue

        try:
            dist = float(entry.get("distance", 9999.0))
        except Exception:
            dist = 9999.0
        if dist > float(max_distance_m):
            continue

        xy = _entry_screen_xy(entry)
        if xy is None:
            continue

        center_dist = ((xy[0] - center_x) ** 2 + (xy[1] - center_y) ** 2) ** 0.5
        candidates.append((center_dist, dist, entry))

    if not candidates:
        return None, f"no_centered_{kind}_within_{int(max_distance_m)}m"

    candidates.sort(key=lambda item: (item[0], item[1]))
    best_center_dist, _, best = candidates[0]
    if best_center_dist > float(max_center_dist_px):
        return None, f"no_centered_{kind}_within_{int(max_distance_m)}m"
    return best, ""


def _find_centered_drawer_entry(
    data: dict | None,
    *,
    max_center_dist_px: float = DRAWER_CENTER_MAX_DIST_PX,
) -> dict | None:
    target, _ = _find_centered_interaction_entry(
        data,
        kind="drawer",
        max_center_dist_px=max_center_dist_px,
    )
    return target


def _find_centered_door_entry(
    data: dict | None,
    *,
    max_center_dist_px: float = DOOR_CENTER_MAX_DIST_PX,
) -> dict | None:
    target, _ = _find_centered_interaction_entry(
        data,
        kind="door",
        max_center_dist_px=max_center_dist_px,
    )
    return target


def _find_matching_entry(data: dict | None, target_entry: dict | None) -> dict | None:
    if not isinstance(target_entry, dict):
        return None
    target_id = target_entry.get("instance_id")
    target_name = _entry_display_name(target_entry).strip().lower()
    if target_id is not None:
        for entry in _realtime_entries(data):
            if entry.get("instance_id") == target_id:
                return entry
    if target_name:
        matches = []
        for entry in _realtime_entries(data):
            if _entry_display_name(entry).strip().lower() == target_name:
                try:
                    dist = float(entry.get("distance", 9999.0))
                except Exception:
                    dist = 9999.0
                matches.append((dist, entry))
        if matches:
            matches.sort(key=lambda item: item[0])
            return matches[0][1]
    return None


def _wait_drawer_open_state(
    target_entry: dict | None,
    *,
    expected_open: bool,
    timeout_s: float = DRAWER_STATE_TIMEOUT_S,
    poll_s: float = DRAWER_STATE_POLL_S,
) -> tuple[bool, bool, dict]:
    deadline = time.time() + max(0.0, float(timeout_s))
    saw_available = False
    last_snapshot = _drawer_entry_snapshot(target_entry)
    while True:
        data = _read_realtime_products_quick(retries=2, sleep_s=0.02)
        available = bool(_realtime_entries(data))
        saw_available = bool(saw_available or available)
        matched = _find_matching_entry(data, target_entry)
        if matched is not None:
            last_snapshot = _drawer_entry_snapshot(matched)
            if bool(matched.get("is_open", False)) == bool(expected_open):
                return True, True, last_snapshot
        if time.time() >= deadline:
            return False, bool(saw_available), last_snapshot
        time.sleep(max(0.01, float(poll_s)))


def _wait_interaction_open_state(
    target_entry: dict | None,
    *,
    expected_open: bool,
    kind: str,
    timeout_s: float = DRAWER_STATE_TIMEOUT_S,
    poll_s: float = DRAWER_STATE_POLL_S,
) -> tuple[bool, bool, dict]:
    deadline = time.time() + max(0.0, float(timeout_s))
    saw_available = False
    last_snapshot = _interaction_target_snapshot(target_entry, kind=kind)
    while True:
        data = _read_realtime_products_quick(retries=2, sleep_s=0.02)
        available = bool(_realtime_entries(data))
        saw_available = bool(saw_available or available)
        matched = _find_matching_entry(data, target_entry)
        if matched is not None:
            last_snapshot = _interaction_target_snapshot(matched, kind=kind)
            if bool(matched.get("is_open", False)) == bool(expected_open):
                return True, True, last_snapshot
        if time.time() >= deadline:
            return False, bool(saw_available), last_snapshot
        time.sleep(max(0.01, float(poll_s)))


def _trace_door_open_state(
    target_entry: dict | None,
    *,
    expected_open: bool,
    timeout_s: float = DOOR_OPEN_ANGLE_TRACE_TIMEOUT_S,
    poll_s: float = DRAWER_STATE_POLL_S,
) -> tuple[bool, bool, dict, list[dict]]:
    deadline = time.time() + max(0.0, float(timeout_s))
    saw_available = False
    last_snapshot = _interaction_target_snapshot(target_entry, kind="door")
    trace: list[dict] = []
    start_ts = time.time()

    while True:
        data = _read_realtime_products_quick(retries=2, sleep_s=0.02)
        available = bool(_realtime_entries(data))
        saw_available = bool(saw_available or available)
        matched = _find_matching_entry(data, target_entry)
        if matched is not None:
            last_snapshot = _interaction_target_snapshot(matched, kind="door")
            trace.append(
                {
                    "t_s": round(max(0.0, time.time() - start_ts), 3),
                    "is_open": bool(matched.get("is_open", False)),
                    "open_angle": _entry_open_angle(matched),
                }
            )
        if time.time() >= deadline:
            break
        time.sleep(max(0.01, float(poll_s)))

    matched_state = bool(last_snapshot.get("is_open", False)) == bool(expected_open)
    return matched_state, bool(saw_available), last_snapshot, trace


def _align_faucet_mode_container_to_pipe(
    *,
    threshold_m: float = FAUCET_FILLING_ALIGN_THRESHOLD_M,
    step_px: int = FAUCET_FILLING_ALIGN_STEP_PX,
    max_steps: int = FAUCET_FILLING_ALIGN_MAX_STEPS,
) -> tuple[bool, dict]:
    direction = 1
    best_distance = float("inf")
    best_meta: dict = {}

    for step_idx in range(max(1, int(max_steps))):
        data = _read_realtime_products_quick(retries=2, sleep_s=0.02)
        source = _find_active_pouring_entry(data)
        pipe = _find_faucet_pipe_entry(data)
        source_pos = _entry_position_xz(source)
        pipe_pos = _entry_position_xz(pipe)
        if source_pos is None:
            return False, {"error": "faucet_alignment_source_not_found", "step": step_idx}
        if pipe_pos is None:
            return False, {"error": "faucet_pipe_not_found", "step": step_idx}

        dx = pipe_pos[0] - source_pos[0]
        dz = pipe_pos[1] - source_pos[1]
        distance = (dx * dx + dz * dz) ** 0.5
        best_distance = min(best_distance, distance)
        best_meta = {
            "source_x": round(source_pos[0], 4),
            "source_z": round(source_pos[1], 4),
            "pipe_x": round(pipe_pos[0], 4),
            "pipe_z": round(pipe_pos[1], 4),
            "distance_xz_m": round(distance, 4),
            "step": step_idx,
        }
        if distance <= float(threshold_m):
            return True, best_meta

        before_distance = distance
        move_faucet_container(direction * int(step_px))
        time.sleep(0.12)

        data_after = _read_realtime_products_quick(retries=2, sleep_s=0.02)
        source_after = _find_active_pouring_entry(data_after)
        pipe_after = _find_faucet_pipe_entry(data_after)
        source_after_pos = _entry_position_xz(source_after)
        pipe_after_pos = _entry_position_xz(pipe_after)
        if source_after_pos is None or pipe_after_pos is None:
            continue

        dx_after = pipe_after_pos[0] - source_after_pos[0]
        dz_after = pipe_after_pos[1] - source_after_pos[1]
        after_distance = (dx_after * dx_after + dz_after * dz_after) ** 0.5
        if after_distance < best_distance:
            best_distance = after_distance
            best_meta = {
                "source_x": round(source_after_pos[0], 4),
                "source_z": round(source_after_pos[1], 4),
                "pipe_x": round(pipe_after_pos[0], 4),
                "pipe_z": round(pipe_after_pos[1], 4),
                "distance_xz_m": round(after_distance, 4),
                "step": step_idx + 1,
            }
        if after_distance <= float(threshold_m):
            return True, best_meta
        if after_distance > (before_distance + 0.002):
            direction *= -1

    return False, {
        "error": "faucet_alignment_not_reached",
        "best_distance_xz_m": round(best_distance, 4),
        **best_meta,
    }


def _trace_faucet_pipe_direction_change(
    reference_entry: dict | None,
    *,
    timeout_s: float = FAUCET_DIRECTION_CHANGE_TIMEOUT_S,
    poll_s: float = FAUCET_DIRECTION_CHANGE_POLL_S,
    min_delta_m: float = FAUCET_DIRECTION_CHANGE_MIN_DELTA_M,
) -> tuple[bool, bool, dict, dict]:
    pre_xz = _entry_position_xz(reference_entry)
    saw_available = False
    last_entry = None
    last_analysis: dict = {}
    moved_distance_m = None
    deadline = time.time() + max(0.0, float(timeout_s))

    while True:
        data = _read_realtime_products_quick(retries=2, sleep_s=0.02)
        if _realtime_entries(data):
            saw_available = True
            entry = _find_matching_faucet_pipe_entry(data, reference_entry)
            if entry is not None:
                last_entry = entry
                last_analysis = _faucet_pipe_side_analysis(data, entry)
                post_xz = _entry_position_xz(entry)
                moved_distance_m = _distance_xz(pre_xz, post_xz)
                if moved_distance_m is not None:
                    last_analysis["moved_distance_m"] = round(float(moved_distance_m), 4)
                    if float(moved_distance_m) >= float(min_delta_m):
                        return True, True, _interaction_target_snapshot(entry, kind="faucet_pipe"), last_analysis

        if time.time() >= deadline:
            break
        time.sleep(max(0.01, float(poll_s)))

    if moved_distance_m is not None:
        last_analysis["moved_distance_m"] = round(float(moved_distance_m), 4)
    return False, bool(saw_available), _interaction_target_snapshot(last_entry, kind="faucet_pipe"), last_analysis


def _ensure_alt_j_interaction_best_effort() -> bool:
    try:
        from epm.cerebellum.game_hotkeys import ensure_alt_j_interaction
        from epm.cerebellum.skills._shared_paths import userdata_root, window_title

        ensure_alt_j_interaction(
            userdata_root=userdata_root(),
            window_title=window_title("CookingSimulator"),
            activate_window=_activate_window,
            io_controller=io_controller,
        )
        return True
    except Exception:
        return False


def _parse_volume_to_ml(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))

    text = str(value).strip()
    if not text:
        return None

    normalized = text.replace(",", ".")
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)?", normalized)
    if not match:
        return None

    try:
        amount = float(match.group(1))
    except Exception:
        return None

    unit = str(match.group(2) or "ml").strip().lower()
    if unit in {"ml", "milliliter", "milliliters", "millilitre", "millilitres"}:
        return amount
    if unit in {"l", "lt", "ltr", "liter", "liters", "litre", "litres"}:
        return amount * 1000.0
    return None


def _parse_container_contents_total_ml(contents: str) -> Optional[float]:
    text = str(contents or "").strip()
    if not text:
        return None

    total_ml = 0.0
    matched = False
    normalized = text.replace(",", ".")
    for match in re.finditer(r"([-+]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)", normalized):
        amount_ml = _parse_volume_to_ml(match.group(0))
        if amount_ml is None:
            continue
        total_ml += amount_ml
        matched = True
    if not matched:
        return None
    return total_ml


def _format_volume_ml(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if abs(float(value)) >= 1000.0:
        return f"{float(value) / 1000.0:.3f} L"
    return f"{float(value):.1f} ml"


def _read_faucet_fill_progress(max_age_s: float = 2.0) -> dict:
    try:
        from epm.cerebellum.interaction_snapshot import read_interaction_snapshot
        from epm.cerebellum.skills._shared_paths import userdata_root
        from epm.cerebellum.skills.auto_navigation.interaction_detector import (
            get_udp_debug_snapshot,
            init_udp_mode,
        )
    except Exception as exc:
        return {
            "error": f"interaction_snapshot_unavailable:{exc.__class__.__name__}",
            "snapshot_available": False,
        }

    def _build_progress(*, source: str, container_name: str, container_contents: str, pour_amount: str, overflow_amount: str, timestamp: str) -> dict:
        return {
            "snapshot_available": True,
            "source": source,
            "container_name": str(container_name or "").strip(),
            "container_contents": str(container_contents or "").strip(),
            "pour_amount": str(pour_amount or "").strip(),
            "overflow_amount": str(overflow_amount or "").strip(),
            "timestamp": str(timestamp or "").strip(),
            "container_total_ml": _parse_container_contents_total_ml(container_contents),
            "pour_amount_ml": _parse_volume_to_ml(pour_amount),
            "overflow_amount_ml": _parse_volume_to_ml(overflow_amount),
        }

    try:
        init_udp_mode()
        udp_snapshot = get_udp_debug_snapshot()
        if isinstance(udp_snapshot, dict) and bool(udp_snapshot.get("running")):
            latest_pour = udp_snapshot.get("latest_pour")
            latest_pour_age_s = udp_snapshot.get("latest_pour_age_s", None)
            if (
                isinstance(latest_pour, dict)
                and latest_pour_age_s is not None
                and 0.0 <= float(latest_pour_age_s) <= float(max_age_s)
            ):
                return _build_progress(
                    source="udp_pour",
                    container_name=latest_pour.get("container_name", ""),
                    container_contents=latest_pour.get("container_contents", ""),
                    pour_amount=latest_pour.get("pour_amount", ""),
                    overflow_amount=latest_pour.get("overflow_amount", ""),
                    timestamp=latest_pour.get("timestamp", ""),
                )

            latest = udp_snapshot.get("latest")
            latest_age_s = udp_snapshot.get("latest_age_s", None)
            if (
                isinstance(latest, dict)
                and latest_age_s is not None
                and 0.0 <= float(latest_age_s) <= float(max_age_s)
            ):
                return _build_progress(
                    source="udp_latest",
                    container_name=latest.get("container_name", ""),
                    container_contents=latest.get("container_contents", ""),
                    pour_amount=latest.get("pour_amount", ""),
                    overflow_amount=latest.get("overflow_amount", ""),
                    timestamp=latest.get("timestamp", ""),
                )
    except Exception:
        pass

    snapshot = read_interaction_snapshot(userdata_root=userdata_root(), max_age_s=max_age_s)
    if snapshot is None:
        return {
            "error": "interaction_snapshot_missing_or_stale",
            "snapshot_available": False,
        }

    return _build_progress(
        source="file_snapshot",
        container_name=getattr(snapshot, "container_name", ""),
        container_contents=getattr(snapshot, "container_contents", ""),
        pour_amount=getattr(snapshot, "pour_amount", ""),
        overflow_amount=getattr(snapshot, "overflow_amount", ""),
        timestamp=getattr(snapshot, "timestamp", ""),
    )


def _read_current_container_liquid_total_ml(max_age_s: float = 2.0) -> tuple[Optional[float], dict]:
    progress = _read_faucet_fill_progress(max_age_s=max_age_s)
    total_ml = progress.get("container_total_ml", None)
    meta = {
        "source": "container_contents",
        "container_name": progress.get("container_name", ""),
        "container_contents": progress.get("container_contents", ""),
        "pour_amount": progress.get("pour_amount", ""),
        "overflow_amount": progress.get("overflow_amount", ""),
        "timestamp": progress.get("timestamp", ""),
    }
    if total_ml is None:
        meta["error"] = str(progress.get("error") or "liquid_amount_not_found_in_alt_j")
    else:
        meta["current_amount_ml"] = round(float(total_ml), 3)
    return (float(total_ml) if total_ml is not None else None), meta


def _has_faucet_fill_progress(progress: dict | None) -> bool:
    if not isinstance(progress, dict):
        return False
    if bool(progress.get("snapshot_available", False)):
        return True
    return any(progress.get(k) is not None for k in ("container_total_ml", "pour_amount_ml", "overflow_amount_ml"))


def _print_faucet_fill_progress(
    *,
    handle_name: str,
    target_delta_ml: float,
    poured_delta_ml: Optional[float],
    container_total_ml: Optional[float],
    container_delta_ml: Optional[float],
    progress: dict,
    end_line: bool = False,
) -> None:
    msg = (
        f"[faucet_fill] handle={handle_name} "
        f"target_delta={_format_volume_ml(target_delta_ml)} "
        f"poured_delta={_format_volume_ml(poured_delta_ml)} "
        f"container_total={_format_volume_ml(container_total_ml)} "
        f"container_delta={_format_volume_ml(container_delta_ml)} "
        f"pour={str(progress.get('pour_amount') or '')!s} "
        f"overflow={str(progress.get('overflow_amount') or '')!s} "
        f"contents={str(progress.get('container_contents') or '')!s}"
    ).strip()
    if end_line:
        print(msg)
    else:
        print(f"\r{msg:<220}", end="", flush=True)


def _auto_fill_faucet_handle(
    *,
    handle_key: Literal["e", "q"],
    handle_name: str,
    target_ml,
    tolerance_ml: float = FAUCET_FILLING_TARGET_TOLERANCE_ML,
) -> dict:
    delta_ml = _parse_volume_to_ml(target_ml)
    if delta_ml is None:
        return {
            "success": False,
            "mode": "faucet_filling_mode",
            "error": "invalid_target_amount",
            "verification": "Failed to parse the target liquid amount.",
            "handle": handle_name,
            "requested_target": target_ml,
        }

    if delta_ml <= 0.0:
        return {
            "success": False,
            "mode": "faucet_filling_mode",
            "error": "invalid_target_increment",
            "verification": "The target liquid increment must be greater than zero.",
            "handle": handle_name,
            "requested_target": target_ml,
        }

    tolerance_value = _parse_volume_to_ml(tolerance_ml)
    tolerance_value = float(tolerance_value if tolerance_value is not None else FAUCET_FILLING_TARGET_TOLERANCE_ML)
    tolerance_value = max(0.0, tolerance_value)

    pre_available, pre = _hold_snapshot()
    ready_ok, ready_meta = _ensure_faucet_fill_ui_ready(
        timeout_s=FAUCET_FILLING_UI_TIMEOUT_S,
        poll_s=FAUCET_FILLING_UI_POLL_S,
    )
    if not ready_ok:
        return {
            "success": False,
            "mode": "faucet_filling_mode",
            "error": str(ready_meta.get("error") or "faucet_fill_mode_not_ready"),
            "verification": "Failed to confirm faucet-filling readiness before opening the handle.",
            "handle": handle_name,
            "requested_target": target_ml,
            "precheck": pre,
            "postcheck": dict(ready_meta.get("mode_snap") or {}),
            "ui_texts_sample": list(ready_meta.get("ui_texts_sample") or []),
            "ui_dump_keys": list(ready_meta.get("ui_dump_keys") or []),
        }

    _ensure_alt_j_interaction_best_effort()
    initial_progress = _read_faucet_fill_progress()
    initial_container_ml = initial_progress.get("container_total_ml", None)
    initial_pour_total_ml = initial_progress.get("pour_amount_ml", None)

    faucet_opened = False
    target_reached = False
    timed_out = False
    no_progress_timeout = False
    last_progress = dict(initial_progress)
    last_container_ml = initial_container_ml
    last_pour_total_ml = initial_pour_total_ml
    last_poured_delta_ml = 0.0
    cached_pour_delta_ml = 0.0
    last_seen_pour_total_ml = initial_pour_total_ml
    last_logged_at = 0.0
    progress_detected = _has_faucet_fill_progress(initial_progress)

    try:
        press_keyboard(handle_key)
        faucet_opened = True

        deadline = time.time() + FAUCET_FILLING_TARGET_TIMEOUT_S
        progress_deadline = time.time() + FAUCET_FILLING_PROGRESS_DETECT_TIMEOUT_S
        while time.time() < deadline:
            RawInputController.check_stop()
            progress = _read_faucet_fill_progress()
            if progress:
                last_progress = dict(progress)
                if _has_faucet_fill_progress(progress):
                    progress_detected = True

            current_pour_total_ml = progress.get("pour_amount_ml", None)
            current_container_ml = progress.get("container_total_ml", None)

            if current_pour_total_ml is not None:
                current_pour_total_ml = float(current_pour_total_ml)
                if last_seen_pour_total_ml is not None and current_pour_total_ml < (float(last_seen_pour_total_ml) - 5.0):
                    cached_pour_delta_ml += max(0.0, float(last_seen_pour_total_ml) - float(initial_pour_total_ml or 0.0))
                    initial_pour_total_ml = current_pour_total_ml
                last_seen_pour_total_ml = current_pour_total_ml
                last_pour_total_ml = current_pour_total_ml
                last_poured_delta_ml = cached_pour_delta_ml + max(0.0, current_pour_total_ml - float(initial_pour_total_ml or 0.0))

            if current_container_ml is not None:
                last_container_ml = float(current_container_ml)

            container_delta_ml = None
            if (initial_container_ml is not None) and (last_container_ml is not None):
                container_delta_ml = max(0.0, float(last_container_ml) - float(initial_container_ml))

            now = time.time()
            if (now - last_logged_at) >= 0.25:
                _print_faucet_fill_progress(
                    handle_name=handle_name,
                    target_delta_ml=float(delta_ml),
                    poured_delta_ml=last_poured_delta_ml,
                    container_total_ml=last_container_ml,
                    container_delta_ml=container_delta_ml,
                    progress=progress,
                    end_line=False,
                )
                last_logged_at = now

            if last_poured_delta_ml >= (float(delta_ml) - tolerance_value):
                target_reached = True
                break
            if (not progress_detected) and time.time() >= progress_deadline:
                no_progress_timeout = True
                break
            time.sleep(FAUCET_FILLING_TARGET_POLL_S)
        if (not target_reached) and (not no_progress_timeout):
            timed_out = True
    finally:
        _print_faucet_fill_progress(
            handle_name=handle_name,
            target_delta_ml=float(delta_ml),
            poured_delta_ml=last_poured_delta_ml,
            container_total_ml=last_container_ml,
            container_delta_ml=(
                max(0.0, float(last_container_ml) - float(initial_container_ml))
                if (initial_container_ml is not None and last_container_ml is not None)
                else None
            ),
            progress=last_progress,
            end_line=True,
        )
        if faucet_opened:
            press_keyboard(handle_key)
            time.sleep(FAUCET_FILLING_STOP_SETTLE_S)

    click_mouse("right")
    exited, post_available, post = _wait_mode_disabled(
        "is_pouring_mode",
        timeout_s=1.2,
        poll_s=0.05,
        settle_s=0.2,
    )

    had_progress = _has_faucet_fill_progress(initial_progress) or _has_faucet_fill_progress(last_progress)
    success = bool(target_reached and exited)
    if target_reached and exited:
        verification = "Filled the container to the target amount and exited water-filling mode."
        error = ""
    elif target_reached:
        verification = "Reached the target amount, but failed to exit water-filling mode."
        error = "faucet_fill_exit_failed"
    elif (not had_progress) and exited:
        verification = "Exited water-filling mode, but no filling progress was detected. Please check whether the container was properly aimed at the faucet."
        error = "faucet_fill_progress_not_detected"
    elif not had_progress:
        verification = "No filling progress was detected. Please check whether the container was properly aimed at the faucet."
        error = "faucet_fill_progress_not_detected"
    elif timed_out and exited:
        verification = "Stopped the faucet and exited water-filling mode, but did not reach the target amount in time."
        error = "target_amount_not_reached_before_timeout"
    elif timed_out:
        verification = "Stopped the faucet, but neither reached the target amount nor exited water-filling mode cleanly."
        error = "target_amount_not_reached_before_timeout"
    elif exited:
        verification = "Stopped the faucet and exited water-filling mode, but failed to read the liquid amount reliably."
        error = str(last_progress.get("error") or "liquid_amount_unavailable")
    else:
        verification = "Stopped the faucet, but failed to read the liquid amount and failed to exit water-filling mode."
        error = str(last_progress.get("error") or "liquid_amount_unavailable")

    actual_container_delta_ml = None
    if (initial_container_ml is not None) and (last_container_ml is not None):
        actual_container_delta_ml = max(0.0, float(last_container_ml) - float(initial_container_ml))

    overflow_amount_ml = last_progress.get("overflow_amount_ml", None)
    mismatch_note = ""
    if actual_container_delta_ml is not None and abs(float(last_poured_delta_ml) - float(actual_container_delta_ml)) > max(20.0, tolerance_value * 2.0):
        mismatch_note = (
            "The poured amount and the actual liquid increase in the container are inconsistent. "
            "The container may not have enough capacity, so some liquid likely spilled or overflowed."
        )
        if overflow_amount_ml is not None and float(overflow_amount_ml) > 0.0:
            mismatch_note = (
                "The poured amount and the actual liquid increase in the container are inconsistent, "
                "and overflow was also detected. The container likely overflowed and some liquid spilled out."
            )

    poured_amount_text = _format_volume_ml(last_poured_delta_ml)
    actual_increment_text = _format_volume_ml(actual_container_delta_ml)
    final_amount_text = _format_volume_ml(last_container_ml)
    result = {
        "success": success,
        "mode": "faucet_filling_mode",
        "error": error,
        "verification": verification,
        "handle": handle_name,
        "target_increment_ml": round(delta_ml, 3),
        "tolerance_ml": round(tolerance_value, 3),
        "initial_amount_ml": (round(float(initial_container_ml), 3) if initial_container_ml is not None else None),
        "final_amount_ml": (round(float(last_container_ml), 3) if last_container_ml is not None else None),
        "poured_amount_ml": round(float(last_poured_delta_ml), 3),
        "actual_container_increment_ml": (round(float(actual_container_delta_ml), 3) if actual_container_delta_ml is not None else None),
        "overflow_amount_ml": (round(float(overflow_amount_ml), 3) if overflow_amount_ml is not None else None),
        "target_reached": bool(target_reached),
        "mode_exited": bool(exited),
        "precheck": pre,
        "postcheck": post,
        "realtime_available": bool(pre_available or post_available),
        "progress": {
            "initial": initial_progress,
            "latest": last_progress,
        },
    }
    if had_progress:
        result["verification"] = (
            f"{result['verification']} "
            f"Measured filled amount: {poured_amount_text}; container total: {final_amount_text}; "
            f"actual increase: {actual_increment_text}."
        )
    if mismatch_note:
        result["consistency_note"] = mismatch_note
        result["verification"] = f"{result['verification']} {mismatch_note}"
    return result


def _auto_fill_faucet_handle_direct(
    *,
    handle_name: str,
    target_ml,
    tolerance_ml: float = FAUCET_FILLING_TARGET_TOLERANCE_ML,
) -> dict:
    delta_ml = _parse_volume_to_ml(target_ml)
    if delta_ml is None:
        return {
            "success": False,
            "mode": "direct_faucet_fill",
            "error": "invalid_target_amount",
            "verification": "Failed to parse the target liquid amount.",
            "handle": handle_name,
            "requested_target": target_ml,
        }

    if delta_ml <= 0.0:
        return {
            "success": False,
            "mode": "direct_faucet_fill",
            "error": "invalid_target_increment",
            "verification": "The target liquid increment must be greater than zero.",
            "handle": handle_name,
            "requested_target": target_ml,
        }

    tolerance_value = _parse_volume_to_ml(tolerance_ml)
    tolerance_value = float(tolerance_value if tolerance_value is not None else FAUCET_FILLING_TARGET_TOLERANCE_ML)
    tolerance_value = max(0.0, tolerance_value)

    _ensure_alt_j_interaction_best_effort()
    initial_progress = _read_faucet_fill_progress()
    initial_container_ml = initial_progress.get("container_total_ml", None)
    initial_pour_total_ml = initial_progress.get("pour_amount_ml", None)

    faucet_opened = False
    target_reached = False
    timed_out = False
    no_progress_timeout = False
    last_progress = dict(initial_progress)
    last_container_ml = initial_container_ml
    last_pour_total_ml = initial_pour_total_ml
    last_poured_delta_ml = 0.0
    cached_pour_delta_ml = 0.0
    last_seen_pour_total_ml = initial_pour_total_ml
    last_logged_at = 0.0
    progress_detected = _has_faucet_fill_progress(initial_progress)

    try:
        click_mouse("left")
        faucet_opened = True

        deadline = time.time() + FAUCET_FILLING_TARGET_TIMEOUT_S
        progress_deadline = time.time() + FAUCET_FILLING_PROGRESS_DETECT_TIMEOUT_S
        while time.time() < deadline:
            RawInputController.check_stop()
            progress = _read_faucet_fill_progress()
            if progress:
                last_progress = dict(progress)
                if _has_faucet_fill_progress(progress):
                    progress_detected = True

            current_pour_total_ml = progress.get("pour_amount_ml", None)
            current_container_ml = progress.get("container_total_ml", None)

            if current_pour_total_ml is not None:
                current_pour_total_ml = float(current_pour_total_ml)
                if last_seen_pour_total_ml is not None and current_pour_total_ml < (float(last_seen_pour_total_ml) - 5.0):
                    cached_pour_delta_ml += max(0.0, float(last_seen_pour_total_ml) - float(initial_pour_total_ml or 0.0))
                    initial_pour_total_ml = current_pour_total_ml
                last_seen_pour_total_ml = current_pour_total_ml
                last_pour_total_ml = current_pour_total_ml
                last_poured_delta_ml = cached_pour_delta_ml + max(0.0, current_pour_total_ml - float(initial_pour_total_ml or 0.0))

            if current_container_ml is not None:
                last_container_ml = float(current_container_ml)

            container_delta_ml = None
            if (initial_container_ml is not None) and (last_container_ml is not None):
                container_delta_ml = max(0.0, float(last_container_ml) - float(initial_container_ml))

            now = time.time()
            if (now - last_logged_at) >= 0.25:
                _print_faucet_fill_progress(
                    handle_name=f"{handle_name}_direct",
                    target_delta_ml=float(delta_ml),
                    poured_delta_ml=last_poured_delta_ml,
                    container_total_ml=last_container_ml,
                    container_delta_ml=container_delta_ml,
                    progress=progress,
                    end_line=False,
                )
                last_logged_at = now

            if last_poured_delta_ml >= (float(delta_ml) - tolerance_value):
                target_reached = True
                break
            if (not progress_detected) and time.time() >= progress_deadline:
                no_progress_timeout = True
                break
            time.sleep(FAUCET_FILLING_TARGET_POLL_S)
        if (not target_reached) and (not no_progress_timeout):
            timed_out = True
    finally:
        _print_faucet_fill_progress(
            handle_name=f"{handle_name}_direct",
            target_delta_ml=float(delta_ml),
            poured_delta_ml=last_poured_delta_ml,
            container_total_ml=last_container_ml,
            container_delta_ml=(
                max(0.0, float(last_container_ml) - float(initial_container_ml))
                if (initial_container_ml is not None and last_container_ml is not None)
                else None
            ),
            progress=last_progress,
            end_line=True,
        )
        if faucet_opened:
            click_mouse("left")
            time.sleep(FAUCET_FILLING_STOP_SETTLE_S)

    actual_container_delta_ml = None
    if (initial_container_ml is not None) and (last_container_ml is not None):
        actual_container_delta_ml = max(0.0, float(last_container_ml) - float(initial_container_ml))

    had_progress = _has_faucet_fill_progress(initial_progress) or _has_faucet_fill_progress(last_progress)
    overflow_amount_ml = last_progress.get("overflow_amount_ml", None)
    mismatch_note = ""
    if actual_container_delta_ml is not None and abs(float(last_poured_delta_ml) - float(actual_container_delta_ml)) > max(20.0, tolerance_value * 2.0):
        mismatch_note = (
            "The poured amount and the actual liquid increase in the container are inconsistent. "
            "The container may not have enough capacity, so some liquid likely spilled or overflowed."
        )
        if overflow_amount_ml is not None and float(overflow_amount_ml) > 0.0:
            mismatch_note = (
                "The poured amount and the actual liquid increase in the container are inconsistent, "
                "and overflow was also detected. The container likely overflowed and some liquid spilled out."
            )

    poured_amount_text = _format_volume_ml(last_poured_delta_ml)
    actual_increment_text = _format_volume_ml(actual_container_delta_ml)
    final_amount_text = _format_volume_ml(last_container_ml)
    if target_reached:
        verification = "Filled the container to the target amount through the faucet."
        error = ""
    elif not had_progress:
        verification = "No filling progress was detected. Please check whether the container was properly aimed at this faucet handle."
        error = "faucet_fill_progress_not_detected"
    else:
        verification = "Filling started, but the target amount was not reached before stopping."
        error = "target_amount_not_reached_before_timeout"

    result = {
        "success": bool(target_reached),
        "mode": "direct_faucet_fill",
        "error": error,
        "verification": verification,
        "handle": handle_name,
        "target_increment_ml": round(delta_ml, 3),
        "tolerance_ml": round(tolerance_value, 3),
        "initial_amount_ml": (round(float(initial_container_ml), 3) if initial_container_ml is not None else None),
        "final_amount_ml": (round(float(last_container_ml), 3) if last_container_ml is not None else None),
        "poured_amount_ml": round(float(last_poured_delta_ml), 3),
        "actual_container_increment_ml": (round(float(actual_container_delta_ml), 3) if actual_container_delta_ml is not None else None),
        "overflow_amount_ml": (round(float(overflow_amount_ml), 3) if overflow_amount_ml is not None else None),
        "target_reached": bool(target_reached),
        "progress": {
            "initial": initial_progress,
            "latest": last_progress,
        },
    }
    if had_progress:
        result["verification"] = (
            f"{result['verification']} "
            f"Measured filled amount: {poured_amount_text}; container total: {final_amount_text}; "
            f"actual increase: {actual_increment_text}."
        )
    if mismatch_note:
        result["consistency_note"] = mismatch_note
        result["verification"] = f"{result['verification']} {mismatch_note}"
    if timed_out and not target_reached:
        result["timed_out"] = True
    return result


def _pour_mode_raise_source(*, pixels: int = 300) -> None:
    """
    Some pourable containers (e.g. salad bowls/containers) can start spilling as soon
    as pouring mode is entered. Immediately lift the source upward a bit to reduce
    premature pouring before the dedicated pour skill starts alignment.
    """
    try:
        hold_mouse("left")
        time.sleep(0.02)
        move_related_mouse(0, -_coerce_mouse_delta(pixels))
        time.sleep(0.05)
    finally:
        try:
            leave_mouse("left")
        except Exception:
            pass

def enter_pouring_mode():
    """
    液体倾倒操作类型动作。手持容器或者液体瓶并且对准另一个容器情况下，对进入液体倾倒模式，这是液体倾倒的前提
    Args:

    """
    pre_available, pre = _hold_snapshot()
    allowed, reason_code, held_name, held_kind = _held_item_allows_pouring(pre)
    if pre_available and not allowed:
        return {
            "success": False,
            "mode": "pouring_mode",
            "mode_entered": False,
            "error": (
                f"enter_pouring_mode_failed:{reason_code}:held_item={held_name!r}:held_kind={held_kind!r}. "
                "Only liquid bottles and containers are allowed for pouring; the current held item is not supported."
            ),
            "feedback_code": "unsupported_pour_source",
            "pour_status": "unsupported_source",
            "feedback_to_planner": (
                f"Only liquid bottles and containers are allowed for pouring. "
                f"The current held item{(' ' + repr(held_name)) if held_name else ''} is not supported."
            ),
            "precheck": pre,
        }
    key="left"
    click_mouse(key)
    # Poll up to 3 seconds after click; do not move the source until mode entry is confirmed.
    ok, snap = _wait_mode_enabled("is_pouring_mode", timeout_s=3.0, poll_s=0.05, settle_s=0.0)
    if ok:
        try:
            _pour_mode_raise_source(pixels=300)
        except Exception:
            pass
        allowed_post, reason_code_post, held_name_post, held_kind_post = _held_item_allows_pouring(snap)
        if not allowed_post:
            # The hold snapshot can briefly drop `is_held` right after mode entry even though
            # the pouring mode is stably enabled and the precheck already confirmed a valid source.
            # In that case, trust the precheck instead of treating it as an accidental mode entry.
            if (
                pre_available
                and allowed
                and reason_code_post in {"no_held_item", "held_item_unknown"}
                and isinstance(snap, dict)
                and bool(snap.get("is_pouring_mode"))
            ):
                return {
                    "success": True,
                    "mode": "pouring_mode",
                    "mode_entered": True,
                    "error": "",
                    "warning": (
                        "enter_pouring_mode_postcheck_relaxed:"
                        f"{reason_code_post}:trusted_precheck_held_item={held_name!r}:held_kind={held_kind!r}"
                    ),
                    "precheck": pre,
                    "postcheck": snap,
                }
            exit_res = {}
            try:
                exit_res = exit_pouring_mode()
            except Exception as e:
                exit_res = {"success": False, "error": f"exit_pouring_mode_failed:{e}"}
            held_name_post = held_name_post or held_name or ""
            held_kind_post = held_kind_post or held_kind or ""
            return {
                "success": False,
                "mode": "pouring_mode",
                "mode_entered": False,
                "error": (
                    f"enter_pouring_mode_failed:accidental_pouring_mode:{reason_code_post}:"
                    f"held_item={held_name_post!r}:held_kind={held_kind_post!r}. "
                    "Detected pouring mode, but the held item is not a liquid bottle or pourable container. "
                    "The action auto-exited pouring mode."
                ),
                "feedback_code": "accidental_pouring_mode",
                "pour_status": "unsupported_source",
                "feedback_to_planner": (
                    "Accidentally entered pouring mode. This usually happens because the held item was aimed at a "
                    "container and the mode was triggered by mistake. Auto-exited pouring mode; only liquid bottles "
                    "and pourable containers are allowed."
                ),
                "exit_mode_attempted": True,
                "exit_mode_result": exit_res,
                "precheck": pre,
                "postcheck": snap,
            }
        return {"success": True, "mode": "pouring_mode", "mode_entered": True, "error": ""}
    return {
        "success": False,
        "mode": "pouring_mode",
        "mode_entered": False,
        "error": "enter_pouring_mode_failed:not_in_pouring_mode",
        "hint": "Before entering pouring mode, hold a container or liquid bottle and aim at another container, then call enter_pouring_mode.",
        "precheck": snap,
    }

def exit_pouring_mode():
    """
    液体倾倒操作类型动作。在液体倾倒模式情况下，退出液体倾倒模式
    Args:

    """
    pre_available, pre = _hold_snapshot()
    pre_pouring = bool(pre_available and bool(pre.get("is_pouring_mode")))

    key="right"
    click_mouse(key)

    ok, post_available, post = _wait_mode_disabled("is_pouring_mode", timeout_s=1.2, poll_s=0.05, settle_s=2.0)
    raw = {
        "mode_exited": bool(ok),
        "pre_active_modes": _snapshot_active_modes(pre),
        "post_active_modes": _snapshot_active_modes(post),
        "precheck": pre,
        "postcheck": post,
    }
    if pre_pouring:
        if ok:
            raw.update({"success": True, "error": ""})
            return raw
        raw.update(
            {
                "success": False,
                "error": (
                    "exit_pouring_mode_post_check_failed:realtime_products_unavailable"
                    if not post_available
                    else "exit_pouring_mode_post_check_failed:pouring_mode_still_active"
                ),
            }
        )
        return raw
    raw.update({"success": True, "error": ""})
    return raw

def exit_current_interaction_mode():
    """
    通用交互模式退出动作。无论当前是倾倒、播撒、切割、搅拌、翻面，还是其他右键可退出的模式，
    默认都先执行一次鼠标右键退出。
    Args:

    """
    pre_available, pre = _hold_snapshot()
    pre_mode_active = bool(pre_available and _interaction_mode_active(pre))

    key="right"
    click_mouse(key)

    ok, post_available, post = _wait_interaction_mode_cleared(timeout_s=1.2, poll_s=0.05, settle_s=2.0)
    if pre_mode_active:
        if ok:
            return {
                "success": True,
                "error": "",
                "mode_exited": True,
                "pre_active_modes": _snapshot_active_modes(pre),
                "post_active_modes": _snapshot_active_modes(post),
                "precheck": pre,
                "postcheck": post,
            }
        return {
            "success": False,
            "error": (
                "exit_current_interaction_mode_post_check_failed:realtime_products_unavailable"
                if not post_available
                else "exit_current_interaction_mode_post_check_failed:interaction_mode_still_active"
            ),
            "pre_active_modes": _snapshot_active_modes(pre),
            "post_active_modes": _snapshot_active_modes(post),
            "precheck": pre,
            "postcheck": post,
        }

    return {
        "success": True,
        "error": "",
        "mode_exited": bool(post_available and (not _interaction_mode_active(post))),
        "pre_active_modes": _snapshot_active_modes(pre),
        "post_active_modes": _snapshot_active_modes(post),
        "precheck": pre,
        "postcheck": post,
    }

def enter_spices_sprinkle_mode():
    """
    调料播撒类型动作。手持调料罐并对准物品或容器情况下，进入香料播撒模式。这是进行香料播撒的前提
    Args:

    """
    key="left"
    click_mouse(key)
    ok, snap = _wait_mode_enabled("is_sprinkle_mode", timeout_s=1.8, poll_s=0.05, settle_s=2.0)
    if ok:
        stable_ok, stable_available, stable_snap = _wait_mode_condition_stable(
            "is_sprinkle_mode",
            expected=True,
            timeout_s=2.4,
            poll_s=0.05,
            stable_s=2.0,
        )
        if stable_available:
            snap = stable_snap
        if not stable_ok:
            ok = False
    if ok:
        return {"success": True, "mode": "sprinkle_mode", "mode_entered": True, "error": ""}
    return {
        "success": False,
        "mode": "sprinkle_mode",
        "mode_entered": False,
        "error": "enter_sprinkle_mode_failed:not_in_sprinkle_mode",
        "hint": "Before entering sprinkle mode, hold a seasoning shaker and aim at an item or container, then call enter_spices_sprinkle_mode.",
        "precheck": snap,
    }

def exit_spices_sprinkle_mode():
    """
    调料播撒类型动作。在香料播撒模式情况下，退出香料播撒模式
    Args:

    """
    pre_available, pre = _hold_snapshot()
    pre_sprinkle = bool(pre_available and bool(pre.get("is_sprinkle_mode")))

    key="right"
    click_mouse(key)

    ok, post_available, post = _wait_mode_disabled("is_sprinkle_mode", timeout_s=1.2, poll_s=0.05, settle_s=2.0)
    raw = {
        "mode_exited": bool(ok),
        "pre_active_modes": _snapshot_active_modes(pre),
        "post_active_modes": _snapshot_active_modes(post),
        "precheck": pre,
        "postcheck": post,
    }
    if pre_sprinkle:
        if ok:
            raw.update({"success": True, "error": ""})
            return raw
        raw.update(
            {
                "success": False,
                "error": (
                    "exit_spices_sprinkle_mode_post_check_failed:realtime_products_unavailable"
                    if not post_available
                    else "exit_spices_sprinkle_mode_post_check_failed:sprinkle_mode_still_active"
                ),
            }
        )
        return raw
    raw.update({"success": True, "error": ""})
    return raw

def adjust_sprinkle_weight():
    """
    调料播撒类型动作。在香料播撒模式下，调整倾倒重量
    Args:

    """
    key="e"
    press_keyboard(key)

########################################################
#                                                      #
#              3.4 旋转与空间控制                      #
#                 (共12个函数)                         #
#                                                      #
########################################################

def rotate_clockwise_along_the_Yaxis():
    """
    旋转与空间控制动作。将物品或工具在能控制对象位姿时，沿Y轴顺时针旋转，通过键盘实现
    Args:

    """
    key="w"
    hold_keyboard(key)
    time.sleep(0.2)
    leave_keyboard(key)

def rotate_counterclockwise_along_the_Yaxis():
    """
    旋转与空间控制动作。将物品或工具在能控制对象位姿时，沿Y轴逆时针旋转，通过键盘实现
    Args:

    """
    key="s"
    hold_keyboard(key)
    time.sleep(0.2)
    leave_keyboard(key)

def rotate_along_the_Yaxis(x):
    """
    旋转与空间控制动作。将物品或工具在能控制对象位姿时，绕Y轴旋转，通过鼠标移动实现
    Args:
      x: 沿y轴顺逆时针的旋转角度。

    """
    key="left"
    hold_mouse(key)
    move_related_mouse(x, 0)

def rotate_clockwise_along_the_Zaxis():
    """
    旋转与空间控制动作。将物品或工具在能控制对象位姿时，沿Z轴顺时针旋转
    Args:

    """
    key="d"
    hold_keyboard(key)
    time.sleep(0.2)
    leave_keyboard(key)

def rotate_counterclockwise_along_the_Zaxis():
    """
    旋转与空间控制动作。将物品或工具在能控制对象位姿时，沿Z轴逆时针旋转
    Args:

    """
    key="a"
    hold_keyboard(key)
    time.sleep(0.2)
    leave_keyboard(key)

def switch_to_advanced_control_mode():
    """
    旋转与空间控制动作。对于手持状态情况下，切换高级控制，以支持空间移动旋转等操作
    Args:


    """
    key="shift"
    hold_keyboard(key)

def horizontal_movement(x, y):
    """
    旋转与空间控制动作。对于特殊模式或者高级控制状态下，对物品进行空间上的水平移动
    Args:
      x: 水平左右移动距离（像素）。
      y: 水平前后移动距离（像素）。

    """
    move_related_mouse(x, y)


# Backward-compat shim: some legacy skills/tools import `Horizontal_movement`.
def Horizontal_movement(x, y):
    return horizontal_movement(x, y)

def spatial_rotation(x, y):
    """
    旋转与空间控制动作。对于高级控制状态下，对于对象进行空间旋转，使其位姿发生变化
    Args:
      x: 物体以视野水平向前为轴顺逆时针旋转角度。
      y: 物体前后翻滚的旋转角度。

    """
    key="right"
    hold_mouse(key)
    move_related_mouse(x, y)

def vertical_movement(y):
    """
    旋转与空间控制动作。对于高级控制状态下，对于对象进行垂直方向上的移动
    Args:
      y: 手持物体垂直移动距离（像素）。

    """
    key="left"
    hold_mouse(key)
    move_related_mouse(0, y)

def switch_to_normal_control_mode():
    """
    旋转与空间控制动作。在高级控制状态下，切换普通控制，以恢复正常的交互状态。
    Args:

    """
    key="shift"
    leave_keyboard(key)

def bottleneck_move_downward(y, ):
    """
    在液体倾倒模式情况下，将瓶口向下倾倒，以流出液体
    Args:
      y: 瓶口上下移动距离（像素）。

    """
    y = _coerce_mouse_delta(y)
    key="left"
    hold_mouse(key)
    move_related_mouse(0, y)

def bottleneck_move_upward(y):
    """
    在液体倾倒模式情况下，将瓶口向上抬起，以停止液体流出
    Args:
      y: 瓶口上下移动距离（像素）。

    """
    y = _coerce_mouse_delta(y)
    key="left"
    hold_mouse(key)
    move_related_mouse(0, -y)

########################################################
#                                                      #
#              3.5 瞄准与精准控制                      #
#                  (共2个函数)                         #
#                                                      #
########################################################
## 这类动作用于需要精准瞄准的场景，切割、倾倒、播撒、翻面等模式下使用，但主要是为了看辅助点的出现与否，暂时不启用具体的瞄准操作
# # 函数 3.5.1 [总函数 49]: 进入瞄准模式
# def Enter_the_aiming_mode(key="w"):
#     """进入瞄准模式"""
#     hold_keyboard(key)
#     # key = w / s

# # 函数 3.5.2 [总函数 50]: 调整瞄准位置
# def Adjust_the_aiming_position(x, y):
#     """调整瞄准位置"""
#     move_related_mouse(x, y)

########################################################
#                                                      #
#                 3.6 切割工具                         #
#                  (共4个函数)                         #
#                                                      #
########################################################

def enter_cutting_mode():
    """
    切割工具类型动作。手持刀时，进入切割模式，以对物品进行瞄准并切割。切割物品时需要先进入切割模式，这是切割的前提
    Args:

    """
    def _is_cutting_board_name(text: str) -> bool:
        raw = str(text or "").strip().lower()
        return any(token in raw for token in ("cutting board", "chopping board", "菜板", "砧板", "cutting_board"))

    try:
        from epm.cerebellum.game_hotkeys import ensure_alt_j_interaction
        from epm.cerebellum.interaction_snapshot import read_interaction_snapshot
        from epm.cerebellum.skills._shared_paths import userdata_root, window_title
        from epm.vision.screen_capture import activate_window

        ensure_alt_j_interaction(
            userdata_root=userdata_root(),
            window_title=window_title("CookingSimulator"),
            activate_window=activate_window,
            io_controller=io_controller,
        )
        snap = read_interaction_snapshot(userdata_root=userdata_root(), max_age_s=2.0)
        if snap is not None:
            pointed_name = str(getattr(snap, "item_name", "") or "").strip()
            container_name = str(getattr(snap, "container_name", "") or "").strip()
            contents = str(getattr(snap, "container_contents", "") or "").strip()
            if _is_cutting_board_name(pointed_name) or _is_cutting_board_name(container_name):
                if not contents:
                    return {
                        "success": False,
                        "mode": "cutting_mode",
                        "mode_entered": False,
                        "error": "enter_cutting_mode_failed:cutting_board_empty",
                        "hint": "The current target is a cutting board, but Alt+J reports no contents on it. Entering cutting mode is meaningless until an ingredient is placed on the board.",
                        "precheck": {
                            "has_target": bool(getattr(snap, "has_target", False)),
                            "item_name": pointed_name,
                            "container_name": container_name,
                            "container_contents": contents,
                            "action": str(getattr(snap, "action", "") or "").strip(),
                            "timestamp": str(getattr(snap, "timestamp", "") or "").strip(),
                        },
                    }
    except Exception:
        pass

    key="left"
    click_mouse(key)
    ok, snap = _wait_mode_enabled("is_cutting_mode", timeout_s=1.8, poll_s=0.05, settle_s=2.0)
    if ok:
        stable_ok, stable_available, stable_snap = _wait_mode_condition_stable(
            "is_cutting_mode",
            expected=True,
            timeout_s=2.4,
            poll_s=0.05,
            stable_s=2.0,
        )
        if stable_available:
            snap = stable_snap
        if not stable_ok:
            ok = False
    if ok:
        return {"success": True, "mode": "cutting_mode", "mode_entered": True, "error": ""}
    return {
        "success": False,
        "mode": "cutting_mode",
        "mode_entered": False,
        "error": "enter_cutting_mode_failed:not_in_cutting_mode",
        "hint": "Before entering cutting mode, hold a knife and aim at a cuttable item or a cutting board, then call enter_cutting_mode.",
        "precheck": snap,
    }

def exit_cutting_mode():
    """
    切割工具类型动作。在切割模式下，切割完毕后，退出切割模式
    Args:

    """
    pre_available, pre = _hold_snapshot()
    pre_cutting = bool(pre_available and bool(pre.get("is_cutting_mode")))

    key="right"
    click_mouse(key)

    ok, post_available, post = _wait_mode_disabled("is_cutting_mode", timeout_s=1.2, poll_s=0.05, settle_s=2.0)
    raw = {
        "mode_exited": bool(ok),
        "pre_active_modes": _snapshot_active_modes(pre),
        "post_active_modes": _snapshot_active_modes(post),
        "precheck": pre,
        "postcheck": post,
    }
    if pre_cutting:
        if ok:
            raw.update({"success": True, "error": ""})
            return raw
        raw.update(
            {
                "success": False,
                "error": (
                    "exit_cutting_mode_post_check_failed:realtime_products_unavailable"
                    if not post_available
                    else "exit_cutting_mode_post_check_failed:cutting_mode_still_active"
                ),
            }
        )
        return raw
    raw.update({"success": True, "error": ""})
    return raw

def knife_cut_down():
    """
    切割工具类型动作。在切割模式下，使刀具垂直下落对正下方物品物品进行切割动作
    Args:

    """
    key="left"
    click_mouse(key)

########################################################
#                                                      #
#                 3.7 液体处理                         #
#                 (共12个函数)                         #
#                                                      #
########################################################

def get_liquid_in_Ladle():
    """
    操作勺子动作。手持勺子时，用勺子盛取特定容量液体。一次性操作的液体容量取决于勺子大小，可以根据勺子名称判断盛取容量
    Args:

    """
    key="left"
    click_mouse(key)

def get_liquid_out_Ladle():
    """
    操作勺子动作。手持勺子且视野对准目标容器时，将勺子中盛取的液体一次性装入目标容器中
    Args:

    """
    key="left"
    click_mouse(key)
    
def pour_out_from_pipette():
    """
    操作移液器动作。手持移液器且视野对准目标容器时，将移液器中的液体装入目标容器中，容量20ml，一次默认操作5ml
    Args:

    """
    key="right"
    click_mouse(key)
    
def pour_in_pipette():
    """
    操作移液器动作。手持移液器且视野对准目标容器时，将目标容器中的液体装入移液器中，容量20ml，一次默认操作5ml
    Args:

    """
    key="right"
    click_mouse(key)

def change_content_5_or_10ml():
    """
    操作移液器动作。手持移液器情况下，将移液器操作的单次液体容量更改装量5ML或者10ML。默认是5ml，再按一次变10ml，再按一次变5ml
    Args:

    """
    key="e"
    press_keyboard(key)

def move_faucet_container(x: int):
    """
    液体容器与水龙头的交互动作。加水模式情况下，移动液体容器位置垂直方向上对准水龙头，以便加水
    Args:
      x: 水平方向上移动距离（像素）。

    """
    move_related_mouse(x, 0)

def change_faucet_pipe_rotation():
    """
    液体容器与水龙头的交互动作。
    可在两种前提下执行：
    1) 已经进入水龙头加水模式；
    2) 已经导航到水龙头区域，并且视角已经对准水龙头出水管。
    执行后会检查水龙头出水管在水平面距离上是否发生偏转，并反馈它当前更靠近左侧还是右侧水池放置点。
    Args:

    """
    data = _read_realtime_products_quick(retries=2, sleep_s=0.02)
    if not _realtime_entries(data):
        return {
            "success": False,
            "error": "change_faucet_pipe_rotation_precheck_failed:realtime_products_unavailable",
            "precheck": {},
            "postcheck": {},
            "verification": "Failed to read faucet state before changing the faucet direction.",
        }

    faucet_mode_active, faucet_mode_meta = _check_faucet_fill_mode_context()
    mode_snap = dict(faucet_mode_meta.get("mode_snap") or _mode_snapshot(data))
    pipe_entry = _find_faucet_pipe_entry(data)
    centered_pipe, centered_reason = _find_centered_faucet_pipe_entry(data)

    if pipe_entry is None:
        return {
            "success": False,
            "error": "change_faucet_pipe_rotation_precheck_failed:faucet_pipe_not_found",
            "precheck": {},
            "postcheck": {},
            "mode_precheck": mode_snap,
            "faucet_mode_precheck": faucet_mode_meta,
            "verification": "Faucet pipe was not found in the current scan, so the faucet direction cannot be verified.",
        }

    precheck = _interaction_target_snapshot(pipe_entry, kind="faucet_pipe")
    pre_side_analysis = _faucet_pipe_side_analysis(data, pipe_entry)

    if (not faucet_mode_active) and (centered_pipe is None):
        return {
            "success": False,
            "error": f"change_faucet_pipe_rotation_precheck_failed:{centered_reason or 'not_in_faucet_mode_or_not_targeting_faucet_pipe'}",
            "precheck": precheck,
            "postcheck": precheck,
            "mode_precheck": mode_snap,
            "faucet_mode_precheck": faucet_mode_meta,
            "pre_side_analysis": pre_side_analysis,
            "verification": "This action requires either faucet-filling mode to be active, or the camera to be aimed at the faucet pipe.",
            "action_performed": False,
        }

    click_mouse("left")
    changed, available, postcheck, post_side_analysis = _trace_faucet_pipe_direction_change(pipe_entry)
    closer_side = str(post_side_analysis.get("closer_sink_place_point") or "unknown")
    side_text = {
        "left": "Left Sink Place Point",
        "right": "Right Sink Place Point",
        "equal": "both sink place points equally",
        "unknown": "an unknown sink place point",
    }.get(closer_side, "an unknown sink place point")

    return {
        "success": bool(changed),
        "error": (
            ""
            if changed
            else (
                "change_faucet_pipe_rotation_post_check_failed:realtime_products_unavailable"
                if not available
                else "change_faucet_pipe_rotation_post_check_failed:faucet_pipe_not_rotated"
            )
        ),
        "precheck": precheck,
        "postcheck": postcheck,
        "mode_precheck": mode_snap,
        "faucet_mode_precheck": faucet_mode_meta,
        "pre_side_analysis": pre_side_analysis,
        "post_side_analysis": post_side_analysis,
        "closest_sink_place_point": closer_side,
        "verification": (
            f"Faucet pipe direction changed successfully. Based on horizontal-plane distance, it is now closer to {side_text}."
            if changed
            else f"Failed to confirm that the faucet pipe direction changed. Based on the latest horizontal-plane distance reading, it is closer to {side_text}."
        ),
        "action_performed": True,
    }

def open_faucet_right_handle_in_faucet_filling_mode(target_ml):
    """
    Faucet filling mode action. Open the right faucet handle in faucet-filling mode to add cold water to the current container.
    `target_ml` is required and is interpreted as the target liquid increment for this fill action.
    After the requested amount is reached, the action stops automatically and exits the faucet interaction.
    The returned feedback reports how much liquid was poured, how much liquid the container actually holds,
    and warns when the two differ significantly, which usually means some liquid spilled or overflowed.
    Args:
      target_ml: Required target liquid increment for this faucet fill. The default unit is milliliters.
        String inputs with units such as "1500 ml" or "1.5 l" are also accepted.
    """
    key="e"
    return _auto_fill_faucet_handle(
        handle_key=key,
        handle_name="right",
        target_ml=target_ml,
    )

def open_faucet_left_handle_in_faucet_filling_mode(target_ml):
    """
    Faucet filling mode action. Open the left faucet handle in faucet-filling mode to add hot water to the current container.
    `target_ml` is required and is interpreted as the target liquid increment for this fill action.
    After the requested amount is reached, the action stops automatically and exits the faucet interaction.
    The returned feedback reports how much liquid was poured, how much liquid the container actually holds,
    and warns when the two differ significantly, which usually means some liquid spilled or overflowed.
    Args:
      target_ml: Required target liquid increment for this faucet fill. The default unit is milliliters.
        String inputs with units such as "1500 ml" or "1.5 l" are also accepted.
    """
    key="q"
    return _auto_fill_faucet_handle(
        handle_key=key,
        handle_name="left",
        target_ml=target_ml,
    )

def screw_faucet_right_handle(target_ml):
    """
    Direct faucet fill action. Use this when the container has already been placed on a sink place point and the camera is already aimed at the right faucet handle. The action starts direct filling from the right handle and stops automatically after the added amount reaches `target_ml`.
    Typical use: first place the container at `Left Sink Place Point` or `Right Sink Place Point`, navigate to the faucet area, aim at the right handle, then call this action.
    Args:
      target_ml: Required target liquid increment for this direct faucet fill. The default unit is milliliters.
        String inputs with units such as "1500 ml" or "1.5 l" are also accepted.
    """
    return _auto_fill_faucet_handle_direct(handle_name="right", target_ml=target_ml)

def screw_faucet_left_handle(target_ml):
    """
    Direct faucet fill action. Use this when the container has already been placed on a sink place point and the camera is already aimed at the left faucet handle. The action starts direct filling from the left handle and stops automatically after the added amount reaches `target_ml`.
    Typical use: first place the container at `Left Sink Place Point` or `Right Sink Place Point`, navigate to the faucet area, aim at the left handle, then call this action.
    Args:
      target_ml: Required target liquid increment for this direct faucet fill. The default unit is milliliters.
        String inputs with units such as "1500 ml" or "1.5 l" are also accepted.
    """
    return _auto_fill_faucet_handle_direct(handle_name="left", target_ml=target_ml)

def enter_the_faucet_filling_mode():
    """
    Faucet filling setup action. Preconditions: hold a pourable container and aim at the Faucet Base area before calling this. It enters faucet-filling mode so you can then use `open_faucet_right_handle_in_faucet_filling_mode`, `open_faucet_left_handle_in_faucet_filling_mode`, or `exit_the_faucet_filling_mode`. Alternative workflow: place the container on a sink place point and navigate to the faucet/sink area, then directly use `screw_faucet_right_handle` or `screw_faucet_left_handle`.
    While holding a liquid container and aiming at the faucet, enter the faucet-filling state and align the container to a usable filling position under the faucet.
    Use this action before calling the faucet-filling-mode handle actions.
    The returned feedback states whether the faucet interaction was entered successfully and whether the container
    was aligned with the faucet.
    Args:

    """
    pre_available, pre = _hold_snapshot()
    allowed, reason_code, held_name, held_kind = _held_item_allows_pouring(pre)
    if pre_available and not allowed:
        return {
            "success": False,
            "mode": "faucet_filling_mode",
            "mode_entered": False,
            "error": "enter_faucet_filling_mode_failed:no_pourable_container_held",
            "verification": "A pourable container or liquid source must be held before entering water-filling mode.",
            "feedback_code": "no_pourable_container_held",
            "feedback_to_planner": (
                "Before entering faucet-filling mode, first hold a pourable container or liquid source. "
                f"Current held item{(' ' + repr(held_name)) if held_name else ''} is not valid."
            ),
            "precheck": pre,
            "held_item_name": held_name,
            "held_item_kind": held_kind,
            "held_item_reason": reason_code,
        }
    click_mouse("left")

    mode_ok, mode_snap = _wait_mode_enabled(
        "is_pouring_mode",
        timeout_s=FAUCET_FILLING_UI_TIMEOUT_S,
        poll_s=FAUCET_FILLING_UI_POLL_S,
        settle_s=0.0,
    )
    if not mode_ok:
        return {
            "success": False,
            "mode": "faucet_filling_mode",
            "mode_entered": False,
            "error": "enter_faucet_filling_mode_failed:is_pouring_mode_not_detected",
            "verification": "Failed to confirm water-filling mode entry.",
            "precheck": pre,
            "postcheck": mode_snap,
        }

    prev_ui_mtime = _trigger_ui_targets_dump()
    ui_ok, ui_texts, ui_data = _wait_ui_dump_contains_texts(
        ("RIGHT HANDLE", "LEFT HANDLE"),
        previous_mtime=prev_ui_mtime,
        timeout_s=FAUCET_FILLING_UI_TIMEOUT_S,
        poll_s=FAUCET_FILLING_UI_POLL_S,
    )
    if not ui_ok:
        return {
            "success": False,
            "mode": "faucet_filling_mode",
            "mode_entered": False,
            "error": "enter_faucet_filling_mode_failed:faucet_handles_ui_not_detected",
            "verification": "Failed to confirm the water-filling UI.",
            "precheck": pre,
            "postcheck": mode_snap,
            "ui_texts_sample": ui_texts[:20],
            "ui_dump_keys": list(ui_data.keys())[:20] if isinstance(ui_data, dict) else [],
        }

    aligned, align_info = _align_faucet_mode_container_to_pipe(
        threshold_m=FAUCET_FILLING_ALIGN_THRESHOLD_M,
        step_px=FAUCET_FILLING_ALIGN_STEP_PX,
        max_steps=FAUCET_FILLING_ALIGN_MAX_STEPS,
    )
    if aligned:
        available, post = _hold_snapshot()
        return {
            "success": True,
            "mode": "faucet_filling_mode",
            "mode_entered": True,
            "error": "",
            "verification": "Entered water-filling mode and aligned the container with the faucet.",
            "precheck": pre,
            "postcheck": post,
            "realtime_available": bool(available),
            "alignment": align_info,
            "ui_handles_detected": ["RIGHT HANDLE", "LEFT HANDLE"],
        }

    available, post = _hold_snapshot()
    return {
        "success": False,
        "mode": "faucet_filling_mode",
        "mode_entered": True,
        "error": str(align_info.get("error") or "enter_faucet_filling_mode_failed:faucet_pipe_alignment_not_reached"),
        "verification": "Entered water-filling mode, but failed to align the container with the faucet.",
        "precheck": pre,
        "postcheck": post,
        "realtime_available": bool(available),
        "alignment": align_info,
        "ui_handles_detected": ["RIGHT HANDLE", "LEFT HANDLE"],
    }

def exit_the_faucet_filling_mode():
    """
    液体容器与水龙头的交互动作。加水模式情况下，退出加水模式，恢复正常手持液体容器状态
    Args:

    """
    pre_available, pre = _hold_snapshot()
    key="right"
    click_mouse(key)
    held_ok, post_available, post = _wait_hold_state(expected_held=True, timeout_s=0.8, poll_s=0.05)
    if held_ok or (not post_available):
        return {
            "success": True,
            "mode": "faucet_filling_mode",
            "mode_exited": True,
            "error": "",
            "verification": "best_effort_no_dedicated_mode_flag",
            "precheck": pre,
            "postcheck": post,
        }
    return {
        "success": False,
        "mode": "faucet_filling_mode",
        "mode_exited": False,
        "error": "exit_faucet_filling_mode_failed:not_holding_after_click",
        "verification": "best_effort_no_dedicated_mode_flag",
        "precheck": pre,
        "postcheck": post,
    }

########################################################
#                                                      #
#                 3.8 选择控制                         #
#                  (共3个函数)                         #
#                                                      #
########################################################
## 这类动作是在拿取容器执行往容器中拾取物品时使用的，主要是通过滚轮来控制选取数量的多少，由于和现实场景操作逻辑关联不大，暂时不启用
# # 函数 3.8.1 [总函数 67]: 选取更少
# def Select_fewer(x, y, key="mid"):
#     """选取更少"""
#     smooth_move_to_position(x, y)
#     Scroll_down_the_wheel()
#     # key = mid

# # 函数 3.8.2 [总函数 68]: 选取更多
# def Select_more(x, y, key="mid"):
#     """选取更多"""
#     smooth_move_to_position(x, y)
#     Scroll_up_the_wheel()
#     # key = mid

# # 函数 3.8.3 [总函数 69]: 更改选取模式
# def Change_selecting_mode(x, y, key="q"):
#     """更改选取模式"""
#     smooth_move_to_position(x, y)
#     press_keyboard(key)
#     # key = q

########################################################
#                                                      #
#              3.9 铲子与夹子操作                      #
#                  (共4个函数)                         #
#                                                      #
########################################################
def enter_the_flipping_mode():
    """
    铲子交互动作。手持铲子且铲起食材后，进入翻动模式，以便对于食材进行翻面操作，这是翻面的前提。比如煎肉饼时需要翻面
    Args:

    """
    key="left"
    click_mouse(key)
    ok, snap = _wait_mode_enabled("is_flip_mode")
    if ok:
        return {"success": True, "mode": "flipping_mode", "mode_entered": True, "error": ""}
    return {
        "success": False,
        "mode": "flipping_mode",
        "mode_entered": False,
        "error": "enter_flipping_mode_failed:not_in_flipping_mode",
        "hint": "Before entering flipping mode, hold a spatula and aim at a platform point on a griddle or grill, then call enter_the_flipping_mode.",
        "precheck": snap,
    }

def exit_the_flipping_mode():
    """
    铲子交互动作。在翻动模式下，退出翻动模式，恢复正常铲子持有状态
    Args:

    """
    pre_available, pre = _hold_snapshot()
    pre_flipping = bool(pre_available and bool(pre.get("is_flip_mode")))

    key="right"
    click_mouse(key)
    
    ok, post_available, post = _wait_mode_disabled("is_flip_mode", timeout_s=1.2, poll_s=0.05, settle_s=2.0)
    raw = {
        "mode_exited": bool(ok),
        "pre_active_modes": _snapshot_active_modes(pre),
        "post_active_modes": _snapshot_active_modes(post),
        "precheck": pre,
        "postcheck": post,
    }
    if pre_flipping:
        if ok:
            raw.update({"success": True, "error": ""})
            return raw
        raw.update(
            {
                "success": False,
                "error": (
                    "exit_the_flipping_mode_post_check_failed:realtime_products_unavailable"
                    if not post_available
                    else "exit_the_flipping_mode_post_check_failed:flipping_mode_still_active"
                ),
            }
        )
        return raw
    raw.update({"success": True, "error": ""})
    return raw
    
def pick_up_with_spatula():
    """
    铲子交互动作。手持铲子情况下对准目标物品，铲起食材
    Args:

    """
    key="left"
    click_mouse(key)

def flip_over():
    """
    铲子交互动作。在翻动模式下，将铲子上的食材翻面到平面上
    Args:

    """
    key="e"
    press_keyboard(key)

def pick_up_with_tongs():
    """
    夹钳交互动作。持有夹钳且对准目标物品情况下，夹取食材，但用不上，可忽略
    Args:

    """
    key="left"
    click_mouse(key)

def release_with_tongs():
    """
    夹钳交互动作。在用夹钳夹住物品时，松开夹钳，但用不上，可忽略
    Args:

    """
    key="e"
    press_keyboard(key)

########################################################
#                                                      #
#              3.10 搅拌机操作                         #
#                  (共6个函数)                         #
#                                                      #
########################################################
def enter_mixing_mode():
    """
    搅拌机交互类动作。持有搅拌机且视野对准容器情况下，进入搅拌模式，以便对容器内物品进行搅拌操作，这是搅拌的前提
    Args:

    """
    key="left"
    click_mouse(key)
    ok, snap = _wait_mode_enabled("is_mixing_mode", timeout_s=1.8, poll_s=0.05, settle_s=2.0)
    if ok:
        stable_ok, stable_available, stable_snap = _wait_mode_condition_stable(
            "is_mixing_mode",
            expected=True,
            timeout_s=2.4,
            poll_s=0.05,
            stable_s=2.0,
        )
        if stable_available:
            snap = stable_snap
        if not stable_ok:
            ok = False
    if ok:
        return {"success": True, "mode": "mixing_mode", "mode_entered": True, "error": ""}
    return {
        "success": False,
        "mode": "mixing_mode",
        "mode_entered": False,
        "error": "enter_mixing_mode_failed:not_in_mixing_mode",
        "hint": "Before entering mixing mode, hold a blender and aim at a container, then call enter_mixing_mode.",
        "precheck": snap,
    }


def _ensure_interaction_mode_for_skill(
    mode_key: str,
    mode_name: str,
    enter_mode_fn,
) -> dict:
    """
    Best-effort mode precheck used by integrated auto_* skills.

    Returns a structured dict:
      {"success": bool, "mode": str, "mode_entered": bool, "error": str, "precheck": dict, "hint": str}
    """
    available, snap = _hold_snapshot()
    if available and bool(snap.get(mode_key, False)):
        stable_ok, stable_available, stable_snap = _wait_mode_condition_stable(
            mode_key,
            expected=True,
            timeout_s=2.4,
            poll_s=0.05,
            stable_s=2.0,
        )
        if stable_available:
            snap = stable_snap
        if stable_ok:
            return {
                "success": True,
                "mode": str(mode_name),
                "mode_entered": False,
                "error": "",
                "precheck": snap,
                "hint": "",
            }
        return {
            "success": False,
            "mode": str(mode_name),
            "mode_entered": False,
            "error": f"enter_{mode_name}_failed:mode_not_stable_for_2s",
            "precheck": snap,
            "hint": f"{mode_name} was detected briefly, but did not stay active for 2 seconds.",
        }
    try:
        result = enter_mode_fn()
    except Exception as e:
        return {
            "success": False,
            "mode": str(mode_name),
            "mode_entered": False,
            "error": f"enter_{mode_name}_exception:{e}",
            "precheck": snap,
            "hint": "",
        }

    if isinstance(result, dict):
        out = dict(result)
        out.setdefault("mode", str(mode_name))
        out.setdefault("mode_entered", bool(out.get("success", False)))
        out.setdefault("error", "")
        out.setdefault("precheck", snap)
        out.setdefault("hint", "")
        return out

    ok = bool(result)
    return {
        "success": bool(ok),
        "mode": str(mode_name),
        "mode_entered": bool(ok),
        "error": "" if ok else f"enter_{mode_name}_failed",
        "precheck": snap,
        "hint": "",
    }
    
def exit_mixing_mode():
    """
    搅拌机交互类动作。搅拌模式情况下，退出搅拌模式，恢复正常搅拌机持有状态
    Args:

    """
    pre_available, pre = _hold_snapshot()
    pre_mixing = bool(pre_available and bool(pre.get("is_mixing_mode")))

    key="right"
    click_mouse(key)
    
    ok, post_available, post = _wait_mode_disabled("is_mixing_mode", timeout_s=1.2, poll_s=0.05, settle_s=2.0)
    raw = {
        "mode_exited": bool(ok),
        "pre_active_modes": _snapshot_active_modes(pre),
        "post_active_modes": _snapshot_active_modes(post),
        "precheck": pre,
        "postcheck": post,
    }
    if pre_mixing:
        if ok:
            raw.update({"success": True, "error": ""})
            return raw
        raw.update(
            {
                "success": False,
                "error": (
                    "exit_mixing_mode_post_check_failed:realtime_products_unavailable"
                    if not post_available
                    else "exit_mixing_mode_post_check_failed:mixing_mode_still_active"
                ),
            }
        )
        return raw
    raw.update({"success": True, "error": ""})
    return raw

def toggle_on_blender():
    """
    搅拌机交互类动作。打开搅拌机, 工具作用为搅拌固体物品为液体
    Args:

    """
    key="e"
    press_keyboard(key)

def toggle_off_blender():
    """
    搅拌机交互类动作。关闭搅拌机
    Args:

    """
    key="e"
    press_keyboard(key)

def blender_downward():
    """
    搅拌机交互类动作。搅拌模式情况下，将搅拌机竖直向下沉，以搅拌到容器底部
    Args:

    """
    key="q"
    press_keyboard(key)

def blender_upward():
    """
    搅拌机交互类动作。搅拌机竖直向上浮，以停止搅拌，也可以直接退出搅拌模式
    Args:

    """
    key="e"
    press_keyboard(key)

########################################################
#                                                      #
#                3.11 瓶子操作                         #
#                  (共4个函数)                         #
#                                                      #
########################################################

def screw_bottle():
    """
    液体瓶交互类动作。持有液体瓶情况下，拧紧瓶盖
    Args:

    """
    key="e"
    press_keyboard(key)

def unscrew_bottle():
    """
    液体瓶交互类动作。持有液体瓶情况下，拧松瓶盖
    Args:

    """
    key="e"
    press_keyboard(key)

def crack_open_the_egg():
    """
    鸡蛋交互类动作。手持鸡蛋时，敲开鸡蛋。最终蛋清在手中，蛋壳垂直掉落
    Args:

    """
    key="e"
    press_keyboard(key)

########################################################
#                                                      #
#              3.12 烹饪设备操作                        #
#                 (共37个函数)                         #
#                                                      #
########################################################

#######################################################
#                                                     #
#           → 3.12.1 炉灶控制                        #
#               (共6个函数)                           #
#                                                     #
#######################################################

def toggle_on_switch():
    """
    电器开关交互类动作。打开一个位置的煤气灶、或者平板煎炉、烤箱、条纹煎炉、炸炉等，开关名都带switch一词，执行此操作前需要导航并对准开关位置
    Args:

    """
    key="left"
    click_mouse(key)

def toggle_off_switch():
    """
    电器开关交互类动作。关闭一个位置的煤气灶、或者平板煎炉、烤箱、条纹煎炉、炸炉等，开关名都带switch一词，执行此操作前需要导航并对准开关位置
    Args:

    """
    key="left"
    click_mouse(key)


# # 函数 3.12.1.2 [总函数 85]: 关闭一个位置的煤气灶
# def Turn_off_the_stove_about_certain_position(x, y, key="left"):
#     """关闭一个位置的煤气灶"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.1.3 [总函数 86]: 打开煤气炉烤箱门
# def Open_the_door_of_the_stove(x, y, key="left"):
#     """打开煤气炉烤箱门"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.1.4 [总函数 87]: 关闭煤气炉烤箱门
# def Close_the_door_of_the_stove(x, y, key="left"):
#     """关闭煤气炉烤箱门"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.1.5 [总函数 88]: 打开煤气炉烤箱
# def Turn_on_the_oven_of_stove(x, y, key="left"):
#     """打开煤气炉烤箱"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.1.6 [总函数 89]: 关闭煤气炉烤箱
# def Turn_off_the_oven_of_stove(x, y, key="left"):
#     """关闭煤气炉烤箱"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

#######################################################
#                                                     #
#           → 3.12.2 油炸锅控制                      #
#               (共4个函数)                           #
#                                                     #
#######################################################

def togggle_add_oil_into_Deep_fryer():
    """
    炸炉专属交互动作。点击工具上Fryer 1/2 Buntton On按钮,炸炉加注食用油。前提是要先对准炸炉加油按钮
    Args:

    """
    key="left"
    click_mouse(key)

def toggle_pour_oil_out_Deep_fryer():
    """
    炸炉专属交互动作。点击工具上Fryer 1/2 Buntton Off按钮,炸炉排出食用油，前提是要先对准炸炉排油按钮
    Args:

    """
    key="left"
    click_mouse(key)


# # 函数 3.12.2.3 [总函数 92]: 打开炸炉加热功能
# def Turn_on_Deep_fryer_heating(x, y, key="left"):
#     """打开炸炉加热功能"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.2.4 [总函数 93]: 关闭炸炉加热功能
# def Turn_off_Deep_fryer_heating(x, y, key="left"):
#     """关闭炸炉加热功能"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

#######################################################
#                                                     #
#           → 3.12.3 烤箱控制                        #
#               (共4个函数)                           #
#                                                     #
#######################################################

# # 函数 3.12.3.1 [总函数 94]: 打开烤箱门
# def Open_the_door_of_the_oven(x, y, key="left"):
#     """打开烤箱门"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.3.2 [总函数 95]: 打开烤箱加热功能
# def Turn_on_oven_heating(x, y, key="left"):
#     """打开烤箱加热功能"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.3.3 [总函数 96]: 关闭烤箱门
# def Close_the_door_of_the_oven(x, y, key="left"):
#     """关闭烤箱门"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.3.4 [总函数 97]: 关闭烤箱加热功能
# def Turn_off_oven_heating(x, y, key="left"):
#     """关闭烤箱加热功能"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

#######################################################
#                                                     #
#         → 3.12.4 平板煎锅与烤架控制                #
#               (共4个函数)                           #
#                                                     #
#######################################################

# # 函数 3.12.4.1 [总函数 98]: 打开平板煎炉加热功能
# def Turn_on_Griddle_heating(x, y, key="left"):
#     """打开平板煎炉加热功能"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.4.2 [总函数 99]: 关闭平板煎炉加热功能
# def Turn_off_Griddle_heating(x, y, key="left"):
#     """关闭平板煎炉加热功能"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.4.3 [总函数 100]: 打开条纹烤炉加热功能
# def Turn_on_Grill_heating(x, y, key="left"):
#     """打开条纹烤炉加热功能"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.4.4 [总函数 101]: 关闭条纹烤炉加热功能
# def Turn_off_Grill_heating(x, y, key="left"):
#     """关闭条纹烤炉加热功能"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

#######################################################
#                                                     #
#           → 3.12.5 冰箱控制                        #
#               (共4个函数)                           #
#                                                     #
#######################################################

def open_door():
    """
    打开一切门类动作，如冰箱左门、右门、烤箱门等,门都带有door一词
    Args:

    """
    data = _read_realtime_products_quick(retries=2, sleep_s=0.02)
    if not _realtime_entries(data):
        return {
            "success": False,
            "error": "open_door_precheck_failed:realtime_products_unavailable",
            "precheck": {},
            "postcheck": {},
            "verification": "Failed to read door state before opening.",
        }

    target, reason = _find_centered_interaction_entry(
        data,
        kind="door",
        max_center_dist_px=DOOR_CENTER_MAX_DIST_PX,
    )
    if target is None:
        return {
            "success": False,
            "error": f"open_door_precheck_failed:{reason or 'no_targeted_door'}",
            "precheck": {},
            "postcheck": {},
            "verification": "No valid door target is both centered in view and within 2 meters, so opening a door is not a reasonable action right now.",
        }

    precheck = _interaction_target_snapshot(target, kind="door")
    if bool(target.get("is_open", False)):
        return {
            "success": True,
            "error": "",
            "precheck": precheck,
            "postcheck": precheck,
            "verification": "Door is already open.",
            "action_performed": False,
        }

    click_mouse("left")
    opened, available, postcheck, angle_trace = _trace_door_open_state(
        target,
        expected_open=True,
    )
    return {
        "success": bool(opened),
        "error": (
            ""
            if opened
            else (
                "open_door_post_check_failed:realtime_products_unavailable"
                if not available
                else "open_door_post_check_failed:door_not_open"
            )
        ),
        "precheck": precheck,
        "postcheck": postcheck,
        "open_angle_trace": angle_trace,
        "verification": (
            "Door opened successfully."
            if opened
            else "Failed to confirm that the door is open."
        ),
        "action_performed": True,
    }

def close_door():
    """
    关闭一切门类动作，如冰箱门、烤箱门等，门都带有door一词
    Args:

    """
    data = _read_realtime_products_quick(retries=2, sleep_s=0.02)
    if not _realtime_entries(data):
        return {
            "success": False,
            "error": "close_door_precheck_failed:realtime_products_unavailable",
            "precheck": {},
            "postcheck": {},
            "verification": "Failed to read door state before closing.",
        }

    target, reason = _find_centered_interaction_entry(
        data,
        kind="door",
        max_center_dist_px=DOOR_CENTER_MAX_DIST_PX,
    )
    if target is None:
        return {
            "success": False,
            "error": f"close_door_precheck_failed:{reason or 'no_targeted_door'}",
            "precheck": {},
            "postcheck": {},
            "verification": "No valid door target is both centered in view and within 2 meters, so closing a door is not a reasonable action right now.",
        }

    precheck = _interaction_target_snapshot(target, kind="door")
    if not bool(target.get("is_open", False)):
        return {
            "success": True,
            "error": "",
            "precheck": precheck,
            "postcheck": precheck,
            "verification": "Door is already closed.",
            "action_performed": False,
        }

    click_mouse("left")
    closed, available, postcheck, angle_trace = _trace_door_open_state(
        target,
        expected_open=False,
    )
    return {
        "success": bool(closed),
        "error": (
            ""
            if closed
            else (
                "close_door_post_check_failed:realtime_products_unavailable"
                if not available
                else "close_door_post_check_failed:door_still_open"
            )
        ),
        "precheck": precheck,
        "postcheck": postcheck,
        "open_angle_trace": angle_trace,
        "verification": (
            "Door closed successfully."
            if closed
            else "Failed to confirm that the door is closed."
        ),
        "action_performed": True,
    }


def open_drawer():
    """
    打开一切抽屉类动作，如冰箱左抽屉、右抽屉等，抽屉都带有drawer一词。
    Args:

    """
    data = _read_realtime_products_quick(retries=2, sleep_s=0.02)
    if not _realtime_entries(data):
        return {
            "success": False,
            "error": "open_drawer_precheck_failed:realtime_products_unavailable",
            "precheck": {},
            "postcheck": {},
            "verification": "Failed to read drawer state before opening.",
        }

    target, reason = _find_centered_interaction_entry(
        data,
        kind="drawer",
        max_center_dist_px=DRAWER_CENTER_MAX_DIST_PX,
    )
    if target is None:
        return {
            "success": False,
            "error": f"open_drawer_precheck_failed:{reason or 'no_targeted_drawer'}",
            "precheck": {},
            "postcheck": {},
            "verification": "No valid drawer target is both centered in view and within 2 meters, so opening a drawer is not a reasonable action right now.",
        }

    precheck = _interaction_target_snapshot(target, kind="drawer")
    if bool(target.get("is_open", False)):
        return {
            "success": True,
            "error": "",
            "precheck": precheck,
            "postcheck": precheck,
            "verification": "Drawer is already open.",
            "action_performed": False,
        }

    click_mouse("left")
    opened, available, postcheck = _wait_interaction_open_state(
        target,
        expected_open=True,
        kind="drawer",
    )
    return {
        "success": bool(opened),
        "error": (
            ""
            if opened
            else (
                "open_drawer_post_check_failed:realtime_products_unavailable"
                if not available
                else "open_drawer_post_check_failed:drawer_not_open"
            )
        ),
        "precheck": precheck,
        "postcheck": postcheck,
        "verification": (
            "Drawer opened successfully."
            if opened
            else "Failed to confirm that the drawer is open."
        ),
        "action_performed": True,
    }


def close_drawer():
    """
    关闭一切抽屉类动作，如冰箱左抽屉、右抽屉等，抽屉都带有drawer一词。
    Args:

    """
    data = _read_realtime_products_quick(retries=2, sleep_s=0.02)
    if not _realtime_entries(data):
        return {
            "success": False,
            "error": "close_drawer_precheck_failed:realtime_products_unavailable",
            "precheck": {},
            "postcheck": {},
            "verification": "Failed to read drawer state before closing.",
        }

    target, reason = _find_centered_interaction_entry(
        data,
        kind="drawer",
        max_center_dist_px=DRAWER_CENTER_MAX_DIST_PX,
    )
    if target is None:
        return {
            "success": False,
            "error": f"close_drawer_precheck_failed:{reason or 'no_targeted_drawer'}",
            "precheck": {},
            "postcheck": {},
            "verification": "No valid drawer target is both centered in view and within 2 meters, so closing a drawer is not a reasonable action right now.",
        }

    precheck = _interaction_target_snapshot(target, kind="drawer")
    if not bool(target.get("is_open", False)):
        return {
            "success": True,
            "error": "",
            "precheck": precheck,
            "postcheck": precheck,
            "verification": "Drawer is already closed.",
            "action_performed": False,
        }

    click_mouse("left")
    closed, available, postcheck = _wait_interaction_open_state(
        target,
        expected_open=False,
        kind="drawer",
    )
    return {
        "success": bool(closed),
        "error": (
            ""
            if closed
            else (
                "close_drawer_post_check_failed:realtime_products_unavailable"
                if not available
                else "close_drawer_post_check_failed:drawer_still_open"
            )
        ),
        "precheck": precheck,
        "postcheck": postcheck,
        "verification": (
            "Drawer closed successfully."
            if closed
            else "Failed to confirm that the drawer is closed."
        ),
        "action_performed": True,
    }

#######################################################
#                                                     #
#         → 3.12.6 食品处理器控制                    #
#               (共2个函数)                           #
#                                                     #
#######################################################

def toggle_on_food_processor():
    """
    食品料理机专属交互动作。打开料理机转头切割功能，将物品由固体打汁成液体
    Args:

    """
    key="left"
    click_mouse(key)

def toggle_off_food_processor():
    """
    食品料理机专属交互动作。关闭料理机转头切割功能
    Args:

    """
    key="left"
    click_mouse(key)

#######################################################
#                                                     #
#           → 3.12.7 微波炉控制                      #
#               (共6个函数)                           #
#                                                     #
#######################################################

# # 函数 3.12.7.1 [总函数 108]: 打开微波炉门
# def Open_Microwave_oven_door(x, y, key="left"):
#     """打开微波炉门"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.7.2 [总函数 109]: 关闭微波炉门
# def Close_Microwave_oven_door(x, y, key="left"):
#     """关闭微波炉门"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.7.3 [总函数 110]: 调整微波炉火力大小
# def Adjust_Microwave_oven_power(x, y, key="left"):
#     """调整微波炉火力大小"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.7.4 [总函数 111]: 开启加热
# def Turn_on_Microwave_oven(x, y, key="left"):
#     """开启加热"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.7.5 [总函数 112]: 加时5秒钟
# def Add_5_sconds(x, y, key="left"):
#     """加时5秒钟"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left

# # 函数 3.12.7.6 [总函数 113]: 终止微波炉加热
# def Turn_off_Microwave_oven(x, y, key="left"):
#     """终止微波炉加热"""
#     smooth_move_to_position(x, y, click=key)
#     # key = left
    
    
##########################################################
##                                                      ##
##                ACTION DISPATCHER (关键)              ##
##                                                      ##
##########################################################

# 这个字典是 local_agent.py 和 local_actions.py 之间的桥梁。
# 它会自动收集当前文件中所有已定义的函数（除了辅助函数和它自己），
# 让AI知道有哪些动作可以调用。

# 自动发现动作的ACTION_DISPATCHER保持不变，它会自动收集所有函数
def _is_action_function(obj):
    return (
        inspect.isfunction(obj)
        and obj.__module__ == __name__
        and not obj.__name__.startswith("_")
        and obj.__name__
        not in {
            # Hidden from action list / GPT tools (internal helper)
            "gui_enable_steady_hands",
            # Hidden from action list / GPT tools (read-only skills exposed as type=skill)
            "auto_perception",
            "query_scene_objects",
            # Hidden from action list / GPT tools (keyboard/mouse primitives)
            "click_mouse",
            "hold_keyboard",
            "hold_mouse",
            "leave_keyboard",
            "leave_mouse",
            "move_related_mouse",
            "press_keyboard",
            "scroll_down_the_wheel",
            "scroll_up_the_wheel",
            "focus_terminal",
            # Hidden from action list / GPT tools (window helper)
            "get_window_center",
        }
    )

ACTION_DISPATCHER = {
    name: func for name, func in inspect.getmembers(sys.modules[__name__], _is_action_function)
}

########################################################
#                                                      #
#              4. 高级功能操作                         #
#                 (新增封装功能)                       #
#                                                      #
########################################################

# def debug_display_and_templates():
#     """调试显示器配置和模板路径"""
#     import mss
#     import os

#     print("\n=== 显示器配置调试 ===")
#     with mss.mss() as sct:
#         print(f"检测到 {len(sct.monitors) - 1} 个显示器:")
#         for i, monitor in enumerate(sct.monitors):
#             if i == 0:
#                 print(f"  monitors[{i}] (全屏): {monitor}")
#             else:
#                 print(f"  monitors[{i}] (显示器{i}): {monitor}")

#     print("\n=== 模板路径调试 ===")
#     from epm.cerebellum.figure_path_mappings import AdvancedFunctionPaths

#     # 检查滚动模板
#     scroll_paths = AdvancedFunctionPaths.get_scroll_search_paths()
#     print("滚动搜索模板:")
#     for name, path in scroll_paths.items():
#         exists = os.path.exists(path)
#         print(f"  {name}: {path} {'✓' if exists else '✗'}")

#     # 检查一些常用的原材料模板
#     print("\n测试原材料模板:")
#     test_ingredients = ["tomato", "onion", "burger_meat"]
#     from figure_path_mappings import OBJECT_ICON_PATHS
#     for ingredient in test_ingredients:
#         if ingredient in OBJECT_ICON_PATHS:
#             path = OBJECT_ICON_PATHS[ingredient]
#             exists = os.path.exists(path)
#             print(f"  {ingredient}: {path} {'✓' if exists else '✗'}")
#         else:
#             print(f"  {ingredient}: 未在路径映射中找到")

#     return True

# def test_tomato_detection():
#     """专门测试tomato.png的检测"""
#     import cv2
#     import os
#     import numpy as np
#     from figure_path_mappings import OBJECT_ICON_PATHS

#     print("\n=== Tomato 检测测试 ===")

#     # 检查tomato模板
#     if "tomato" not in OBJECT_ICON_PATHS:
#         print("[ERROR] tomato未在OBJECT_ICON_PATHS中找到")
#         return False

#     tomato_path = OBJECT_ICON_PATHS["tomato"]
#     print(f"Tomato模板路径: {tomato_path}")

#     if not os.path.exists(tomato_path):
#         print(f"[ERROR] 模板文件不存在: {tomato_path}")
#         return False

#     # 激活游戏窗口
#     try:
#         _activate_window()
#         window_rect = _get_window_rect()
#         print(f"游戏窗口位置: {window_rect}")

#         # 截图
#         screenshot = _capture_screenshot_mss_numpy(window_rect)
#         print(f"截图成功，尺寸: {screenshot.shape}")

#         # 加载模板
#         template = cv2.imread(tomato_path, cv2.IMREAD_GRAYSCALE)
#         if template is None:
#             print(f"[ERROR] 无法加载模板: {tomato_path}")
#             return False

#         print(f"模板尺寸: {template.shape}")

#         # 转换截图为灰度
#         screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

#         # 执行模板匹配
#         result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
#         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

#         print(f"匹配结果 - 最大值: {max_val:.3f}, 位置: {max_loc}")

#         # 尝试不同阈值
#         thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
#         for threshold in thresholds:
#             locations = np.where(result >= threshold)
#             match_count = len(locations[0])
#             print(f"阈值 {threshold}: 找到 {match_count} 个匹配")

#             if match_count > 0:
#                 # 显示前几个匹配
#                 for i, (y, x) in enumerate(zip(locations[0][:3], locations[1][:3])):
#                     confidence = result[y, x]
#                     print(f"  匹配 {i+1}: 位置({x}, {y}), 置信度: {confidence:.3f}")

#         return max_val > 0.5

#     except Exception as e:
#         print(f"[ERROR] 测试过程中出现错误: {e}")
#         return False

# def scroll_and_find_ingredient(ingredient_template_path: str,
#                               top_template_path: str = None,
#                               bottom_template_path: str = None,
#                               max_scrolls: int = 100,
#                               window_title: str = 'CookingSimulator'):
#     """
#     【功能7】滚动鼠标遍历寻找特定原材料条目并点击
#     在商店界面中滚动查找指定的原材料，找到后自动点击

#     Args:
#         ingredient_template_path: 要查找的原材料模板图片路径
#         top_template_path: 进度条顶部模板路径（可选，使用集中化配置）
#         bottom_template_path: 进度条底部模板路径（可选，使用集中化配置）
#         max_scrolls: 最大滚动次数
#         window_title: 游戏窗口标题

#     Returns:
#         bool: 是否成功找到并点击了原材料
#     """
#     import cv2
#     import numpy as np
#     import time

#     try:
#         # 使用集中化路径配置，如果没有提供则使用默认值
#         if top_template_path is None or bottom_template_path is None:
#             scroll_paths = AdvancedFunctionPaths.get_scroll_search_paths()
#             if top_template_path is None:
#                 top_template_path = scroll_paths["top_template"]
#             if bottom_template_path is None:
#                 bottom_template_path = scroll_paths["bottom_template"]

#         print(f"[*] 使用滚动模板: 顶部={top_template_path}, 底部={bottom_template_path}")

#         # 激活游戏窗口
#         _activate_window(window_title)
#         window_rect = _get_window_rect(window_title)

#         def find_button_in_image(screenshot, template_path):
#             # 直接复制原始代码的实现，增加调试信息
#             template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)  # 加载按钮模板
#             if template is None:
#                 print(f"[ERROR] 无法加载模板图片: {template_path}")
#                 return []

#             print(f"[DEBUG] 模板路径: {template_path}")
#             print(f"[DEBUG] 模板尺寸: {template.shape}")

#             screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)  # 将截图转换为灰度图像
#             print(f"[DEBUG] 截图灰度尺寸: {screenshot_gray.shape}")

#             # 执行模板匹配
#             result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)

#             # 获取最大匹配值用于调试
#             min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#             print(f"[DEBUG] 最大匹配值: {max_val:.3f}")

#             # 降低匹配阈值并提供灵活性
#             threshold = 0.7  # 降低阈值，原来是0.95
#             locations = np.where(result >= threshold)
#             print(f"[DEBUG] 阈值 {threshold} 下找到 {len(locations[0])} 个匹配")

#             # 如果没找到，尝试更低的阈值
#             if len(locations[0]) == 0:
#                 threshold = 0.5
#                 locations = np.where(result >= threshold)
#                 print(f"[DEBUG] 降低阈值到 {threshold}，找到 {len(locations[0])} 个匹配")

#             # 获取匹配区域的矩形框
#             match_rectangles = []
#             for pt in zip(*locations[::-1]):
#                 x, y = pt
#                 w, h = template.shape[1], template.shape[0]  # 获取模板的宽度和高度
#                 match_rectangles.append((x, y, w, h))  # 存储矩形框坐标和大小
#                 print(f"[DEBUG] 找到匹配区域: ({x}, {y}, {w}, {h})")

#             return match_rectangles

#         def detect_scroll_position(top_path, bottom_path):
#             screenshot = _capture_screenshot_mss_numpy(window_rect)

#             if find_button_in_image(screenshot, top_path):
#                 return "top"
#             if find_button_in_image(screenshot, bottom_path):
#                 return "bottom"
#             return "middle"

#         def scroll_and_search(direction="down"):
#             for _ in range(max_scrolls):
#                 screenshot = _capture_screenshot_mss_numpy(window_rect)
#                 matches = find_button_in_image(screenshot, ingredient_template_path)

#                 if matches:
#                     x, y, w, h = matches[0]
#                     center_x = window_rect['left'] + x + w // 2
#                     center_y = window_rect['top'] + y + h // 2

#                     io_controller.mouse_move_absolute(center_x, center_y)
#                     time.sleep(0.1)
#                     io_controller.click('left')
#                     print(f"成功找到并点击原材料: ({center_x}, {center_y})")
#                     return True

#                 # 滚动
#                 if direction == "down":
#                     io_controller.scroll_wheel(-1)
#                     if detect_scroll_position(top_template_path, bottom_template_path) == "bottom":
#                         break
#                 else:
#                     io_controller.scroll_wheel(1)
#                     if detect_scroll_position(top_template_path, bottom_template_path) == "top":
#                         break

#                 time.sleep(0.1)

#             return False

#         # 检查当前位置并决定滚动策略
#         position = detect_scroll_position(top_template_path, bottom_template_path)

#         if position == "top":
#             print("从顶部开始向下滚动搜索")
#             return scroll_and_search("down")
#         elif position == "bottom":
#             print("从底部开始向上滚动搜索")
#             return scroll_and_search("up")
#         else:
#             print("从中间位置先向上滚动到顶部，再向下搜索")
#             # 先滚动到顶部
#             for _ in range(max_scrolls):
#                 io_controller.scroll_wheel(1)
#                 if detect_scroll_position(top_template_path, bottom_template_path) == "top":
#                     break
#                 time.sleep(0.1)
#             # 然后向下搜索
#             return scroll_and_search("down")

#     except Exception as e:
#         print(f"滚动查找原材料时出错: {e}")
#         return False

# def order_dish(dish_template_path: str,
#                order_button_template_path: str = None,
#                window_title: str = 'CookingSimulator'):
#     """
#     【功能9】订菜操作
#     在菜单界面找到指定菜品并点击订购按钮

#     Args:
#         dish_template_path: 菜品模板图片路径
#         order_button_template_path: 订购按钮模板图片路径（可选，使用集中化配置）
#         window_title: 游戏窗口标题

#     Returns:
#         bool: 是否成功订购菜品
#     """
#     import cv2
#     import numpy as np
#     import time

#     try:
#         # 使用集中化路径配置，如果没有提供则使用默认值
#         if order_button_template_path is None:
#             order_paths = AdvancedFunctionPaths.get_order_dish_paths()
#             order_button_template_path = order_paths["order_button_template"]

#         print(f"[*] 使用订单模板: 订购按钮={order_button_template_path}")

#         # 激活游戏窗口
#         _activate_window(window_title)
#         window_rect = _get_window_rect(window_title)

#         # 截取当前屏幕
#         screenshot = _capture_screenshot_mss_numpy(window_rect)
#         screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

#         # 加载模板
#         dish_template = cv2.imread(dish_template_path, cv2.IMREAD_GRAYSCALE)
#         order_template = cv2.imread(order_button_template_path, cv2.IMREAD_GRAYSCALE)

#         # 查找菜品
#         dish_result = cv2.matchTemplate(screenshot_gray, dish_template, cv2.TM_CCOEFF_NORMED)
#         threshold = 0.95
#         dish_locations = np.where(dish_result >= threshold)

#         if len(dish_locations[0]) == 0:
#             print("未找到指定菜品")
#             return False

#         # 计算订购按钮位置（基于菜品位置）
#         for pt in zip(*dish_locations[::-1]):
#             x, y = pt
#             dish_w, dish_h = dish_template.shape[1], dish_template.shape[0]
#             order_w, order_h = order_template.shape[1], order_template.shape[0]

#             # 订购按钮通常在菜品图片的右下角
#             order_x = x + dish_w - order_w
#             order_y = y + dish_h

#             # 转换为屏幕绝对坐标
#             screen_x = window_rect['left'] + order_x + order_w // 2
#             screen_y = window_rect['top'] + order_y + order_h // 2

#             # 点击订购按钮
#             io_controller.mouse_move_absolute(screen_x, screen_y)
#             time.sleep(0.2)
#             io_controller.click('left')
#             print(f"成功点击订购按钮: ({screen_x}, {screen_y})")
#             return True

#     except Exception as e:
#         print(f"订菜操作时出错: {e}")
#         return False

# def recognize_star_rating(taste_label_path: str = None,
#                          time_label_path: str = None,
#                          score_label_path: str = None,
#                          star_template_path: str = None,
#                          window_title: str = 'CookingSimulator'):
#     """
#     【功能19】识别评分星星得分
#     识别游戏中的星级评分（口味、时间、总分）

#     Args:
#         taste_label_path: 口味标签模板路径（可选，使用集中化配置）
#         time_label_path: 时间标签模板路径（可选，使用集中化配置）
#         score_label_path: 总分标签模板路径（可选，使用集中化配置）
#         star_template_path: 星星模板路径（可选，使用集中化配置）
#         window_title: 游戏窗口标题

#     Returns:
#         dict: 包含各项评分的字典，例如 {'taste': 4.2, 'time': 3.8, 'score': 4.0}
#     """
#     import cv2
#     import numpy as np

#     try:
#         # 使用集中化路径配置，如果没有提供则使用默认值
#         if any(path is None for path in [taste_label_path, time_label_path, score_label_path, star_template_path]):
#             rating_paths = AdvancedFunctionPaths.get_star_rating_paths()
#             if taste_label_path is None:
#                 taste_label_path = rating_paths["taste_label_path"]
#             if time_label_path is None:
#                 time_label_path = rating_paths["time_label_path"]
#             if score_label_path is None:
#                 score_label_path = rating_paths["score_label_path"]
#             if star_template_path is None:
#                 star_template_path = rating_paths["star_template_path"]

#         print(f"[*] 使用评分模板: 口味={taste_label_path}, 时间={time_label_path}, 总分={score_label_path}, 星星={star_template_path}")

#         # 激活游戏窗口
#         _activate_window(window_title)
#         window_rect = _get_window_rect(window_title)

#         # 截取游戏窗口
#         screenshot = _capture_screenshot_mss_numpy(window_rect)

#         # 加载模板
#         taste_img = cv2.imread(taste_label_path)
#         time_img = cv2.imread(time_label_path)
#         score_img = cv2.imread(score_label_path)
#         star_img = cv2.imread(star_template_path)

#         if any(img is None for img in [taste_img, time_img, score_img, star_img]):
#             print("无法读取模板图片，请检查路径")
#             return {}

#         # 获取星星尺寸
#         star_h, star_w = star_img.shape[:2]

#         def match_label(big_img, label_img, threshold=0.8):
#             gray_big = cv2.cvtColor(big_img, cv2.COLOR_BGR2GRAY)
#             gray_lbl = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)

#             res = cv2.matchTemplate(gray_big, gray_lbl, cv2.TM_CCOEFF_NORMED)
#             loc = np.where(res >= threshold)
#             if len(loc[0]) == 0:
#                 return None
#             y, x = loc[0][0], loc[1][0]
#             h, w = gray_lbl.shape[:2]
#             return (x, y, w, h)

#         def calc_star_fill_ratio(star_roi, line_idx=1, threshold=18):
#             # 计算星星填充度的核心算法
#             reshaped = star_roi.reshape(-1, 3).astype(np.float32)

#             # 定义颜色（BGR格式）
#             base_color_bgr = np.array([66, 63, 63], dtype=np.float32)
#             bg_color_bgr = np.array([38, 29, 30], dtype=np.float32)

#             if line_idx in [1, 2]:
#                 fill_color_bgr = np.array([45, 149, 214], dtype=np.float32)
#             else:
#                 fill_color_bgr = np.array([36, 192, 185], dtype=np.float32)

#             # 计算距离
#             dist_base = np.linalg.norm(reshaped - base_color_bgr, axis=1)
#             dist_bg = np.linalg.norm(reshaped - bg_color_bgr, axis=1)
#             dist_fill = np.linalg.norm(reshaped - fill_color_bgr, axis=1)

#             base_count = fill_count = 0

#             for i in range(len(reshaped)):
#                 db, dg, df = dist_base[i], dist_bg[i], dist_fill[i]
#                 min_dist = min(db, dg, df)

#                 if min_dist == dg and dg < threshold:
#                     pass  # 背景像素
#                 elif min_dist == db and db < threshold:
#                     base_count += 1
#                 elif min_dist == df and df < threshold:
#                     fill_count += 1

#             star_area = base_count + fill_count
#             return fill_count / star_area if star_area > 0 else 0.0

#         def detect_stars_in_line(big_img, label_img, star_w, star_h, offset_x=0, offset_y=0, gap=5, line_idx=1):
#             matched = match_label(big_img, label_img)
#             if not matched:
#                 return None

#             lx, ly, lw, lh = matched
#             star_x1 = lx + lw + offset_x
#             star_y1 = ly + offset_y

#             ratios = []
#             for i in range(5):
#                 sx = star_x1 + i * (star_w + gap)
#                 sy = star_y1
#                 star_roi = big_img[sy:sy + star_h, sx:sx + star_w]

#                 if star_roi.size == 0:
#                     ratios.append(0.0)
#                 else:
#                     ratio = calc_star_fill_ratio(star_roi, line_idx=line_idx)
#                     ratios.append(ratio)

#             return ratios

#         # 检测三行星星评分
#         taste_ratios = detect_stars_in_line(screenshot, taste_img, star_w, star_h, line_idx=1)
#         time_ratios = detect_stars_in_line(screenshot, time_img, star_w, star_h, line_idx=2)
#         score_ratios = detect_stars_in_line(screenshot, score_img, star_w, star_h, line_idx=3)

#         results = {}

#         if taste_ratios:
#             taste_score = sum(taste_ratios)
#             results['taste'] = round(taste_score, 2)
#             print(f"口味评分: {taste_score:.2f}/5.0")

#         if time_ratios:
#             time_score = sum(time_ratios)
#             results['time'] = round(time_score, 2)
#             print(f"时间评分: {time_score:.2f}/5.0")

#         if score_ratios:
#             overall_score = sum(score_ratios)
#             results['score'] = round(overall_score, 2)
#             print(f"总体评分: {overall_score:.2f}/5.0")

#         return results

#     except Exception as e:
#         print(f"识别星级评分时出错: {e}")
#         return {}

# def get_customer_feedback(flavor_template_path: str = None,
#                          technique_template_path: str = None,
#                          temperature_template_path: str = None,
#                          assess_icon_path: str = None,
#                          guest_complaints_path: str = None,
#                          window_title: str = 'CookingSimulator'):
#     """
#     【功能20】识别客人反馈信息
#     获取客人对菜品的详细反馈信息

#     Args:
#         flavor_template_path: 口味模板路径（可选，使用集中化配置）
#         technique_template_path: 技术模板路径（可选，使用集中化配置）
#         temperature_template_path: 温度模板路径（可选，使用集中化配置）
#         assess_icon_path: 评估图标路径（可选，使用集中化配置）
#         guest_complaints_path: 客人投诉按钮路径（可选，使用集中化配置）
#         window_title: 游戏窗口标题

#     Returns:
#         list: 客人反馈信息列表
#     """
#     import cv2
#     import numpy as np
#     import time
#     from paddleocr import PaddleOCR
#     import re
#     import wordninja

#     try:
#         # 使用集中化路径配置，如果没有提供则使用默认值
#         if any(path is None for path in [flavor_template_path, technique_template_path, temperature_template_path, assess_icon_path, guest_complaints_path]):
#             feedback_paths = AdvancedFunctionPaths.get_customer_feedback_paths()
#             if flavor_template_path is None:
#                 flavor_template_path = feedback_paths["flavor_template_path"]
#             if technique_template_path is None:
#                 technique_template_path = feedback_paths["technique_template_path"]
#             if temperature_template_path is None:
#                 temperature_template_path = feedback_paths["temperature_template_path"]
#             if assess_icon_path is None:
#                 assess_icon_path = feedback_paths["assess_icon_path"]
#             if guest_complaints_path is None:
#                 guest_complaints_path = feedback_paths["guest_complaints_path"]

#         print(f"[*] 使用反馈模板: 口味={flavor_template_path}, 技术={technique_template_path}, 温度={temperature_template_path}")
#         print(f"[*] 使用反馈图标: 评估={assess_icon_path}, 投诉={guest_complaints_path}")

#         # 初始化OCR
#         ocr = PaddleOCR(use_angle_cls=True, lang="ch")

#         # 激活游戏窗口
#         _activate_window(window_title)
#         window_rect = _get_window_rect(window_title)

#         def find_button_in_image(screenshot, template_path, threshold=0.6):
#             template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
#             screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

#             result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
#             loc = np.where(result >= threshold)

#             h, w = template.shape[:2]
#             match_rectangles = []
#             for pt in zip(*loc[::-1]):
#                 x, y = pt
#                 match_rectangles.append((x, y, w, h))

#             return match_rectangles

#         def recognize_text(img):
#             result = ocr.ocr(img, cls=True)
#             return "\n".join([word_info[1][0] for line in result for word_info in line])

#         def process_text(text):
#             # 文本处理逻辑
#             text = text.replace("\n", " ")
#             words = wordninja.split(text)

#             block_words = ["flavor", "technique"]
#             filtered_words = [
#                 word for word in words
#                 if len(word) > 1 and word.isalpha() and
#                 not any(word.lower() in block_word for block_word in block_words)
#             ]

#             if len(filtered_words) == 1:
#                 return ""

#             return " ".join(filtered_words)

#         def capture_region(x, y, offset=(-200, -65, 190, 50)):
#             region = {
#                 "left": x + offset[0],
#                 "top": y + offset[1],
#                 "width": offset[2],
#                 "height": offset[3]
#             }
#             img = _capture_screenshot_mss_numpy(region)
#             return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

#         def compare_images(img1, img2, threshold=0.9):
#             from skimage.metrics import structural_similarity as ssim
#             gray1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
#             gray2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)
#             score, _ = ssim(gray1, gray2, full=True)
#             return score < threshold

#         # 点击客人投诉按钮
#         screenshot = _capture_screenshot_mss_numpy(window_rect)
#         complaints_buttons = find_button_in_image(screenshot, guest_complaints_path, threshold=0.9)

#         if not complaints_buttons:
#             print("未找到客人投诉按钮")
#             return []

#         # 点击投诉按钮
#         x, y, w, h = complaints_buttons[0]
#         center_x = window_rect['left'] + x + w // 2
#         center_y = window_rect['top'] + y + h // 2
#         io_controller.mouse_move_absolute(center_x, center_y)
#         time.sleep(0.2)
#         io_controller.click('left')
#         time.sleep(1.0)

#         # 获取新的截图
#         screenshot = _capture_screenshot_mss_numpy(window_rect)

#         feedback_results = []

#         # 检查各个评估项目
#         templates = [
#             (flavor_template_path, "口味"),
#             (technique_template_path, "技术"),
#             (temperature_template_path, "温度")
#         ]

#         for template_path, category in templates:
#             matches = find_button_in_image(screenshot, template_path)

#             if matches:
#                 x, y, w, h = matches[0]

#                 # 计算评估图标位置
#                 assess_icon_img = cv2.imread(assess_icon_path, cv2.IMREAD_GRAYSCALE)
#                 icon_h, icon_w = assess_icon_img.shape[:2]

#                 # 计算第一个图标中心位置（相对于模板底部）
#                 bottom_middle_x = window_rect['left'] + x + w // 2
#                 bottom_middle_y = window_rect['top'] + y + h

#                 # 生成3个图标位置
#                 icon_centers = []
#                 offset_right, offset_down = 15, 20
#                 first_center = (bottom_middle_x + offset_right, bottom_middle_y + offset_down)
#                 icon_centers.append(first_center)

#                 for i in range(1, 3):
#                     center_x = first_center[0] + i * icon_w
#                     center_y = first_center[1]
#                     icon_centers.append((center_x, center_y))

#                 # 依次点击图标获取反馈
#                 for i, (abs_x, abs_y) in enumerate(icon_centers, 1):
#                     # 点击前截图
#                     before_img = capture_region(abs_x, abs_y)
#                     time.sleep(0.5)

#                     # 点击图标
#                     io_controller.mouse_move_absolute(abs_x, abs_y)
#                     time.sleep(0.2)
#                     io_controller.click('left')
#                     time.sleep(1)

#                     # 点击后截图
#                     after_img = capture_region(abs_x, abs_y)

#                     # 检查是否有弹窗
#                     popup_found = compare_images(before_img, after_img)

#                     if popup_found:
#                         detected_text = recognize_text(after_img)
#                         final_text = process_text(detected_text)
#                         if final_text:
#                             feedback_results.append(f"{category}评估{i}: {final_text}")
#                             print(f"检测到{category}反馈: {final_text}")

#                     time.sleep(1)

#         return feedback_results

#     except Exception as e:
#         print(f"获取客人反馈时出错: {e}")
#         return []

##########################################################
##                                                      ##
##               Integrated Skills (auto_*)              ##
##                                                      ##
##########################################################

# Make the vendored `auto_*` skill packages importable for legacy-style imports
# like `import auto_cutting...`.
import os as _os

_CEREBELLUM_DIR = _os.path.dirname(__file__)
_CEREBELLUM_SKILLS_DIR = _os.path.join(_CEREBELLUM_DIR, "skills")
if _CEREBELLUM_SKILLS_DIR not in sys.path:
    sys.path.insert(0, _CEREBELLUM_SKILLS_DIR)


def _skill_auto_cutting(
    item_name: str,
    cut_count: int = 3,
    cut_interval: float = 0.25,
    use_closed_loop: bool = True,
    equal_volume: bool = False,
    item_instance_id: int | None = None,
) -> dict:
    """
    Run `auto_cutting.auto_cutting.auto_cut`.

    Manual test:
      python epm/test/manual_input_validation.py --action skill_auto_cutting --focus item_name="lemon" cut_count=3 cut_interval=0.25 use_closed_loop=true
    """
    try:
        from epm.cerebellum.skills.auto_cutting.auto_cutting import auto_cut, get_last_auto_cut_failure_info  # type: ignore
    except Exception:  # pragma: no cover
        from auto_cutting.auto_cutting import auto_cut, get_last_auto_cut_failure_info  # type: ignore  # lazy import to avoid cycles

    from pathlib import Path

    from epm.cerebellum.realtime_products import best_match_by_name, extract_items, read_realtime_products
    from epm.cerebellum.skills._shared_paths import realtime_products_json

    ok = False
    err = ""
    tb = ""
    available, snap = _hold_snapshot()
    cutting_active = bool(available and bool(snap.get("is_cutting_mode", False)))
    if not cutting_active:
        return {
            "success": False,
            "item_name": str(item_name),
            "cut_num": int(cut_count),
            "weight": None,
            "error": "auto_cut_requires_cutting_mode",
            "mode": "cutting_mode",
            "mode_entered": False,
            "hint": "Call enter_cutting_mode first. auto_cut is only valid when cutting mode is already active.",
            "precheck": snap if isinstance(snap, dict) else {},
            "traceback": "",
        }
    stable_ok, stable_available, stable_snap = _wait_mode_condition_stable(
        "is_cutting_mode",
        expected=True,
        timeout_s=2.4,
        poll_s=0.05,
        stable_s=2.0,
    )
    if stable_available:
        snap = stable_snap
    if not stable_ok:
        return {
            "success": False,
            "item_name": str(item_name),
            "cut_num": int(cut_count),
            "weight": None,
            "error": "auto_cut_requires_stable_cutting_mode",
            "mode": "cutting_mode",
            "mode_entered": False,
            "hint": "Cutting mode was not stable for 2 seconds. Re-enter cutting mode and keep aiming at the cuttable target before auto_cut.",
            "precheck": snap if isinstance(snap, dict) else {},
            "traceback": "",
        }
    try:
        ok = bool(
            auto_cut(
                item_name=item_name,
                item_instance_id=(int(item_instance_id) if item_instance_id is not None else None),
                cut_count=int(cut_count),
                cut_interval=float(cut_interval),
                use_closed_loop=bool(use_closed_loop),
                equal_volume=bool(equal_volume),
            )
        )
    except Exception as e:
        ok = False
        err = str(e)
        try:
            import traceback

            tb = traceback.format_exc()
        except Exception:
            tb = ""

    failure_info = {}
    try:
        failure_info = get_last_auto_cut_failure_info()
    except Exception:
        failure_info = {}

    weight = None
    try:
        rt_path: Path = realtime_products_json()
        data = read_realtime_products(rt_path)
        items = extract_items(data)
        m = best_match_by_name(items, item_name)
        if m and str(m.get("kind", "") or "").strip().lower() == "products":
            weight = m.get("weight", None)
    except Exception:
        weight = None

    return {
        "success": bool(ok),
        "item_name": str(item_name),
        "cut_num": int(cut_count),
        "weight": weight,
        "error": (
            ""
            if ok
            else (
                "auto_cut_target_out_of_cutting_scope"
                if failure_info.get("code") == "target_out_of_cutting_scope"
                else (err or "cut_failed")
            )
        ),
        "mode": "cutting_mode",
        "mode_entered": False,
        "hint": (
            ""
            if ok
            else (
                "The target item seems too far for the current cutting-mode working area. "
                "Re-aim the knife at the intended item or cutting board before entering cutting mode, "
                "then try cutting again."
                if failure_info.get("code") == "target_out_of_cutting_scope"
                else ""
            )
        ),
        "precheck": snap if isinstance(snap, dict) else {},
        "failure_info": failure_info if (not ok and isinstance(failure_info, dict) and failure_info) else {},
        # Debug-only: keep the traceback (when available) so planner can report a precise failure.
        # This is safe because it only appears on failure and is truncated by feedback rendering.
        "traceback": (tb.strip()[:4000] if (not ok and tb) else ""),
    }


def _skill_auto_pouring(
    container_name: str,
    target_ml: float | None = None,
    tolerance: float = 1.0,
    skip_alignment: bool = False,
    container_instance_id: int | None = None,
) -> dict:
    """
    Run `auto_pouring.auto_pouring.auto_pour`.

    Manual test:
      python epm/test/manual_input_validation.py --action skill_auto_pouring --focus container_name="Paella Pan" target_ml=50 tolerance=1 skip_alignment=false
    """
    try:
        from epm.cerebellum.skills.auto_pouring.auto_pouring import auto_pour  # type: ignore
    except Exception:  # pragma: no cover
        from auto_pouring.auto_pouring import auto_pour  # type: ignore  # lazy import to avoid cycles

    resolved_target_ml = 50.0 if target_ml in (None, "") else float(target_ml)

    try:
        result = auto_pour(
            container_name=container_name,
            container_instance_id=(int(container_instance_id) if container_instance_id is not None else None),
            target_ml=resolved_target_ml,
            tolerance=float(tolerance),
            skip_alignment=bool(skip_alignment),
        )
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "container_name": str(container_name),
            "target_ml": float(resolved_target_ml),
            "tolerance": float(tolerance),
            "poured_ml": 0.0,
            "mode": "pouring_mode",
            "mode_entered": None,
        }

    # auto_pour now returns a dict with poured_ml; keep a small backward-compatible shim.
    if isinstance(result, dict):
        # Add a small hint for prompt feedback on mode failures.
        out = dict(result)
        out.setdefault("mode", "pouring_mode")
        out.setdefault("mode_entered", None)
        err = str(out.get("error") or "").strip()
        if err.startswith("bottle_empty:"):
            out.setdefault("feedback_code", "bottle_empty")
            out.setdefault("pour_status", "empty_bottle")
            out.setdefault(
                "feedback_to_planner",
                "The held liquid bottle appears empty. Do not keep pouring from this bottle; switch to another liquid source or continue with the next step.",
            )
        elif err.startswith("no_liquid_flow:"):
            out.setdefault("feedback_code", "no_liquid_flow")
            out.setdefault("pour_status", "no_liquid_flow")
            out.setdefault(
                "feedback_to_planner",
                "No liquid is flowing. Re-aim the bottle mouth over the container and retry once; if it still fails under a steep tilt, treat the bottle as empty.",
            )
        elif err.startswith("unsupported_pour_source:"):
            out.setdefault("feedback_code", "unsupported_pour_source")
            out.setdefault("pour_status", "unsupported_source")
            out.setdefault(
                "feedback_to_planner",
                "Only liquid bottles and containers are allowed for pouring. The current held item is not supported.",
            )
        return out

    ok = bool(result)
    return {
        "success": bool(ok),
        "error": "" if ok else "auto_pour_failed (check: holding bottle, pouring mode, Alt+J enabled, F12 scan running)",
        "container_name": str(container_name),
        "target_ml": float(resolved_target_ml),
        "tolerance": float(tolerance),
        "poured_ml": None,
        "mode": "pouring_mode",
        "mode_entered": None,
    }


def _skill_auto_sprinkling(
    target_item: str,
    sprinkle_num: int = 3,
    align_to_target: bool = False,
    exit_after: bool = True,
    aim_each_time: bool = True,
    target_instance_id: int | None = None,
) -> dict:
    """Run `auto_sprinkling.auto_sprinkling.sprinkle` and return completed count.

    Note: If `aim_each_time=False`, the skill will skip auto-alignment and only
    execute timed sprinkle clicks. This is useful when calibration is unreliable;
    aim the crosshair at the target manually before running.
    """
    # Best-effort: ensure Alt+J interaction stream is enabled (for grams/interaction monitoring),
    # and ensure F12 realtime scan is running (for target lookup).
    try:
        from epm.cerebellum.game_hotkeys import ensure_alt_j_interaction, ensure_f12_products_scan
        from epm.cerebellum.skills._shared_paths import userdata_root, window_title
        from epm.vision.screen_capture import activate_window

        ensure_f12_products_scan(
            realtime_products_path=userdata_root() / "realtime_products.json",
            window_title=window_title("CookingSimulator"),
            activate_window=activate_window,
            io_controller=io_controller,
        )
        ensure_alt_j_interaction(
            userdata_root=userdata_root(),
            window_title=window_title("CookingSimulator"),
            activate_window=activate_window,
            io_controller=io_controller,
        )
    except Exception:
        pass
    try:
        from epm.cerebellum.skills.auto_sprinkling.auto_sprinkling import sprinkle  # type: ignore
    except Exception:  # pragma: no cover
        from auto_sprinkling.auto_sprinkling import sprinkle  # type: ignore  # lazy import to avoid cycles

    try:
        completed = int(
            sprinkle(
                target_item=target_item,
                target_instance_id=(int(target_instance_id) if target_instance_id is not None else None),
                sprinkle_num=int(sprinkle_num),
                align_to_target=bool(align_to_target),
                exit_after=bool(exit_after),
                aim_each_time=bool(aim_each_time),
            )
        )
        return {
            "success": bool(completed > 0),
            "target": str(target_item),
            "requested": int(sprinkle_num),
            "completed": int(completed),
            "error": "" if completed > 0 else "sprinkle_completed_0",
            "mode": "sprinkle_mode",
            "mode_entered": None,
        }
    except Exception as e:
        return {
            "success": False,
            "target": str(target_item),
            "requested": int(sprinkle_num),
            "completed": 0,
            "error": str(e),
            "mode": "sprinkle_mode",
            "mode_entered": False,
        }


def _deprecated_auto_goto(object: str) -> dict:
    """
    DEPRECATED: use `auto_navigation(target=...)`.

    This function is kept in source for reference only and is not exposed as an action anymore.
    """
    return auto_navigation(target=str(object))


def _norm_name(s: object) -> str:
    return str(s or "").strip().lower()

def auto_cut(object: str, cut_num: int, object_instance_id: int | None = None) -> dict:
    """
    切割物品技能。在切割模式下，对物品切 `cut_num` 刀，切完后自动退出切割模式，恢复为手持刀的状态。

    Args:
      object: The name of the item to be cut (e.g. "lemon", "garlic").
      cut_num: Number of cuts.
      object_instance_id: Optional `instance_id` in realtime_products.json. Use it to disambiguate when multiple
        items share the same name (recommended for robust execution).
        To find candidate `instance_id`s, call the read-only skill `query_scene_objects(query="...")`, e.g.:
          query_scene_objects(query="lemon")

    Returns:
      {"success": bool, "item_name": str, "cut_num": int, "error": str}
    """
    return _skill_auto_cutting(
        item_name=str(object),
        cut_count=int(cut_num),
        equal_volume=False,
        item_instance_id=(int(object_instance_id) if object_instance_id is not None else None),
    )


def auto_pour(container: str, pour_ml: float | None = None, container_instance_id: int | None = None) -> dict:
    """
    倾倒技能。在倾倒模式下，将液体自动倾倒到 `container` 中，直到达到 `pour_ml` 毫升。

    Args:
      container: Name of the target container for dumping (e.g. "Paella Pan", "Pot").
      pour_ml: Optional target milliliters to pour (e.g. 10, 50). Defaults to 50 if omitted.
      container_instance_id: Optional `instance_id` in realtime_products.json for the target container. Use it to
        disambiguate when multiple containers share the same name.
        To find candidate `instance_id`s, call the read-only skill `query_scene_objects(query="...")`, e.g.:
          query_scene_objects(query="Pan")

    Returns (includes feedback signal for the planner):
      {"success": bool, "poured_ml": float, "target_ml": float, "tolerance": float, "container_name": str, "error": str}
    """
    return _skill_auto_pouring(
        container_name=str(container),
        target_ml=(50.0 if pour_ml in (None, "") else float(pour_ml)),
        tolerance=1.0,
        skip_alignment=False,
        container_instance_id=(int(container_instance_id) if container_instance_id is not None else None),
    )


def auto_sprinkle(target: str, sprinkle_num: int, target_instance_id: int | None = None) -> dict:
    """
    播撒技能。在播撒模式下，将调料罐内调料播撒到 `target` 上 `sprinkle_num` 次。默认一次约 1g（除非已执行改变播撒克数的动作）。

    Manual test:
      python epm/test/manual_input_validation.py --action auto_sprinkle --focus target="Steak" sprinkle_num=3

    Args:
      target: Target name to sprinkle onto (usually a food item or container).
      sprinkle_num: number of times the sprinkling action is executed  (int).
      target_instance_id: Optional `instance_id` in realtime_products.json for the sprinkle target. Use it to
        disambiguate when multiple items share the same name.
        To find candidate `instance_id`s, call the read-only skill `query_scene_objects(query="...")`, e.g.:
          query_scene_objects(query="Steak")

    Returns:
      {"success": bool, "target": str, "requested": int, "completed": int, "error": str}

    GPT tool guidance:
      Purpose: add seasoning by repeating the sprinkle interaction.
      When to call: target is visible/interactable and sprinkling action/tool is available.
      Side effects: sends keyboard/mouse events.
      Final state: sprinkle interaction executed ~`sprinkle_num` times.
    """
    return _skill_auto_sprinkling(
        target_item=str(target),
        sprinkle_num=int(sprinkle_num),
        aim_each_time=True,
        target_instance_id=(int(target_instance_id) if target_instance_id is not None else None),
    )


def auto_flip(meat: str, put_place: str, meat_instance_id: int | None = None, put_place_instance_id: int | None = None) -> dict:
    """
    铲子翻面技能。在翻面模式下，将铲子上的 `meat` 移动到 `put_place` 位置给肉翻面，
    并在结束后自动退出翻面状态，恢复为手持铲子的状态。

    Args:
      meat: Meat/food item name on the spatula (e.g. "T-bone Steak").
      put_place: Place point / surface name (e.g. "Griddle Surface-1", "Plate").
      meat_instance_id: Optional `instance_id` in realtime_products.json for the meat item.
      put_place_instance_id: Optional `instance_id` in realtime_products.json for the placement target.
        Note: this action expects a real `instance_id` (NOT a platform index). For platform category+index shorthand
        like put_place="Side Table", put_place_instance_id=3, use the `--skill auto_filp` wrapper instead.
        To find candidate `instance_id`s:
          - Meat: query_scene_objects(query="cod")
          - Put-place (platform points): query_scene_objects(query="Griddle Surface-3")

    Returns:
      {"success": bool, "meat": str, "put_place": str, "error": str}
    """
    return _skill_auto_flipping(
        meat_name=str(meat),
        put_place_name=str(put_place),
        meat_instance_id=(int(meat_instance_id) if meat_instance_id is not None else None),
        put_place_instance_id=(int(put_place_instance_id) if put_place_instance_id is not None else None),
    )


def _auto_filp(meat: str, put_place: str) -> dict:
    """Backward-compat internal alias of `auto_flip` (typo)."""
    return auto_flip(meat=meat, put_place=put_place)


def auto_mix(container: str, container_instance_id: int | None = None) -> dict:
    """
    搅拌技能。在搅拌模式下，对 `container` 内物品进行自动搅拌操作，
    固体经过搅拌后成液体；搅拌完后自动退出搅拌模式，恢复为手持搅拌机的状态。

    Manual test:
      python epm/test/manual_input_validation.py --action auto_mix --focus container="Big Pot"

    Args:
      container: Container name to be mixed (e.g. "Bowl", "Paella Pan", "Pot").
      container_instance_id: Optional `instance_id` in realtime_products.json for the container. Use it to disambiguate
        when multiple containers share the same name.
        To find candidate `instance_id`s, call the read-only skill `query_scene_objects(query="...")`, e.g.:
          query_scene_objects(query="Big Pot")

    Returns:
      {"success": bool, "container": str, "error": str}
    """
    result = _skill_auto_mix(
        container_name=str(container),
        container_instance_id=(int(container_instance_id) if container_instance_id is not None else None),
    )
    if isinstance(result, dict):
        return result
    return {"success": False, "container": str(container), "error": "auto_mix_failed (non-dict return)"}


def _deprecated_gui_order_dish(dish_name: str) -> dict:
    """
    DEPRECATED: use `gui_order_dish_via_computer` (world-level flow) or call `order_dish_by_search` directly.

    Kept for reference only; not exposed as an action.
    """
    _ = dish_name
    return {"success": False, "error": "deprecated"}


def gui_buy_new_item(item: str) -> dict:
    """
    购买物品技能。对于获取场景中不存在的物品，执行此技能。空手状态下，自动导航至 Carton Box，在商店 UI 界面自动购买物品。
    购买后智能体处于手持对应物品的状态；再次购买前请先放置好物品确保空手，否则会失败。

    Manual test:
      python epm/test/manual_input_validation.py --action gui_buy_new_item --focus item="water"

    Args:
      item: object name (English/Chinese) resolvable via `epm/data/object_en_ch_mapping.txt`.
            Category and icon template are auto-resolved.

    Returns:
      {"success": bool, "item": str, "category": str, "error": str}
    """
    pre_available, pre = _hold_snapshot()
    if pre_available and bool(pre.get("is_held", False)):
        held_name_en = str(pre.get("held_name_en") or "")
        held_name_cn = str(pre.get("held_name_cn") or "")
        held_kind = str(pre.get("held_kind") or "")
        held_name = held_name_en or held_name_cn or "unknown"
        return {
            "success": False,
            "error": "gui_buy_new_item_precheck_failed:hands_not_empty",
            "hint": "put_down_or_use_current_item_before_buying",
            "item": str(item),
            "held_item": held_name,
            "held_item_en": held_name_en,
            "held_item_cn": held_name_cn,
            "held_item_kind": held_kind,
            "details": (
                f"cannot buy {str(item)} while holding {held_name}; "
                "gui_buy_new_item requires empty hands before entering the store flow"
            ),
            "precheck": pre,
        }

    from epm.cerebellum.gui_actions.store_flow import buy_new_item

    return buy_new_item(item=str(item), category=None)


def gui_enable_steady_hands() -> dict:
    """
    Internal GUI helper: enable perk "Steady hands".

    NOTE: This function is intentionally hidden from ACTION_DISPATCHER / GPT tool list.
    Use it from code (e.g. order flows) if needed; do not expose as a standalone action.
    """
    from epm.cerebellum.gui_actions.perks_gui import enable_steady_hands_perk

    return enable_steady_hands_perk()


def _deprecated_gui_get_recipe_feedback() -> dict:
    """
    DEPRECATED: use `gui_submit_dish_via_checkout_stand` (which includes feedback parsing).

    Kept for reference only; not exposed as an action.
    """
    return {"success": False, "error": "deprecated"}


def _deprecated_gui_submit_dish_and_evaluate(dish_name: str) -> dict:
    """
    DEPRECATED: use `gui_submit_dish_via_checkout_stand`.

    Kept for reference only; not exposed as an action.
    """
    _ = dish_name
    return {"success": False, "error": "deprecated"}


def gui_submit_dish_via_checkout_stand(
    dish_name: str,
    *,
    checkout_target: str = "Checkout Stand",
    _debug_submit_only: bool = False,
    _debug_skip_holding_check: bool = False,
) -> dict:
    """
    上菜评价技能。每个任务结束后必须执行该技能：手持菜品状态下，
    自动导航至 Checkout Stand，在上菜 UI 界面自动上交菜品，获取所作菜的评分以及反馈。

    Flow:
    - goto(checkout_target)
    - left click to enter serving UI
    - wait ~3s for UI
    - click submit entry (relative mouse move heuristic)
    - Alt+Y dump + click dish entry
    - Alt+K parse feedback
    - right click to exit UI

    Manual test:
      python epm/test/manual_input_validation.py --action gui_submit_dish_via_checkout_stand --focus dish_name="Lemon Tart"
    Args:
      dish_name: 要提交并评价的菜品名称。

    """
    def _is_submit_holdable_name(name: str | None) -> bool:
        if not name:
            return False
        allow = {
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
            "big pot",
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
            "tart",
            "挞",
        }
        s = str(name).strip()
        if not s:
            return False
        primary = s.split(",", 1)[0].strip().lower()
        return primary in allow

    def _read_realtime_products() -> dict | None:
        try:
            from epm.cerebellum.skills._shared_paths import realtime_products_json
            import json as _json
            path = realtime_products_json()
            if not path.exists():
                return None
            for enc in ("utf-8-sig", "utf-8", "gbk"):
                try:
                    raw = path.read_text(encoding=enc, errors="ignore")
                    data = _json.loads(raw)
                    return data if isinstance(data, dict) else None
                except Exception:
                    continue
        except Exception:
            return None
        return None

    data = _read_realtime_products()
    if not (bool(_debug_submit_only) or bool(_debug_skip_holding_check)):
        if not data or "products" not in data:
            return {
                "success": False,
                "error": "realtime_products_unavailable",
                "hint": "run_f12_scan_or_check_mod",
            }

        held = None
        try:
            for p in data.get("products", []):
                if isinstance(p, dict) and p.get("is_held", False):
                    held = p
                    break
        except Exception:
            held = None

        if not held:
            return {
                "success": False,
                "error": "not_holding_anything",
                "products_count": len(data.get("products", [])) if isinstance(data.get("products", None), list) else 0,
                "hint": "pick_up_container_first",
            }

        held_name = held.get("name_en") or held.get("name") or held.get("name_cn")
        held_kind = str(held.get("kind") or "").strip().lower()
        if not _is_submit_holdable_name(str(held_name or "")):
            return {
                "success": False,
                "error": "held_item_not_submit_holdable",
                "held_item": held_name,
                "held_kind": held.get("kind"),
                "held_instance_id": held.get("instance_id"),
            }

    from epm.cerebellum.gui_actions.serve_flow import submit_dish_via_checkout_stand
    return submit_dish_via_checkout_stand(
        dish_name=str(dish_name),
        checkout_target=str(checkout_target),
        _debug_submit_only=bool(_debug_submit_only),
    )


def gui_order_dish_via_computer(dish_name: str) -> dict:
    """
    下订单技能。每个任务开始前必须先执行该技能：空手状态下，
    自动导航至 Computer，在电脑 UI 界面自动下订单；点完订单后智能体处于空手状态。
    如果要再次下订单要留意确保空手，否则会失败。

    Flow:
    - goto(computer_target) to reach the in-world Computer
    - click to enter the computer UI
    - order dish via `gui_order_dish`
    - right click to exit the computer UI
    Args:
      dish_name: 要下订单的菜品名称。

    """
    from epm.cerebellum.gui_actions.order_flow import order_dish_via_computer
    computer_target = "Computer"
    return order_dish_via_computer(dish_name=str(dish_name), computer_target=str(computer_target))

def _skill_auto_mix(container_name: str, container_instance_id: int | None = None) -> dict:
    """
    Mix inside a container using Blender mixing mode.

    Internal helper (not exposed as an action):
    - Use `auto_mix(container=...)` as the public entrypoint.
    """
    from epm.cerebellum.skills.auto_mix.auto_mix import mix_in_container  # type: ignore
    mode_precheck = _ensure_interaction_mode_for_skill(
        "is_mixing_mode",
        "mixing_mode",
        enter_mixing_mode,
    )
    if not bool(mode_precheck.get("success", False)):
        return {
            "success": False,
            "error": str(mode_precheck.get("error") or "enter_mixing_mode_failed"),
            "container_name": str(container_name),
            "mode": "mixing_mode",
            "mode_entered": bool(mode_precheck.get("mode_entered", False)),
            "hint": str(mode_precheck.get("hint") or ""),
            "precheck": mode_precheck.get("precheck") or {},
        }

    try:
        result = mix_in_container(
            container_name=str(container_name),
            container_instance_id=(int(container_instance_id) if container_instance_id is not None else None),
            verbose=True,
        )
        if not isinstance(result, dict):
            return {
                "success": False,
                "error": "auto_mix returned non-dict result",
                "container_name": str(container_name),
                "mode": "mixing_mode",
                "mode_entered": bool(mode_precheck.get("mode_entered", False)),
                "hint": str(mode_precheck.get("hint") or ""),
                "precheck": mode_precheck.get("precheck") or {},
            }
        out = dict(result)
        out.setdefault("mode", "mixing_mode")
        out.setdefault("mode_entered", bool(mode_precheck.get("mode_entered", False)))
        out.setdefault("hint", str(mode_precheck.get("hint") or ""))
        out.setdefault("precheck", mode_precheck.get("precheck") or {})
        return out
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "container_name": str(container_name),
            "mode": "mixing_mode",
            "mode_entered": bool(mode_precheck.get("mode_entered", False)),
            "hint": str(mode_precheck.get("hint") or ""),
            "precheck": mode_precheck.get("precheck") or {},
        }


def _skill_auto_flipping(
    meat_name: str,
    put_place_name: str,
    place_radius_m: float = 0.12,
    verbose: bool = True,
    meat_instance_id: int | None = None,
    put_place_instance_id: int | None = None,
) -> dict:
    """Run `auto_flipping.auto_flipping.auto_flip` with structured error feedback."""
    try:
        from epm.cerebellum.skills.auto_flipping import auto_flipping as flipping  # type: ignore
    except Exception:  # pragma: no cover
        from auto_flipping import auto_flipping as flipping  # type: ignore  # lazy import to avoid cycles

    mode_precheck = _ensure_interaction_mode_for_skill(
        "is_flip_mode",
        "flipping_mode",
        enter_the_flipping_mode,
    )
    if not bool(mode_precheck.get("success", False)):
        return {
            "success": False,
            "meat": str(meat_name),
            "put_place": str(put_place_name),
            "place_radius_m": float(place_radius_m),
            "meat_instance_id": (int(meat_instance_id) if meat_instance_id is not None else None),
            "put_place_instance_id": (int(put_place_instance_id) if put_place_instance_id is not None else None),
            "error": str(mode_precheck.get("error") or "enter_flipping_mode_failed"),
            "mode": "flipping_mode",
            "mode_entered": bool(mode_precheck.get("mode_entered", False)),
            "hint": str(mode_precheck.get("hint") or ""),
            "precheck": mode_precheck.get("precheck") or {},
        }

    ok = bool(
        flipping.auto_flip(
            meat_name=str(meat_name),
            put_place_name=str(put_place_name),
            meat_instance_id=(int(meat_instance_id) if meat_instance_id is not None else None),
            put_place_instance_id=(int(put_place_instance_id) if put_place_instance_id is not None else None),
            place_radius_m=float(place_radius_m),
            verbose=bool(verbose),
        )
    )
    err = ""
    if not ok:
        try:
            err = str(flipping.get_last_error() or "").strip()
        except Exception:
            err = ""
    return {
        "success": bool(ok),
        "meat": str(meat_name),
        "put_place": str(put_place_name),
        "place_radius_m": float(place_radius_m),
        "meat_instance_id": (int(meat_instance_id) if meat_instance_id is not None else None),
        "put_place_instance_id": (int(put_place_instance_id) if put_place_instance_id is not None else None),
        "error": "" if ok else (err or "auto_flip_failed"),
        "mode": "flipping_mode",
        "mode_entered": bool(mode_precheck.get("mode_entered", False)),
        "hint": str(mode_precheck.get("hint") or ""),
        "precheck": mode_precheck.get("precheck") or {},
    }


def _deprecated_skill_astar_navigate_to_object(map_file: str, object_name: str, auto_activate_window: bool = True) -> bool:
    """
    DEPRECATED: use `auto_navigation(target=...)`.

    Run A* navigation to an object using the consolidated navigator backend.

    Note: This used to call `auto_navigation.astar_navigator.AStarNavigator`, but that
    implementation was redundant with the maintained `SimpleAstarRadarNavigator`.
    """
    from epm.cerebellum.skills.auto_navigation.skill import NavigateArgs, run as run_nav  # type: ignore

    res = run_nav(None, NavigateArgs(target=str(object_name), map_file=str(map_file), auto_activate_window=bool(auto_activate_window)))  # type: ignore[arg-type]
    return bool(res.success)


def _deprecated_skill_auto_navigation(target: str, auto_activate_window: bool = True) -> dict:
    """
    DEPRECATED: use `auto_navigation(target=...)`.

    This function is kept in source for reference only and is not exposed as an action anymore.
    """
    _ = auto_activate_window
    return auto_navigation(target=str(target))


def auto_navigation(target: str, target_instance_id: int | None = None) -> dict:
    """
    导航技能。导航到 `target` 前并将视野正对 `target`。

    Manual test:
      python epm/test/manual_input_validation.py --action auto_navigation --focus target=lemon
    Args:
      target: 目标名称，允许三类输入：
        1) 普通物体名称（例如 "lemon"）。
        2) 平台放置点类别名称（例如 "Side Table"）。
        3) 工具交互点类别名称（例如 "Stove Switch"）。
      target_instance_id: 语义取决于 target 类型：
        1) 普通物体：realtime_products 中对应条目的 `instance_id`（用于同名多实例消歧）。
        2) 平台类别（kind="platform_point"）：当 target 是平台类别名时，`target_instance_id` 作为平台序号 index
          （需落在允许范围内），并自动展开为具体平台点名。
           例如：target="Side Table", target_instance_id=3  ->  "Side Table-3"。
        3) 工具交互点类别（kind="tool_point"）：当 target 是工具交互点类别名时，`target_instance_id` 作为类别序号 index
          （需落在允许范围内），并自动展开为具体工具点名（如 "Stove Switch 1"）。若该类别不接受 index，则不要传。
        如果 `target` 不是平台/工具类别名，则 `target_instance_id` 按普通物体的 instance_id 解释。
      主动查找候选：
        - 普通物体：用 `query_scene_objects(query="...")` 获取候选及 instance_id。
        - 平台/工具类别：用 `list_supported_items(query="...")` 获取可用类别与命名，再决定是否需要 index。

    """
    from epm.cerebellum.skills.auto_navigation.skill import NavigateArgs, run as run_nav  # type: ignore

    res = run_nav(
        None,
        NavigateArgs(
            target=str(target),
            auto_activate_window=True,
            target_instance_id=(int(target_instance_id) if target_instance_id is not None else None),
        ),
    )  # type: ignore[arg-type]

    return {"success": bool(res.success), "target": str(target), "error": "" if bool(res.success) else (res.error or "auto_navigation_failed")}


def auto_perception(only_on_screen: bool = False, max_items: int = 200) -> dict:
    """
    自动感知技能：获取当前视野内的物品列表（读取 realtime_products.json，按距离排序）。

    Args:
      only_on_screen: 是否只保留 `is_on_screen=True` 的物体（默认 False）。
      max_items: 最多返回多少个物体（默认 200）。
    """
    from epm.cerebellum.skills._shared_paths import realtime_products_json
    from epm.cerebellum.skills.auto_perception.skill import PerceptionArgs, run as run_perception

    res = run_perception(
        PerceptionArgs(
            realtime_products_path=realtime_products_json(),
            only_on_screen=bool(only_on_screen),
            max_items=int(max_items),
        )
    )

    visible_items = []
    if isinstance(getattr(res, "raw", None), dict) and isinstance(res.raw.get("visible_items"), list):
        visible_items = list(res.raw.get("visible_items") or [])

    return {
        "success": bool(res.success),
        "visible_items": visible_items,
        "error": "" if bool(res.success) else (res.error or "auto_perception_failed"),
    }


def query_scene_objects(query: str, only_on_screen: bool = False, max_items: int = 100, max_distance: float = -1) -> dict:
    """
    检索技能：在 realtime_products.json 中按名称子串（不区分大小写）检索物品信息。
    `query` 支持单个字符串、以 `;` / `,` 分隔的多个查询词，或 JSON 列表字符串。

    Args:
      query: 名称子串（英文；匹配 name/name_en/name_cn 等字段）。
      only_on_screen: 是否只检索当前视野内物体,这个参数一般不需要动（默认 False）。
      max_items: 最多返回多少个匹配项，这个参数一般不需要动（默认 100）。
      max_distance: 最大距离过滤，这个参数不需要动（默认 -1；表示不进行过滤）。
    """
    from epm.cerebellum.skills._shared_paths import realtime_products_json
    from epm.cerebellum.skills.query_scene_objects.skill import QuerySceneObjectsArgs, run as run_query

    res = run_query(
        realtime_products_path=realtime_products_json(),
        args=QuerySceneObjectsArgs(
            query=str(query),
            only_on_screen=bool(only_on_screen),
            max_items=int(max_items),
            max_distance=float(max_distance),
        ),
    )

    results = []
    if isinstance(getattr(res, "raw", None), dict) and isinstance(res.raw.get("results"), list):
        results = list(res.raw.get("results") or [])
    results_by_query = {}
    if isinstance(getattr(res, "raw", None), dict) and isinstance(res.raw.get("results_by_query"), dict):
        results_by_query = dict(res.raw.get("results_by_query") or {})
    queries = []
    if isinstance(getattr(res, "raw", None), dict) and isinstance(res.raw.get("queries"), list):
        queries = list(res.raw.get("queries") or [])

    return {
        "success": bool(res.success),
        "query": str(query),
        "queries": queries,
        "results": results,
        "results_by_query": results_by_query,
        "error": "" if bool(res.success) else (res.error or "query_scene_objects_failed"),
    }


def list_supported_items(
    query: str = "",
    include_mapping: bool = True,
    include_put_place: bool = True,
    include_tool_points: bool = True,
) -> dict:
    """
    列出“环境支持的物品/平台名称”（不依赖当前场景是否存在该物品）。

    用途：当你不确定某个名称是否是合法输入时，先调用它再决定下一步动作。

    示例（函数调用风格）：
      - list_supported_items()
      - list_supported_items(query="steak")
      - list_supported_items(query="platform")

    结合实例选择（当前场景具体 instance_id）：
      - query_scene_objects(query="lemon")  # 返回 realtime_products.json 中的候选实例及 instance_id

    Args:
      query: 子串搜索（对 name_en / name_cn / category / object_id 进行匹配；不区分大小写）。
      include_mapping: 是否返回 object_en_ch_mapping.txt 的全量支持物品列表（默认 True）。
      include_put_place: 是否返回 epm/data/put_place_list.json 中的平台类别信息（默认 True）。
      include_tool_points: 是否返回 epm/data/tool_interaction_point_list.json 中的工具交互点类别信息（默认 True）。

    Returns:
      dict: {"success": bool, "query": str, "items": list[dict], "platforms": list[dict], "tool_points": list[dict], "error": str}
    """
    from epm.cerebellum.skills.list_supported_items.skill import ListSupportedItemsArgs, run as run_list_supported  # type: ignore

    res = run_list_supported(
        args=ListSupportedItemsArgs(
            query=str(query),
            include_mapping=bool(include_mapping),
            include_put_place=bool(include_put_place),
            include_tool_points=bool(include_tool_points),
        )
    )

    items = []
    platforms = []
    tool_points = []
    if isinstance(getattr(res, "raw", None), dict):
        if isinstance(res.raw.get("items"), list):
            items = list(res.raw.get("items") or [])
        if isinstance(res.raw.get("platforms"), list):
            platforms = list(res.raw.get("platforms") or [])
        if isinstance(res.raw.get("tool_points"), list):
            tool_points = list(res.raw.get("tool_points") or [])

    return {
        "success": bool(res.success),
        "query": str(query),
        "items": items,
        "platforms": platforms,
        "tool_points": tool_points,
        "error": "" if bool(res.success) else (res.error or "list_supported_items_failed"),
    }


# Rebuild dispatcher at the end so the new skill wrappers are included.
def _is_action_function(obj):
    return (
        inspect.isfunction(obj)
        and obj.__module__ == __name__
        and not obj.__name__.startswith("_")
        and obj.__name__
        not in {
            # Hidden from action list / GPT tools (internal helper)
            "gui_enable_steady_hands",
            # Hidden from action list / GPT tools (read-only skills exposed as type=skill)
            "auto_perception",
            "query_scene_objects",
            "list_supported_items",
            # Hidden from action list / GPT tools (keyboard/mouse primitives)
            "click_mouse",
            "hold_keyboard",
            "hold_mouse",
            "leave_keyboard",
            "leave_mouse",
            "move_related_mouse",
            "press_keyboard",
            "scroll_down_the_wheel",
            "scroll_up_the_wheel",
            "focus_terminal",
            # Hidden from action list / GPT tools (window helper)
            "get_window_center",
        }
    )


ACTION_DISPATCHER = {
    name: func for name, func in inspect.getmembers(sys.modules[__name__], _is_action_function)
}


print(f"[*] local_actions.py loaded with RawInputController. Found {len(ACTION_DISPATCHER)} actions.")


if __name__ == "__main__":
    _activate_window()
    move_forward()

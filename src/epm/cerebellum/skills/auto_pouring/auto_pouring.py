# auto_pouring.py
# ================================================================
# Cooking Simulator Auto-Pouring Module v1.6
# ================================================================
#
# 功能：
#   1. 位置对准：液体瓶口对准容器中心（闭环控制）
#   2. 角度倾斜控制：监控倾倒量，达到目标量后停止
#   3. 退出倾倒模式
#
# 前提条件：
#   - 已手持液体瓶并进入倾倒模式
#   - 游戏中已按 Alt+J 开启交互检测（用于监控倾倒量）
#   - 游戏中已按 F12 开启产品扫描（用于获取位置信息）
#
# 使用方法：
#   python auto_pouring.py pour <容器名称> <目标倾倒量ml>
#   例如: python auto_pouring.py pour "Paella Pan" 50
#
# 依赖：
#   - C# Mod: CS_CamDump (v4.60+) 支持瓶口位置(spout_position)输出
#   - interaction_detector.py 提供倾倒量监控
#   - local_actions.py 提供基础操作函数
#
# v1.6更新：
#   - 新增全局累计量管理器，统一跟踪倾倒量（处理弹窗重置）
#   - 分离监控和决策：达标后直接返回，由调用者退出
#   - exit_pouring_mode 等待液体停止(is_pouring=false)后返回最终精确值
#   - 解决检测到49ml但实际倾倒50ml的时序差问题
#
# v1.5更新：
#   - 使用 spout_position（实际出液点位置）替代 bounds 中心计算
#   - 解决瓶子倾斜时瓶口位置计算不准确的问题
#
# v1.4更新：
#   - 试探标定法：通过实际移动测试鼠标与3D空间的映射关系
#   - 解决不同相机姿态下映射关系不固定的问题
#   - 使用2x2变换矩阵计算精确的鼠标移动量
#
# v1.3更新：
#   - 修复：倾斜控制使用左键（不是右键）
#   - 修复：液体瓶检测支持倾倒模式
#
# v1.2更新：
#   - 支持 is_pouring_mode 和 is_pouring 字段（C# Mod v4.55）
#   - 添加倾倒模式液体瓶检测
#   - 实现单行覆盖更新输出（使用 \r）
#
# v1.1更新：
#   - 简化逻辑：C# Mod v4.53已将Pan/Grill Pan的position修正为bounds中心
#   - Python只需直接使用position字段，无需额外计算
#
# ================================================================

import os
import sys
import json
import time
import math
import re
from typing import Any, Optional, Dict, List, Tuple, Union

# 添加相关模块到 Python 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO_NAV_PATH = os.path.join(SCRIPT_DIR, "..", "auto_navigation")
DEPLOY_AGENT_PATH = os.path.join(SCRIPT_DIR, "..", "..", "deploy_agent")

# 先添加本地目录，确保导入本地 config.py
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

for path in [AUTO_NAV_PATH, DEPLOY_AGENT_PATH]:
    if path not in sys.path:
        sys.path.append(path)  # 用 append 而不是 insert，避免覆盖本地模块

# 从本包 config.py 导入配置
try:
    from .config import (
        USERDATA_PATH,
        REALTIME_PRODUCTS_JSON,
        GAME_WINDOW_TITLE,
        DEFAULT_POUR_AMOUNT,
        DEFAULT_POUR_TOLERANCE,
    )
except Exception:  # pragma: no cover
    from epm.cerebellum.skills.auto_pouring.config import (  # type: ignore
        USERDATA_PATH,
        REALTIME_PRODUCTS_JSON,
        GAME_WINDOW_TITLE,
        DEFAULT_POUR_AMOUNT,
        DEFAULT_POUR_TOLERANCE,
    )

# ================================================================
# 闭环控制参数
# ================================================================
POSITION_TOLERANCE = 0.02      # 位置容差 2cm（液体瓶对准容器中心）
MAX_ITERATIONS = 50            # 最大迭代次数（位置对准）

# 动作参数
ACTION_DELAY = 0.05            # 每个动作后的等待时间
HORIZONTAL_MOVE_DELAY = 0.8    # 水平移动后额外等待，给扫描文件留出更新时间
CALIBRATION_SETTLE_DELAY = 0.8  # 标定阶段额外等待，让位置文件稳定更新
TILT_STEP_DURATION = 0.1       # 每次倾斜步进的时长（秒）
TILT_CHECK_INTERVAL = 0.05     # 检查倾倒量的间隔（秒）
TILT_MICRO_STEP_PIXELS = 4     # 单次倾斜动作拆成更小的微步，避免倾角突变
TILT_MICRO_STEP_DELAY = 0.03   # 微步之间的短暂停顿

# 动态移动步长
MOVE_LARGE_THRESHOLD = 0.08    # 大误差阈值（米）
MOVE_MEDIUM_THRESHOLD = 0.04   # 中误差阈值（米）
MOVE_LARGE_PIXELS = 150        # 大误差时的像素数
MOVE_MEDIUM_PIXELS = 80        # 中误差时的像素数
MOVE_SMALL_PIXELS = 30         # 小误差时的像素数

# 倾斜控制参数
TILT_PIXELS_PER_STEP = 20      # 每次倾斜的鼠标移动像素
MAX_TILT_STEPS = 600           # 最大倾斜步数（防止无限倾斜）
CONTAINER_POUR_ROTATION_THRESHOLD = 120.0  # container->container 常规分支目标角（raw 120）
CONTAINER_POT_ROTATION_THRESHOLD = 160.0   # pot：专门倾斜到 160 度
CONTAINER_PLATE_ROTATION_Z_THRESHOLD = 90.0  # plate-like source：监控 rotation.z，到 90 度停止
CONTAINER_POUR_ROTATION_THRESHOLD_MIRRORED = 210.0  # container->container 180 分支目标角（raw 210）
CONTAINER_BAKE_TRAY_ROTATION_THRESHOLD = 70.0  # bake tray 作为倾倒源时，专门倾斜到 70 度
CONTAINER_POUR_MAX_STEPS = 250             # container->container 最多按 S 的次数；单步变小后进一步放宽上限
CONTAINER_POUR_KEY_HOLD_SECONDS = 0.015   # container->container 每次 W 的短按时长；进一步减小单步倾斜幅度
CONTAINER_PAN_PRETILT_KEY_HOLD_SECONDS = 0.015  # pan-like 容器首下轻微预倾斜，先让出液边显现
CONTAINER_PAN_FIXED_BIAS_METERS = 0.20    # pan-like：初期固定左偏 20cm
CONTAINER_PLATE_FIXED_BIAS_METERS = 0.10  # 其他 plate-like：初期固定左偏 10cm
CONTAINER_LARGE_PLATE_FIXED_BIAS_METERS = 0.15  # large plate：初期固定左偏 15cm
CONTAINER_BOWL_FIXED_BIAS_METERS = 0.03   # bowl：初期固定左偏 3cm
CONTAINER_SMALL_POT_FIXED_BIAS_METERS = 0.05  # small pot：初期固定左偏 5cm
CONTAINER_BIG_POT_FIXED_BIAS_METERS = 0.15    # big pot：初期固定左偏 15cm
CONTAINER_BAKE_TRAY_FIXED_BIAS_METERS = 0.30  # bake tray：初期固定左偏 30cm
BOX_SOURCE_RIGHT_BIAS_METERS = 0.18       # box-like：固定偏置距离，当前为 18cm
BOX_SOURCE_CAMERA_BACK_ANGLE_DEGREES = 55.0  # box-like：从屏幕右方向朝屏幕下方向偏 55°
BOX_SOURCE_MAX_TILT_DEGREES = 70.0        # box-like：最大倾斜角 70 度
CONTAINER_POUR_STEP_DELAY = 0.12           # 每次按 S 后等待姿态稳定
CONTAINER_ALIGN_THRESHOLD = 0.00           # container->container 外层阈值为 0：每次按 W 前都先做位置对齐
CONTAINER_ALIGN_MOVE_PIXELS = 30          # container->container 对准时固定每步 30px

# 数据更新等待参数
DATA_UPDATE_TIMEOUT = 3.0      # 等待数据更新的最大时间（秒）
DATA_CHECK_INTERVAL = 0.05     # 检查数据更新的间隔（秒）
POSE_SETTLE_TIMEOUT = 3.5      # 水平移动后等待源物体位置稳定的最大时间
POSE_CHANGE_EPSILON = 0.002    # 认为位置已发生变化的最小距离（米）
POSE_STABLE_EPSILON = 0.0015   # 认为位置已稳定的最大抖动（米）

# 标定参数
CALIBRATION_PIXELS = 30        # 标定时的试探移动像素数
CALIBRATION_MIN_DISPLACEMENT = 0.005  # 标定有效的最小3D位移（米）
PROBE_RECOVERY_MIN_MOVEMENT = 0.004   # 探测某个方向是否“真的能动”的最小位移（米）
PROBE_RECOVERY_MIN_IMPROVEMENT = 0.003  # 探测后认为“更接近目标”的最小改进量（米）
CONTAINER_AXIS_FLIP_EPSILON = 0.0005  # fixed-step container feedback: single-axis worsening threshold
CONTAINER_DEAD_ZONE_POS_EPSILON = 0.003   # fixed-step container feedback: treat <3mm as nearly not moving
CONTAINER_DEAD_ZONE_DIST_EPSILON = 0.002  # fixed-step container feedback: distance change under 2mm counts as stalled


# ================================================================
# 累计倾倒量管理器
# ================================================================
# 全局状态：跟踪整个倾倒过程的累计量
_pour_state = {
    'cached_total': 0.0,        # 弹窗重置前的累计量
    'last_pour_amount': 0.0,    # 上次读取的 pour_amount
    'session_start': 0.0,       # 当前会话起始值
    'is_tracking': False,       # 是否正在跟踪
}
_last_pouring_source_log_signature: Optional[Tuple[Any, str, float]] = None


def reset_pour_state():
    """重置倾倒量跟踪状态"""
    global _pour_state
    _pour_state = {
        'cached_total': 0.0,
        'last_pour_amount': 0.0,
        'session_start': 0.0,
        'is_tracking': False,
    }


def start_pour_tracking(initial_amount: float = 0.0):
    """
    开始倾倒量跟踪

    Args:
        initial_amount: 初始倾倒量（当前弹窗显示的值）
    """
    global _pour_state
    _pour_state['cached_total'] = 0.0
    _pour_state['last_pour_amount'] = initial_amount
    _pour_state['session_start'] = initial_amount
    _pour_state['is_tracking'] = True


def update_pour_tracking(current_pour_amount: float) -> float:
    """
    更新倾倒量跟踪（每次读取数据时调用）

    Args:
        current_pour_amount: 当前弹窗显示的倾倒量

    Returns:
        当前累计倾倒量
    """
    global _pour_state

    if not _pour_state['is_tracking']:
        return 0.0

    # 检测弹窗重置：pour_amount 突然变小很多（超过5ml）
    if current_pour_amount < _pour_state['last_pour_amount'] - 5.0:
        # 弹窗重置了，把之前的量累加到缓存
        session_poured = _pour_state['last_pour_amount'] - _pour_state['session_start']
        if session_poured > 0:
            _pour_state['cached_total'] += session_poured
            print_status(f"[弹窗重置] 缓存累计: {_pour_state['cached_total']:.1f} ml", end_line=True)
        # 重置会话起始量
        _pour_state['session_start'] = current_pour_amount

    _pour_state['last_pour_amount'] = current_pour_amount

    return get_pour_total()


def get_pour_total() -> float:
    """
    获取当前累计倾倒量

    Returns:
        累计倾倒量（缓存量 + 当前会话量）
    """
    global _pour_state

    if not _pour_state['is_tracking']:
        return 0.0

    session_poured = _pour_state['last_pour_amount'] - _pour_state['session_start']
    return _pour_state['cached_total'] + max(0, session_poured)


def finalize_pour_tracking() -> float:
    """
    结束倾倒量跟踪，返回最终累计量

    在 exit_pouring_mode 时调用，确保返回准确的最终值

    Returns:
        最终累计倾倒量
    """
    global _pour_state

    final_total = get_pour_total()
    _pour_state['is_tracking'] = False

    return final_total


# ================================================================
# 单行更新输出辅助函数
# ================================================================

def print_status(message: str, end_line: bool = False):
    """
    输出状态信息（使用 \\r 实现单行覆盖更新）

    Args:
        message: 状态信息
        end_line: True=换行（状态结束），False=覆盖当前行
    """
    # 清除当前行（最多80字符）
    if end_line:
        print(f"\r{message:<80}")
    else:
        print(f"\r{message:<80}", end="", flush=True)


def print_pouring_status(poured: float, target: float, is_pouring: bool = False):
    """
    输出倾倒状态（单行覆盖更新）

    Args:
        poured: 已倾倒量 (ml)
        target: 目标量 (ml)
        is_pouring: 是否正在倾倒
    """
    progress = min(poured / target * 100, 100) if target > 0 else 0
    bar_len = 20
    filled = int(bar_len * progress / 100)
    bar = "█" * filled + "░" * (bar_len - filled)

    status = "倾倒中" if is_pouring else "等待"
    msg = f"[{bar}] {poured:.1f}/{target:.1f} ml ({progress:.0f}%) - {status}"
    print_status(msg)


# ================================================================
# JSON文件读取函数
# ================================================================

def get_json_file_mtime() -> float:
    """获取JSON文件的修改时间戳"""
    try:
        if os.path.exists(REALTIME_PRODUCTS_JSON):
            return os.path.getmtime(REALTIME_PRODUCTS_JSON)
        return 0.0
    except Exception:
        return 0.0


def wait_for_data_update(old_mtime: float, timeout: float = DATA_UPDATE_TIMEOUT,
                          verbose: bool = False) -> bool:
    """等待JSON文件更新（时间戳变化）"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        new_mtime = get_json_file_mtime()
        if new_mtime > old_mtime:
            return True
        time.sleep(DATA_CHECK_INTERVAL)
    return False


def get_active_source_top(data: Optional[Dict] = None) -> Optional[Tuple[float, float, float]]:
    """读取当前倾倒源的瓶口/壶口位置。"""
    if data is None:
        data = read_realtime_products()
    if data is None:
        return None
    bottle = find_active_pour_source(data)
    if bottle is None:
        return None
    return get_bottle_top_position(bottle)


def _xz_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2)


def wait_for_source_pose_settle(
    old_mtime: float,
    before_top: Optional[Tuple[float, float, float]],
    timeout: float = POSE_SETTLE_TIMEOUT,
) -> Optional[Tuple[float, float, float]]:
    """
    等到扫描文件更新，且当前倾倒源的位置真正变化并基本稳定后再返回。
    如果因为边界等原因没有明显移动，则超时后返回最后观测值。
    """
    start_time = time.time()
    last_top: Optional[Tuple[float, float, float]] = None
    saw_change = False

    while time.time() - start_time < timeout:
        new_mtime = get_json_file_mtime()
        if new_mtime <= old_mtime:
            time.sleep(DATA_CHECK_INTERVAL)
            continue

        data = read_realtime_products()
        top = get_active_source_top(data)
        if top is None:
            time.sleep(DATA_CHECK_INTERVAL)
            continue

        if before_top is not None and _xz_distance(top, before_top) >= POSE_CHANGE_EPSILON:
            saw_change = True

        if last_top is not None:
            settled = _xz_distance(top, last_top) <= POSE_STABLE_EPSILON
            if saw_change and settled:
                return top

        last_top = top
        time.sleep(0.12)

    if last_top is not None:
        return last_top
    return get_active_source_top()


def read_realtime_products() -> Optional[Dict]:
    """读取实时物品扫描数据"""
    try:
        if not os.path.exists(REALTIME_PRODUCTS_JSON):
            print(f"[!] 扫描文件不存在: {REALTIME_PRODUCTS_JSON}")
            print("[!] 请确保游戏运行中并按 F12 开启扫描")
            return None

        with open(REALTIME_PRODUCTS_JSON, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        return data
    except json.JSONDecodeError as e:
        print(f"[!] JSON 解析错误: {e}")
        return None
    except Exception as e:
        print(f"[!] 读取扫描数据失败: {e}")
        return None


# ================================================================
# 物品查找函数
# ================================================================

def find_held_item(data: Dict = None, instance_id: Optional[int] = None) -> Optional[Dict]:
    """查找当前手持的物品（液体瓶）"""
    if data is None:
        data = read_realtime_products()

    if data is None or "products" not in data:
        return None

    for product in data["products"]:
        if instance_id is not None:
            try:
                if int(product.get("instance_id")) != int(instance_id):  # type: ignore[arg-type]
                    continue
            except Exception:
                continue
        if product.get("is_held", False):
            return product

    return None


def find_pouring_bottle(data: Dict = None) -> Optional[Dict]:
    """
    查找当前进入倾倒模式的液体瓶

    v1.4: 当有多个 is_pouring_mode=True 的物品时，
    优先选择 is_on_screen=True 的。

    Returns:
        dict: 液体瓶信息（包含 is_pouring_mode=True 的物品）
        None: 未找到
    """
    if data is None:
        data = read_realtime_products()

    if data is None or "products" not in data:
        return None

    def _position_y(item: Dict) -> float:
        pos = item.get("position")
        if isinstance(pos, dict):
            try:
                return float(pos.get("y"))
            except Exception:
                return float("-inf")
        return float("-inf")

    # 收集所有倾倒模式的物品
    pouring_items = []
    for product in data["products"]:
        if product.get("is_pouring_mode", False):
            pouring_items.append(product)

    if not pouring_items:
        return None

    if len(pouring_items) == 1:
        return pouring_items[0]

    global _last_pouring_source_log_signature
    chosen = max(
        pouring_items,
        key=lambda p: (
            _position_y(p),
            1 if bool(p.get("is_on_screen", False)) else 0,
            1 if bool(p.get("is_held", False)) else 0,
        ),
    )
    chosen_name = str(chosen.get("name_en") or chosen.get("name_cn") or chosen.get("name") or "").strip()
    chosen_y = _position_y(chosen)
    chosen_signature = (
        chosen.get("instance_id"),
        chosen_name,
        round(float(chosen_y), 4),
    )
    if _last_pouring_source_log_signature != chosen_signature:
        print(
            f"[*] 检测到 {len(pouring_items)} 个 is_pouring_mode=true 的物体；"
            f"按 position.y 最大选择倾倒源: {chosen_name or 'unknown'} (y={chosen_y:.3f})"
        )
        _last_pouring_source_log_signature = chosen_signature
    return chosen


def _get_item_kind(item: Optional[Dict]) -> str:
    """返回扫描结果中的 kind，小写。"""
    if not item:
        return ""
    return str(item.get("kind", "") or "").strip().lower()


def is_container_item(item: Optional[Dict]) -> bool:
    kind = _get_item_kind(item)
    if kind in {"container", "containers"}:
        return True
    # Dishware like Plate / Bowl / Casserole can appear as tools in scan data,
    # but when they enter pouring mode we still want them to follow the
    # container->container pouring path.
    return _is_plate_like_container(item)


def is_liquid_item(item: Optional[Dict]) -> bool:
    return _get_item_kind(item) in {"liquid", "liquids"}


def classify_active_pour_source(item: Optional[Dict]) -> str:
    """识别当前处于 is_pouring_mode / is_held 的源物体类型。"""
    if item is None:
        return "unknown"
    if is_liquid_item(item):
        return "liquid"
    if is_container_item(item):
        return "container"
    return "unknown"


def _is_cutting_board_like(item: Optional[Dict]) -> bool:
    if not isinstance(item, dict):
        return False
    text = " ".join(
        str(item.get(k, "") or "")
        for k in ("name", "name_en", "name_cn", "display_name", "prefab_name")
    ).strip().lower()
    return any(token in text for token in ("cutting board", "chopping board", "菜板", "砧板", "cutting_board"))


def _is_pan_like_container(item: Optional[Dict]) -> bool:
    if not isinstance(item, dict):
        return False
    text = " ".join(
        str(item.get(k, "") or "")
        for k in ("name", "name_en", "name_cn", "display_name", "prefab_name", "component_type")
    ).strip().lower()
    if "pancontainer" in text:
        return True
    if any(token in text for token in ("frying pan", "grill pan", "paella pan", "煎锅")):
        return True
    return re.search(r"\bpan\b", text) is not None


def _is_bake_tray_container(item: Optional[Dict]) -> bool:
    if not isinstance(item, dict):
        return False
    text = " ".join(
        str(item.get(k, "") or "")
        for k in ("name", "name_en", "name_cn", "display_name", "prefab_name", "component_type")
    ).strip().lower()
    return ("bake tray" in text) or ("baking tray" in text) or ("烘烤托盘" in text) or ("烤盘" in text)


def _is_small_pot_container(item: Optional[Dict]) -> bool:
    if not isinstance(item, dict):
        return False
    text = " ".join(
        str(item.get(k, "") or "")
        for k in ("name", "name_en", "name_cn", "display_name", "prefab_name", "component_type")
    ).strip().lower()
    return ("small pot" in text) or ("小锅" in text)


def _is_big_pot_container(item: Optional[Dict]) -> bool:
    if not isinstance(item, dict):
        return False
    text = " ".join(
        str(item.get(k, "") or "")
        for k in ("name", "name_en", "name_cn", "display_name", "prefab_name", "component_type")
    ).strip().lower()
    return ("big pot" in text) or ("大锅" in text)


def _is_pot_like_container(item: Optional[Dict]) -> bool:
    if _is_small_pot_container(item) or _is_big_pot_container(item):
        return True
    if not isinstance(item, dict):
        return False
    text = " ".join(
        str(item.get(k, "") or "")
        for k in ("name", "name_en", "name_cn", "display_name", "prefab_name", "component_type")
    ).strip().lower()
    return ("potcontainer" in text) or re.search(r"\bpot\b", text) is not None


def _is_plate_like_container(item: Optional[Dict]) -> bool:
    if not isinstance(item, dict):
        return False
    if _is_bake_tray_container(item) or _is_pan_like_container(item) or _is_pot_like_container(item):
        return False
    text = " ".join(
        str(item.get(k, "") or "")
        for k in ("name", "name_en", "name_cn", "display_name", "prefab_name", "component_type", "game_object")
    ).strip().lower()
    plate_tokens = (
        "deep plate",
        "large plate",
        "small plate",
        "square plate",
        "plastic bowl",
        "casserole",
        "bowl",
        "plate",
        "深盘",
        "大盘",
        "小盘",
        "方盘",
        "塑料碗",
        "砂锅",
        "碗",
        "盘",
    )
    return any(token in text for token in plate_tokens)


def _is_bowl_like_container(item: Optional[Dict]) -> bool:
    if not isinstance(item, dict):
        return False
    text = " ".join(
        str(item.get(k, "") or "")
        for k in ("name", "name_en", "name_cn", "display_name", "prefab_name", "component_type", "game_object")
    ).strip().lower()
    bowl_tokens = (
        "plastic bowl",
        "bowl",
        "塑料碗",
        "碗",
    )
    return any(token in text for token in bowl_tokens)


def _is_large_plate_container(item: Optional[Dict]) -> bool:
    if not isinstance(item, dict):
        return False
    text = " ".join(
        str(item.get(k, "") or "")
        for k in ("name", "name_en", "name_cn", "display_name", "prefab_name", "component_type", "game_object")
    ).strip().lower()
    large_plate_tokens = (
        "large plate",
        "大盘",
    )
    return any(token in text for token in large_plate_tokens)


def _is_box_like_pour_source(item: Optional[Dict]) -> bool:
    if not isinstance(item, dict):
        return False
    text = " ".join(
        str(item.get(k, "") or "")
        for k in ("name", "name_en", "name_cn", "display_name", "prefab_name")
    ).strip().lower()
    return "box" in text or "盒" in text


def _is_supported_pour_source(item: Optional[Dict]) -> tuple[bool, str]:
    source_type = classify_active_pour_source(item)
    if source_type == "liquid":
        return True, ""
    if source_type == "container":
        if _is_cutting_board_like(item):
            return False, "cutting_board_not_supported_for_auto_pour"
        return True, ""
    return False, f"unsupported_pour_source_type:{source_type}"


def _pouring_mode_lost_reason(item: Optional[Dict]) -> str:
    held_name = ""
    held_kind = ""
    if isinstance(item, dict):
        held_name = str(item.get("name_en") or item.get("name_cn") or item.get("name") or "").strip()
        held_kind = str(item.get("kind") or "").strip()
    suffix = ""
    if held_name:
        suffix += f":held_item={held_name!r}"
    if held_kind:
        suffix += f":held_kind={held_kind!r}"
    return f"pouring_mode_lost:is_pouring_mode_became_false{suffix}"


def find_active_pour_source(data: Dict = None) -> Optional[Dict]:
    """
    查找当前真实的倾倒源物体。

    优先选择 is_pouring_mode=True 的物体；如果还没进入倾倒模式，则回退到当前手持物体。
    这里不再假设它一定是液体瓶，因为现在也支持 container->container。
    """
    if data is None:
        data = read_realtime_products()

    if data is None:
        return None

    pouring_item = find_pouring_bottle(data)
    if pouring_item:
        return pouring_item

    return find_held_item(data)


def wait_for_pour_source_ready(
    *,
    require_pouring_mode: bool,
    timeout_s: float = 4.5,
    stable_s: float = 2.0,
    poll_s: float = 0.05,
) -> Tuple[Optional[Dict], Optional[Dict], str]:
    """
    Wait until the active pour source exists and stays in the expected state.

    Returns:
      (data, source_item, reason)
      `reason == ""` means the source stayed ready for `stable_s` seconds.
    """
    deadline = time.time() + max(float(timeout_s), float(stable_s))
    stable_needed = max(0.0, float(stable_s))
    stable_since: Optional[float] = None
    last_data: Optional[Dict] = None
    last_source: Optional[Dict] = None
    last_reason = "realtime_products_unavailable"

    while True:
        data = read_realtime_products()
        last_data = data

        if data is None:
            stable_since = None
            last_reason = "realtime_products_unavailable"
        else:
            source_item = find_active_pour_source(data)
            last_source = source_item
            if source_item is None:
                stable_since = None
                last_reason = "pour_source_not_found"
            else:
                supported, unsupported_reason = _is_supported_pour_source(source_item)
                in_pouring_mode = bool(source_item.get("is_pouring_mode", False))
                if not supported:
                    stable_since = None
                    last_reason = f"unsupported_pour_source:{unsupported_reason}"
                elif require_pouring_mode and not in_pouring_mode:
                    stable_since = None
                    last_reason = "not_in_pouring_mode"
                else:
                    now = time.time()
                    if stable_needed <= 0.0:
                        return data, source_item, ""
                    if stable_since is None:
                        stable_since = now
                    elif (now - stable_since) >= stable_needed:
                        return data, source_item, ""

        if time.time() >= deadline:
            return last_data, last_source, last_reason
        time.sleep(max(0.01, float(poll_s)))


def find_liquid_bottle(data: Dict = None) -> Optional[Dict]:
    """
    查找可用的液体瓶（手持或倾倒模式）

    优先返回倾倒模式的液体瓶，其次返回手持的物品

    Returns:
        dict: 液体瓶信息
        None: 未找到
    """
    if data is None:
        data = read_realtime_products()

    if data is None:
        return None

    # 优先检查倾倒模式
    pouring_bottle = find_pouring_bottle(data)
    if is_liquid_item(pouring_bottle):
        return pouring_bottle

    # 其次检查手持物品
    held_item = find_held_item(data)
    if is_liquid_item(held_item):
        return held_item

    return None


def find_container_by_name(
    container_name: str,
    data: Dict = None,
    must_be_on_screen: bool = False,
    instance_id: Optional[int] = None,
) -> Optional[Dict]:
    """
    根据名称查找容器

    Args:
        container_name: 容器名称（中文或英文）
        data: 扫描数据（可选）
        must_be_on_screen: 是否必须在屏幕内

    Returns:
        dict: 容器信息
        None: 未找到
    """
    if data is None:
        data = read_realtime_products()

    if data is None or "products" not in data:
        return None

    container_name_lower = container_name.lower().strip()

    # 收集所有匹配的容器
    matches = []
    for product in data["products"]:
        if instance_id is not None:
            try:
                if int(product.get("instance_id")) != int(instance_id):  # type: ignore[arg-type]
                    continue
            except Exception:
                continue
        # 排除手持物品
        if product.get("is_held", False):
            continue

        name_cn = product.get("name_cn", "").lower()
        name_en = product.get("name_en", "").lower()

        # 匹配
        if (container_name_lower in name_cn or container_name_lower in name_en or
            name_cn in container_name_lower or name_en in container_name_lower):
            if not must_be_on_screen or product.get("is_on_screen", False):
                matches.append(product)

    if not matches:
        return None

    # 返回距离最近的
    matches.sort(key=lambda x: x.get("distance", 999))
    return matches[0]


def _any_product_with_instance_id(data: Dict, instance_id: int) -> Optional[Dict]:
    if not isinstance(data, dict) or "products" not in data or not isinstance(data.get("products"), list):
        return None
    for p in data.get("products") or []:
        if not isinstance(p, dict):
            continue
        try:
            if int(p.get("instance_id")) == int(instance_id):  # type: ignore[arg-type]
                return p
        except Exception:
            continue
    return None


def _best_effort_aim_at_container(container: Dict) -> None:
    """
    Best-effort: rotate view towards the container using its screen_x/screen_y (if present),
    then a right-click can enter pouring mode.
    """
    try:
        from epm.cerebellum.local_actions import move_related_mouse
    except ImportError:
        return

    sx = container.get("screen_x")
    sy = container.get("screen_y")
    if not isinstance(sx, (int, float)) or not isinstance(sy, (int, float)):
        return

    # Heuristic: realtime_products screen coords use 1600x900-ish convention; center (800,450).
    dx = int(round(float(sx) - 800.0))
    dy = int(round(float(sy) - 450.0))

    # Clamp so we don't spin wildly on bad readings.
    dx = max(-400, min(400, dx))
    dy = max(-300, min(300, dy))
    if dx == 0 and dy == 0:
        return

    move_related_mouse(dx, dy)
    time.sleep(0.05)


# ================================================================
# 位置和边界辅助函数
# ================================================================

def get_position(item: Dict) -> Tuple[float, float, float]:
    """获取物品的3D位置"""
    pos = item.get("position", {})
    return (
        pos.get("x", 0),
        pos.get("y", 0),
        pos.get("z", 0)
    )


def get_bounds(item: Dict) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    获取物品的边界框

    Returns:
        tuple: ((min_x, min_y, min_z), (max_x, max_y, max_z))
        None: 没有边界框信息
    """
    bounds_min = item.get("bounds_min")
    bounds_max = item.get("bounds_max")

    if bounds_min is None or bounds_max is None:
        return None

    return (
        (bounds_min.get("x", 0), bounds_min.get("y", 0), bounds_min.get("z", 0)),
        (bounds_max.get("x", 0), bounds_max.get("y", 0), bounds_max.get("z", 0))
    )


def get_pan_like_base_bias_distance(source_item: Dict) -> float:
    """
    pan-like / pot-like / plate-like 倾倒源的基础左偏距离，只在开始时读取一次。
    """
    if _is_bake_tray_container(source_item):
        return CONTAINER_BAKE_TRAY_FIXED_BIAS_METERS
    if _is_small_pot_container(source_item):
        return CONTAINER_SMALL_POT_FIXED_BIAS_METERS
    if _is_big_pot_container(source_item):
        return CONTAINER_BIG_POT_FIXED_BIAS_METERS
    if _is_plate_like_container(source_item):
        if _is_bowl_like_container(source_item):
            return CONTAINER_BOWL_FIXED_BIAS_METERS
        if _is_large_plate_container(source_item):
            return CONTAINER_LARGE_PLATE_FIXED_BIAS_METERS
        return CONTAINER_PLATE_FIXED_BIAS_METERS
    if not _is_pan_like_container(source_item):
        return 0.0

    return CONTAINER_PAN_FIXED_BIAS_METERS


def get_container_pour_center(container: Dict) -> Tuple[float, float, float]:
    """
    获取容器的倾倒目标中心位置

    v4.53+ C# Mod 已将 Pan/Grill Pan 的 position 修正为 bounds 几何中心，
    所以直接返回 position 即可。

    Args:
        container: 容器信息

    Returns:
        tuple: (x, y, z) 倾倒目标中心位置
    """
    # C# Mod v4.53+ 已修正 Pan/Grill Pan 的 position，直接使用
    return get_position(container)


def is_within_bounds(x: float, z: float, container: Dict, margin: float = 0.0) -> bool:
    """
    检查位置是否在容器边界内

    Args:
        x, z: 位置坐标（水平面）
        container: 容器信息
        margin: 边距（正值=允许超出边界一点）

    Returns:
        bool: 是否在边界内
    """
    bounds = get_bounds(container)
    if bounds is None:
        return True  # 无边界信息时假设在范围内

    (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
    return (min_x - margin <= x <= max_x + margin and
            min_z - margin <= z <= max_z + margin)


def get_bottle_top_position(bottle: Dict) -> Tuple[float, float, float]:
    """
    获取液体瓶瓶口位置（出液点位置）

    优先使用 spout_position（C# Mod v4.60 输出的实际出液点位置），
    如果没有则回退到 bounds 计算。

    Args:
        bottle: 液体瓶信息

    Returns:
        tuple: (x, y, z) 瓶口位置
    """
    if is_container_item(bottle):
        return get_position(bottle)

    # v1.5: 优先使用 spout_position（游戏内部的实际出液点位置）
    spout = bottle.get("spout_position")
    if spout and isinstance(spout, dict):
        return (spout.get("x", 0), spout.get("y", 0), spout.get("z", 0))

    # 回退：使用 bounds 计算（不够精确，瓶子倾斜时瓶口不在bounds中心）
    bounds = get_bounds(bottle)
    if bounds:
        (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
        # 瓶口在顶部中心（仅当瓶子竖直时准确）
        return (
            (min_x + max_x) / 2,
            max_y,  # 顶部
            (min_z + max_z) / 2
        )
    else:
        # 无bounds时使用位置作为近似
        return get_position(bottle)


def get_item_rotation_axes(item: Optional[Dict]) -> Tuple[float, float, float]:
    """返回物体 rotation 的 x/y/z，缺失时按 0 处理。"""
    if not item:
        return 0.0, 0.0, 0.0

    rotation = item.get("rotation", {})
    if not isinstance(rotation, dict):
        return 0.0, 0.0, 0.0

    try:
        return (
            float(rotation.get("x", 0) or 0.0),
            float(rotation.get("y", 0) or 0.0),
            float(rotation.get("z", 0) or 0.0),
        )
    except Exception:
        return 0.0, 0.0, 0.0


def get_item_tilt_angle(item: Optional[Dict]) -> float:
    """
    计算当前倾倒物体相对竖直方向的倾斜角。

    这里沿用液体瓶原来的判断方式：分别看 rotation.x / rotation.z
    相对 0/360 的最小夹角，再取较大者作为当前倾斜程度。
    """
    rotation_x, _, rotation_z = get_item_rotation_axes(item)
    tilt_from_x = rotation_x if rotation_x <= 180 else 360 - rotation_x
    tilt_from_z = rotation_z if rotation_z <= 180 else 360 - rotation_z
    return float(max(tilt_from_x, tilt_from_z))


def _normalize_rotation_angle(angle: float) -> float:
    angle = float(angle) % 360.0
    return angle if angle >= 0.0 else angle + 360.0


def _is_rotation_near_180_branch(angle: float) -> bool:
    angle = _normalize_rotation_angle(angle)
    dist_to_zero_branch = min(angle, 360.0 - angle)
    dist_to_180 = abs(angle - 180.0)
    return dist_to_180 < dist_to_zero_branch


def get_container_pour_target_state(
    item: Optional[Dict],
) -> Tuple[str, float, float, float, float, float]:
    """
    为 container->container 倾倒选择监控轴与目标角。

    默认监控 z 轴；plate-like source 也监控 z 轴，但使用单独阈值。

    返回:
      (axis_name, current_raw_angle, target_raw_angle, raw_x, raw_y, raw_z)
    """
    raw_x, raw_y, raw_z = get_item_rotation_axes(item)
    raw_x = _normalize_rotation_angle(raw_x)
    raw_y = _normalize_rotation_angle(raw_y)
    raw_z = _normalize_rotation_angle(raw_z)

    axis_name = "z"
    current_raw = min(raw_z, 360.0 - raw_z)
    target_raw = CONTAINER_POUR_ROTATION_THRESHOLD
    if _is_bake_tray_container(item):
        target_raw = CONTAINER_BAKE_TRAY_ROTATION_THRESHOLD
    elif _is_pot_like_container(item):
        target_raw = CONTAINER_POT_ROTATION_THRESHOLD
    elif _is_plate_like_container(item):
        axis_name = "z"
        current_raw = min(raw_z, 360.0 - raw_z)
        target_raw = CONTAINER_PLATE_ROTATION_Z_THRESHOLD

    return axis_name, current_raw, target_raw, raw_x, raw_y, raw_z


def is_container_pour_target_reached(
    item: Optional[Dict],
    axis_name: str,
    target_raw_angle: float,
) -> Tuple[bool, float]:
    raw_x, raw_y, raw_z = get_item_rotation_axes(item)
    axis_value = raw_z
    if axis_name == "x":
        axis_value = raw_x
    elif axis_name == "y":
        axis_value = raw_y
    current_raw = _normalize_rotation_angle(axis_value)

    current_progress = min(current_raw, 360.0 - current_raw)
    return current_progress >= target_raw_angle, current_progress


# ================================================================
# 倾倒量监控函数
# ================================================================

def parse_pour_amount(pour_str: str) -> Optional[float]:
    """
    解析倾倒量字符串为数值

    Args:
        pour_str: 倾倒量字符串，如 "42 ml"

    Returns:
        float: 倾倒量（ml），解析失败返回 None
    """
    if not pour_str:
        return None

    try:
        text = str(pour_str).strip()
        if not text:
            return None

        normalized = text.replace(",", ".")
        match = re.search(r'([\d.]+)\s*([a-zA-Z]+)?', normalized)
        if match:
            value = float(match.group(1))
            unit = str(match.group(2) or "ml").strip().lower()
            if unit in {"l", "lt", "ltr", "liter", "liters", "litre", "litres"}:
                return value * 1000.0
            return value
        return None
    except Exception:
        return None


def get_current_pour_amount() -> Optional[float]:
    """
    获取当前倾倒量（通过 interaction_detector）

    Returns:
        float: 当前倾倒量（ml），获取失败返回 None
    """
    try:
        from epm.cerebellum.skills.auto_navigation.interaction_detector import read_interaction_info_prefer_pour

        # 尝试读取交互信息
        info = read_interaction_info_prefer_pour(use_udp=True, max_age_s=0.1)
        if info and info.pour_amount:
            return parse_pour_amount(info.pour_amount)
        return None
    except ImportError:
        print("[!] 无法导入 interaction_detector")
        return None
    except Exception as e:
        print(f"[!] 获取倾倒量失败: {e}")
        return None


def init_pour_monitoring():
    """初始化倾倒量监控（启动UDP接收器）"""
    try:
        from epm.cerebellum.skills.auto_navigation.interaction_detector import init_udp_mode
        init_udp_mode()
        print("[*] 倾倒量监控已启动（UDP模式）")
        time.sleep(0.3)  # 等待接收器启动
    except ImportError:
        print("[!] 无法导入 interaction_detector，将使用文件模式")
    except Exception as e:
        print(f"[!] 初始化监控失败: {e}")


def stop_pour_monitoring():
    """停止倾倒量监控"""
    try:
        from epm.cerebellum.skills.auto_navigation.interaction_detector import stop_udp_mode
        stop_udp_mode()
    except:
        pass


# ================================================================
# 移动控制函数
# ================================================================

def get_move_pixels(error_distance: float) -> int:
    """根据位置误差大小返回合适的移动像素数"""
    if error_distance >= MOVE_LARGE_THRESHOLD:
        return MOVE_LARGE_PIXELS
    elif error_distance >= MOVE_MEDIUM_THRESHOLD:
        return MOVE_MEDIUM_PIXELS
    else:
        return MOVE_SMALL_PIXELS


def move_bottle_horizontal(dx: int, dy: int) -> None:
    """
    在倾倒模式下水平移动液体瓶（直接鼠标移动）

    Args:
        dx: X方向移动像素（正=右，负=左）
        dy: Y方向移动像素（正=下，负=上，对应游戏Z轴）
    """
    try:
        from epm.cerebellum.local_actions import horizontal_movement
    except ImportError:
        print("[!] 无法导入 local_actions")
        return

    # 直接鼠标移动控制液体瓶水平位置
    horizontal_movement(dx, dy)
    time.sleep(HORIZONTAL_MOVE_DELAY)


def tilt_bottle(pixels: int) -> None:
    """
    倾斜液体瓶（左键 + 鼠标上下移动）

    Args:
        pixels: 倾斜像素数（正=向下倾斜/开始倾倒，负=向上恢复）
    """
    try:
        from epm.cerebellum.local_actions import hold_mouse, leave_mouse, move_related_mouse
    except ImportError:
        print("[!] 无法导入 local_actions")
        return

    # 左键 + 鼠标移动控制倾斜角度。
    # 为了避免倾角一下跳大，这里把一次大位移拆成多个小微步。
    hold_mouse("left")
    time.sleep(0.02)
    remaining = int(pixels)
    step_sign = 1 if remaining >= 0 else -1
    remaining_abs = abs(remaining)
    micro_step = max(1, int(TILT_MICRO_STEP_PIXELS))

    while remaining_abs > 0:
        cur_step = min(micro_step, remaining_abs)
        move_related_mouse(0, step_sign * cur_step)  # Y方向移动控制倾斜
        remaining_abs -= cur_step
        if remaining_abs > 0:
            time.sleep(TILT_MICRO_STEP_DELAY)

    time.sleep(ACTION_DELAY)
    leave_mouse("left")
    time.sleep(ACTION_DELAY)


def press_container_pour_step(hold_seconds: Optional[float] = None) -> None:
    """
    container->container 倾倒时，按一次 S 键推进一次翻转。

    用户要求这里不要持续按住，而是每按一次 S，就重新做一次对齐。
    """
    try:
        from epm.cerebellum.local_actions import hold_keyboard, leave_keyboard
    except ImportError:
        print("[!] 无法导入 local_actions")
        return

    hold_keyboard("s")
    time.sleep(float(hold_seconds or CONTAINER_POUR_KEY_HOLD_SECONDS))
    leave_keyboard("s")
    time.sleep(CONTAINER_POUR_STEP_DELAY)


def exit_pouring_mode() -> float:
    """
    退出倾倒模式，返回最终累计倾倒量

    先执行退出动作（停止倾斜），然后等待液体停止流动，读取最终值

    Returns:
        最终累计倾倒量（ml）
    """
    try:
        from epm.cerebellum.local_actions import click_mouse
    except ImportError:
        print("[!] 无法导入 local_actions")
        return finalize_pour_tracking()

    print("[*] 退出倾倒模式...")

    # 1. 先执行退出动作（停止倾斜）
    click_mouse("right")
    time.sleep(0.2)

    # 2. 等待液体停止流动，读取最终值（最多等待2秒）
    max_wait_iterations = 20
    for i in range(max_wait_iterations):
        data = read_realtime_products()
        is_pouring = False
        if data:
            source_item = find_active_pour_source(data)
            if source_item:
                is_pouring = source_item.get("is_pouring", False)

        # 读取最新的倾倒量并更新跟踪
        current_amount = get_current_pour_amount()
        if current_amount is not None:
            update_pour_tracking(current_amount)

        if not is_pouring:
            # 液体已停止流动
            break

        time.sleep(0.1)

    # 3. 获取最终累计量
    final_total = finalize_pour_tracking()

    # 4. 停止监控
    stop_pour_monitoring()

    return final_total


# ================================================================
# 标定函数
# ================================================================

def calibrate_mouse_to_3d_mapping(
    verbose: bool = True,
    probe_pixels: Optional[int] = None,
    retry_depth: int = 0,
) -> Optional[Tuple[List[float], List[float]]]:
    """
    试探标定法：测试鼠标移动与3D空间位移的映射关系

    通过两次试探移动（X方向和Y方向），测量实际的3D位移，
    计算鼠标像素与3D空间的变换系数。

    Returns:
        tuple: (x_coeffs, z_coeffs) 其中：
            x_coeffs = [dx_per_mouse_x, dx_per_mouse_y]  # 鼠标移动对3D X轴的影响
            z_coeffs = [dz_per_mouse_x, dz_per_mouse_y]  # 鼠标移动对3D Z轴的影响
        None: 标定失败
    """
    pixels = int(probe_pixels or CALIBRATION_PIXELS)

    if verbose:
        print(f"[*] 开始鼠标-3D映射标定... (probe={pixels}px)")

    # 读取初始位置
    data = read_realtime_products()
    if data is None:
        print("[!] 标定失败：无法读取扫描数据")
        return None

    bottle = find_active_pour_source(data)
    if bottle is None:
        print("[!] 标定失败：未检测到液体瓶")
        return None

    initial_pos = get_bottle_top_position(bottle)
    init_x, init_y, init_z = initial_pos

    if verbose:
        print(f"  初始位置: ({init_x:.4f}, {init_z:.4f})")

    # ===== 试探1：鼠标 X 方向移动 =====
    pre_mtime = get_json_file_mtime()
    move_bottle_horizontal(pixels, 0)
    updated = wait_for_data_update(pre_mtime, timeout=2.0)
    if not updated:
        print("[!] 警告：X移动后JSON未更新")
    time.sleep(CALIBRATION_SETTLE_DELAY)
    wait_for_source_pose_settle(pre_mtime, initial_pos, timeout=max(2.0, CALIBRATION_SETTLE_DELAY + 1.0))

    data = read_realtime_products()
    bottle = find_active_pour_source(data)
    if bottle is None:
        print("[!] 标定失败：X移动后未检测到液体瓶")
        return None

    pos_after_x = get_bottle_top_position(bottle)
    dx1 = pos_after_x[0] - init_x  # 3D X位移
    dz1 = pos_after_x[2] - init_z  # 3D Z位移

    if verbose:
        print(f"  鼠标X+{pixels}px → 3D位移: (Δx={dx1:.4f}, Δz={dz1:.4f})")

    # 回到原位
    pre_mtime = get_json_file_mtime()
    move_bottle_horizontal(-pixels, 0)
    wait_for_data_update(pre_mtime, timeout=2.0)
    time.sleep(CALIBRATION_SETTLE_DELAY)
    wait_for_source_pose_settle(pre_mtime, pos_after_x, timeout=max(2.0, CALIBRATION_SETTLE_DELAY + 1.0))

    # ===== 试探2：鼠标 Y 方向移动 =====
    pre_mtime = get_json_file_mtime()
    data = read_realtime_products()
    bottle = find_active_pour_source(data)
    if bottle is None:
        print("[!] 标定失败：回位后未检测到液体瓶")
        return None

    pos_before_y = get_bottle_top_position(bottle)

    move_bottle_horizontal(0, pixels)
    updated = wait_for_data_update(pre_mtime, timeout=2.0)
    if not updated:
        print("[!] 警告：Y移动后JSON未更新")
    time.sleep(CALIBRATION_SETTLE_DELAY)
    wait_for_source_pose_settle(pre_mtime, pos_before_y, timeout=max(2.0, CALIBRATION_SETTLE_DELAY + 1.0))

    data = read_realtime_products()
    bottle = find_active_pour_source(data)
    if bottle is None:
        print("[!] 标定失败：Y移动后未检测到液体瓶")
        return None

    pos_after_y = get_bottle_top_position(bottle)
    dx2 = pos_after_y[0] - pos_before_y[0]  # 3D X位移
    dz2 = pos_after_y[2] - pos_before_y[2]  # 3D Z位移

    if verbose:
        print(f"  鼠标Y+{pixels}px → 3D位移: (Δx={dx2:.4f}, Δz={dz2:.4f})")

    # 回到原位
    pre_mtime = get_json_file_mtime()
    move_bottle_horizontal(0, -pixels)
    wait_for_data_update(pre_mtime, timeout=2.0)
    wait_for_source_pose_settle(pre_mtime, pos_after_y, timeout=max(2.0, CALIBRATION_SETTLE_DELAY + 1.0))

    # 检查标定是否有效（位移足够大）
    total_displacement = math.sqrt(dx1*dx1 + dz1*dz1) + math.sqrt(dx2*dx2 + dz2*dz2)
    min_required_displacement = 1e-6
    if total_displacement <= min_required_displacement:
        print(
            f"[!] 标定失败：位移过小 ({total_displacement:.4f}m < "
            f"{min_required_displacement:.4f}m)，可能当前倾倒模式下鼠标位移响应太弱"
        )
        return None

    # 计算每像素的3D位移系数
    # x_coeffs[0] = 鼠标X移动1像素导致的3D X位移
    # x_coeffs[1] = 鼠标Y移动1像素导致的3D X位移
    x_coeffs = [dx1 / pixels, dx2 / pixels]
    z_coeffs = [dz1 / pixels, dz2 / pixels]

    x_probe_displacement = math.sqrt(dx1 * dx1 + dz1 * dz1)
    y_probe_displacement = math.sqrt(dx2 * dx2 + dz2 * dz2)
    det = x_coeffs[0] * z_coeffs[1] - x_coeffs[1] * z_coeffs[0]

    if x_probe_displacement <= 1e-6 or y_probe_displacement <= 1e-6 or abs(det) < 1e-10:
        if retry_depth < 2:
            next_pixels = min(160, pixels * 2)
            if verbose:
                print(
                    f"[!] 标定退化：x_probe={x_probe_displacement:.4f}m "
                    f"y_probe={y_probe_displacement:.4f}m det={det:.6e}，"
                    f"改用更大步长 {next_pixels}px 重试"
                )
            return calibrate_mouse_to_3d_mapping(
                verbose=verbose,
                probe_pixels=next_pixels,
                retry_depth=retry_depth + 1,
            )
        print(
            f"[!] 标定失败：映射退化 "
            f"(x_probe={x_probe_displacement:.4f}m, y_probe={y_probe_displacement:.4f}m, det={det:.6e})"
        )
        return None

    if verbose:
        print(f"  标定完成:")
        print(f"    鼠标X移动1px → 3D位移 ({x_coeffs[0]:.6f}, {z_coeffs[0]:.6f})")
        print(f"    鼠标Y移动1px → 3D位移 ({x_coeffs[1]:.6f}, {z_coeffs[1]:.6f})")

    return (x_coeffs, z_coeffs)


def compute_mouse_movement_vector(
    err_x: float,
    err_z: float,
    x_coeffs: List[float],
    z_coeffs: List[float],
) -> Tuple[float, float]:
    """
    根据3D误差和标定系数，计算需要的鼠标移动量

    使用2x2矩阵求逆来计算：
    [err_x]   [x_coeffs[0]  x_coeffs[1]] [mouse_dx]
    [err_z] = [z_coeffs[0]  z_coeffs[1]] [mouse_dy]

    Args:
        err_x: 3D X方向误差（目标-当前）
        err_z: 3D Z方向误差（目标-当前）
        x_coeffs: [dx_per_mouse_x, dx_per_mouse_y]
        z_coeffs: [dz_per_mouse_x, dz_per_mouse_y]

    Returns:
        tuple: (mouse_dx, mouse_dy) 鼠标移动像素数（浮点）
    """
    # 矩阵 A = [[a, b], [c, d]]
    a, b = x_coeffs[0], x_coeffs[1]
    c, d = z_coeffs[0], z_coeffs[1]

    # 行列式
    det = a * d - b * c

    if abs(det) < 1e-10:
        # 矩阵奇异，使用简单的比例计算
        mouse_dx = (err_x / x_coeffs[0]) if abs(x_coeffs[0]) > 1e-6 else 0.0
        mouse_dy = (err_z / z_coeffs[1]) if abs(z_coeffs[1]) > 1e-6 else 0.0
        return (mouse_dx, mouse_dy)

    # 逆矩阵 A^-1 = (1/det) * [[d, -b], [-c, a]]
    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det

    # 计算鼠标移动量
    mouse_dx = inv_a * err_x + inv_b * err_z
    mouse_dy = inv_c * err_x + inv_d * err_z

    return (mouse_dx, mouse_dy)


def compute_mouse_movement(err_x: float, err_z: float,
                           x_coeffs: List[float], z_coeffs: List[float]) -> Tuple[int, int]:
    mouse_dx, mouse_dy = compute_mouse_movement_vector(err_x, err_z, x_coeffs, z_coeffs)
    return (int(round(mouse_dx)), int(round(mouse_dy)))


def get_pan_like_source_offset(
    calibration: Tuple[List[float], List[float]],
    current_raw_angle: float,
    target_raw_angle: float,
    base_bias_distance: float,
) -> Tuple[Tuple[float, float, float], float]:
    """
    为 pan-like / pot-like 倾倒源构造“初期左偏，后期穿过中心到半幅右偏”的
    source position 偏移。

    “左”方向来自标定阶段测得的屏幕右方向向量取反，
    因此会跟随当前相机视角，而不是写死世界坐标轴。
    """
    if base_bias_distance <= 1e-6:
        return (0.0, 0.0, 0.0), 0.0

    x_coeffs, z_coeffs = calibration
    screen_right_x = x_coeffs[0]
    screen_right_z = z_coeffs[0]
    screen_right_norm = math.sqrt(screen_right_x * screen_right_x + screen_right_z * screen_right_z)
    if screen_right_norm <= 1e-6:
        return (0.0, 0.0, 0.0), 0.0

    progress = 0.0
    if target_raw_angle > 1e-6:
        progress = max(0.0, min(1.0, current_raw_angle / target_raw_angle))

    # progress=0 时保留初始左偏；progress=1 时走到反方向一半偏置。
    start_bias_distance = float(base_bias_distance)
    end_bias_distance = -0.5 * float(base_bias_distance)
    bias_distance = start_bias_distance + (end_bias_distance - start_bias_distance) * progress

    left_unit_x = -screen_right_x / screen_right_norm
    left_unit_z = -screen_right_z / screen_right_norm
    return (
        left_unit_x * bias_distance,
        0.0,
        left_unit_z * bias_distance,
    ), bias_distance


def get_box_like_source_offset(
    source_item: Optional[Dict],
    calibration: Optional[Tuple[List[float], List[float]]] = None,
    bias_distance: float = BOX_SOURCE_RIGHT_BIAS_METERS,
) -> Tuple[Tuple[float, float, float], float]:
    """
    为 box-like 倾倒源构造固定偏置距离的 source position 偏移。

    使用相机标定得到的屏幕右方向与屏幕下方向，
    在 XZ 平面上构造“从屏幕右方向朝屏幕下方向偏 40°”的偏置方向。

    也就是整体仍然是相机视角下的右后方，只是更偏右一些，而不是 45° 对角线。
    """
    if bias_distance <= 1e-6:
        return (0.0, 0.0, 0.0), 0.0

    if calibration is None:
        raise ValueError("box_like_source_missing_calibration")

    x_coeffs, z_coeffs = calibration
    screen_right_x = x_coeffs[0]
    screen_right_z = z_coeffs[0]
    screen_back_x = x_coeffs[1]
    screen_back_z = z_coeffs[1]

    screen_right_norm = math.sqrt(screen_right_x * screen_right_x + screen_right_z * screen_right_z)
    screen_back_norm = math.sqrt(screen_back_x * screen_back_x + screen_back_z * screen_back_z)
    if screen_right_norm <= 1e-6 or screen_back_norm <= 1e-6:
        raise ValueError("box_like_source_invalid_calibration_axes")

    right_unit_x = screen_right_x / screen_right_norm
    right_unit_z = screen_right_z / screen_right_norm
    back_unit_x = screen_back_x / screen_back_norm
    back_unit_z = screen_back_z / screen_back_norm

    angle_rad = math.radians(float(BOX_SOURCE_CAMERA_BACK_ANGLE_DEGREES))
    dir_x = math.cos(angle_rad) * right_unit_x + math.sin(angle_rad) * back_unit_x
    dir_z = math.cos(angle_rad) * right_unit_z + math.sin(angle_rad) * back_unit_z
    dir_norm = math.sqrt(dir_x * dir_x + dir_z * dir_z)
    if dir_norm <= 1e-6:
        raise ValueError("box_like_source_invalid_camera_bias_direction")

    return (
        (dir_x / dir_norm) * bias_distance,
        0.0,
        (dir_z / dir_norm) * bias_distance,
    ), bias_distance


def log_box_like_alignment_reference(
    source_item: Optional[Dict],
    container: Dict,
    source_offset: Optional[Tuple[float, float, float]],
    *,
    prefix: str = "[box_align]",
) -> None:
    """
    打印 box-like 倾倒源的偏置参考点对齐信息，便于确认是否真的在用偏置点对齐。
    """
    if source_item is None:
        print(f"{prefix} source_item=missing")
        return

    raw_top = get_bottle_top_position(source_item)
    target_center = get_container_pour_center(container)
    off_x = source_offset[0] if source_offset is not None else 0.0
    off_z = source_offset[2] if source_offset is not None else 0.0
    effective_x = raw_top[0] + off_x
    effective_z = raw_top[2] + off_z
    print(
        f"{prefix} raw_source=({raw_top[0]:.3f}, {raw_top[2]:.3f}) "
        f"source_offset=({off_x:.3f}, {off_z:.3f}) "
        f"effective_source=({effective_x:.3f}, {effective_z:.3f}) "
        f"target_center=({target_center[0]:.3f}, {target_center[2]:.3f})"
    )


def get_source_target_offset_distance(
    source_item: Optional[Dict],
    container: Optional[Dict],
    source_offset: Optional[Tuple[float, float, float]] = None,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    计算当前倾倒源参考点与目标容器中心的水平偏差。

    Returns:
        (offset_distance, effective_x, effective_z, target_x, target_z, raw_y) 或 None
    """
    if source_item is None or container is None:
        return None

    bottle_top = get_bottle_top_position(source_item)
    target_center = get_container_pour_center(container)
    effective_x = bottle_top[0] + (source_offset[0] if source_offset is not None else 0.0)
    effective_z = bottle_top[2] + (source_offset[2] if source_offset is not None else 0.0)
    offset_x = effective_x - target_center[0]
    offset_z = effective_z - target_center[2]
    offset_distance = math.sqrt(offset_x * offset_x + offset_z * offset_z)
    return (
        offset_distance,
        effective_x,
        effective_z,
        target_center[0],
        target_center[2],
        bottle_top[1],
    )


def probe_best_alignment_direction(
    container: Dict,
    current_distance: float,
    target_override: Optional[Tuple[float, float, float]] = None,
    source_offset: Optional[Tuple[float, float, float]] = None,
    probe_pixels: int = CONTAINER_ALIGN_MOVE_PIXELS,
    verbose: bool = False,
) -> Optional[Tuple[int, int, float]]:
    """
    在四个基本方向上做一次小范围试探，找出“既能动，又能更接近目标”的方向。
    """
    data = read_realtime_products()
    if data is None:
        return None

    bottle = find_active_pour_source(data)
    if bottle is None:
        return None

    start_top = get_bottle_top_position(bottle)
    start_x, _, start_z = start_top
    source_offset_x = source_offset[0] if source_offset is not None else 0.0
    source_offset_z = source_offset[2] if source_offset is not None else 0.0
    target_x, _, target_z = target_override or get_container_pour_center(container)
    start_effective_x = start_x + source_offset_x
    start_effective_z = start_z + source_offset_z

    candidates = [
        (-probe_pixels, 0),
        (probe_pixels, 0),
        (0, -probe_pixels),
        (0, probe_pixels),
    ]

    best_move: Optional[Tuple[int, int, float]] = None
    best_distance = current_distance

    for move_x, move_y in candidates:
        pre_action_mtime = get_json_file_mtime()
        move_bottle_horizontal(move_x, move_y)
        wait_for_data_update(pre_action_mtime, timeout=1.0, verbose=False)
        moved_top = wait_for_source_pose_settle(pre_action_mtime, start_top, timeout=POSE_SETTLE_TIMEOUT)

        if moved_top is not None:
            moved_x, _, moved_z = moved_top
            moved_effective_x = moved_x + source_offset_x
            moved_effective_z = moved_z + source_offset_z
            moved_distance = math.sqrt((target_x - moved_effective_x) ** 2 + (target_z - moved_effective_z) ** 2)
            moved_amount = math.sqrt((moved_effective_x - start_effective_x) ** 2 + (moved_effective_z - start_effective_z) ** 2)
        else:
            moved_distance = float("inf")
            moved_amount = 0.0

        pre_action_mtime = get_json_file_mtime()
        move_bottle_horizontal(-move_x, -move_y)
        wait_for_data_update(pre_action_mtime, timeout=1.0, verbose=False)
        wait_for_source_pose_settle(pre_action_mtime, moved_top or start_top, timeout=POSE_SETTLE_TIMEOUT)

        if verbose:
            print(
                f"  [方向探测] move=({move_x},{move_y}) "
                f"delta={moved_amount:.4f}m dist={moved_distance:.4f}m"
            )

        improved_enough = moved_distance < (best_distance - PROBE_RECOVERY_MIN_IMPROVEMENT)
        moved_enough = moved_amount >= PROBE_RECOVERY_MIN_MOVEMENT
        if moved_enough and improved_enough:
            best_distance = moved_distance
            best_move = (move_x, move_y, moved_distance)

    return best_move


# ================================================================
# 闭环控制函数
# ================================================================

def align_bottle_to_container(
    container: Dict,
    verbose: bool = True,
    calibration: Optional[Tuple[List[float], List[float]]] = None,
    target_override: Optional[Tuple[float, float, float]] = None,
    source_offset: Optional[Tuple[float, float, float]] = None,
    max_iterations: Optional[int] = None,
    allow_direction_flip: bool = True,
    flip_only_on_regression: bool = False,
    use_probe_recovery: bool = False,
    fixed_move_pixels: Optional[int] = None,
    return_reason: bool = False,
) -> Union[bool, Tuple[bool, str]]:
    """
    闭环控制：将液体瓶口对准容器倾倒中心

    v1.4: 使用试探标定法，先测试鼠标与3D空间的映射关系，
    然后使用变换矩阵计算精确的鼠标移动量。

    Args:
        container: 容器信息（包含position/bounds）
        verbose: 是否输出详细日志

    Returns:
        bool: 是否成功对准
    """
    def _ret(success: bool, reason: str = "") -> Union[bool, Tuple[bool, str]]:
        return (success, reason) if return_reason else success

    # 获取容器倾倒中心位置
    target_x, target_y, target_z = target_override or get_container_pour_center(container)
    source_offset_x = source_offset[0] if source_offset is not None else 0.0
    source_offset_z = source_offset[2] if source_offset is not None else 0.0

    if verbose:
        print(f"[*] 倾倒目标: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")

    # ===== 步骤1: 标定鼠标-3D映射 =====
    if calibration is None:
        calibration = calibrate_mouse_to_3d_mapping(verbose=verbose)

    if calibration is None:
        print("[!] 标定失败，停止对准；已禁用默认映射回退")
        return False

    x_coeffs, z_coeffs = calibration

    # ===== 步骤2: 闭环对准 =====
    if verbose:
        print("[*] 开始闭环对准...")

    iteration_limit = int(max_iterations) if max_iterations is not None else MAX_ITERATIONS
    best_distance = float("inf")
    last_distance: Optional[float] = None
    last_x: Optional[float] = None
    last_z: Optional[float] = None
    stagnant_rounds = 0
    regression_rounds = 0
    distance_delta: Optional[float] = None
    direction_mode = 0
    direction_modes: list[tuple[int, int]] = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    step_scale = 1.0
    axis_direction_x = 1
    axis_direction_y = 1
    last_move_x = 0
    last_move_y = 0
    last_err_abs_x: Optional[float] = None
    last_err_abs_z: Optional[float] = None

    for iteration in range(iteration_limit):
        # 记录动作前的时间戳
        pre_action_mtime = get_json_file_mtime()

        # 读取当前数据
        data = read_realtime_products()
        if data is None:
            print("[!] 无法读取扫描数据")
            return _ret(False, "realtime_products_unavailable")

        # 获取液体瓶（倾倒模式或手持）
        bottle = find_active_pour_source(data)
        if bottle is None:
            print("[!] 未检测到液体瓶（倾倒模式或手持）")
            return _ret(False, "active_pour_source_not_found")

        # 获取瓶口位置
        bottle_top = get_bottle_top_position(bottle)
        cur_x, cur_y, cur_z = bottle_top
        effective_x = cur_x + source_offset_x
        effective_z = cur_z + source_offset_z

        # 计算位置误差（水平面）
        err_x = target_x - effective_x
        err_z = target_z - effective_z
        distance = math.sqrt(err_x * err_x + err_z * err_z)
        distance_delta = None if last_distance is None else (distance - last_distance)

        if (
            fixed_move_pixels is not None
            and allow_direction_flip
            and last_err_abs_x is not None
            and last_err_abs_z is not None
        ):
            x_worse = last_move_x != 0 and abs(err_x) > (last_err_abs_x + CONTAINER_AXIS_FLIP_EPSILON)
            z_worse = last_move_y != 0 and abs(err_z) > (last_err_abs_z + CONTAINER_AXIS_FLIP_EPSILON)

            if x_worse:
                axis_direction_x *= -1
                if verbose:
                    print(f"  [axis_flip] axis=x err_abs={abs(err_x):.4f}")

            if z_worse:
                axis_direction_y *= -1
                if verbose:
                    print(f"  [axis_flip] axis=z err_abs={abs(err_z):.4f}")

            if distance_delta is not None and distance_delta > 0.003 and not x_worse and not z_worse:
                if abs(err_x) >= abs(err_z):
                    axis_direction_x *= -1
                    if verbose:
                        print(f"  [axis_flip] axis=x fallback distance_delta={distance_delta:+.4f}m")
                else:
                    axis_direction_y *= -1
                    if verbose:
                        print(f"  [axis_flip] axis=z fallback distance_delta={distance_delta:+.4f}m")

        improved = distance < (best_distance - 0.01)
        if improved:
            best_distance = distance
        if last_distance is not None and last_x is not None and last_z is not None:
            pos_delta = math.sqrt((effective_x - last_x) ** 2 + (effective_z - last_z) ** 2)
            dist_delta = abs(distance - last_distance)
            got_worse = distance_delta is not None and distance_delta > 0.003
            dead_zone_stall = (
                fixed_move_pixels is not None
                and pos_delta < CONTAINER_DEAD_ZONE_POS_EPSILON
                and dist_delta < CONTAINER_DEAD_ZONE_DIST_EPSILON
            )

            if dead_zone_stall:
                stagnant_rounds += 1
                if verbose:
                    print(
                        f"  [dead_zone] pos_delta={pos_delta:.4f}m dist_delta={dist_delta:.4f}m "
                        f"round={stagnant_rounds}"
                    )

                if stagnant_rounds >= 2:
                    probe_pixels = max(1, int(fixed_move_pixels))
                    best_probe = probe_best_alignment_direction(
                        container,
                        current_distance=distance,
                        target_override=(target_x, target_y, target_z),
                        source_offset=source_offset,
                        probe_pixels=probe_pixels,
                        verbose=verbose,
                    )
                    if best_probe is None:
                        if verbose:
                            print("  [blocked] dead-zone probe found no reachable improving direction")
                        return _ret(False, "blocked/unreachable")

                    probe_move_x, probe_move_y, probe_distance = best_probe
                    if verbose:
                        print(
                            f"  [dead_zone_recover] move=({probe_move_x},{probe_move_y}) "
                            f"distance={probe_distance:.4f}m"
                        )
                    pre_probe_mtime = get_json_file_mtime()
                    move_bottle_horizontal(probe_move_x, probe_move_y)
                    wait_for_data_update(pre_probe_mtime, timeout=1.0, verbose=False)
                    wait_for_source_pose_settle(pre_probe_mtime, bottle_top, timeout=POSE_SETTLE_TIMEOUT)
                    stagnant_rounds = 0
                    regression_rounds = 0
                    last_distance = None
                    last_x = None
                    last_z = None
                    last_move_x = probe_move_x
                    last_move_y = probe_move_y
                    last_err_abs_x = None
                    last_err_abs_z = None
                    continue

            if not dead_zone_stall and pos_delta < 0.008 and dist_delta < 0.01:
                stagnant_rounds += 1
            elif improved:
                stagnant_rounds = 0
            else:
                stagnant_rounds = 0

            if got_worse:
                regression_rounds += 1
            elif improved:
                regression_rounds = 0

            should_flip = allow_direction_flip and (
                (flip_only_on_regression and regression_rounds >= 2)
                or ((not flip_only_on_regression) and stagnant_rounds >= 2)
            )

            should_probe_recover = use_probe_recovery and (got_worse or stagnant_rounds >= 2)
            if should_probe_recover:
                probe_pixels = fixed_move_pixels if fixed_move_pixels is not None else max(24, min(60, get_move_pixels(distance)))
                best_probe = probe_best_alignment_direction(
                    container,
                    current_distance=distance,
                    target_override=(target_x, target_y, target_z),
                    source_offset=source_offset,
                    probe_pixels=probe_pixels,
                    verbose=verbose,
                )
                if best_probe is not None:
                    probe_move_x, probe_move_y, probe_distance = best_probe
                    if verbose:
                        print(
                            f"  [probe_recover] move=({probe_move_x},{probe_move_y}) "
                            f"distance={probe_distance:.4f}m"
                        )
                    pre_probe_mtime = get_json_file_mtime()
                    move_bottle_horizontal(probe_move_x, probe_move_y)
                    wait_for_data_update(pre_probe_mtime, timeout=1.0, verbose=False)
                    wait_for_source_pose_settle(pre_probe_mtime, bottle_top, timeout=POSE_SETTLE_TIMEOUT)
                    stagnant_rounds = 0
                    regression_rounds = 0
                    last_distance = None
                    last_x = None
                    last_z = None
                    step_scale = 1.0
                    continue

                if got_worse:
                    direction_mode = (direction_mode + 1) % len(direction_modes)
                    step_scale = 1.0
                    if verbose:
                        sign_x, sign_y = direction_modes[direction_mode]
                        print(
                            f"  [flip_now] distance_delta={distance_delta:+.4f}m "
                            f"mode={direction_mode} sign=({sign_x},{sign_y})"
                        )
                    stagnant_rounds = 0
                    regression_rounds = 0
                    last_distance = distance
                    last_x = cur_x
                    last_z = cur_z
                    continue

            if got_worse and allow_direction_flip and not use_probe_recovery:
                direction_mode = (direction_mode + 1) % len(direction_modes)
                step_scale = 1.0
                if verbose:
                    sign_x, sign_y = direction_modes[direction_mode]
                    print(
                        f"  [flip_now] distance_delta={distance_delta:+.4f}m "
                        f"mode={direction_mode} sign=({sign_x},{sign_y})"
                    )
                stagnant_rounds = 0
                regression_rounds = 0
                last_distance = distance
                last_x = cur_x
                last_z = cur_z
                continue

            if should_flip:
                direction_mode = (direction_mode + 1) % len(direction_modes)
                step_scale = 1.0
                if verbose:
                    sign_x, sign_y = direction_modes[direction_mode]
                    print(
                        f"  [converge_adjust] regression={regression_rounds} stagnant={stagnant_rounds} "
                        f"mode={direction_mode} sign=({sign_x},{sign_y}) scale={step_scale:.2f}"
                    )
                stagnant_rounds = 0
                regression_rounds = 0
            elif stagnant_rounds >= 2:
                if allow_direction_flip:
                    if direction_mode == 0:
                        step_scale = max(0.35, step_scale * 0.7)
                    else:
                        step_scale = max(0.35, step_scale * 0.85)
                    if verbose:
                        sign_x, sign_y = direction_modes[direction_mode]
                        print(
                            f"  [收敛调整] stagnant={stagnant_rounds} "
                            f"mode={direction_mode} sign=({sign_x},{sign_y}) scale={step_scale:.2f}"
                        )
                else:
                    step_scale = max(0.25, step_scale * 0.7)
                    if verbose:
                        print(
                            f"  [收敛调整] stagnant={stagnant_rounds} "
                            f"keep_direction=True scale={step_scale:.2f}"
                        )
                stagnant_rounds = 0

        if verbose:
            if distance_delta is None:
                trend = "start"
                delta_str = "n/a"
            else:
                if distance_delta > 0.003:
                    trend = "farther"
                elif distance_delta < -0.003:
                    trend = "closer"
                else:
                    trend = "flat"
                delta_str = f"{distance_delta:+.4f}m"
            print(
                f"  [iter {iteration}] top=({effective_x:.4f}, {effective_z:.4f}), "
                f"err=({err_x:.4f}, {err_z:.4f}), dist={distance:.4f}m, "
                f"delta={delta_str}, trend={trend}"
            )

        # 检查是否到达目标
        if abs(err_x) < POSITION_TOLERANCE and abs(err_z) < POSITION_TOLERANCE:
            if verbose:
                print(f"  [对准完成] 迭代 {iteration} 次")
                print(f"    最终瓶口: ({effective_x:.3f}, {effective_z:.3f})")
                print(f"    目标位置: ({target_x:.3f}, {target_z:.3f})")
                print(f"    最终误差: {distance:.4f}m")
            return _ret(True, "")

        # 使用标定结果计算鼠标移动量
        raw_move_x, raw_move_y = compute_mouse_movement_vector(err_x, err_z, x_coeffs, z_coeffs)
        move_x = int(round(raw_move_x))
        move_y = int(round(raw_move_y))
        sign_x, sign_y = direction_modes[direction_mode]
        raw_move_x *= sign_x
        raw_move_y *= sign_y
        move_x *= sign_x
        move_y *= sign_y

        if fixed_move_pixels is not None:
            effective_fixed_pixels = max(1, int(round(fixed_move_pixels * max(0.25, step_scale))))
            raw_move_x *= axis_direction_x
            raw_move_y *= axis_direction_y

            raw_norm = math.sqrt(raw_move_x * raw_move_x + raw_move_y * raw_move_y)
            if raw_norm <= 1e-6:
                fallback_x = axis_direction_x if abs(err_x) >= POSITION_TOLERANCE else 0.0
                fallback_y = axis_direction_y if abs(err_z) >= POSITION_TOLERANCE else 0.0
                raw_norm = math.sqrt(fallback_x * fallback_x + fallback_y * fallback_y)
                raw_move_x = fallback_x
                raw_move_y = fallback_y

            if raw_norm <= 1e-6:
                move_x = 0
                move_y = 0
            else:
                scale = effective_fixed_pixels / raw_norm
                move_x = int(round(raw_move_x * scale))
                move_y = int(round(raw_move_y * scale))

                if move_x == 0 and abs(raw_move_x) > 1e-6:
                    move_x = 1 if raw_move_x > 0 else -1
                if move_y == 0 and abs(raw_move_y) > 1e-6:
                    move_y = 1 if raw_move_y > 0 else -1
        else:
            # 限制单次移动幅度（防止过冲）
            max_pixels = max(12, int(round(get_move_pixels(distance) * step_scale)))
            if abs(move_x) > max_pixels:
                move_x = max_pixels if move_x > 0 else -max_pixels
            if abs(move_y) > max_pixels:
                move_y = max_pixels if move_y > 0 else -max_pixels

            # 确保至少有最小移动量
            min_pixels = max(3, int(round(5 * step_scale)))
            if abs(move_x) < min_pixels and abs(err_x) >= POSITION_TOLERANCE:
                move_x = min_pixels if err_x > 0 else -min_pixels
            if abs(move_y) < min_pixels and abs(err_z) >= POSITION_TOLERANCE:
                move_y = min_pixels if err_z > 0 else -min_pixels

        if move_x != 0 or move_y != 0:
            if verbose:
                print(f"  [移动] dx={move_x}, dy={move_y} (标定计算)")
            move_bottle_horizontal(move_x, move_y)

            # 等待数据更新
            wait_for_data_update(pre_action_mtime, verbose=False)
            wait_for_source_pose_settle(pre_action_mtime, bottle_top, timeout=POSE_SETTLE_TIMEOUT)

        last_move_x = move_x
        last_move_y = move_y
        last_err_abs_x = abs(err_x)
        last_err_abs_z = abs(err_z)
        last_distance = distance
        last_x = effective_x
        last_z = effective_z

    print(f"[!] 达到最大迭代次数 {iteration_limit}，未能完成对准")
    return _ret(False, "max_iterations_reached")


def pour_with_monitoring(
    target_ml: float,
    tolerance: float = DEFAULT_POUR_TOLERANCE,
    container: Dict = None,
    verbose: bool = True,
    source_offset: Optional[Tuple[float, float, float]] = None,
    max_tilt_deg: Optional[float] = None,
) -> Tuple[bool, float, str]:
    """
    带监控的倾倒操作：倾斜液体瓶直到达到目标倾倒量

    v1.6 更新：
    - 使用全局累计量管理器统一跟踪倾倒量
    - 达标后直接返回，由调用者调用 exit_pouring_mode 退出
    - 最终精确值由 exit_pouring_mode 返回（等待液体停止后）

    v1.3 更新：
    - 修复：倾斜控制使用左键（不是右键）
    - 水平位置调整：直接移动鼠标（不需要按键）
    - 基于 is_pouring 的开关控制（液体流出时保持，未流出时继续倾斜）
    - 累计倾倒量处理（弹窗重置时累加之前的量）

    Args:
        target_ml: 目标倾倒量（ml）
        tolerance: 倾倒量容差（ml）
        container: 容器信息（用于边界检查）
        verbose: 是否输出详细日志

    Returns:
        tuple: (是否达到目标, 当前累计倾倒量)
        注意：最终精确值应从 exit_pouring_mode() 获取
    """
    if verbose:
        print(f"[*] 开始倾倒，目标: {target_ml} ml (容差: ±{tolerance} ml)")

    # 初始化监控
    init_pour_monitoring()
    time.sleep(0.2)

    # 使用全局累计量管理器
    initial_amount = get_current_pour_amount() or 0.0
    start_pour_tracking(initial_amount)

    if verbose:
        print(f"[*] 初始显示量: {initial_amount} ml")

    tilt_steps = 0
    max_iterations = MAX_TILT_STEPS * 3  # 总迭代上限（因为不是每次都倾斜）
    # Bottle-empty / no-flow guard:
    # Use poured amount delta (Alt+J/UDP) instead of only `is_pouring` boolean, which can be unreliable.
    last_total = float(get_pour_total() or 0.0)
    last_increase_ts = time.time()
    min_increase_ml = 0.2
    no_increase_timeout_s = 2.0
    empty_tilt_deg = float(max_tilt_deg) if max_tilt_deg is not None else 110.0

    for iteration in range(max_iterations):
        # 获取当前倾倒状态（从 JSON 读取 is_pouring 和 rotation）
        data = read_realtime_products()
        is_pouring_now = False
        bottle_tilt_angle = 0.0  # 瓶子倾斜角度
        rotation_x = 0.0
        rotation_z = 0.0

        if data is None:
            # JSON读取失败时，短暂等待后重试
            time.sleep(0.05)
            continue

        bottle = find_active_pour_source(data)
        if bottle:
            if not bool(bottle.get("is_pouring_mode", False)):
                final_total = float(get_pour_total() or 0.0)
                if verbose:
                    print_status("[倾倒模式丢失] 检测到 is_pouring_mode=False，立即停止并报错退出", end_line=True)
                return False, final_total, _pouring_mode_lost_reason(bottle)
            is_pouring_now = bottle.get("is_pouring", False)
            # 获取瓶子倾斜角度（rotation.x 和 rotation.z）
            rotation = bottle.get("rotation", {})
            if isinstance(rotation, dict):
                rotation_x = rotation.get("x", 0)
                rotation_z = rotation.get("z", 0)
            # 计算与垂直方向的夹角（同时检查x和z，取较大值）
            tilt_from_x = rotation_x if rotation_x <= 180 else 360 - rotation_x
            tilt_from_z = rotation_z if rotation_z <= 180 else 360 - rotation_z
            bottle_tilt_angle = max(tilt_from_x, tilt_from_z)
        else:
            # 没找到瓶子时仍然继续，但输出调试信息
            if verbose and iteration % 20 == 0:
                print(f"\n[DEBUG iter={iteration}] 未找到瓶子对象")

        # 获取当前显示的倾倒量并更新跟踪
        current_display = get_current_pour_amount()
        current_total = 0.0

        if current_display is not None:
            # 使用全局管理器更新跟踪（自动处理弹窗重置）
            current_total = update_pour_tracking(current_display)

            # 单行覆盖更新输出
            if verbose:
                print_pouring_status(current_total, target_ml, is_pouring_now)

            # 检查是否达到目标 - 达标后直接返回，由调用者决定退出
            if current_total >= target_ml - tolerance:
                if verbose:
                    print_status(f"[达到目标] 累计倾倒 {current_total:.1f} ml", end_line=True)
                return True, current_total, ""
        else:
            # If display is missing, still keep a best-effort total for empty/no-flow checks.
            current_total = float(get_pour_total() or 0.0)

        # 调试：每隔10次迭代输出瓶子状态
        if verbose and iteration % 10 == 0:
            print(f"\n[DEBUG iter={iteration}] rot.x={rotation_x:.1f}° rot.z={rotation_z:.1f}° tilt={bottle_tilt_angle:.1f}° pouring={is_pouring_now}")

        # Empty/no-flow detection: if tilt is high but poured amount doesn't increase for a while.
        now_ts = time.time()
        if current_total > last_total + float(min_increase_ml):
            last_total = float(current_total)
            last_increase_ts = now_ts

        if float(bottle_tilt_angle) >= float(empty_tilt_deg) and not bool(is_pouring_now):
            if (now_ts - last_increase_ts) >= float(no_increase_timeout_s):
                final_total = float(get_pour_total() or current_total or 0.0)
                if verbose:
                    if final_total < 0.5:
                        print_status(
                            f"[无液体流出] 倾斜{bottle_tilt_angle:.1f}° 且倾倒量长期不增加（累计 {final_total:.1f} ml）。"
                            "请确认：已对准容器进入倾倒模式、瓶口在容器上方、以及瓶子未空。",
                            end_line=True,
                        )
                    else:
                        print_status(
                            f"[瓶子已空] 倾斜{bottle_tilt_angle:.1f}° 且倾倒量不再增加，累计倾倒 {final_total:.1f} ml",
                            end_line=True,
                        )
                if final_total < 0.5:
                    msg = (
                        f"[无液体流出] 倾斜{bottle_tilt_angle:.1f}° 且倾倒量长期不增加（累计 {final_total:.1f} ml）。"
                        "请确认：已对准容器进入倾倒模式、瓶口在容器上方、以及瓶子未空。"
                    )
                    return False, final_total, f"no_liquid_flow:{msg}"
                msg = f"[瓶子已空] 倾斜{bottle_tilt_angle:.1f}° 且倾倒量不再增加，累计倾倒 {final_total:.1f} ml"
                return False, final_total, f"bottle_empty:{msg}"

        # 在继续增大倾角前，优先保证参考点与目标点已基本重合。
        if container and bottle and not is_pouring_now:
            offset_info = get_source_target_offset_distance(
                bottle,
                container,
                source_offset=source_offset,
            )
            if offset_info is not None:
                offset_distance, effective_x, effective_z, target_x, target_z, _ = offset_info
                if offset_distance > POSITION_TOLERANCE:
                    if verbose:
                        print_status(
                            "[预对齐] 参考点未重合："
                            f"偏离 {offset_distance*100:.1f}cm > {POSITION_TOLERANCE*100:.1f}cm，先回正再继续增倾",
                            end_line=True,
                        )
                        print(
                            f"  [pre_tilt_align] effective_source=({effective_x:.3f}, {effective_z:.3f}) "
                            f"target_center=({target_x:.3f}, {target_z:.3f})"
                        )
                        if source_offset is not None and _is_box_like_pour_source(bottle):
                            log_box_like_alignment_reference(
                                bottle,
                                container,
                                source_offset,
                                prefix="  [box_align.pre_tilt]",
                            )
                    align_bottle_to_container(
                        container,
                        verbose=False,
                        source_offset=source_offset,
                    )
                    time.sleep(TILT_CHECK_INTERVAL)
                    continue

        # 基于 is_pouring 的开关控制
        if is_pouring_now:
            # 液体正在流出，保持当前角度（不操作）
            time.sleep(TILT_CHECK_INTERVAL)
        else:
            # 液体未流出，继续增加倾斜
            if float(max_tilt_deg or 0.0) > 0.0 and float(bottle_tilt_angle) >= float(max_tilt_deg):
                time.sleep(TILT_CHECK_INTERVAL)
            elif tilt_steps < MAX_TILT_STEPS:
                tilt_bottle(TILT_PIXELS_PER_STEP)
                tilt_steps += 1
                time.sleep(TILT_CHECK_INTERVAL)
            else:
                # 已达最大倾斜，等待
                time.sleep(TILT_CHECK_INTERVAL)

        # 检查瓶口是否偏离目标位置，如果偏离超过5cm则回正
        if container and data:
            bottle = find_active_pour_source(data)
            if bottle:
                offset_info = get_source_target_offset_distance(
                    bottle,
                    container,
                    source_offset=source_offset,
                )
                if offset_info is not None:
                    offset_distance, _, _, _, _, _ = offset_info
                    # 如果偏离超过5cm，回正位置
                    if offset_distance > 0.05:
                        if verbose:
                            print_status(f"[位置修正] 瓶口偏离{offset_distance*100:.1f}cm，回正中...", end_line=True)
                            if source_offset is not None and _is_box_like_pour_source(bottle):
                                log_box_like_alignment_reference(
                                    bottle,
                                    container,
                                    source_offset,
                                    prefix="[box_align.recover]",
                                )
                        # 重新对准
                        align_bottle_to_container(
                            container,
                            verbose=False,
                            source_offset=source_offset,
                        )

    # 达到最大迭代次数
    final_total = get_pour_total()
    print_status(f"[!] 达到最大迭代次数，累计倾倒: {final_total:.1f} ml", end_line=True)
    return False, float(final_total or 0.0), "target_not_reached"


# ================================================================
# 主倾倒函数
# ================================================================


def pour_container_with_step_alignment(
    container: Dict,
    verbose: bool = True,
) -> Tuple[bool, float, str]:
    """
    Container-to-container pouring.

    The game uses a different control path here:
    align first, then press `S` once, and repeat
    until the source container rotation reaches the target threshold.
    """
    calibration = calibrate_mouse_to_3d_mapping(verbose=verbose)
    if calibration is None:
        return False, 0.0, "container_alignment_calibration_failed"

    data = read_realtime_products()
    source_item = find_active_pour_source(data)
    source_type = classify_active_pour_source(source_item)
    if source_item is None:
        return False, 0.0, "active_pour_source_not_found"
    if source_type != "container":
        return False, 0.0, f"active_pour_source_is_not_container:{source_type}"

    axis_name, current_raw_angle, target_raw_angle, raw_x, raw_y, raw_z = get_container_pour_target_state(source_item)
    pan_base_bias_distance = get_pan_like_base_bias_distance(source_item)
    if verbose:
        print(
            f"[*] container pour mode: calibrate once, reuse that mapping for all later moves, "
            f"feedback-drive fixed {CONTAINER_ALIGN_MOVE_PIXELS}px steps, "
            f"align before each short S tap "
            f"({CONTAINER_POUR_KEY_HOLD_SECONDS:.2f}s), "
            f"monitor axis={axis_name}, current_raw={current_raw_angle:.1f}, "
            f"target_raw={target_raw_angle:.1f} (raw_x={raw_x:.1f}, raw_y={raw_y:.1f}, raw_z={raw_z:.1f})"
        )

    reached, current_raw_angle = is_container_pour_target_reached(
        source_item,
        axis_name=axis_name,
        target_raw_angle=target_raw_angle,
    )
    if reached:
        return True, current_raw_angle, ""

    base_target_center = get_container_pour_center(container)

    pour_steps_taken = 0
    align_round_idx = 0
    did_pan_pretilt = False
    if _is_pan_like_container(source_item) and verbose:
        print(
            f"[*] pan-like source detected; align first, then apply one gentle pre-tilt "
            f"({CONTAINER_PAN_PRETILT_KEY_HOLD_SECONDS:.3f}s) [base_bias={pan_base_bias_distance:.3f}m]"
        )
    elif _is_pot_like_container(source_item) and verbose and pan_base_bias_distance > 0.0:
        print(
            f"[*] pot-like source detected; use pan-like left-bias strategy "
            f"[base_bias={pan_base_bias_distance:.3f}m, target_raw={target_raw_angle:.1f}]"
        )
    elif _is_plate_like_container(source_item) and verbose:
        print(
            f"[*] plate-like source detected; stop by rotation.z "
            f"[base_bias={pan_base_bias_distance:.3f}m, target_raw={target_raw_angle:.1f}]"
        )

    while pour_steps_taken < CONTAINER_POUR_MAX_STEPS:
        align_round_idx += 1
        data = read_realtime_products()
        source_item = find_active_pour_source(data)
        source_type = classify_active_pour_source(source_item)

        if source_item is None:
            return False, 0.0, "active_pour_source_not_found"
        if source_type != "container":
            return False, 0.0, f"active_pour_source_is_not_container:{source_type}"
        if not bool(source_item.get("is_pouring_mode", False)):
            return False, current_raw_angle, _pouring_mode_lost_reason(source_item)

        reached, current_raw_angle = is_container_pour_target_reached(
            source_item,
            axis_name=axis_name,
            target_raw_angle=target_raw_angle,
        )
        if reached:
            return True, current_raw_angle, ""

        source_alignment_offset, pan_bias_distance = get_pan_like_source_offset(
            calibration=calibration,
            current_raw_angle=current_raw_angle,
            target_raw_angle=target_raw_angle,
            base_bias_distance=pan_base_bias_distance,
        )

        bottle_top = get_bottle_top_position(source_item)
        effective_source_x = bottle_top[0] + source_alignment_offset[0]
        effective_source_z = bottle_top[2] + source_alignment_offset[2]
        offset_x = effective_source_x - base_target_center[0]
        offset_z = effective_source_z - base_target_center[2]
        offset_distance = math.sqrt(offset_x**2 + offset_z**2)

        aligned, align_reason = align_bottle_to_container(
            container,
            verbose=(verbose and align_round_idx == 1),
            calibration=calibration,
            target_override=base_target_center,
            source_offset=source_alignment_offset,
            max_iterations=MAX_ITERATIONS,
            allow_direction_flip=False,
            flip_only_on_regression=False,
            use_probe_recovery=False,
            fixed_move_pixels=CONTAINER_ALIGN_MOVE_PIXELS,
            return_reason=True,
        )

        data = read_realtime_products()
        source_item = find_active_pour_source(data)
        source_type = classify_active_pour_source(source_item)
        if source_item is None:
            return False, 0.0, "active_pour_source_not_found"
        if source_type != "container":
            return False, 0.0, f"active_pour_source_is_not_container:{source_type}"
        if not bool(source_item.get("is_pouring_mode", False)):
            return False, current_raw_angle, _pouring_mode_lost_reason(source_item)
        bottle_top = get_bottle_top_position(source_item)
        effective_source_x = bottle_top[0] + source_alignment_offset[0]
        effective_source_z = bottle_top[2] + source_alignment_offset[2]
        offset_x = effective_source_x - base_target_center[0]
        offset_z = effective_source_z - base_target_center[2]
        offset_distance = math.sqrt(offset_x**2 + offset_z**2)

        if verbose:
            pan_bias_suffix = ""
            if abs(pan_bias_distance) > 1e-6:
                pan_bias_suffix = (
                    f" bias={pan_bias_distance:.3f}m "
                    f"source=({bottle_top[0]:.3f}, {bottle_top[2]:.3f}) "
                    f"effective=({effective_source_x:.3f}, {effective_source_z:.3f}) "
                    f"target=({base_target_center[0]:.3f}, {base_target_center[2]:.3f})"
                )
            print(
                f"  [container_align] round={align_round_idx}/{CONTAINER_POUR_MAX_STEPS} "
                f"aligned={aligned} offset={offset_distance:.3f}m reason={align_reason or 'n/a'}"
                f"{pan_bias_suffix}"
            )

        # align_bottle_to_container 的成功判定本身就是轴向 POSITION_TOLERANCE。
        # 这里必须尊重 aligned 结果，不能再用欧氏距离重复否决，
        # 否则会出现“已对准但始终不按 S”，最终把上限全耗光。
        if not aligned:
            if align_reason == "blocked/unreachable":
                return False, current_raw_angle, "container_alignment_blocked_unreachable"
            continue

        if _is_pan_like_container(source_item) and not did_pan_pretilt:
            if verbose:
                print(
                    f"  [pan_pretilt] aligned_once=1 hold={CONTAINER_PAN_PRETILT_KEY_HOLD_SECONDS:.3f}s; "
                    f"re-align before first real pour step"
                )
            pre_action_mtime = get_json_file_mtime()
            press_container_pour_step(hold_seconds=CONTAINER_PAN_PRETILT_KEY_HOLD_SECONDS)
            wait_for_data_update(pre_action_mtime, timeout=1.0, verbose=False)
            did_pan_pretilt = True
            continue

        pre_action_mtime = get_json_file_mtime()
        press_container_pour_step()
        pour_steps_taken += 1
        wait_for_data_update(pre_action_mtime, timeout=1.0, verbose=False)

        if verbose:
            data = read_realtime_products()
            source_item = find_active_pour_source(data)
            reached, current_raw_angle = is_container_pour_target_reached(
                source_item,
                axis_name=axis_name,
                target_raw_angle=target_raw_angle,
            )
            print(
                f"  [container_pour] step={pour_steps_taken}/{CONTAINER_POUR_MAX_STEPS} "
                f"axis={axis_name} raw={current_raw_angle:.1f}/{target_raw_angle:.1f}"
                f"{' pretilt=1' if did_pan_pretilt and pour_steps_taken == 1 else ''}"
            )

    data = read_realtime_products()
    final_source = find_active_pour_source(data)
    _, final_raw_angle = is_container_pour_target_reached(
        final_source,
        axis_name=axis_name,
        target_raw_angle=target_raw_angle,
    )
    return False, final_raw_angle, f"container_rotation_not_reached:{final_raw_angle:.1f}"

def auto_pour(
    container_name: str,
    container_instance_id: Optional[int] = None,
    target_ml: Optional[float] = DEFAULT_POUR_AMOUNT,
    tolerance: float = DEFAULT_POUR_TOLERANCE,
    skip_alignment: bool = False,
) -> Dict[str, Any]:
    """
    自动倾倒指定量的液体到容器中

    前提：已手持液体瓶并进入倾倒模式

    Args:
        container_name: 目标容器名称
        target_ml: 目标倾倒量（ml）
        tolerance: 倾倒量容差（ml）
        skip_alignment: 是否跳过位置对准（如果已经对准）

    Returns:
        dict: result payload, including final poured amount.
    """

    resolved_target_ml = DEFAULT_POUR_AMOUNT if target_ml in (None, "") else float(target_ml)

    def _fail(msg: str, *, poured_ml: float = 0.0) -> Dict[str, Any]:
        return {
            "success": False,
            "error": str(msg),
            "container_name": container_name,
            "target_ml": float(resolved_target_ml),
            "tolerance": float(tolerance),
            "poured_ml": float(poured_ml),
        }

    try:
        from epm.cerebellum.local_actions import _activate_window, click_mouse
    except ImportError as e:
        print(f"[!] 无法导入 local_actions: {e}")
        return _fail(f"import_local_actions_failed:{e}")

    print("=" * 50)
    print("  自动倾倒模块 v1.4")
    print("=" * 50)

    # 读取扫描数据
    data = read_realtime_products()
    if data is None:
        print("[!] 无法读取扫描数据，请确保 F12 扫描已开启")
        return _fail("realtime_products_unavailable (press F12 in-game)")

    # 查找容器
    container = find_container_by_name(container_name, data, instance_id=container_instance_id)
    if container is None:
        if container_instance_id is not None:
            inst = None
            try:
                inst = int(container_instance_id)
            except Exception:
                inst = None
            if inst is not None:
                hit = _any_product_with_instance_id(data, inst)
                if hit is None:
                    return _fail(f"container_instance_id_not_found: instance_id={inst} name={container_name!r}")
                return _fail(
                    f"container_instance_id_name_mismatch: instance_id={inst} name={container_name!r} hit_name_en={hit.get('name_en')!r} hit_name_cn={hit.get('name_cn')!r}"
                )
        print(f"[!] 未找到容器: {container_name}")
        print("[*] 屏幕上可见的物品:")
        for p in data.get("products", [])[:10]:
            if p.get("is_on_screen") and not p.get("is_held") and not p.get("is_pouring_mode"):
                print(f"    - {p.get('name_en')} ({p.get('name_cn')})")
        return _fail(f"container_not_found:{container_name}")

    container_pos = get_position(container)
    print(f"\n[*] 目标容器: {container.get('name_en')} ({container.get('name_cn')})")
    print(f"[*] 容器位置: ({container_pos[0]:.3f}, {container_pos[1]:.3f}, {container_pos[2]:.3f})")

    bounds = get_bounds(container)
    if bounds:
        (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
        print(f"[*] 容器尺寸: x={max_x-min_x:.3f}m, y={max_y-min_y:.3f}m, z={max_z-min_z:.3f}m")

    # 查找当前真实的倾倒源：可能是 liquid，也可能是 container
    bottle = find_active_pour_source(data)
    if bottle is None:
        data_ready, bottle_ready, ready_reason = wait_for_pour_source_ready(
            require_pouring_mode=True,
            timeout_s=4.5,
            stable_s=2.0,
            poll_s=0.05,
        )
        if bottle_ready is None:
            print("[!] 未检测到稳定的倾倒源物体")
            print("[*] 请确保:")
            print("    1. 已手持可倾倒物体")
            print("    2. 已进入倾倒模式，或至少保持手持状态")
            return _fail(
                "pour_source_not_found "
                "(waited 2.0s for a stable pouring-mode source, hold a pourable item and/or enter pouring mode)"
            )
        if ready_reason:
            return _fail(
                "enter_pouring_mode_failed: not in pouring mode after waiting 2.0s for a stable source. "
                "Please aim at the container and right-click to enter pouring mode."
            )
        data = data_ready or data
        bottle = bottle_ready

    source_type = classify_active_pour_source(bottle)

    # 显示当前倾倒源状态
    is_pouring_mode = bottle.get("is_pouring_mode", False)
    is_pouring = bottle.get("is_pouring", False)

    print(f"[*] 倾倒源: {bottle.get('name_en')} ({bottle.get('name_cn')})")
    print(f"[*] 倾倒源类型: {source_type}")
    supported_source, unsupported_reason = _is_supported_pour_source(bottle)
    if not supported_source:
        exit_res: Dict[str, Any] = {}
        if bool(is_pouring_mode):
            print("[!] 当前手持物体不允许 auto_pour：将先右键退出倾倒模式...")
            try:
                exit_res = exit_pouring_mode()
            except Exception as e:
                exit_res = {"success": False, "error": f"exit_pouring_mode_failed:{e}"}
        held_name = str(bottle.get("name_en") or bottle.get("name_cn") or bottle.get("name") or "").strip()
        precheck_reason = "not_in_pouring_mode" if not bool(is_pouring_mode) else "accidental_pouring_mode"
        error_msg = (
            f"unsupported_pour_source:{precheck_reason}:{unsupported_reason}:held_item={held_name!r}. "
            "Detected pouring mode, but the held item is not a liquid bottle or pourable container. "
            "This is likely an accidental mode entry caused by aiming the held item at another container; "
            "the action auto-exited pouring mode."
        )
        return {
            "success": False,
            "error": error_msg,
            "container_name": container_name,
            "target_ml": float(resolved_target_ml),
            "tolerance": float(tolerance),
            "poured_ml": 0.0,
            "feedback_code": "unsupported_pour_source",
            "pour_status": "unsupported_source",
            "feedback_to_planner": (
                "Accidentally entered pouring mode. This usually happens because the held item was aimed at a "
                "container and the mode was triggered by mistake. Auto-exited pouring mode; only liquid bottles "
                f"and pourable containers are allowed. Current held item{(' ' + repr(held_name)) if held_name else ''} is not supported."
            ),
            "mode": "pouring_mode",
            "mode_entered": False if not bool(is_pouring_mode) else None,
            "exit_mode_attempted": bool(is_pouring_mode),
            "exit_mode_result": exit_res,
            "held_item_name": held_name,
            "held_item_kind": str(bottle.get("kind") or "").strip(),
            "source_type": source_type,
        }
    if is_pouring_mode:
        print(f"[*] 状态: 倾倒模式" + (" (正在流出)" if is_pouring else " (待命)"))
        data_ready, bottle_ready, ready_reason = wait_for_pour_source_ready(
            require_pouring_mode=True,
            timeout_s=4.5,
            stable_s=2.0,
            poll_s=0.05,
        )
        if ready_reason:
            return _fail(
                "enter_pouring_mode_failed: not in pouring mode after waiting 2.0s for a stable source. "
                "Please aim at the container and right-click to enter pouring mode."
            )
        data = data_ready or data
        bottle = bottle_ready or bottle
        source_type = classify_active_pour_source(bottle)
        is_pouring_mode = bool(bottle.get("is_pouring_mode", False))
        is_pouring = bool(bottle.get("is_pouring", False))
    else:
        print(f"[*] 状态: 手持中")
    print(f"[*] 目标倾倒量: {resolved_target_ml} ml")

    # 激活游戏窗口
    try:
        _activate_window(GAME_WINDOW_TITLE)
    except Exception as e:
        print(f"[!] 激活窗口失败: {e}")
        return _fail(f"activate_window_failed:{e}")

    time.sleep(0.3)

    liquid_alignment_calibration: Optional[Tuple[List[float], List[float]]] = None
    liquid_source_offset: Optional[Tuple[float, float, float]] = None
    liquid_max_tilt_deg: Optional[float] = None
    if source_type == "liquid" and _is_box_like_pour_source(bottle):
        liquid_alignment_calibration = calibrate_mouse_to_3d_mapping(verbose=False)
        if liquid_alignment_calibration is not None:
            try:
                liquid_source_offset, box_bias_distance = get_box_like_source_offset(
                    bottle,
                    calibration=liquid_alignment_calibration,
                    bias_distance=BOX_SOURCE_RIGHT_BIAS_METERS,
                )
            except ValueError as e:
                print(f"[!] {e}")
                return _fail(str(e))
            liquid_max_tilt_deg = BOX_SOURCE_MAX_TILT_DEGREES
            print(
                f"[*] box-like source reference bias enabled: camera_bias_offset={box_bias_distance:.3f}m, "
                f"camera_right_to_back_angle={BOX_SOURCE_CAMERA_BACK_ANGLE_DEGREES:.1f}°, "
                f"offset_xz=({liquid_source_offset[0]:.3f}, {liquid_source_offset[2]:.3f}), "
                f"max_tilt={liquid_max_tilt_deg:.1f}°"
            )
        else:
            print("[!] box-like source bias skipped: calibration_failed")

    # auto_pour no longer enters pouring mode on its own.
    if bool(bottle.get("is_held", False)) and not bool(is_pouring_mode):
        return _fail(
            "auto_pour_requires_pouring_mode: not in pouring mode. "
            "Call action:enter_pouring_mode first, then retry auto_pour."
        )

    # ===== 步骤1: 位置对准 =====
    if source_type == "container":
        print(f"\n[1] 容器倾倒使用连续对准逻辑：对准成功后才按一次 S")
    elif source_type == "liquid" and _is_box_like_pour_source(bottle) and liquid_source_offset is not None:
        print(f"\n[1] Box-like 偏置参考点对准...")
        log_box_like_alignment_reference(
            bottle,
            container,
            liquid_source_offset,
            prefix="  [box_align.pre]",
        )
        if not align_bottle_to_container(
            container,
            verbose=True,
            calibration=liquid_alignment_calibration,
            source_offset=liquid_source_offset,
        ):
            print("[!] Box-like 偏置参考点对准失败")
    elif not skip_alignment:
        print(f"\n[1] 对准容器中心...")
        if not align_bottle_to_container(
            container,
            verbose=True,
            calibration=liquid_alignment_calibration,
            source_offset=liquid_source_offset,
        ):
            print("[!] 位置对准失败")
            # 继续尝试倾倒
    else:
        print(f"\n[1] 跳过位置对准")

    # ===== 步骤2: 倾倒 =====
    print(f"\n[2] 开始倾倒...")
    if source_type == "container":
        reached, _, reason = pour_container_with_step_alignment(
            container=container,
            verbose=True,
        )
    elif source_type == "liquid":
        reached, _, reason = pour_with_monitoring(
            target_ml=resolved_target_ml,
            tolerance=tolerance,
            container=container,
            verbose=True,
            source_offset=liquid_source_offset,
            max_tilt_deg=liquid_max_tilt_deg,
        )
    else:
        reached, _, reason = False, 0.0, f"unsupported_pour_source_type:{source_type}"

    # ===== 步骤3: 退出倾倒模式 =====
    print(f"\n[3] 退出倾倒模式...")
    # exit_pouring_mode 返回最终精确的累计倾倒量
    final_amount = exit_pouring_mode()

    if reached:
        print(f"\n[+] 倾倒完成! 累计倾倒: {final_amount:.1f} ml")
    else:
        print(f"\n[!] 倾倒未完成，累计倾倒: {final_amount:.1f} ml")

    print("=" * 50)
    return {
        "success": bool(reached),
        "error": "" if reached else (str(reason) or "target_not_reached"),
        "container_name": container_name,
        "source_type": source_type,
        "target_ml": float(resolved_target_ml),
        "tolerance": float(tolerance),
        "poured_ml": float(final_amount),
    }


# ================================================================
# 调试和测试函数
# ================================================================

def list_visible_items() -> List[Dict]:
    """列出当前屏幕上可见的所有物品"""
    data = read_realtime_products()
    if data is None:
        print("[!] 无法读取扫描数据")
        return []

    visible_items = []
    for product in data.get("products", []):
        if product.get("is_on_screen", False):
            visible_items.append(product)

    visible_items.sort(key=lambda x: x.get("distance", 999))

    print(f"\n屏幕上可见的物品 ({len(visible_items)} 个):")
    print("-" * 70)
    for i, item in enumerate(visible_items[:20], 1):
        pos = item.get('position', {})
        # 状态标记
        status_marks = []
        if item.get('is_held'):
            status_marks.append("[手持]")
        if item.get('is_pouring_mode'):
            if item.get('is_pouring'):
                status_marks.append("[倾倒中]")
            else:
                status_marks.append("[倾倒模式]")
        if item.get('bounds_min'):
            status_marks.append("[有bounds]")
        status_str = " ".join(status_marks)
        print(f"{i}. {item.get('name_en')} ({item.get('name_cn')}) {status_str}")
        print(f"   位置: ({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f})")
    print("-" * 70)

    return visible_items


def show_container_info(container_name: str):
    """显示容器详细信息"""
    print(f"\n=== 容器信息: {container_name} ===\n")

    data = read_realtime_products()
    if data is None:
        print("[!] 无法读取扫描数据")
        return

    container = find_container_by_name(container_name, data, must_be_on_screen=False, instance_id=container_instance_id)
    if container:
        pos = container.get("position", {})
        print(f"容器: {container.get('name_en')} ({container.get('name_cn')})")
        print(f"  位置: x={pos.get('x', 0):.3f}, y={pos.get('y', 0):.3f}, z={pos.get('z', 0):.3f}")
        print(f"  距离: {container.get('distance', 0):.2f}m")

        bounds = get_bounds(container)
        if bounds:
            (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
            print(f"  bounds_min: ({min_x:.3f}, {min_y:.3f}, {min_z:.3f})")
            print(f"  bounds_max: ({max_x:.3f}, {max_y:.3f}, {max_z:.3f})")
            print(f"  尺寸: x={max_x-min_x:.3f}m, y={max_y-min_y:.3f}m, z={max_z-min_z:.3f}m")
        else:
            print("  [无bounds信息]")
    else:
        print(f"[!] 未找到容器: {container_name}")

    # 显示液体瓶信息（手持或倾倒模式）
    bottle = find_active_pour_source(data)
    if bottle:
        is_pouring_mode = bottle.get('is_pouring_mode', False)
        is_pouring = bottle.get('is_pouring', False)
        status = "倾倒中" if is_pouring else ("倾倒模式" if is_pouring_mode else "手持中")
        print(f"\n液体瓶: {bottle.get('name_en')} ({bottle.get('name_cn')}) [{status}]")
        bottle_top = get_bottle_top_position(bottle)
        print(f"  瓶口位置: ({bottle_top[0]:.3f}, {bottle_top[1]:.3f}, {bottle_top[2]:.3f})")

        if container:
            pour_center = get_container_pour_center(container)
            dx = pour_center[0] - bottle_top[0]
            dz = pour_center[2] - bottle_top[2]
            print(f"\n与倾倒目标的差值:")
            print(f"  dx = {dx:.3f}m (正=右, 负=左)")
            print(f"  dz = {dz:.3f}m (正=前, 负=后)")
            print(f"  水平距离 = {math.sqrt(dx*dx + dz*dz):.3f}m")

            # 检查是否在有效区域内
            in_bounds = is_within_bounds(bottle_top[0], bottle_top[2], container)
            print(f"  瓶口在有效区域内: {'是' if in_bounds else '否'}")


def test_pour_monitoring():
    """测试倾倒量监控"""
    print("\n=== 测试倾倒量监控 ===\n")
    print("请确保已按 Alt+J 开启交互检测")
    print("按 Ctrl+C 停止\n")

    init_pour_monitoring()
    try:
        while True:
            amount = get_current_pour_amount()
            if amount is not None:
                print(f"当前倾倒量: {amount:.1f} ml")
            else:
                print("未检测到倾倒量")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n停止监控")
    finally:
        stop_pour_monitoring()


# ================================================================
# 主程序入口
# ================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  Cooking Simulator 自动倾倒模块 v1.4")
    print("=" * 50)
    print()
    print("可用命令:")
    print("  python auto_pouring.py list                       - 列出可见物品")
    print("  python auto_pouring.py info <容器名>              - 显示容器信息")
    print("  python auto_pouring.py monitor                    - 测试倾倒量监控")
    print("  python auto_pouring.py pour <容器名> <目标ml>     - 倾倒液体")
    print()
    print("示例:")
    print("  python auto_pouring.py info \"Paella Pan\"")
    print("  python auto_pouring.py pour \"Paella Pan\" 50")
    print()

    if len(sys.argv) < 2:
        print("[*] 默认执行: 列出可见物品")
        list_visible_items()
    else:
        cmd = sys.argv[1].lower()

        if cmd == "list":
            list_visible_items()

        elif cmd == "info":
            if len(sys.argv) >= 3:
                show_container_info(sys.argv[2])
            else:
                print("[!] 用法: python auto_pouring.py info <容器名>")

        elif cmd == "monitor":
            test_pour_monitoring()

        elif cmd == "pour":
            if len(sys.argv) >= 4:
                container_name = sys.argv[2]
                try:
                    target_ml = float(sys.argv[3])
                except ValueError:
                    print(f"[!] 无效的倾倒量: {sys.argv[3]}")
                    sys.exit(1)
                auto_pour(container_name=container_name, target_ml=target_ml)
            elif len(sys.argv) >= 3:
                container_name = sys.argv[2]
                auto_pour(container_name=container_name, target_ml=DEFAULT_POUR_AMOUNT)
            else:
                print("[!] 用法: python auto_pouring.py pour <容器名> <目标ml>")

        else:
            print(f"[!] 未知命令: {cmd}")

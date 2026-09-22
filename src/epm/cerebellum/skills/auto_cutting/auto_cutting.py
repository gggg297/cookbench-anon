# auto_cutting.py
# ================================================================
# Cooking Simulator Auto-Cutting Module v3.6
# ================================================================
#
# 功能：
#   1. 基于3D位置和闭环反馈控制自动切割
#   2. 使用is_held字段识别手持刀具
#   3. 使用is_cutting字段识别切割模式的刀（v3.2新增）
#   4. 使用bounds字段计算切割位置
#   5. 闭环控制确保刀具精确到位
#   6. 切割位置包含目标旋转角度，支持刀具旋转控制
#
# 使用方法：
#   python auto_cutting.py cut <物品名称> <切割次数>
#   例如: python auto_cutting.py cut lemon 3
#
# 依赖：
#   - C# Mod: CS_CamDump (v4.30+) 提供is_held/is_cutting/bounds/rotation字段
#   - local_actions.py 提供基础操作函数
#
# ================================================================

import os
import sys
import json
import time
import math
from typing import Optional, Dict, List, Tuple

# 从本包 config 导入配置（EPM-adapted）
from .config import (
    USERDATA_PATH,
    REALTIME_PRODUCTS_JSON,
    GAME_WINDOW_TITLE,
    DEFAULT_CUT_COUNT,
    DEFAULT_CUT_INTERVAL
)


# ================================================================
# 闭环控制参数
# ================================================================
POSITION_TOLERANCE = 0.01      # 位置容差 1cm
ROTATION_TOLERANCE = 2.0       # 角度容差 2度
MAX_ITERATIONS = 100           # 最大迭代次数
MAX_TARGET_MISSING_STREAK = 3  # 目标连续丢失多少轮后立即终止
MAX_SCOPE_STALL_STREAK = 4     # 连续多少轮“几乎不动且无进展”后判定超出当前切割作用域

# 离散动作参数（类似导航的离散化控制）
ACTION_DELAY = 0.05            # 每个动作后的等待时间
JSON_UPDATE_DELAY = 0.2        # 等待JSON文件更新的时间（旧参数，保留兼容）
MOVE_SETTLE_DELAY = 1.50       # 移动后额外等待物理/扫描稳定，避免读到滞后位置

# v3.5: 动态旋转步长（根据误差大小调整）
ROTATION_LARGE_THRESHOLD = 30.0   # 大误差阈值（度）
ROTATION_MEDIUM_THRESHOLD = 10.0  # 中误差阈值（度）
ROTATION_LARGE_DURATION = 0.30    # 大误差时的按键时长（秒）
ROTATION_MEDIUM_DURATION = 0.10   # 中误差时的按键时长（秒）
ROTATION_SMALL_DURATION = 0.05    # 小误差时的按键时长（秒）

# v3.5: 动态移动步长（根据位置误差大小调整）
MOVE_LARGE_THRESHOLD = 0.10       # 大误差阈值（米）
MOVE_MEDIUM_THRESHOLD = 0.05      # 中误差阈值（米）
MOVE_LARGE_PIXELS = 300           # 大误差时的像素数
MOVE_MEDIUM_PIXELS = 200          # 中误差时的像素数
MOVE_SMALL_PIXELS = 50            # 小误差时的像素数
MOVE_PROBE_PIXELS = 20            # 无可用标定时的保守探测步长
SCOPE_STALL_DISTANCE_THRESHOLD_M = POSITION_TOLERANCE  # 与切割到位判定保持一致，避免中间灰区
SCOPE_STALL_BLADE_MOVE_THRESHOLD_M = 0.003
SCOPE_STALL_PROGRESS_THRESHOLD_M = 0.002

# v3.7: 鼠标水平移动与世界坐标(x/z)的动态标定（参考 auto_pouring 的 2x2 映射）
CALIBRATION_PIXELS = 120
RECALIBRATE_AFTER_ROTATION_DEG = 8.0
CALIBRATION_ANOMALY_MIN_EXPECTED = 0.004
CALIBRATION_ANOMALY_MIN_ACTUAL = 0.0015
CALIBRATION_ANOMALY_COS_THRESHOLD = 0.15
CALIBRATION_ANOMALY_STREAK = 2   # 连续两次异常再重标定，优先复用上一刀标定

# 切割安全边距：只在物体长轴的中间 15%~85% 区间内落刀，避免外侧切空。
CUT_AXIS_START_RATIO = 0.15
CUT_AXIS_END_RATIO = 0.85

# 刀锋偏移参数（刀的有效切割点不在几何中心，而在刀锋1/4处）
KNIFE_BLADE_OFFSET_RATIO = 0.4   # 刀锋中点相对于刀长的偏移比例
KNIFE_DEFAULT_LENGTH = 0.8       # 默认刀长（米），用于没有bounds时

# v3.5: 基于时间戳的数据更新等待参数
DATA_UPDATE_TIMEOUT = 5.0      # 等待数据更新的最大时间（秒）
DATA_CHECK_INTERVAL = 0.05     # 检查数据更新的间隔（秒）

# v3.6: 物理稳定等待参数
PHYSICS_STABLE_THRESHOLD = 1.0   # 角度变化小于此值认为稳定（度）
PHYSICS_STABLE_CHECKS = 2        # 需要连续稳定的次数
PHYSICS_CHECK_INTERVAL = 0.08    # 每次检查的间隔（秒）
PHYSICS_STABLE_TIMEOUT = 1.0     # 等待稳定的最大时间（秒）


# ================================================================
# JSON文件时间戳函数 (v3.5 新增)
# ================================================================

def get_json_file_mtime() -> float:
    """
    获取JSON文件的修改时间戳

    Returns:
        float: 文件修改时间戳（秒），文件不存在返回 0.0
    """
    try:
        if os.path.exists(REALTIME_PRODUCTS_JSON):
            return os.path.getmtime(REALTIME_PRODUCTS_JSON)
        return 0.0
    except Exception:
        return 0.0


def wait_for_data_update(old_mtime: float, timeout: float = DATA_UPDATE_TIMEOUT,
                          verbose: bool = False) -> bool:
    """
    等待JSON文件更新（时间戳变化）

    v3.5: 方案A+C实现 - 监控文件时间戳，等待数据更新后再读取

    Args:
        old_mtime: 动作执行前的文件时间戳
        timeout: 最大等待时间（秒）
        verbose: 是否输出详细日志

    Returns:
        bool: True 表示检测到更新，False 表示超时
    """
    start_time = time.time()
    check_count = 0

    while time.time() - start_time < timeout:
        new_mtime = get_json_file_mtime()
        check_count += 1

        if new_mtime > old_mtime:
            if verbose:
                elapsed = time.time() - start_time
                print(f"  [数据更新] 检测到更新，耗时 {elapsed:.2f}s ({check_count}次检查)")
            return True

        time.sleep(DATA_CHECK_INTERVAL)

    if verbose:
        print(f"  [数据更新] 超时 ({timeout}s)，未检测到更新")
    return False


def wait_for_physics_stable(cutting_target: Dict = None, verbose: bool = False) -> Tuple[Optional[Dict], float]:
    """
    等待刀的物理状态稳定（v3.6 新增）

    通过连续读取角度值，确认角度变化小于阈值后认为稳定。
    解决游戏物理惯性导致的数据滞后问题。

    Args:
        cutting_target: 被切割的目标物品（用于定位刀）
        verbose: 是否输出详细日志

    Returns:
        tuple: (稳定后的刀数据, 稳定时的角度)
               如果超时未稳定，返回最后读取的数据
    """
    start_time = time.time()
    prev_angle = None
    stable_count = 0
    last_knife = None
    last_data = None

    while time.time() - start_time < PHYSICS_STABLE_TIMEOUT:
        # 读取当前数据
        data = read_realtime_products()
        if data is None:
            time.sleep(PHYSICS_CHECK_INTERVAL)
            continue

        # 查找刀
        knife = find_cutting_knife(data, cutting_target)
        if knife is None:
            time.sleep(PHYSICS_CHECK_INTERVAL)
            continue

        last_knife = knife
        last_data = data
        cur_angle = get_rotation_y(knife)

        if cur_angle is None:
            time.sleep(PHYSICS_CHECK_INTERVAL)
            continue

        # 检查角度是否稳定
        if prev_angle is not None:
            angle_change = abs(normalize_angle(cur_angle - prev_angle))
            if angle_change < PHYSICS_STABLE_THRESHOLD:
                stable_count += 1
                if verbose:
                    print(f"  [物理稳定] 角度={cur_angle:.1f}°, 变化={angle_change:.2f}°, 稳定次数={stable_count}")
                if stable_count >= PHYSICS_STABLE_CHECKS:
                    if verbose:
                        elapsed = time.time() - start_time
                        print(f"  [物理稳定] 确认稳定，耗时 {elapsed:.2f}s")
                    return last_data, cur_angle
            else:
                stable_count = 0  # 重置稳定计数
                if verbose:
                    print(f"  [物理稳定] 角度={cur_angle:.1f}°, 变化={angle_change:.2f}° (未稳定)")

        prev_angle = cur_angle
        time.sleep(PHYSICS_CHECK_INTERVAL)

    if verbose:
        print(f"  [物理稳定] 超时 ({PHYSICS_STABLE_TIMEOUT}s)，使用当前数据")

    return last_data, prev_angle if prev_angle else 0.0


# ================================================================
# 数据读取函数
# ================================================================

def read_realtime_products() -> Optional[Dict]:
    """
    读取实时物品扫描数据

    Returns:
        dict: 扫描数据，包含 products 列表
        None: 读取失败
    """
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
# 物品查找函数 (v3.0 使用 is_held 字段)
# ================================================================

def find_held_item(data: Dict = None, instance_id: Optional[int] = None) -> Optional[Dict]:
    """
    查找当前手持的物品（使用 is_held 字段）

    Returns:
        dict: 手持物品信息
        None: 未手持任何物品
    """
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


def find_held_knife(data: Dict = None) -> Optional[Dict]:
    """
    查找手持的刀（使用 is_held 字段，兼容旧逻辑）

    Returns:
        dict: 刀的信息
        None: 未找到或手持的不是刀
    """
    held_item = find_held_item(data)
    if held_item is None:
        return None

    # 检查是否是刀
    name_en = held_item.get("name_en", "").lower()
    if "knife" in name_en:
        return held_item

    return None


def find_knife_with_status(data: Dict = None) -> tuple:
    """
    查找刀并返回状态（优先 is_cutting，其次 is_held）

    Returns:
        tuple: (knife_dict, status)
            - knife_dict: 刀的信息，None表示未找到可用的刀
            - status: 状态字符串
                - "cutting": 刀在切割模式
                - "held": 刀在手中但未进入切割模式
                - "not_found": 未找到刀
    """
    if data is None:
        data = read_realtime_products()

    if data is None or "products" not in data:
        return None, "not_found"

    cutting_knife = None
    held_knife = None

    for product in data["products"]:
        name_en = product.get("name_en", "").lower()
        if "knife" not in name_en:
            continue

        # 优先检查切割模式
        if product.get("is_cut_mode", product.get("is_cutting", False)):
            cutting_knife = product
            break  # 找到切割模式的刀，直接返回

        # 记录手持的刀（备选）
        if product.get("is_held", False) and held_knife is None:
            held_knife = product

    # 返回结果
    if cutting_knife is not None:
        return cutting_knife, "cutting"
    elif held_knife is not None:
        return held_knife, "held"
    else:
        return None, "not_found"


# 模块级变量：记录切割刀的 instance_id
_cutting_knife_instance_id: Optional[int] = None
_last_auto_cut_failure_info: Dict[str, object] = {}


def _set_last_auto_cut_failure_info(info: Optional[Dict[str, object]]) -> None:
    global _last_auto_cut_failure_info
    _last_auto_cut_failure_info = dict(info or {})


def get_last_auto_cut_failure_info() -> Dict[str, object]:
    return dict(_last_auto_cut_failure_info)


def set_cutting_knife_id(instance_id: int) -> None:
    """设置切割刀的 instance_id（进入切割模式前调用）"""
    global _cutting_knife_instance_id
    _cutting_knife_instance_id = instance_id
    print(f"[*] 已记录切割刀 instance_id: {instance_id}")


def find_knife_by_instance_id(data: Dict, instance_id: int) -> Optional[Dict]:
    """根据 instance_id 查找刀"""
    if data is None or "products" not in data:
        return None

    for product in data["products"]:
        if product.get("instance_id") == instance_id:
            return product
    return None


def find_cutting_knife(data: Dict = None, target: Dict = None) -> Optional[Dict]:
    """
    查找切割模式下的刀

    策略优先级（v3.2更新）：
    1. 找 is_cutting=true 的刀（最可靠，C# v4.30通过Outline组件检测）
    2. 如果有记录的 instance_id，用它查找
    3. 找 is_held=true 的刀
    4. 找距离玩家最近的刀（屏幕上可见的）

    Args:
        data: 扫描数据
        target: 目标物品（未使用，保留接口兼容）

    Returns:
        dict: 刀的信息
    """
    global _cutting_knife_instance_id

    if data is None:
        data = read_realtime_products()

    if data is None or "products" not in data:
        return None

    # 方法1: v3.2新增 - 找 is_cutting=true 的刀（最可靠）
    for product in data["products"]:
        if product.get("is_cut_mode", product.get("is_cutting", False)):
            _cutting_knife_instance_id = product.get("instance_id")
            return product

    # 方法2: 用记录的 instance_id 查找
    if _cutting_knife_instance_id is not None:
        knife = find_knife_by_instance_id(data, _cutting_knife_instance_id)
        if knife is not None and knife.get("is_cut_mode", knife.get("is_cutting", False)):
            return knife

    return None


def find_item_by_name(
    item_name: str,
    data: Dict = None,
    must_be_on_screen: bool = False,
    exclude_held: bool = True,
    instance_id: Optional[int] = None,
) -> Optional[Dict]:
    """
    根据物品名称查找物品信息

    Args:
        item_name: 物品名称（中文或英文）
        data: 扫描数据（可选）
        must_be_on_screen: 是否必须在屏幕内
        exclude_held: 是否排除手持物品

    Returns:
        dict: 物品信息
        None: 未找到
    """
    if data is None:
        data = read_realtime_products()

    if data is None or "products" not in data:
        return None

    item_name_lower = item_name.lower().strip()

    # 收集所有匹配的物品
    matches = []
    for product in data["products"]:
        # 排除手持物品
        if exclude_held and product.get("is_held", False):
            continue

        if instance_id is not None:
            try:
                if int(product.get("instance_id")) != int(instance_id):
                    continue
            except Exception:
                continue

        name_cn = product.get("name_cn", "").lower()
        name_en = product.get("name_en", "").lower()

        # 精确匹配优先
        if name_en == item_name_lower or name_cn == item_name_lower:
            if not must_be_on_screen or product.get("is_on_screen", False):
                matches.append(product)
        # 部分匹配
        elif (item_name_lower in name_cn or item_name_lower in name_en):
            if not must_be_on_screen or product.get("is_on_screen", False):
                matches.append(product)

    if not matches:
        return None

    # 返回距离最近的
    matches.sort(key=lambda x: x.get("distance", 999))
    return matches[0]


def refresh_cutting_target(target: Dict, data: Dict = None) -> Optional[Dict]:
    """按 instance_id 优先、名称兜底，刷新当前切割目标。"""
    if not isinstance(target, dict):
        return None
    target_name = str(target.get("name_en") or target.get("name_cn") or "").strip()
    target_instance_id = target.get("instance_id")
    resolved_instance_id: Optional[int] = None
    if target_instance_id is not None and str(target_instance_id).strip():
        try:
            resolved_instance_id = int(target_instance_id)
        except Exception:
            resolved_instance_id = None
    if not target_name and resolved_instance_id is None:
        return None
    return find_item_by_name(
        target_name,
        data=data,
        exclude_held=True,
        instance_id=resolved_instance_id,
    )


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


def get_rotation(item: Dict) -> Optional[Tuple[float, float, float]]:
    """获取物品的旋转角度 (x, y, z)"""
    rotation = item.get("rotation")
    if rotation is None:
        return None
    return (
        rotation.get("x", 0),
        rotation.get("y", 0),
        rotation.get("z", 0)
    )


def get_rotation_z(item: Dict) -> Optional[float]:
    """获取物品的Z轴旋转角度"""
    rotation = item.get("rotation")
    if rotation is None:
        return None
    return rotation.get("z", 0)


def get_rotation_y(item: Dict) -> Optional[float]:
    """获取物品的Y轴旋转角度（用于确定切割方向）"""
    rotation = item.get("rotation")
    if rotation is None:
        return None
    return rotation.get("y", 0)


def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-180, 180] 范围"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def angle_difference(target: float, current: float) -> float:
    """计算从 current 到 target 需要旋转的角度（正值=顺时针，负值=逆时针）"""
    diff = normalize_angle(target - current)
    return diff


def angle_difference_symmetric(target: float, current: float) -> float:
    """
    计算对称目标角度的最小旋转角度

    v3.5: 刀刃是对称的，目标角度θ和θ+180°等效
    选择旋转角度最小的方向

    Args:
        target: 目标角度（度）
        current: 当前角度（度）

    Returns:
        float: 最小旋转角度（正值=顺时针，负值=逆时针）
    """
    # 计算到目标角度的旋转
    diff1 = normalize_angle(target - current)
    # 计算到目标+180°的旋转
    diff2 = normalize_angle(target + 180.0 - current)

    # 选择绝对值更小的
    if abs(diff1) <= abs(diff2):
        return diff1
    else:
        return diff2


def calculate_horizontal_distance(pos1: Tuple[float, float, float],
                                   pos2: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    计算两个位置在水平面（XZ平面）上的差值

    Returns:
        tuple: (dx, dz) 从 pos1 到 pos2 的差值
    """
    dx = pos2[0] - pos1[0]
    dz = pos2[2] - pos1[2]
    return (dx, dz)


def calculate_ellipsoid_equal_volume_cuts(cut_num: int) -> List[float]:
    """
    计算椭球形物体的等体积切割位置（归一化到 [-1, 1]）

    椭球体积公式: V = (4/3) * π * a * b * c
    沿长轴(a)切割时，位置x处的横截面积: A(x) = π * b * c * (1 - x²/a²)

    从位置 x1 到 x2 的体积:
    V(x1, x2) = ∫[x1,x2] A(x) dx = π*b*c * [x - x³/(3a²)] 从x1到x2

    对于归一化椭球 (a=1)，总体积 V_total = (4/3) * π * b * c
    我们需要找到切割位置使得每块体积 = V_total / (cut_num + 1)

    Args:
        cut_num: 切割次数

    Returns:
        list: 归一化切割位置 [-1, 1] 范围内的值
    """
    # 归一化椭球，a=1，总体积正比于 4/3
    # 从 -1 到 x 的累积体积函数 (归一化后):
    # F(x) = (x - x³/3 + 2/3) / (4/3) = (3x - x³ + 2) / 4
    # 范围: F(-1) = 0, F(1) = 1

    def cumulative_volume(x):
        """累积体积函数，归一化到 [0, 1]"""
        return (3 * x - x ** 3 + 2) / 4

    def inverse_cumulative_volume(v):
        """反函数：给定目标累积体积比例，求位置x"""
        # 求解 (3x - x³ + 2) / 4 = v
        # 即 x³ - 3x + (4v - 2) = 0
        # Use simple bisection (no scipy/numpy dependency).
        lo, hi = -1.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if cumulative_volume(mid) < v:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # 计算等体积切割位置
    positions = []
    for i in range(1, cut_num + 1):
        target_volume_ratio = i / (cut_num + 1)
        x = inverse_cumulative_volume(target_volume_ratio)
        positions.append(x)

    return positions


def calculate_cut_positions(target: Dict, cut_num: int,
                              knife_y_rotation: float = 0.0,
                              equal_volume: bool = True) -> List[Tuple[float, float, float]]:
    """
    基于目标物品的bounds和旋转计算切割位置列表

    v3.4: 正确处理物品旋转和长短边判断
    v3.6: 支持等体积切割（假设椭球形物体）

    切割原理：
        1. 从 AABB 和物品 Y 旋转反推物品的本地长短边
        2. 计算世界坐标系中的长轴方向
        3. 切割点沿长轴分布:
           - equal_volume=True: 等体积切割（椭球假设）
           - equal_volume=False: 等距切割
        4. 刀刃垂直于长轴（与短边平行）

    Args:
        target: 目标物品信息（需包含bounds和rotation）
        cut_num: 切割次数（产生 cut_num+1 片）
        knife_y_rotation: 刀当前的Y轴旋转角度（度）- 用于参考
        equal_volume: 是否使用等体积切割（默认True）

    Returns:
        list: [(x, z, knife_target_y_rotation), ...] 切割位置和刀的目标Y旋转角度
    """
    import math

    bounds = get_bounds(target)
    target_rotation = target.get("rotation", {})
    target_y_rot = target_rotation.get("y", 0.0)

    if bounds is None:
        pos = get_position(target)
        # 无bounds时，刀垂直于物品朝向
        return [(pos[0], pos[2], (target_y_rot + 90.0) % 360)]

    (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
    center_x = (min_x + max_x) / 2
    center_z = (min_z + max_z) / 2
    size_x = max_x - min_x
    size_z = max_z - min_z

    # 归一化 Y 旋转到 [0, 360)
    theta = target_y_rot % 360
    theta_rad = math.radians(theta)

    # 反推本地尺寸（近似）
    # 当 theta 接近 0°/180°：AABB ≈ 本地尺寸
    # 当 theta 接近 90°/270°：AABB 的 X/Z 交换
    cos_theta = abs(math.cos(theta_rad))
    sin_theta = abs(math.sin(theta_rad))

    if cos_theta > sin_theta:
        # theta 更接近 0° 或 180°
        local_size_x = size_x
        local_size_z = size_z
    else:
        # theta 更接近 90° 或 270°
        local_size_x = size_z
        local_size_z = size_x

    # 判断本地长轴方向
    if local_size_x > local_size_z:
        local_long_axis = "X"
        local_long_length = local_size_x
    else:
        local_long_axis = "Z"
        local_long_length = local_size_z

    # 计算世界坐标系中的长轴角度
    # 本地 Z 轴经过 Y 旋转后：世界角度 = theta
    # 本地 X 轴经过 Y 旋转后：世界角度 = theta + 90°
    if local_long_axis == "Z":
        world_long_angle = theta
    else:
        world_long_angle = (theta + 90.0) % 360

    # 刀刃垂直于长轴（横切物品）：刀身方向 = 长轴方向 + 90°
    knife_target_y_rot = (world_long_angle + 90.0) % 360

    # 长轴方向的单位向量
    world_long_rad = math.radians(world_long_angle)
    long_axis_x = math.sin(world_long_rad)
    long_axis_z = math.cos(world_long_rad)

    # 沿长轴方向的切割长度
    cut_length = local_long_length

    positions = []
    usable_axis_ratio = max(0.0, min(1.0, CUT_AXIS_END_RATIO) - max(0.0, CUT_AXIS_START_RATIO))
    usable_axis_ratio = usable_axis_ratio if usable_axis_ratio > 0 else 1.0
    trimmed_half = (cut_length * usable_axis_ratio) / 2.0
    if equal_volume:
        # Equal-volume cuts on a normalized ellipsoid, mapped to the long axis.
        # Returned t values are in [-1, 1] where -1/+1 are the two ends.
        t_list = calculate_ellipsoid_equal_volume_cuts(int(cut_num))
        for t in t_list:
            safe_t = float(t) * usable_axis_ratio
            x = center_x + safe_t * (cut_length / 2.0) * long_axis_x
            z = center_z + safe_t * (cut_length / 2.0) * long_axis_z
            positions.append((x, z, knife_target_y_rot))
    else:
        # Equal-distance cuts along the trimmed long axis span.
        step = (trimmed_half * 2.0) / (cut_num + 1)
        start_x = center_x - trimmed_half * long_axis_x
        start_z = center_z - trimmed_half * long_axis_z
        for i in range(1, cut_num + 1):
            x = start_x + step * i * long_axis_x
            z = start_z + step * i * long_axis_z
            positions.append((x, z, knife_target_y_rot))

    print(f"  [切割分析] 物品Y旋转={target_y_rot:.1f}°, 本地长轴={local_long_axis}")
    print(f"  [切割分析] 世界长轴角度={world_long_angle:.1f}°, 刀目标Y旋转={knife_target_y_rot:.1f}° (刀刃垂直长轴)")
    print(f"  [切割分析] 长轴方向=({long_axis_x:.2f}, {long_axis_z:.2f}), 切割长度={cut_length:.3f}m")
    print(f"  [切割分析] 切割策略={'equal_volume' if equal_volume else 'equal_distance'}")
    print(
        f"  [切割分析] 有效切割区间={CUT_AXIS_START_RATIO*100:.0f}%~{CUT_AXIS_END_RATIO*100:.0f}% "
        f"(有效长度={cut_length * usable_axis_ratio:.3f}m)"
    )
    if len(positions) >= 2:
        # Project successive points onto the long axis to show actual spacing along the cut axis.
        ds = []
        for (x1, z1, _), (x2, z2, _) in zip(positions, positions[1:]):
            ds.append((x2 - x1) * long_axis_x + (z2 - z1) * long_axis_z)
        ds_str = ", ".join([f"{d*100:.1f}cm" for d in ds])
        print(f"  [切割分析] 相邻切割点沿长轴间距: {ds_str}")

    return positions


def calculate_cut_direction(target: Dict, knife: Dict = None) -> str:
    """
    分析切割方向

    Returns:
        str: "along_x" 或 "along_z"，表示切割线沿哪个轴
    """
    bounds = get_bounds(target)
    if bounds is None:
        return "unknown"

    (min_x, _, min_z), (max_x, _, max_z) = bounds
    size_x = max_x - min_x
    size_z = max_z - min_z

    return "along_x" if size_x >= size_z else "along_z"


def get_screen_center(data: Dict = None) -> Tuple[int, int]:
    """获取屏幕中心坐标"""
    if data is None:
        data = read_realtime_products()

    if data is None:
        return (960, 540)

    width = data.get("screen_width", 1920)
    height = data.get("screen_height", 1080)
    return (width // 2, height // 2)


# ================================================================
# 闭环控制移动函数
# ================================================================

def get_knife_blade_position(knife: Dict) -> Tuple[float, float, float]:
    """
    计算刀锋有效切割点的位置（而非几何中心）

    v3.5: 刀的有效切割点在刀锋的1/4处，不是几何中心

    刀的结构：
    - 几何中心是刀身的中点
    - 刀锋（刀刃）在刀的前端，只有这部分能切割
    - 有效切割点 = 几何中心 + 刀长/4 * 刀身朝向

    Args:
        knife: 刀的信息（需包含position, rotation, 可选bounds）

    Returns:
        tuple: (x, y, z) 刀锋有效切割点的位置
    """
    # 获取刀的几何中心
    center_x, center_y, center_z = get_position(knife)

    return (center_x, center_y, center_z)


def get_rotation_duration(error_deg: float) -> float:
    """
    根据旋转误差大小返回合适的按键时长

    v3.5: 动态旋转步长策略
    - 误差 > 30°：大步长，快速接近
    - 误差 10-30°：中步长
    - 误差 < 10°：小步长，精细调整

    Args:
        error_deg: 旋转误差（度，绝对值）

    Returns:
        float: 按键时长（秒）
    """
    error_abs = abs(error_deg)
    if error_abs >= ROTATION_LARGE_THRESHOLD:
        return ROTATION_LARGE_DURATION
    elif error_abs >= ROTATION_MEDIUM_THRESHOLD:
        return ROTATION_MEDIUM_DURATION
    else:
        return ROTATION_SMALL_DURATION


def get_move_pixels(error_distance: float) -> int:
    """
    根据位置误差大小返回合适的移动像素数

    v3.5: 动态移动步长策略
    - 误差 > 0.10m：大步长 200像素
    - 误差 0.05-0.10m：中步长 100像素
    - 误差 < 0.05m：小步长 50像素

    Args:
        error_distance: 位置误差（米）

    Returns:
        int: 移动像素数
    """
    if error_distance >= MOVE_LARGE_THRESHOLD:
        return MOVE_LARGE_PIXELS
    elif error_distance >= MOVE_MEDIUM_THRESHOLD:
        return MOVE_MEDIUM_PIXELS
    else:
        return MOVE_SMALL_PIXELS


def compute_mouse_movement(err_x: float, err_z: float,
                           x_coeffs: List[float], z_coeffs: List[float]) -> Tuple[float, float]:
    """
    v3.7: 使用 2x2 线性模型把 (err_x, err_z) 转为鼠标水平移动量 (dx, dy)。

    线性模型：
      Δx = a*dx + b*dy
      Δz = c*dx + d*dy
    """
    a, b = x_coeffs
    c, d = z_coeffs
    det = a * d - b * c

    if abs(det) < 1e-9:
        mouse_dx = (err_x / a) if abs(a) > 1e-9 else 0.0
        mouse_dy = (err_z / d) if abs(d) > 1e-9 else 0.0
        return mouse_dx, mouse_dy

    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det

    mouse_dx = inv_a * err_x + inv_b * err_z
    mouse_dy = inv_c * err_x + inv_d * err_z
    return mouse_dx, mouse_dy


def is_calibration_usable(x_coeffs: List[float], z_coeffs: List[float], *, eps: float = 1e-5) -> bool:
    """
    判断标定矩阵是否足以提供二维位置控制。

    需要满足：
    1. dx、dy 两列都对世界坐标有非平凡影响
    2. 2x2 矩阵不是近似奇异
    """
    if not isinstance(x_coeffs, list) or not isinstance(z_coeffs, list):
        return False
    if len(x_coeffs) != 2 or len(z_coeffs) != 2:
        return False

    a, b = float(x_coeffs[0]), float(x_coeffs[1])
    c, d = float(z_coeffs[0]), float(z_coeffs[1])
    col_dx = math.sqrt(a * a + c * c)
    col_dy = math.sqrt(b * b + d * d)
    det = abs(a * d - b * c)
    return col_dx > eps and col_dy > eps and det > eps * eps


def create_calibration_state(
    calibration: Optional[Tuple[List[float], List[float]]] = None
) -> Dict[str, object]:
    """保存切割阶段跨刀复用的标定状态。"""
    state: Dict[str, object] = {
        "x_coeffs": None,
        "z_coeffs": None,
        "needs_recalibration": False,
        "recalibration_reason": "",
        "anomaly_streak": 0,
    }
    if calibration is not None:
        x_coeffs, z_coeffs = calibration
        if is_calibration_usable(x_coeffs, z_coeffs):
            state["x_coeffs"] = list(x_coeffs)
            state["z_coeffs"] = list(z_coeffs)
    return state


def _load_calibration_from_state(
    calibration_state: Optional[Dict[str, object]]
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    if not calibration_state:
        return None, None
    x_coeffs = calibration_state.get("x_coeffs")
    z_coeffs = calibration_state.get("z_coeffs")
    if is_calibration_usable(x_coeffs, z_coeffs):
        return list(x_coeffs), list(z_coeffs)
    return None, None


def _store_calibration_in_state(
    calibration_state: Optional[Dict[str, object]],
    x_coeffs: Optional[List[float]],
    z_coeffs: Optional[List[float]],
    *,
    clear_recalibration_flag: bool = True,
) -> None:
    if calibration_state is None:
        return
    if is_calibration_usable(x_coeffs, z_coeffs):
        calibration_state["x_coeffs"] = list(x_coeffs)
        calibration_state["z_coeffs"] = list(z_coeffs)
        calibration_state["anomaly_streak"] = 0
        if clear_recalibration_flag:
            calibration_state["needs_recalibration"] = False
            calibration_state["recalibration_reason"] = ""
    else:
        calibration_state["x_coeffs"] = None
        calibration_state["z_coeffs"] = None


def _request_recalibration(
    calibration_state: Optional[Dict[str, object]],
    reason: str,
    *,
    verbose: bool = False,
) -> None:
    if calibration_state is None:
        return
    calibration_state["x_coeffs"] = None
    calibration_state["z_coeffs"] = None
    calibration_state["needs_recalibration"] = True
    calibration_state["recalibration_reason"] = reason
    if verbose:
        print(f"  [标定] 标记重标定: {reason}")


def detect_calibration_anomaly(
    *,
    prev_blade_x: float,
    prev_blade_z: float,
    cur_blade_x: float,
    cur_blade_z: float,
    move_x: int,
    move_y: int,
    x_coeffs: List[float],
    z_coeffs: List[float],
) -> Optional[str]:
    """比较预测位移与实际位移，判断当前复用标定是否已失真。"""
    predicted_x = x_coeffs[0] * move_x + x_coeffs[1] * move_y
    predicted_z = z_coeffs[0] * move_x + z_coeffs[1] * move_y
    actual_x = cur_blade_x - prev_blade_x
    actual_z = cur_blade_z - prev_blade_z

    predicted_norm = math.sqrt(predicted_x * predicted_x + predicted_z * predicted_z)
    actual_norm = math.sqrt(actual_x * actual_x + actual_z * actual_z)

    if predicted_norm < CALIBRATION_ANOMALY_MIN_EXPECTED:
        return None
    if actual_norm < CALIBRATION_ANOMALY_MIN_ACTUAL:
        return f"movement_response_too_small:pred={predicted_norm:.4f}m,actual={actual_norm:.4f}m"

    dot = predicted_x * actual_x + predicted_z * actual_z
    cos_sim = dot / max(predicted_norm * actual_norm, 1e-9)
    if cos_sim < CALIBRATION_ANOMALY_COS_THRESHOLD:
        return f"movement_direction_mismatch:cos={cos_sim:.3f}"

    return None


def _get_cutting_knife_blade_xz(data: Dict, cutting_target: Dict = None) -> Optional[Tuple[float, float]]:
    knife = find_cutting_knife(data, cutting_target)
    if knife is None:
        return None
    blade_x, _, blade_z = get_knife_blade_position(knife)
    return blade_x, blade_z


def calibrate_mouse_to_3d_mapping_for_knife(cutting_target: Dict = None,
                                            verbose: bool = True) -> Optional[Tuple[List[float], List[float]]]:
    """
    v3.7: 动态标定 Horizontal_movement(dx,dy) 与世界坐标(Δx,Δz)的映射关系。
    参考 auto_pouring 的做法：对 dx、dy 分别做一次试探移动，测量刀刃位置变化。
    """
    data = read_realtime_products()
    if data is None:
        return None

    init = _get_cutting_knife_blade_xz(data, cutting_target=cutting_target)
    if init is None:
        return None
    init_x, init_z = init

    if verbose:
        print("[*] v3.7 标定：开始测量鼠标移动 -> 刀刃(x,z)位移映射...")
        print(f"  初始刀刃: ({init_x:.4f}, {init_z:.4f})")

    # 试探1：dx
    pre_mtime = get_json_file_mtime()
    move_knife_step(CALIBRATION_PIXELS, 0)
    wait_for_data_update(pre_mtime, timeout=2.0, verbose=False)
    wait_for_move_settle()

    data = read_realtime_products()
    after_dx = _get_cutting_knife_blade_xz(data or {}, cutting_target=cutting_target)
    if after_dx is None:
        return None
    dx1 = after_dx[0] - init_x
    dz1 = after_dx[1] - init_z

    # 回位
    pre_mtime = get_json_file_mtime()
    move_knife_step(-CALIBRATION_PIXELS, 0)
    wait_for_data_update(pre_mtime, timeout=2.0, verbose=False)
    wait_for_move_settle()

    # 试探2：dy
    data = read_realtime_products()
    before_dy = _get_cutting_knife_blade_xz(data or {}, cutting_target=cutting_target)
    if before_dy is None:
        return None

    pre_mtime = get_json_file_mtime()
    move_knife_step(0, CALIBRATION_PIXELS)
    wait_for_data_update(pre_mtime, timeout=2.0, verbose=False)
    wait_for_move_settle()

    data = read_realtime_products()
    after_dy = _get_cutting_knife_blade_xz(data or {}, cutting_target=cutting_target)
    if after_dy is None:
        return None

    dx2 = after_dy[0] - before_dy[0]
    dz2 = after_dy[1] - before_dy[1]

    # 回位
    pre_mtime = get_json_file_mtime()
    move_knife_step(0, -CALIBRATION_PIXELS)
    wait_for_data_update(pre_mtime, timeout=2.0, verbose=False)
    wait_for_move_settle()

    x_coeffs = [dx1 / CALIBRATION_PIXELS, dx2 / CALIBRATION_PIXELS]
    z_coeffs = [dz1 / CALIBRATION_PIXELS, dz2 / CALIBRATION_PIXELS]

    if verbose:
        print("  标定结果:")
        print(f"    Δx = {x_coeffs[0]:.6f}*dx + {x_coeffs[1]:.6f}*dy")
        print(f"    Δz = {z_coeffs[0]:.6f}*dx + {z_coeffs[1]:.6f}*dy")

    return x_coeffs, z_coeffs


def rotate_knife_step(direction: str, duration: float = None, error_deg: float = None) -> None:
    """
    旋转刀一小步（离散动作）

    v3.5: 支持动态步长，根据误差大小自动调整

    Args:
        direction: "cw" (顺时针，D键) 或 "ccw" (逆时针，A键)
        duration: 按键持续时间（秒），None则根据error_deg自动计算
        error_deg: 当前旋转误差（度），用于动态计算步长
    """
    try:
        from epm.cerebellum.local_actions import hold_keyboard, leave_keyboard
    except ImportError:
        return

    # 动态计算步长
    if duration is None:
        if error_deg is not None:
            duration = get_rotation_duration(error_deg)
        else:
            duration = ROTATION_MEDIUM_DURATION  # 默认中等步长

    key = "d" if direction == "cw" else "a"
    hold_keyboard(key)       # 按下
    time.sleep(duration)     # 保持按住
    leave_keyboard(key)      # 松开
    time.sleep(ACTION_DELAY) # 等待动作生效


def move_knife_step(dx: int, dy: int) -> None:
    """
    移动刀一小步（离散动作，固定像素数）

    Args:
        dx: X方向移动像素（正=右，负=左）
        dy: Y方向移动像素（正=下，负=上）
    """
    try:
        from epm.cerebellum.local_actions import horizontal_movement
    except ImportError:
        return

    horizontal_movement(dx, dy)
    time.sleep(ACTION_DELAY)


def wait_for_json_update() -> None:
    """等待JSON文件更新"""
    time.sleep(JSON_UPDATE_DELAY)


def wait_for_move_settle() -> None:
    """移动后额外等待一小段时间，避免读取到刚更新但仍滞后的位姿。"""
    time.sleep(MOVE_SETTLE_DELAY)


def move_knife_to_target_closed_loop(target_x: float, target_z: float,
                                      target_rot_y: float = None,
                                      cutting_target: Dict = None,
                                      check_target_presence: bool = True,
                                      verbose: bool = True,
                                      calibration: Optional[Tuple[List[float], List[float]]] = None,
                                      calibration_state: Optional[Dict[str, object]] = None) -> bool:
    """
    使用闭环反馈控制将刀移动到目标位置（含旋转调整）

    v3.5: 基于文件时间戳等待数据更新，避免读取过期数据导致的振荡
          方案A+C: 监控文件时间戳 + 等待数据更新确认后再行动

    Args:
        target_x: 目标X坐标（游戏世界坐标）
        target_z: 目标Z坐标（游戏世界坐标）
        target_rot_y: 目标Y轴旋转角度（度），None表示不调整旋转
        cutting_target: 被切割的目标物品（用于在切割模式下定位刀）
        check_target_presence: 是否在闭环过程中持续刷新并检查切割目标是否仍存在
        verbose: 是否输出详细日志

    Returns:
        bool: 是否成功到达目标位置
    """
    # v3.7: 动态标定映射（用于避免不同朝向导致轴交换/符号翻转）
    # 注意：刀具旋转后，位置映射可能变化，因此这里只把外部标定当作“候选初值”。
    x_coeffs = None
    z_coeffs = None
    if calibration_state is not None:
        x_coeffs, z_coeffs = _load_calibration_from_state(calibration_state)
        if calibration_state.get("needs_recalibration") and verbose:
            reason = calibration_state.get("recalibration_reason", "")
            if reason:
                print(f"  [标定] 使用前检测到需重标定: {reason}")
    if calibration is not None and (x_coeffs is None or z_coeffs is None):
        x_coeffs, z_coeffs = calibration
        if not is_calibration_usable(x_coeffs, z_coeffs):
            if verbose:
                print("  [标定] 外部传入的标定矩阵退化，忽略并等待旋转到位后重标定")
            x_coeffs, z_coeffs = None, None
        else:
            _store_calibration_in_state(calibration_state, x_coeffs, z_coeffs)

    # 首次读取数据，获取初始时间戳
    data = read_realtime_products()
    last_mtime = get_json_file_mtime()
    missing_target_streak = 0
    scope_stall_streak = 0

    for iteration in range(MAX_ITERATIONS):
        refreshed_target = (
            refresh_cutting_target(cutting_target, data=data)
            if (check_target_presence and cutting_target is not None)
            else None
        )
        if check_target_presence and cutting_target is not None:
            if refreshed_target is None:
                missing_target_streak += 1
                if verbose:
                    print(
                        f"  [目标] 当前未找到切割目标，missing_streak="
                        f"{missing_target_streak}/{MAX_TARGET_MISSING_STREAK}"
                    )
                if missing_target_streak >= MAX_TARGET_MISSING_STREAK:
                    print("[!] 闭环控制：切割目标连续丢失，终止对齐")
                    return False
            else:
                cutting_target = refreshed_target
                missing_target_streak = 0

        # 在切割模式下用 find_cutting_knife（因为 is_held 可能为 false）
        knife = find_cutting_knife(data, cutting_target)

        if knife is None:
            print("[!] 闭环控制：未找到刀")
            return False

        # 获取刀的Y轴旋转（用于控制对准与姿态）
        cur_rot_y = get_rotation_y(knife)  # Y轴旋转（水平面内）

        # 当前版本使用刀中心作为对准点
        blade_x, blade_y, blade_z = get_knife_blade_position(knife)
        center_x, center_y, center_z = get_position(knife)

        # 计算位置误差（当前等价于刀中心误差）
        err_x = target_x - blade_x
        err_z = target_z - blade_z
        distance = math.sqrt(err_x * err_x + err_z * err_z)
        center_err_x = target_x - center_x
        center_err_z = target_z - center_z

        # 计算旋转误差（使用对称角度，选择最小旋转）
        rot_err = None
        if target_rot_y is not None and cur_rot_y is not None:
            rot_err = angle_difference_symmetric(target_rot_y, cur_rot_y)

        # 只有在目标朝向基本对齐后，鼠标移动->世界坐标映射才相对稳定。
        # 否则很容易出现“标定时一个方向可动，旋转后该方向失效”的问题。
        if (rot_err is None) or (abs(rot_err) < ROTATION_TOLERANCE):
            if x_coeffs is None or z_coeffs is None:
                calibration_now = calibrate_mouse_to_3d_mapping_for_knife(
                    cutting_target=cutting_target,
                    verbose=verbose,
                )
                if calibration_now is not None:
                    cand_x_coeffs, cand_z_coeffs = calibration_now
                    if is_calibration_usable(cand_x_coeffs, cand_z_coeffs):
                        x_coeffs, z_coeffs = cand_x_coeffs, cand_z_coeffs
                        _store_calibration_in_state(calibration_state, x_coeffs, z_coeffs)
                    else:
                        if verbose:
                            print("  [标定] 当前朝向下标定矩阵退化，暂时使用保守探测移动")
                        x_coeffs, z_coeffs = None, None

        if verbose and iteration % 3 == 0:
            knife_instance_id = knife.get("instance_id", "N/A")
            rot_info = f", Y旋转={cur_rot_y:.1f}°" if cur_rot_y is not None else ""
            target_rot_info = f" (目标={target_rot_y:.1f}°/±180°)" if target_rot_y is not None else ""
            print(
                f"  [迭代 {iteration}] knife_id={knife_instance_id} "
                f"刀中心=({center_x:.3f}, {center_z:.3f}){rot_info}{target_rot_info}, "
                f"center_err=({center_err_x:.3f}, {center_err_z:.3f}) "
                f"target=({target_x:.3f}, {target_z:.3f})"
            )

        # 检查是否到达（单轴分别判断，曼哈顿距离）
        position_ok = (abs(err_x) < POSITION_TOLERANCE) and (abs(err_z) < POSITION_TOLERANCE)
        rotation_ok = (rot_err is None) or (abs(rot_err) < ROTATION_TOLERANCE)

        if position_ok and rotation_ok:
            if verbose:
                print(f"  [闭环] 到达目标，迭代 {iteration} 次，误差 {distance:.4f}m")
            return True

        prev_blade_x, prev_blade_z = blade_x, blade_z

        # 记录动作前的时间戳
        pre_action_mtime = get_json_file_mtime()

        # 优先调整旋转（如果需要）
        action_performed = False
        rotation_performed = False  # v3.6: 标记是否执行了旋转
        if rot_err is not None and abs(rot_err) >= ROTATION_TOLERANCE:
            direction = "cw" if rot_err > 0 else "ccw"
            duration = get_rotation_duration(rot_err)
            if verbose:
                step_type = "大" if abs(rot_err) >= ROTATION_LARGE_THRESHOLD else ("中" if abs(rot_err) >= ROTATION_MEDIUM_THRESHOLD else "小")
                print(f"  [旋转] {direction} (误差={rot_err:.1f}°, {step_type}步长={duration:.3f}s)")
            rotate_knife_step(direction, error_deg=rot_err)
            action_performed = True
            rotation_performed = True
        else:
            # 执行离散移动（动态步长）
            move_x = 0
            move_y = 0

            if x_coeffs is not None and z_coeffs is not None:
                # v3.7: 使用标定矩阵反算鼠标移动，自动适配轴/符号变化
                raw_move_x, raw_move_y = compute_mouse_movement(err_x, err_z, x_coeffs, z_coeffs)
                move_x = int(round(raw_move_x))
                move_y = int(round(raw_move_y))

                # 限制单次移动幅度（防止过冲）
                max_pixels = get_move_pixels(distance)
                if abs(move_x) > max_pixels:
                    move_x = max_pixels if move_x > 0 else -max_pixels
                if abs(move_y) > max_pixels:
                    move_y = max_pixels if move_y > 0 else -max_pixels

                # 确保至少有最小移动量，避免四舍五入为0导致卡住
                min_pixels = 5
                if (move_x == 0 and move_y == 0) and distance >= POSITION_TOLERANCE:
                    if abs(raw_move_x) >= abs(raw_move_y) and abs(raw_move_x) > 1e-9:
                        move_x = min_pixels if raw_move_x > 0 else -min_pixels
                    elif abs(raw_move_y) > 1e-9:
                        move_y = min_pixels if raw_move_y > 0 else -min_pixels
            else:
                # 没有可用标定时，只做保守探测，避免粗暴的固定 50/50 漂移。
                if abs(err_x) >= abs(err_z) and abs(err_x) >= POSITION_TOLERANCE:
                    move_x = MOVE_PROBE_PIXELS if err_x > 0 else -MOVE_PROBE_PIXELS
                elif abs(err_z) >= POSITION_TOLERANCE:
                    move_y = -MOVE_PROBE_PIXELS if err_z > 0 else MOVE_PROBE_PIXELS

            if move_x != 0 or move_y != 0:
                if verbose:
                    if x_coeffs is not None and z_coeffs is not None:
                        step_type = "大" if distance >= MOVE_LARGE_THRESHOLD else ("中" if distance >= MOVE_MEDIUM_THRESHOLD else "小")
                        method = "标定"
                    else:
                        step_type = "探测"
                        method = "无标定"
                    print(f"  [移动-{method}] dx={move_x}, dy={move_y} ({step_type}步长, 距离={distance:.3f}m)")
                move_knife_step(move_x, move_y)
                action_performed = True

        # v3.5: 等待数据更新后再读取（方案A+C）
        # v3.6: 旋转后额外等待物理稳定
        if action_performed:
            # 等待JSON文件时间戳变化，表示有新数据
            updated = wait_for_data_update(pre_action_mtime, verbose=False)
            if not updated and verbose:
                print(f"  [警告] 等待数据更新超时，继续使用当前数据")

            # v3.6: 旋转操作后，等待物理稳定再继续
            if rotation_performed:
                stable_data, stable_angle = wait_for_physics_stable(
                    cutting_target=cutting_target,
                    verbose=verbose
                )
                if stable_data is not None:
                    data = stable_data
                else:
                    data = read_realtime_products()

                if rot_err is not None and abs(rot_err) >= RECALIBRATE_AFTER_ROTATION_DEG:
                    x_coeffs, z_coeffs = None, None
                    _request_recalibration(
                        calibration_state,
                        f"rotation_shift:{abs(rot_err):.1f}deg",
                        verbose=verbose,
                    )
            else:
                # 移动操作后，额外等一小段稳定时间再读取，避免数据更新已到但位置仍滞后。
                wait_for_move_settle()
                data = read_realtime_products()
                scope_like_anomaly = False
                anomaly_reason = ""
                if x_coeffs is not None and z_coeffs is not None and data is not None:
                    cur_blade = _get_cutting_knife_blade_xz(data, cutting_target=cutting_target)
                    if cur_blade is not None:
                        anomaly_reason = detect_calibration_anomaly(
                            prev_blade_x=prev_blade_x,
                            prev_blade_z=prev_blade_z,
                            cur_blade_x=cur_blade[0],
                            cur_blade_z=cur_blade[1],
                            move_x=move_x,
                            move_y=move_y,
                            x_coeffs=x_coeffs,
                            z_coeffs=z_coeffs,
                        )
                        if anomaly_reason:
                            if calibration_state is not None:
                                streak = int(calibration_state.get("anomaly_streak", 0)) + 1
                                calibration_state["anomaly_streak"] = streak
                            else:
                                streak = CALIBRATION_ANOMALY_STREAK
                            scope_like_anomaly = anomaly_reason.startswith("movement_response_too_small") or anomaly_reason.startswith("movement_direction_mismatch")
                            if streak >= CALIBRATION_ANOMALY_STREAK:
                                x_coeffs, z_coeffs = None, None
                                _request_recalibration(
                                    calibration_state,
                                    anomaly_reason,
                                    verbose=verbose,
                                )
                        elif calibration_state is not None:
                            calibration_state["anomaly_streak"] = 0
                            _store_calibration_in_state(calibration_state, x_coeffs, z_coeffs)
                cur_blade = _get_cutting_knife_blade_xz(data, cutting_target=cutting_target) if data is not None else None
                if cur_blade is not None and (move_x != 0 or move_y != 0):
                    blade_move = math.sqrt((cur_blade[0] - prev_blade_x) ** 2 + (cur_blade[1] - prev_blade_z) ** 2)
                    post_err_x = target_x - cur_blade[0]
                    post_err_z = target_z - cur_blade[1]
                    post_distance = math.sqrt(post_err_x * post_err_x + post_err_z * post_err_z)
                    progress = distance - post_distance
                    if (
                        distance >= SCOPE_STALL_DISTANCE_THRESHOLD_M
                        and blade_move <= SCOPE_STALL_BLADE_MOVE_THRESHOLD_M
                        and progress <= SCOPE_STALL_PROGRESS_THRESHOLD_M
                    ):
                        scope_stall_streak += 1
                        if verbose:
                            print(
                                "  [作用域检测] 刀移动几乎无效，"
                                f"blade_move={blade_move:.4f}m, progress={progress:.4f}m, "
                                f"stall_streak={scope_stall_streak}/{MAX_SCOPE_STALL_STREAK}"
                            )
                        if scope_stall_streak >= MAX_SCOPE_STALL_STREAK:
                            _set_last_auto_cut_failure_info(
                                {
                                    "code": "target_out_of_cutting_scope",
                                    "reason": "stalled_near_cutting_deadzone",
                                    "iteration": int(iteration),
                                    "target_distance_before_m": round(float(distance), 4),
                                    "target_distance_after_m": round(float(post_distance), 4),
                                    "blade_move_m": round(float(blade_move), 4),
                                    "progress_m": round(float(progress), 4),
                                    "move_x": int(move_x),
                                    "move_y": int(move_y),
                                }
                            )
                            print("[!] 闭环控制：目标超出当前切割模式作用域，停止对齐")
                            return False
                    elif (
                        distance >= SCOPE_STALL_DISTANCE_THRESHOLD_M
                        and scope_like_anomaly
                    ):
                        scope_stall_streak += 1
                        if verbose:
                            print(
                                "  [作用域检测] 连续检测到移动响应异常，"
                                f"anomaly={anomaly_reason}, "
                                f"stall_streak={scope_stall_streak}/{MAX_SCOPE_STALL_STREAK}"
                            )
                        if scope_stall_streak >= MAX_SCOPE_STALL_STREAK:
                            _set_last_auto_cut_failure_info(
                                {
                                    "code": "target_out_of_cutting_scope",
                                    "reason": "repeated_movement_response_anomaly",
                                    "iteration": int(iteration),
                                    "target_distance_before_m": round(float(distance), 4),
                                    "target_distance_after_m": round(float(post_distance), 4),
                                    "blade_move_m": round(float(blade_move), 4),
                                    "progress_m": round(float(progress), 4),
                                    "anomaly_reason": str(anomaly_reason),
                                    "move_x": int(move_x),
                                    "move_y": int(move_y),
                                }
                            )
                            print("[!] 闭环控制：目标超出当前切割模式作用域，停止对齐")
                            return False
                    else:
                        scope_stall_streak = 0

            last_mtime = get_json_file_mtime()

    print(f"[!] 闭环控制：达到最大迭代次数 {MAX_ITERATIONS}，未能到达目标")
    return False


# ================================================================
# 自动切割执行函数 (v3.0 闭环控制)
# ================================================================

def auto_cut(item_name: str, item_instance_id: Optional[int] = None, cut_count: int = DEFAULT_CUT_COUNT,
             cut_interval: float = DEFAULT_CUT_INTERVAL,
             use_closed_loop: bool = True,
             equal_volume: bool = True) -> bool:
    """
    自动切割指定物品 (v3.1 闭环控制版，含旋转控制)

    基于闭环反馈控制：
    1. 找到手持的刀（使用 is_held 字段）
    2. 找到目标物品（获取 bounds 信息）
    3. 计算切割位置列表（含目标旋转角度）
    4. 对每个切割位置：闭环移动刀+调整旋转 -> 执行切割

    Args:
        item_name: 要切割的物品名称
        cut_count: 切割次数
        cut_interval: 每刀之间的间隔（秒）
        use_closed_loop: 是否使用闭环控制

    Returns:
        bool: 是否成功完成切割
    """
    try:
        from epm.cerebellum.local_actions import (
            _activate_window,
            click_mouse,
            horizontal_movement,
            io_controller,
            hold_keyboard,
            leave_keyboard
        )
    except ImportError as e:
        print(f"[!] 无法导入 local_actions: {e}")
        return False

    _set_last_auto_cut_failure_info(None)

    print("=" * 50)
    print("  自动切割模块 v3.1 (闭环+旋转控制)")
    print("=" * 50)

    # 读取扫描数据
    data = read_realtime_products()
    if data is None:
        print("[!] 无法读取扫描数据，请确保 F12 扫描已开启")
        return False

    knife_now, knife_status = find_knife_with_status(data)
    if knife_status != "cutting":
        print("[!] auto_cut 前置条件不满足：当前未处于切割模式")
        if knife_now is not None:
            print(f"[*] 当前刀状态: {knife_status}")
        print("[*] 需要先手持刀并调用 enter_cutting_mode，确认 is_cutting_mode=true 后再执行 auto_cut")
        return False

    # 查找目标物品（先找目标，用于定位切割模式的刀）
    target = find_item_by_name(item_name, data, exclude_held=True, instance_id=item_instance_id)
    if target is None:
        print(f"[!] 未找到物品: {item_name}")
        print("[*] 屏幕上可见的物品:")
        for p in data.get("products", [])[:10]:
            if p.get("is_on_screen") and not p.get("is_held"):
                print(f"    - {p.get('name_en')} ({p.get('name_cn')})")
        return False

    # 查找切割模式下的刀（切割模式 is_held 可能为 false）
    knife = find_cutting_knife(data, target)
    if knife is None:
        print("[!] 未找到刀具，请确保已拿起刀并进入切割模式")
        return False

    # 显示刀的检测方式
    if knife.get("is_held"):
        print("[*] 刀检测方式: is_held=True")
    else:
        print("[*] 刀检测方式: 位置匹配（切割模式）")

    knife_pos = get_position(knife)
    knife_instance_id = knife.get("instance_id", "N/A")
    print(f"[*] 刀 instance_id: {knife_instance_id}")
    print(f"[*] 刀的位置: ({knife_pos[0]:.3f}, {knife_pos[1]:.3f}, {knife_pos[2]:.3f})")
    print(f"[*] 刀 is_held: {knife.get('is_held', 'N/A')}")

    if knife.get("rotation"):
        rot = knife["rotation"]
        print(f"[*] 刀的旋转: ({rot.get('x', 0):.1f}, {rot.get('y', 0):.1f}, {rot.get('z', 0):.1f})")

    # 目标物品已在前面找到
    target_pos = get_position(target)
    target_name = target.get("name_en", "Unknown")
    print(f"\n[*] 目标物品: {target_name}")
    print(f"[*] 目标位置: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")

    # 获取目标的bounds信息
    bounds = get_bounds(target)
    if bounds:
        (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
        print(f"[*] 目标边界: min=({min_x:.3f}, {min_y:.3f}, {min_z:.3f})")
        print(f"             max=({max_x:.3f}, {max_y:.3f}, {max_z:.3f})")
        print(f"[*] 尺寸: x={max_x-min_x:.3f}m, y={max_y-min_y:.3f}m, z={max_z-min_z:.3f}m")
    else:
        print("[*] 目标无bounds信息，将使用单点切割")

    print(f"[*] 切割次数: {cut_count}")

    # 获取刀的当前Y轴旋转（用于切割方向分析）
    knife_y_rot = get_rotation_y(knife) or 0.0

    # 计算切割位置列表（含旋转角度）
    cut_positions = calculate_cut_positions(target, cut_count, knife_y_rot, equal_volume=bool(equal_volume))

    # 分析切割方向
    cut_dir = calculate_cut_direction(target)
    print(f"[*] 切割方向: {cut_dir} (沿{'X' if cut_dir == 'along_x' else 'Z'}轴)")

    print(f"[*] 切割位置: {len(cut_positions)} 个 (x, z, 目标Z旋转°)")
    for i, (x, z, rot_z) in enumerate(cut_positions, 1):
        print(f"    {i}. ({x:.3f}, {z:.3f}, {rot_z:.1f}°)")

    # 激活游戏窗口
    try:
        _activate_window(GAME_WINDOW_TITLE)
    except Exception as e:
        print(f"[!] 激活窗口失败: {e}")
        return False

    time.sleep(0.3)

    # ===== 步骤1: 按住Shift并进入切割模式 =====
    print(f"\n[1] 按住Shift，进入切割模式...")
    hold_keyboard("shift")  # 按住Shift键
    time.sleep(0.1)
    # io_controller.click("left")  # 点击进入切割模式
    time.sleep(0.5)

    # v3.7: 标定鼠标水平移动与世界坐标映射（用于适配不同朝向/符号翻转/轴交换）
    calibration = None
    calibration_state = create_calibration_state()
    if use_closed_loop:
        try:
            calibration = calibrate_mouse_to_3d_mapping_for_knife(cutting_target=target, verbose=True)
        except Exception:
            calibration = None
        if calibration is None:
            print("[!] 第一刀预标定失败，将在闭环中等待旋转到位后再尝试完整重标定")
        elif is_calibration_usable(calibration[0], calibration[1]):
            calibration_state = create_calibration_state(calibration)
            print("[*] 第一刀使用完整标定；后续切割默认复用上一刀标定")
        else:
            calibration = None
            print("[!] 第一刀预标定矩阵退化，将在闭环中等待旋转到位后再尝试完整重标定")

    # ===== 步骤2: 执行切割序列 =====
    for i, (target_x, target_z, target_rot_y) in enumerate(cut_positions, 1):
        print(f"\n[2.{i}] 切割位置 {i}/{len(cut_positions)}: "
              f"({target_x:.3f}, {target_z:.3f}, Y旋转={target_rot_y:.1f}°)")
        check_target_presence = (i == 1)

        if use_closed_loop:
            # 闭环移动到目标位置（含旋转调整）
            success = move_knife_to_target_closed_loop(
                target_x, target_z, target_rot_y,
                cutting_target=target,
                check_target_presence=check_target_presence,
                verbose=True,
                calibration=calibration,
                calibration_state=calibration_state,
            )
            if not success:
                print(f"[!] 无法到达切割位置 {i}")
                failure_info = get_last_auto_cut_failure_info()
                if failure_info.get("code") == "target_out_of_cutting_scope":
                    print("[!] 当前目标对当前切割模式作用域来说有些远，停止 auto_cut")
                    leave_keyboard("shift")
                    return False
                if check_target_presence:
                    latest_data = read_realtime_products()
                    latest_target = refresh_cutting_target(target, data=latest_data)
                    if latest_target is None:
                        print("[!] 切割目标已不存在或无法重新定位，停止 auto_cut")
                        leave_keyboard("shift")
                        return False
                    target = latest_target
                # 继续尝试下一个位置
            calibration = None
        else:
            # 开环移动（旧方式，不支持旋转）
            data = read_realtime_products()
            knife = find_cutting_knife(data)  # 使用 find_cutting_knife 因为已进入切割模式
            if knife:
                cur_x, _, cur_z = get_position(knife)
                dx = target_x - cur_x
                dz = target_z - cur_z
                # 简单的开环移动
                mouse_dx = int(dx * 500)
                mouse_dy = int(-dz * 500)
                horizontal_movement(mouse_dx, mouse_dy)
                time.sleep(0.3)

        # 执行切割
        print(f"  执行切割...")
        io_controller.click("left")
        time.sleep(cut_interval)

    # ===== 步骤3: 退出切割模式并松开Shift =====
    print(f"\n[3] 退出切割模式，松开Shift...")
    time.sleep(0.3)
    click_mouse("right")  # 右键退出切割模式
    time.sleep(0.1)
    leave_keyboard("shift")  # 松开Shift键

    print("\n[+] 切割完成!")
    print("=" * 50)
    _set_last_auto_cut_failure_info(None)
    return True


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
            pos = product.get("position", {})
            visible_items.append({
                "name_cn": product.get("name_cn"),
                "name_en": product.get("name_en"),
                "position": pos,
                "distance": product.get("distance"),
                "is_held": product.get("is_held", False),
                "has_bounds": product.get("bounds_min") is not None
            })

    visible_items.sort(key=lambda x: x.get("distance", 999))

    print(f"\n屏幕上可见的物品 ({len(visible_items)} 个):")
    print("-" * 70)
    for i, item in enumerate(visible_items[:20], 1):
        pos = item['position']
        held_mark = " [手持]" if item['is_held'] else ""
        bounds_mark = " [有bounds]" if item['has_bounds'] else ""
        print(f"{i}. {item['name_en']} ({item['name_cn']}){held_mark}{bounds_mark}")
        print(f"   位置: ({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f})")
        print(f"   距离: {item['distance']:.2f}m")
    print("-" * 70)

    return visible_items


def show_knife_and_target(item_name: str):
    """显示刀和目标物品的详细信息"""
    print(f"\n=== 详细信息: {item_name} ===\n")

    data = read_realtime_products()
    if data is None:
        print("[!] 无法读取扫描数据")
        return

    # 查找刀（优先检查切割模式）
    knife, knife_status = find_knife_with_status(data)
    if knife:
        status_text = "切割模式" if knife_status == "cutting" else "手持中"
        pos = knife.get("position", {})
        print(f"刀 ({knife.get('name_en')}) [{status_text}]:")
        print(f"  位置: x={pos.get('x', 0):.3f}, y={pos.get('y', 0):.3f}, z={pos.get('z', 0):.3f}")
        print(f"  距离: {knife.get('distance', 0):.2f}m")
        print(f"  is_held: {knife.get('is_held', False)}, is_cut_mode: {knife.get('is_cut_mode', knife.get('is_cutting', False))}")

        if knife.get("rotation"):
            rot = knife["rotation"]
            print(f"  旋转: ({rot.get('x', 0):.1f}, {rot.get('y', 0):.1f}, {rot.get('z', 0):.1f})")

        bounds = get_bounds(knife)
        if bounds:
            (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
            print(f"  bounds: ({min_x:.3f},{min_y:.3f},{min_z:.3f}) - ({max_x:.3f},{max_y:.3f},{max_z:.3f})")

        # 如果刀在手中但未进入切割模式，给出提示
        if knife_status == "held":
            print("\n[!] 刀在手中但未进入切割模式，请将刀靠近食物进入切割模式")
    else:
        print("[!] 未找到刀，请先拿起刀")

    # 查找目标
    target = find_item_by_name(item_name, data, exclude_held=True)
    if target:
        pos = target.get("position", {})
        print(f"\n目标 ({target.get('name_en')}):")
        print(f"  位置: x={pos.get('x', 0):.3f}, y={pos.get('y', 0):.3f}, z={pos.get('z', 0):.3f}")
        print(f"  距离: {target.get('distance', 0):.2f}m")
        print(f"  is_held: {target.get('is_held', False)}")

        bounds = get_bounds(target)
        if bounds:
            (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
            print(f"  bounds_min: ({min_x:.3f}, {min_y:.3f}, {min_z:.3f})")
            print(f"  bounds_max: ({max_x:.3f}, {max_y:.3f}, {max_z:.3f})")
            print(f"  尺寸: x={max_x-min_x:.3f}m, y={max_y-min_y:.3f}m, z={max_z-min_z:.3f}m")

            # 显示切割位置
            print(f"\n  切割位置 (3刀):")
            positions = calculate_cut_positions(target, 3)
            for i, pos in enumerate(positions, 1):
                if len(pos) == 3:
                    x, z, rot_z = pos
                    print(f"    {i}. ({x:.3f}, {z:.3f}, rot={rot_z:.1f}°)")
                else:
                    x, z = pos[:2]
                    print(f"    {i}. ({x:.3f}, {z:.3f})")
        else:
            print("  [无bounds信息]")

        # 计算差值
        if knife:
            knife_pos = get_position(knife)
            target_pos = get_position(target)
            dx, dz = calculate_horizontal_distance(knife_pos, target_pos)
            dy = target_pos[1] - knife_pos[1]

            print(f"\n差值:")
            print(f"  dx = {dx:.3f}m (正=右, 负=左)")
            print(f"  dy = {dy:.3f}m (正=上, 负=下)")
            print(f"  dz = {dz:.3f}m (正=前, 负=后)")
            print(f"  水平距离 = {math.sqrt(dx*dx + dz*dz):.3f}m")
    else:
        print(f"\n[!] 未找到目标: {item_name}")


def test_closed_loop(item_name: str):
    """测试闭环控制移动到物品中心"""
    try:
        from epm.cerebellum.local_actions import _activate_window
    except ImportError as e:
        print(f"[!] 无法导入 local_actions: {e}")
        return

    print(f"\n=== 闭环控制测试: 移动到 {item_name} ===\n")

    data = read_realtime_products()
    if data is None:
        print("[!] 无法读取扫描数据")
        return

    target = find_item_by_name(item_name, data, exclude_held=True)
    if target is None:
        print(f"[!] 未找到: {item_name}")
        return

    # 计算第一个切割位置和目标旋转角度
    cut_positions = calculate_cut_positions(target, cut_num=1)
    if cut_positions:
        target_x, target_z, target_rot_y = cut_positions[0]
        print(f"[*] 目标位置: ({target_x:.3f}, {target_z:.3f}), 目标Y旋转={target_rot_y:.1f}°")
    else:
        target_pos = get_position(target)
        target_x, target_z = target_pos[0], target_pos[2]
        target_rot_y = None
        print(f"[*] 目标位置: ({target_x:.3f}, {target_z:.3f})")

    _activate_window(GAME_WINDOW_TITLE)
    time.sleep(0.3)

    success = move_knife_to_target_closed_loop(target_x, target_z, target_rot_y, verbose=True)

    if success:
        print("\n[+] 闭环控制成功到达目标位置!")
    else:
        print("\n[!] 闭环控制未能到达目标位置")


# ================================================================
# 主程序入口
# ================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  Cooking Simulator 自动切割模块 v3.0")
    print("  (闭环反馈控制)")
    print("=" * 50)
    print()
    print("可用命令:")
    print("  python auto_cutting.py list                    - 列出可见物品")
    print("  python auto_cutting.py info <物品名>           - 显示详细信息")
    print("  python auto_cutting.py test <物品名>           - 测试闭环移动")
    print("  python auto_cutting.py cut <物品名> <刀数>     - 切割物品")
    print()
    print("示例:")
    print("  python auto_cutting.py info lemon")
    print("  python auto_cutting.py test lemon")
    print("  python auto_cutting.py cut lemon 3")
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
                show_knife_and_target(sys.argv[2])
            else:
                print("[!] 用法: python auto_cutting.py info <物品名>")

        elif cmd == "test":
            if len(sys.argv) >= 3:
                test_closed_loop(sys.argv[2])
            else:
                print("[!] 用法: python auto_cutting.py test <物品名>")

        elif cmd == "cut":
            if len(sys.argv) >= 4:
                item_name = sys.argv[2]
                try:
                    cut_count = int(sys.argv[3])
                except ValueError:
                    print(f"[!] 无效的切割次数: {sys.argv[3]}")
                    sys.exit(1)
                auto_cut(item_name=item_name, cut_count=cut_count)
            elif len(sys.argv) >= 3:
                item_name = sys.argv[2]
                auto_cut(item_name=item_name, cut_count=DEFAULT_CUT_COUNT)
            else:
                print("[!] 用法: python auto_cutting.py cut <物品名> <刀数>")

        else:
            print(f"[!] 未知命令: {cmd}")

# -*- coding: utf-8 -*-
"""
实时交互检测器 - 读取游戏中玩家当前指向的物品信息

支持两种模式：
1. UDP模式（推荐）：低延迟，实时性好
2. 文件模式（备用）：兼容性好

使用方法：
1. 在游戏中按 Alt+J 开启实时交互检测
2. 运行此脚本或导入模块使用
"""
import os
import time
import socket
import threading
from datetime import datetime
from typing import Optional, NamedTuple, Callable

# ============== 配置 ==============
UDP_PORT = 52525
UDP_RECVFROM_BUFFER_SIZE = 65535  # max safe UDP datagram read size; avoids WinError 10040
UDP_SOCKET_RCVBUF_SIZE = 1024 * 1024  # enlarge socket receive buffer for large/high-frequency packets  # UDP端口（需与C# Mod一致）
try:
    from epm.cerebellum.skills._shared_paths import userdata_root

    _UD = str(userdata_root())
except Exception:  # pragma: no cover
    # Neutral fallback: point COOKGAME_USERDATA_ROOT at your own game UserData folder.
    _env_ud = os.environ.get("COOKGAME_USERDATA_ROOT", "").strip()
    _UD = (
        _env_ud
        if _env_ud
        else os.path.join(os.path.expanduser("~"), "CookingSimulator", "UserData")
    )

INTERACTION_INFO_PATH = os.path.join(_UD, "realtime_interaction_info.txt")


def _autodetect_interaction_info_path(configured: str) -> str:
    """
    自动探测 realtime_interaction_info.txt 的真实路径。

    说明：不同机器/磁盘安装位置不同（例如 D:/F: 等盘符）；若路径不对，UDP 未开启或未收到包时，文件回退会失败。
    """
    candidates = []

    env_userdata = os.environ.get("COOKSIM_USERDATA", "").strip()
    if env_userdata:
        candidates.append(os.path.join(env_userdata, "realtime_interaction_info.txt"))

    if configured:
        candidates.append(configured)

    # 常见 Steam 安装目录
    candidates.append(r"C:\Program Files (x86)\Steam\steamapps\common\CookingSimulator\UserData\realtime_interaction_info.txt")
    candidates.append(r"C:\Program Files\Steam\steamapps\common\CookingSimulator\UserData\realtime_interaction_info.txt")

    # 尝试在常见盘符下探测（避免硬编码盘符）
    suffix = r"Software\Steam\steamapps\common\CookingSimulator\UserData\realtime_interaction_info.txt"
    for drive in ("D:", "E:", "F:", "G:", "C:"):
        candidates.append(os.path.join(drive + "\\", suffix))

    for p in candidates:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            continue

    return configured


INTERACTION_INFO_PATH = _autodetect_interaction_info_path(INTERACTION_INFO_PATH)

# ============== 数据结构 ==============
class InteractionInfo(NamedTuple):
    """交互信息数据结构"""
    has_target: bool
    item_name: str
    action: str
    weight: str
    pour_amount: str        # 倾倒量 (如 "42 ml") - 从瓶子倒出的量（较大值）
    overflow_amount: str    # 溢出量 (如 "5 ml") - 从容器溢出的量（较小值）
    container_name: str     # 容器名称 (如 "Paella Pan")
    container_contents: str # 容器内容 (如 "Wine:113 ml,Water:50 ml")
    timestamp: str


def _parse_content(content: str) -> Optional[InteractionInfo]:
    """解析交互信息内容"""
    try:
        lines = content.strip().split('\n')
        data = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()

        return InteractionInfo(
            has_target=data.get('HasTarget', 'False') == 'True',
            item_name=data.get('ItemName', ''),
            action=data.get('Action', ''),
            weight=data.get('Weight', ''),
            pour_amount=data.get('PourAmount', ''),
            overflow_amount=data.get('OverflowAmount', ''),
            container_name=data.get('ContainerName', ''),
            container_contents=data.get('ContainerContents', ''),
            timestamp=data.get('Timestamp', '')
        )
    except Exception:
        return None


# ============== 文件模式（保留兼容） ==============
def read_interaction_info_file() -> Optional[InteractionInfo]:
    """
    从文件读取当前交互信息（文件模式）

    Returns:
        InteractionInfo 或 None
    """
    if not os.path.exists(INTERACTION_INFO_PATH):
        return None

    try:
        with open(INTERACTION_INFO_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        return _parse_content(content)
    except Exception as e:
        print(f"读取交互信息失败: {e}")
        return None


# ============== UDP模式（推荐） ==============
class UDPInteractionReceiver:
    """UDP交互信息接收器（低延迟）"""

    def __init__(self, port: int = UDP_PORT):
        self.port = port
        self.sock = None
        self.running = False
        self.thread = None
        self.latest_info: Optional[InteractionInfo] = None
        # Cache the latest "pouring" info to avoid being overwritten by interleaved
        # non-pouring packets (which often have empty PourAmount).
        self.latest_pour_info: Optional[InteractionInfo] = None
        self._latest_received_at: Optional[float] = None
        self._latest_pour_received_at: Optional[float] = None
        self.callback: Optional[Callable[[InteractionInfo], None]] = None
        self._lock = threading.Lock()

    def start(self, callback: Optional[Callable[[InteractionInfo], None]] = None):
        """启动UDP接收器"""
        if self.running:
            return

        self.callback = callback
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, UDP_SOCKET_RCVBUF_SIZE)
        except Exception:
            pass
        self.sock.bind(('127.0.0.1', self.port))
        self.sock.settimeout(0.5)  # 设置超时以便能够停止

        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"✅ UDP接收器已启动 (端口: {self.port})")

    def stop(self):
        """停止UDP接收器"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.sock:
            self.sock.close()
            self.sock = None
        print("⏹ UDP接收器已停止")

    def _receive_loop(self):
        """接收循环"""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(UDP_RECVFROM_BUFFER_SIZE)
                content = data.decode('utf-8')
                info = _parse_content(content)
                if info:
                    with self._lock:
                        self.latest_info = info
                        self._latest_received_at = time.time()
                        action_l = (info.action or "").strip().lower()
                        pour_amount = (info.pour_amount or "").strip()
                        if pour_amount or ("pour" in action_l):
                            self.latest_pour_info = info
                            self._latest_pour_received_at = time.time()
                    if self.callback:
                        self.callback(info)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"UDP接收错误: {e}")

    def get_latest(self) -> Optional[InteractionInfo]:
        """获取最新的交互信息"""
        with self._lock:
            return self.latest_info

    def get_latest_pour(self, max_age_s: float = 0.6) -> Optional[InteractionInfo]:
        """
        获取最新的倾倒信息（优先返回带 PourAmount 的包）。

        Args:
            max_age_s: 允许的缓存最大年龄（秒），避免返回过旧的倾倒数据
        """
        now = time.time()
        with self._lock:
            if not self.latest_pour_info or not self._latest_pour_received_at:
                return None
            if now - self._latest_pour_received_at > float(max_age_s):
                return None
            return self.latest_pour_info

    def get_debug_snapshot(self) -> dict:
        """返回用于调试的快照（不保证字段齐全，尽量不抛异常）"""
        with self._lock:
            latest = self.latest_info
            latest_pour = self.latest_pour_info
            return {
                "running": bool(self.running),
                "latest": latest._asdict() if latest else None,
                "latest_pour": latest_pour._asdict() if latest_pour else None,
                "latest_age_s": (
                    (time.time() - float(self._latest_received_at))
                    if self._latest_received_at
                    else None
                ),
                "latest_pour_age_s": (
                    (time.time() - float(self._latest_pour_received_at))
                    if self._latest_pour_received_at
                    else None
                ),
            }


# ============== 统一接口 ==============
# 全局UDP接收器实例
_udp_receiver: Optional[UDPInteractionReceiver] = None


def init_udp_mode(callback: Optional[Callable[[InteractionInfo], None]] = None):
    """
    初始化UDP模式

    Args:
        callback: 可选的回调函数，当收到新数据时调用
    """
    global _udp_receiver
    if _udp_receiver is None:
        _udp_receiver = UDPInteractionReceiver()
    _udp_receiver.start(callback)


def stop_udp_mode():
    """停止UDP模式"""
    global _udp_receiver
    if _udp_receiver:
        _udp_receiver.stop()


def read_interaction_info(use_udp: bool = True) -> Optional[InteractionInfo]:
    """
    读取当前交互信息

    Args:
        use_udp: 是否优先使用UDP模式（默认True）

    Returns:
        InteractionInfo 或 None
    """
    global _udp_receiver

    # 优先使用UDP模式
    if use_udp and _udp_receiver and _udp_receiver.running:
        info = _udp_receiver.get_latest()
        if info:
            return info

    # 回退到文件模式
    return read_interaction_info_file()


def read_interaction_info_prefer_pour(use_udp: bool = True, max_age_s: float = 0.6) -> Optional[InteractionInfo]:
    """
    读取当前交互信息，但在 UDP 模式下优先返回“倾倒信息”（PourAmount 非空或 Action 包含 Pour）。

    目的：避免倾倒 UI 的数据被交错出现的非倾倒 UDP 包覆盖，导致读取 PourAmount 为空。
    """
    global _udp_receiver

    if use_udp and _udp_receiver and _udp_receiver.running:
        info = _udp_receiver.get_latest_pour(max_age_s=max_age_s)
        if info:
            return info
        info = _udp_receiver.get_latest()
        if info:
            return info

    return read_interaction_info_file()


def get_udp_debug_snapshot() -> Optional[dict]:
    """获取 UDP 接收器调试快照；未初始化/未运行时返回 None。"""
    global _udp_receiver
    try:
        if _udp_receiver and _udp_receiver.running:
            return _udp_receiver.get_debug_snapshot()
    except Exception:
        return None
    return None


def get_current_target(use_udp: bool = True) -> Optional[str]:
    """
    获取当前指向的物品名称

    Args:
        use_udp: 是否优先使用UDP模式

    Returns:
        物品名称字符串，或 None
    """
    info = read_interaction_info(use_udp)
    if info and info.has_target and info.item_name:
        return info.item_name
    return None


def is_pointing_at(item_name: str, fuzzy: bool = True, use_udp: bool = True) -> bool:
    """
    检查是否正在指向指定物品

    Args:
        item_name: 要检查的物品名称
        fuzzy: 是否模糊匹配（不区分大小写，包含即可）
        use_udp: 是否优先使用UDP模式

    Returns:
        是否指向该物品
    """
    current = get_current_target(use_udp)
    if not current:
        return False

    if fuzzy:
        return item_name.lower() in current.lower()
    else:
        return item_name == current


def wait_for_target(item_name: str, timeout: float = 10.0, fuzzy: bool = True, use_udp: bool = True) -> bool:
    """
    等待直到指向指定物品

    Args:
        item_name: 目标物品名称
        timeout: 超时时间（秒）
        fuzzy: 是否模糊匹配
        use_udp: 是否优先使用UDP模式

    Returns:
        是否成功指向目标物品
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_pointing_at(item_name, fuzzy, use_udp):
            return True
        time.sleep(0.05)  # UDP模式可以用更短的间隔
    return False


def monitor_interaction(callback: Optional[Callable[[InteractionInfo], None]] = None,
                        use_udp: bool = True,
                        interval: float = 0.1,
                        realtime_display: bool = False):
    """
    持续监控交互状态（阻塞式）

    Args:
        callback: 当交互状态变化时调用的回调函数
        use_udp: 是否使用UDP模式
        interval: 检查间隔（秒），仅文件模式使用
        realtime_display: 是否实时刷新显示（同一行更新）
    """
    import sys

    last_item = ""
    mode_str = "UDP" if use_udp else "文件"
    display_str = "实时刷新" if realtime_display else "逐行"
    print(f"开始监控交互状态 ({mode_str}模式, {display_str}显示)... (Ctrl+C 停止)")
    print()

    def display_info(info: InteractionInfo, changed: bool):
        """显示信息"""
        nonlocal last_item
        current_item = info.item_name if info.has_target else ""

        # 构建详细信息字符串
        details = []
        if info.weight:
            details.append(info.weight)
        if info.container_name and info.container_name != current_item:
            details.append(f"容器:{info.container_name}")
        if info.pour_amount:
            details.append(f"倾倒:{info.pour_amount}")
        if info.overflow_amount:
            details.append(f"溢出:{info.overflow_amount}")
        if info.container_contents:
            details.append(f"内容:{info.container_contents}")
        if info.action:
            details.append(f"动作:{info.action}")
        detail_str = " | ".join(details) if details else ""

        if realtime_display:
            # 实时刷新模式：同一行更新
            if current_item:
                text = f"指向: {current_item:<25} {detail_str:<50}"
            else:
                text = f"指向: (无)                                                              "
            sys.stdout.write(f"\r{text}")
            sys.stdout.flush()
        else:
            # 逐行模式：只在变化时打印（倾倒时始终打印）
            if changed or info.pour_amount:
                if current_item:
                    print(f"[{info.timestamp}] 指向: {current_item} ({detail_str})")
                else:
                    print(f"[{info.timestamp}] 未指向任何物品")

    if use_udp:
        # UDP模式：使用回调
        def on_receive(info: InteractionInfo):
            nonlocal last_item
            current_item = info.item_name if info.has_target else ""
            changed = current_item != last_item
            display_info(info, changed)
            if changed and callback:
                callback(info)
            last_item = current_item

        init_udp_mode(on_receive)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_udp_mode()
            print("\n监控已停止")
    else:
        # 文件模式：轮询
        try:
            while True:
                info = read_interaction_info_file()
                if info:
                    current_item = info.item_name if info.has_target else ""
                    changed = current_item != last_item
                    display_info(info, changed)
                    if changed and callback:
                        callback(info)
                    last_item = current_item
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n监控已停止")


# ============== 测试代码 ==============
if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("  实时交互检测器测试")
    print("=" * 50)

    # 检查命令行参数
    use_udp = "--file" not in sys.argv
    realtime = "--realtime" in sys.argv or "-r" in sys.argv

    mode_str = "UDP" if use_udp else "文件"
    display_str = "实时刷新" if realtime else "逐行"

    print(f"通信模式: {mode_str}")
    print(f"显示模式: {display_str}")
    print(f"UDP端口: {UDP_PORT}")
    print()
    print("参数说明:")
    print("  --file     使用文件模式（默认UDP）")
    print("  --realtime 或 -r  实时刷新显示（同一行更新）")
    print()
    print("开始持续监控...")
    monitor_interaction(use_udp=use_udp, realtime_display=realtime)

# -*- coding: utf-8 -*-
"""
实时雷达数据读取器
读取C# mod生成的realtime_radar_scan.txt文件
基于Unity Physics.Raycast的360度实时检测，100%准确
"""
import os
import re
from typing import Optional, List, Tuple
from dataclasses import dataclass

try:
    from .config import DATA_ROOT  # EPM-adapted default
except Exception:  # pragma: no cover
    # Neutral fallback: point COOKGAME_USERDATA_ROOT at your own game UserData folder.
    DATA_ROOT = os.environ.get("COOKGAME_USERDATA_ROOT", "").strip() or os.path.join(
        os.path.expanduser("~"), "CookingSimulator", "UserData"
    )


@dataclass
class RealtimeRadarReading:
    """实时雷达读数"""
    direction_name: str
    angle: float
    distance: Optional[float]  # None表示无障碍，>=0表示距离
    obstacle_name: str

    def is_clear(self, safe_distance: float = 0.5) -> bool:
        """检查该方向是否安全"""
        return self.distance is None or self.distance >= safe_distance


class RealtimeRadarReader:
    """
    实时雷达数据读取器

    读取C# mod生成的realtime_radar_scan.txt文件
    该文件包含基于Unity Physics.Raycast的360度实时检测结果
    """

    def __init__(self, game_userdata_path: str = None, radar_file_path: str = None):
        """
        初始化读取器

        Args:
            game_userdata_path: 游戏UserData目录路径，如果为None则自动查找
        """
        if radar_file_path:
            self.radar_file_path = str(radar_file_path)
        else:
            if game_userdata_path is None:
                game_userdata_path = DATA_ROOT
            self.radar_file_path = os.path.join(game_userdata_path, "realtime_radar_scan.txt")

    def read_radar_data(self) -> Optional[Tuple[List[RealtimeRadarReading], dict]]:
        """
        读取实时雷达数据

        Returns:
            (readings, metadata): 雷达读数列表和元数据字典，如果文件不存在或解析失败则返回None
        """
        if not os.path.exists(self.radar_file_path):
            return None

        try:
            with open(self.radar_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析元数据
            metadata = {}
            for line in content.split('\n'):
                if line.startswith('# PlayerPosition:'):
                    pos_str = line.split(':', 1)[1].strip()
                    pos_parts = [float(x.strip()) for x in pos_str.split(',')]
                    metadata['player_position'] = tuple(pos_parts)
                elif line.startswith('# PlayerYaw:'):
                    metadata['player_yaw'] = float(line.split(':', 1)[1].strip())
                elif line.startswith('# DetectionRange:'):
                    metadata['detection_range'] = float(line.split(':', 1)[1].strip())
                elif line.startswith('# Timestamp:'):
                    metadata['timestamp'] = line.split(':', 1)[1].strip()

            # 解析雷达读数
            readings = []
            # 匹配格式：[正北 ↑] Angle=0, Distance=None, Obstacle=None
            pattern = r'\[(.+?)\]\s+Angle=([\d.]+),\s+Distance=(\S+),\s+Obstacle=(.+)'

            for match in re.finditer(pattern, content):
                dir_name = match.group(1)
                angle = float(match.group(2))
                distance_str = match.group(3)
                obstacle_name = match.group(4)

                # 解析距离
                if distance_str == 'None':
                    distance = None
                else:
                    try:
                        distance = float(distance_str)
                    except:
                        distance = None

                reading = RealtimeRadarReading(
                    direction_name=dir_name,
                    angle=angle,
                    distance=distance,
                    obstacle_name=obstacle_name
                )
                readings.append(reading)

            return (readings, metadata)

        except Exception as e:
            print(f"❌ 读取实时雷达数据失败: {e}")
            return None

    def get_front_clearance(
        self,
        player_yaw: float,
        safe_distance: float = 0.5,
        check_angle_range: float = 135.0
    ) -> bool:
        """
        检查正前方是否安全（使用扇形区域检测）

        Args:
            player_yaw: 玩家朝向（度）
            safe_distance: 安全距离阈值（米）
            check_angle_range: 前方检查的角度范围（度），默认135度（±67.5°，覆盖前、前左、前右3个方向）

        Returns:
            bool: 前方是否安全
        """
        data = self.read_radar_data()
        if data is None:
            return True  # 如果无数据，假设安全

        readings, metadata = data

        # 如果readings为空，假设安全
        if not readings:
            return True

        player_yaw = player_yaw % 360

        # 计算角度差的辅助函数
        def angle_diff(d1, d2):
            diff = abs(d1 - d2)
            return min(diff, 360 - diff)

        # 找出前方扇区范围内的所有方向（±67.5°，覆盖3个雷达方向）
        front_directions = [
            r for r in readings
            if angle_diff(r.angle, player_yaw) <= check_angle_range / 2
        ]

        if not front_directions:
            # 如果没有找到前方方向，使用最接近的方向
            front_reading = min(readings, key=lambda r: angle_diff(r.angle, player_yaw))
            return front_reading.is_clear(safe_distance)

        # 在前方扇区内找最小距离（最保守策略）
        min_distance = float('inf')
        closest_obstacle = None

        for reading in front_directions:
            if reading.distance is not None:
                if reading.distance < min_distance:
                    min_distance = reading.distance
                    closest_obstacle = reading

        # 如果前方扇区内都没有障碍物
        if min_distance == float('inf'):
            return True

        # 检查最小距离是否安全
        return min_distance >= safe_distance

    def get_safest_direction(
        self,
        target_direction: Optional[float] = None,
        safe_distance: float = 0.5
    ) -> Optional[float]:
        """
        获取最安全的移动方向

        Args:
            target_direction: 目标方向（度，世界坐标系），如果为None则返回距离最远的方向
            safe_distance: 安全距离阈值（米）

        Returns:
            float: 最安全方向的角度（世界坐标系），如果所有方向都不安全则返回None
        """
        data = self.read_radar_data()
        if data is None:
            return None

        readings, metadata = data

        # 如果readings为空，返回None
        if not readings:
            return None

        # 筛选安全方向
        safe_directions = [r for r in readings if r.is_clear(safe_distance)]

        if not safe_directions:
            return None

        if target_direction is None:
            # 返回距离最远的方向
            return max(safe_directions, key=lambda r: r.distance or float('inf')).angle

        # 返回与目标方向最接近的安全方向
        target_direction = target_direction % 360

        def angle_diff(d1, d2):
            diff = abs(d1 - d2)
            return min(diff, 360 - diff)

        closest = min(safe_directions, key=lambda r: angle_diff(r.angle, target_direction))
        return closest.angle

    def get_all_readings(self) -> Optional[List[RealtimeRadarReading]]:
        """获取所有方向的雷达读数"""
        data = self.read_radar_data()
        if data is None:
            return None
        return data[0]


# 测试代码
if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass

    print("=" * 80)
    print("实时雷达数据读取测试")
    print("=" * 80)

    reader = RealtimeRadarReader()

    print(f"\n雷达数据文件: {reader.radar_file_path}")

    if not os.path.exists(reader.radar_file_path):
        print("\n⚠️ 雷达数据文件不存在")
        print("   请确保：")
        print("   1. 游戏已启动")
        print("   2. 在游戏中按 F11 开启实时雷达扫描")
        sys.exit(1)

    data = reader.read_radar_data()
    if data is None:
        print("\n❌ 读取雷达数据失败")
        sys.exit(1)

    readings, metadata = data

    print("\n【元数据】")
    print(f"  时间戳: {metadata.get('timestamp', 'N/A')}")
    print(f"  玩家位置: {metadata.get('player_position', 'N/A')}")
    print(f"  玩家朝向: {metadata.get('player_yaw', 'N/A')}°")
    print(f"  检测范围: {metadata.get('detection_range', 'N/A')}m")

    print("\n【雷达读数】")
    print(f"{'方向':<15} {'角度':<8} {'距离':<12} {'障碍物':<30}")
    print("-" * 80)

    for reading in readings:
        distance_str = f"{reading.distance:.3f}m" if reading.distance is not None else "None"
        safe_str = "✅" if reading.is_clear(0.5) else "❌"
        print(f"{reading.direction_name:<15} {reading.angle:<8.0f} {distance_str:<12} {reading.obstacle_name:<30} {safe_str}")

    # 测试前方安全检测
    player_yaw = metadata.get('player_yaw', 0)
    print(f"\n【前方安全检测】")
    is_front_clear = reader.get_front_clearance(player_yaw, safe_distance=0.5, check_angle_range=150.0)
    print(f"  玩家朝向: {player_yaw:.1f}°")
    print(f"  前方安全: {'✅ 是' if is_front_clear else '❌ 否'}")

    # 测试最安全方向
    print(f"\n【最安全方向】")
    safest_dir = reader.get_safest_direction(target_direction=None, safe_distance=0.5)
    if safest_dir is not None:
        print(f"  最安全方向: {safest_dir:.0f}°")
    else:
        print(f"  ❌ 周围都有障碍物，无安全方向")

    print("\n" + "=" * 80)

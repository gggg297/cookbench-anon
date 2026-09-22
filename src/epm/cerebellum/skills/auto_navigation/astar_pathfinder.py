# -*- coding: utf-8 -*-
"""
A*路径规划器 - 基于障碍物地图的最优路径查找
"""
import math
import heapq
import json
import os
import io
from collections import deque
from contextlib import redirect_stdout
from typing import Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field


@dataclass
class PathNode:
    """A*算法的路径节点"""
    grid_x: int
    grid_z: int
    g_cost: float = float('inf')  # 从起点到当前点的实际代价
    h_cost: float = 0.0            # 从当前点到终点的启发式代价
    f_cost: float = float('inf')  # f = g + h
    parent: Optional['PathNode'] = None

    def __lt__(self, other):
        """Heap ordering for A* open-set nodes."""
        if not isinstance(other, PathNode):
            print(
                "[A*][WARN] invalid_heap_compare "
                f"self=PathNode(grid=({self.grid_x},{self.grid_z}), g={self.g_cost:.5f}, "
                f"h={self.h_cost:.5f}, f={self.f_cost:.5f}) "
                f"other_type={type(other).__name__} other_repr={other!r}"
            )
            raise TypeError(
                f"[A*] invalid_heap_compare other_type={type(other).__name__}"
            )
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        """节点相等判断"""
        if not isinstance(other, PathNode):
            return False
        return self.grid_x == other.grid_x and self.grid_z == other.grid_z

    def __hash__(self):
        """哈希支持"""
        return hash((self.grid_x, self.grid_z))


class AStarPathfinder:
    """A*路径规划器"""
    _MAP_CACHE: dict[str, dict[str, Any]] = {}

    def __init__(self, map_file: str = None):
        """
        初始化路径规划器

        Args:
            map_file: 障碍物地图JSON文件路径
        """
        self.grid_size = 0.02  # 栅格大小（默认2cm，会从地图文件中覆盖）
        self.obstacle_grid: Set[Tuple[int, int]] = set()  # 障碍物栅格集合
        self.map_loaded = False
        self.player_clearance_pos: Optional[Tuple[float, float]] = None  # 玩家位置（用于50cm清除半径）
        self.inner_obstacle_rects: List[dict] = []  # 除外墙外的内部障碍矩形包围盒
        self.last_path_debug: dict[str, Any] = {}
        self.last_docking_debug: dict[str, Any] = {}
        self._is_obstacle_calls: int = 0
        self._get_neighbors_calls: int = 0

        if map_file:
            self.load_map(map_file)

    def _assert_valid_instance(self, *, caller: str) -> None:
        if isinstance(self, AStarPathfinder):
            return
        raise TypeError(
            "[A*] invalid_pathfinder_instance:"
            f" caller={caller}"
            f" self_type={type(self).__name__}"
            f" self_repr={self!r}. "
            "Expected an AStarPathfinder instance. "
            "Likely causes: using `AStarPathfinder` instead of `AStarPathfinder()`, "
            "overwriting `self.pathfinder`, or calling an unbound instance method incorrectly."
        )

    def _repair_open_heap(self, open_set: list, open_dict: dict, *, context: str) -> int:
        """
        Best-effort repair for corrupted A* open heap.
        If non-PathNode values appear, rebuild the heap from valid open_dict nodes.
        """
        invalid_heap = [item for item in open_set if not isinstance(item, PathNode)]
        invalid_dict_keys = [grid for grid, node in list(open_dict.items()) if not isinstance(node, PathNode)]
        invalid_count = len(invalid_heap) + len(invalid_dict_keys)
        if invalid_count <= 0:
            return 0

        print(f"[A*][WARN] Heap corruption detected at {context}: invalid_heap={len(invalid_heap)} invalid_dict={len(invalid_dict_keys)}")
        for idx, item in enumerate(invalid_heap[:5], start=1):
            print(f"  [A*][WARN] invalid_heap[{idx}]={item!r} type={type(item).__name__}")
        for idx, grid in enumerate(invalid_dict_keys[:5], start=1):
            node = open_dict.get(grid)
            print(f"  [A*][WARN] invalid_dict[{idx}] grid={grid!r} value={node!r} type={type(node).__name__}")
        self._heap_debug_snapshot(open_set, open_dict, context=f"repair:{context}")

        valid_nodes = [node for node in open_dict.values() if isinstance(node, PathNode)]
        for grid in invalid_dict_keys:
            open_dict.pop(grid, None)
        open_set[:] = valid_nodes
        try:
            heapq.heapify(open_set)
        except Exception as e:
            raise RuntimeError(
                f"[A*] heap_rebuild_failed context={context} "
                f"valid_nodes={len(valid_nodes)} error={type(e).__name__}:{e}"
            ) from e
        print(f"[A*][WARN] Heap rebuilt from valid open_dict nodes at {context}: remaining_nodes={len(open_set)}")
        return invalid_count

    def _format_heap_item(self, item: Any) -> str:
        if isinstance(item, PathNode):
            parent = None
            if isinstance(item.parent, PathNode):
                parent = (item.parent.grid_x, item.parent.grid_z)
            return (
                f"PathNode(grid=({item.grid_x},{item.grid_z}), "
                f"g={item.g_cost:.5f}, h={item.h_cost:.5f}, f={item.f_cost:.5f}, "
                f"parent={parent})"
            )
        return f"{type(item).__name__}:{item!r}"

    def _heap_debug_snapshot(
        self,
        open_set: list,
        open_dict: dict,
        *,
        context: str,
        current_grid: Optional[Tuple[int, int]] = None,
        goal_grid: Optional[Tuple[int, int]] = None,
        limit: int = 5,
    ) -> None:
        try:
            valid_open_set = sum(1 for item in open_set if isinstance(item, PathNode))
            valid_open_dict = sum(1 for node in open_dict.values() if isinstance(node, PathNode))
            print(
                f"[A*][DEBUG] heap_snapshot context={context} "
                f"open_set_len={len(open_set)} valid_open_set={valid_open_set} "
                f"open_dict_len={len(open_dict)} valid_open_dict={valid_open_dict} "
                f"current_grid={current_grid} goal_grid={goal_grid}"
            )
            for idx, item in enumerate(list(open_set)[:limit], start=1):
                print(f"  [A*][DEBUG] open_set[{idx}]={self._format_heap_item(item)}")
            for idx, (grid, node) in enumerate(list(open_dict.items())[:limit], start=1):
                print(f"  [A*][DEBUG] open_dict[{idx}] grid={grid} value={self._format_heap_item(node)}")
        except Exception as e:
            print(f"[A*][WARN] heap_snapshot_failed context={context} error={type(e).__name__}:{e}")

    def _safe_heapify_open_set(self, open_set: list, open_dict: dict, *, context: str) -> int:
        """
        Heapify with one repair retry if invalid non-PathNode values slipped into open_set.
        """
        valid_open_set = sum(1 for item in open_set if isinstance(item, PathNode))
        valid_open_dict = sum(1 for node in open_dict.values() if isinstance(node, PathNode))
        print(
            f"[A*][DEBUG] heapify_enter context={context} "
            f"open_set_len={len(open_set)} valid_open_set={valid_open_set} "
            f"open_dict_len={len(open_dict)} valid_open_dict={valid_open_dict}"
        )
        repaired = self._repair_open_heap(open_set, open_dict, context=f"{context}:precheck")
        try:
            heapq.heapify(open_set)
            return repaired
        except Exception as e:
            self._heap_debug_snapshot(open_set, open_dict, context=f"heapify_exception:{context}")
            print(f"[A*][WARN] heapify failed at {context}: {e!r}; trying repair/rebuild once")
            repaired += self._repair_open_heap(open_set, open_dict, context=f"{context}:exception")
            heapq.heapify(open_set)
            return repaired

    def _safe_heappush_open_set(self, open_set: list, node: PathNode, *, context: str) -> None:
        if not isinstance(node, PathNode):
            raise TypeError(
                f"[A*] invalid_heappush_node context={context} node_type={type(node).__name__}"
            )
        try:
            heapq.heappush(open_set, node)
        except Exception as e:
            self._heap_debug_snapshot(open_set, {}, context=f"heappush_failed:{context}")
            raise RuntimeError(
                f"[A*] heappush_failed context={context} "
                f"node=({node.grid_x},{node.grid_z}) error={type(e).__name__}:{e}"
            ) from e

    def _safe_heappop_open_set(
        self,
        open_set: list,
        *,
        context: str,
        open_dict: Optional[dict] = None,
        current_grid: Optional[Tuple[int, int]] = None,
        goal_grid: Optional[Tuple[int, int]] = None,
    ) -> PathNode:
        try:
            current = heapq.heappop(open_set)
        except Exception as e:
            self._heap_debug_snapshot(
                open_set,
                open_dict or {},
                context=f"heappop_failed:{context}",
                current_grid=current_grid,
                goal_grid=goal_grid,
            )
            raise RuntimeError(
                f"[A*] heappop_failed context={context} error={type(e).__name__}:{e}"
            ) from e
        if not isinstance(current, PathNode):
            self._heap_debug_snapshot(
                open_set,
                open_dict or {},
                context=f"invalid_heappop_node:{context}",
                current_grid=current_grid,
                goal_grid=goal_grid,
            )
            raise TypeError(
                f"[A*] invalid_heappop_node context={context} node_type={type(current).__name__}"
            )
        return current

    def load_map(self, map_file: str) -> bool:
        """
        加载障碍物地图

        Args:
            map_file: 地图JSON文件路径

        Returns:
            是否加载成功
        """
        try:
            map_path = os.path.abspath(map_file)
            stat = os.stat(map_path)
            cached = self._MAP_CACHE.get(map_path)
            if (
                isinstance(cached, dict)
                and cached.get("mtime_ns") == int(stat.st_mtime_ns)
                and cached.get("size") == int(stat.st_size)
            ):
                self.grid_size = float(cached["grid_size"])
                self.obstacle_grid = cached["obstacle_grid"]
                self.inner_obstacle_rects = cached["inner_obstacle_rects"]
                self.map_loaded = True
                print(f"[地图加载] 使用缓存地图: {map_path}")
                print(f"[地图加载] 成功加载 {len(self.obstacle_grid)} 个障碍物栅格")
                print(f"[地图加载] 栅格大小: {self.grid_size}m")
                if self.inner_obstacle_rects:
                    print(f"[地图加载] 检测到 {len(self.inner_obstacle_rects)} 个内部障碍矩形")
                return True

            with open(map_path, 'r', encoding='utf-8') as f:
                map_data = json.load(f)

            self.grid_size = map_data['metadata'].get('grid_size', 0.2)
            obstacle_grid: Set[Tuple[int, int]] = set()

            # 加载障碍物栅格
            for cell in map_data['grid']:
                if cell['is_obstacle']:
                    obstacle_grid.add((cell['x'], cell['z']))

            self.obstacle_grid = obstacle_grid
            self.inner_obstacle_rects = self._extract_inner_obstacle_rects()
            self.map_loaded = True
            self._MAP_CACHE[map_path] = {
                "mtime_ns": int(stat.st_mtime_ns),
                "size": int(stat.st_size),
                "grid_size": float(self.grid_size),
                "obstacle_grid": self.obstacle_grid,
                "inner_obstacle_rects": self.inner_obstacle_rects,
            }
            print(f"[地图加载] 成功加载 {len(self.obstacle_grid)} 个障碍物栅格")
            print(f"[地图加载] 栅格大小: {self.grid_size}m")
            if self.inner_obstacle_rects:
                print(f"[地图加载] 检测到 {len(self.inner_obstacle_rects)} 个内部障碍矩形")
            return True

        except Exception as e:
            print(f"[地图加载] 失败: {e}")
            return False

    def _extract_inner_obstacle_rects(self, min_cells: int = 200) -> List[dict]:
        """提取除最大外墙连通域外的内部障碍物矩形包围盒。"""
        if not self.obstacle_grid:
            return []

        remaining = set(self.obstacle_grid)
        components: List[List[Tuple[int, int]]] = []

        while remaining:
            start = next(iter(remaining))
            queue = deque([start])
            remaining.remove(start)
            comp: List[Tuple[int, int]] = []

            while queue:
                x, z = queue.popleft()
                comp.append((x, z))
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (x + dx, z + dz)
                    if nb in remaining:
                        remaining.remove(nb)
                        queue.append(nb)

            components.append(comp)

        components.sort(key=len, reverse=True)
        rects: List[dict] = []

        for comp in components[1:]:
            if len(comp) < min_cells:
                continue

            xs = [x for x, _ in comp]
            zs = [z for _, z in comp]
            xmin, xmax = min(xs), max(xs)
            zmin, zmax = min(zs), max(zs)
            rects.append({
                "cell_count": len(comp),
                "xmin": xmin,
                "xmax": xmax,
                "zmin": zmin,
                "zmax": zmax,
                "xmin_world": xmin * self.grid_size,
                "xmax_world": (xmax + 1) * self.grid_size,
                "zmin_world": zmin * self.grid_size,
                "zmax_world": (zmax + 1) * self.grid_size,
            })

        rects.sort(key=lambda rect: rect["cell_count"], reverse=True)
        return rects

    def _find_inner_obstacle_rect_for_target(self, target_world_pos: Tuple[float, float]) -> Optional[dict]:
        """若目标落在某个内部障碍物包围盒中，则返回该矩形。"""
        target_gx, target_gz = self.world_to_grid(target_world_pos[0], target_world_pos[1])
        for rect in self.inner_obstacle_rects:
            if rect["xmin"] <= target_gx <= rect["xmax"] and rect["zmin"] <= target_gz <= rect["zmax"]:
                return rect
        return None

    def _calculate_rect_internal_target_docking_pose(
        self,
        target_world_pos: Tuple[float, float],
        rect: dict,
        safe_distance: float,
        strategy: str = "nearest_edge",
        start_world_pos: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """
        目标点落在矩形障碍物内部时，按最近边投影到该边外法线方向生成停靠位姿。
        """
        tx, tz = target_world_pos
        edge_margin = max(0.12, self.grid_size * 3)
        min_outer_offset = 0.30

        edge_distance_map = {
            ("left", tx - rect["xmin_world"]),
            ("right", rect["xmax_world"] - tx),
            ("bottom", tz - rect["zmin_world"]),
            ("top", rect["zmax_world"] - tz),
        }
        edge_distance_items = sorted(edge_distance_map, key=lambda item: item[1])

        def prioritized_edges() -> List[Tuple[str, float]]:
            mode = str(strategy or "nearest_edge").strip().lower()
            if mode in {"ratio", "normalized_ratio", "proportion", "proportional"}:
                cx = (rect["xmin_world"] + rect["xmax_world"]) * 0.5
                cz = (rect["zmin_world"] + rect["zmax_world"]) * 0.5
                hx = max(1e-6, (rect["xmax_world"] - rect["xmin_world"]) * 0.5)
                hz = max(1e-6, (rect["zmax_world"] - rect["zmin_world"]) * 0.5)
                preferred_edges: list[str] = []

                norm_x = abs((tx - cx) / hx)
                norm_z = abs((tz - cz) / hz)
                if norm_x >= norm_z:
                    target_primary = "right" if tx >= cx else "left"
                else:
                    target_primary = "top" if tz >= cz else "bottom"
                if target_primary not in preferred_edges:
                    preferred_edges.append(target_primary)

                ordered: List[Tuple[str, float]] = []
                for edge_name in preferred_edges:
                    ordered.extend(item for item in edge_distance_items if item[0] == edge_name)
                ordered.extend(item for item in edge_distance_items if item[0] not in preferred_edges)
                return ordered

            return edge_distance_items

        def clamp(value: float, low: float, high: float) -> float:
            if low > high:
                return (low + high) * 0.5
            return max(low, min(high, value))

        def try_edge(edge_name: str, edge_dist: float) -> Optional[Tuple[Tuple[float, float], float]]:
            outer_offset = min(safe_distance, max(min_outer_offset, safe_distance - max(0.0, edge_dist)))
            if edge_name in ("left", "right"):
                tangent_low = rect["zmin_world"] + edge_margin
                tangent_high = rect["zmax_world"] - edge_margin
                tangent_axis = "z"
                tangent_target = clamp(tz, tangent_low, tangent_high)
                fixed_value = rect["xmin_world"] - outer_offset if edge_name == "left" else rect["xmax_world"] + outer_offset
            else:
                tangent_low = rect["xmin_world"] + edge_margin
                tangent_high = rect["xmax_world"] - edge_margin
                tangent_axis = "x"
                tangent_target = clamp(tx, tangent_low, tangent_high)
                fixed_value = rect["zmin_world"] - outer_offset if edge_name == "bottom" else rect["zmax_world"] + outer_offset

            offsets = [0.0]
            tangent_span = max(0.0, tangent_high - tangent_low)
            max_offset = tangent_span * 0.5
            step = max(self.grid_size, 0.02)
            cur = step
            while cur <= max_offset + 1e-9:
                offsets.extend([cur, -cur])
                cur += step

            tested = set()
            for offset in offsets:
                tangent_value = clamp(tangent_target + offset, tangent_low, tangent_high)
                tangent_key = round(tangent_value, 4)
                if tangent_key in tested:
                    continue
                tested.add(tangent_key)

                if tangent_axis == "z":
                    ideal_x, ideal_z = fixed_value, tangent_value
                else:
                    ideal_x, ideal_z = tangent_value, fixed_value

                test_gx, test_gz = self.world_to_grid(ideal_x, ideal_z)
                if self.is_obstacle(test_gx, test_gz):
                    continue

                yaw_dx = tx - ideal_x
                yaw_dz = tz - ideal_z
                ideal_yaw = math.degrees(math.atan2(yaw_dx, yaw_dz)) % 360
                actual_distance = math.sqrt(yaw_dx * yaw_dx + yaw_dz * yaw_dz)
                return ((ideal_x, ideal_z), ideal_yaw, outer_offset, actual_distance)

            return None

        for edge_name, edge_dist in prioritized_edges():
            pose = try_edge(edge_name, edge_dist)
            if pose is None:
                continue

            print(
                f"[位姿计算v2] 目标位于内部矩形障碍中，使用最近边特判: "
                f"edge={edge_name}, edge_dist={edge_dist:.3f}m, strategy={strategy}"
            )
            print(
                f"[位姿计算v2] 内部障碍包围盒: "
                f"x=[{rect['xmin_world']:.2f}, {rect['xmax_world']:.2f}], "
                f"z=[{rect['zmin_world']:.2f}, {rect['zmax_world']:.2f}]"
            )
            print(
                f"[位姿计算v2] 特判停靠位姿: "
                f"pos=({pose[0][0]:.3f}, {pose[0][1]:.3f}), yaw={pose[1]:.1f}°, "
                f"outer_offset={pose[2]:.3f}m, target_distance={pose[3]:.3f}m"
            )
            self.last_docking_debug = {
                "mode": "internal_rect_special_case",
                "strategy": str(strategy or "nearest_edge"),
                "start_world_pos": (
                    [round(float(start_world_pos[0]), 4), round(float(start_world_pos[1]), 4)]
                    if start_world_pos is not None
                    else None
                ),
                "target_world_pos": [round(float(tx), 4), round(float(tz), 4)],
                "rect_world_bounds": {
                    "xmin": round(float(rect["xmin_world"]), 4),
                    "xmax": round(float(rect["xmax_world"]), 4),
                    "zmin": round(float(rect["zmin_world"]), 4),
                    "zmax": round(float(rect["zmax_world"]), 4),
                },
                "chosen": {
                    "edge": edge_name,
                    "ideal_world_pos": [round(float(pose[0][0]), 4), round(float(pose[0][1]), 4)],
                    "ideal_yaw": round(float(pose[1]), 4),
                    "outer_offset_m": round(float(pose[2]), 4),
                    "target_distance_m": round(float(pose[3]), 4),
                },
            }
            return (pose[0], pose[1])

        print("[位姿计算v2] 目标虽在内部矩形障碍内，但沿最近边未找到可行停靠点，回退通用逻辑")
        return None

    def _calculate_container_bbox_edge_docking_pose(
        self,
        *,
        target_world_pos: Tuple[float, float],
        container_bounds_min: dict,
        container_bounds_max: dict,
        start_world_pos: Optional[Tuple[float, float]] = None,
        outward_margin_m: float = 0.05,
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """
        对容器内目标，直接在容器 bbox 对应的局部障碍边界上找最近可站位。

        思路：
        1. 只在 container bbox 覆盖的局部地图障碍格中搜索，避免被外围大障碍误导。
        2. 找该局部障碍簇的边界格（相邻4邻域存在自由格）。
        3. 从边界格朝自由格外推 `outward_margin_m` 作为候选站位。
        4. 选择距离目标最近的候选站位，直接朝向目标。
        """
        try:
            xmin = min(float(container_bounds_min.get("x")), float(container_bounds_max.get("x")))
            xmax = max(float(container_bounds_min.get("x")), float(container_bounds_max.get("x")))
            zmin = min(float(container_bounds_min.get("z")), float(container_bounds_max.get("z")))
            zmax = max(float(container_bounds_min.get("z")), float(container_bounds_max.get("z")))
        except Exception:
            return None

        grid_margin_cells = max(2, int(math.ceil(outward_margin_m / self.grid_size)) + 1)
        min_gx = int(math.floor(xmin / self.grid_size)) - grid_margin_cells
        max_gx = int(math.ceil(xmax / self.grid_size)) + grid_margin_cells
        min_gz = int(math.floor(zmin / self.grid_size)) - grid_margin_cells
        max_gz = int(math.ceil(zmax / self.grid_size)) + grid_margin_cells

        local_obstacles: set[Tuple[int, int]] = set()
        for gx in range(min_gx, max_gx + 1):
            for gz in range(min_gz, max_gz + 1):
                if self.is_hard_obstacle(gx, gz):
                    local_obstacles.add((gx, gz))

        if not local_obstacles:
            print("[位姿计算v2] 容器 bbox 特判未命中局部障碍格，回退通用逻辑")
            return None

        target_grid = self.world_to_grid(target_world_pos[0], target_world_pos[1])
        if target_grid in local_obstacles:
            queue = deque([target_grid])
            component = {target_grid}
        else:
            nearest_seed = min(
                local_obstacles,
                key=lambda g: (g[0] - target_grid[0]) ** 2 + (g[1] - target_grid[1]) ** 2,
            )
            queue = deque([nearest_seed])
            component = {nearest_seed}

        while queue:
            gx, gz = queue.popleft()
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (gx + dx, gz + dz)
                if nb in local_obstacles and nb not in component:
                    component.add(nb)
                    queue.append(nb)

        axis_dirs = (
            (-1, 0, "left"),
            (1, 0, "right"),
            (0, -1, "front"),
            (0, 1, "back"),
        )
        candidate_entries: list[tuple[float, float, Tuple[float, float], float, dict[str, Any]]] = []
        candidate_debug: list[dict[str, Any]] = []

        for gx, gz in component:
            obstacle_center = self.grid_to_world(gx, gz)
            for dx, dz, edge_name in axis_dirs:
                free_grid = (gx + dx, gz + dz)
                if self.is_hard_obstacle(*free_grid):
                    continue

                free_center = self.grid_to_world(*free_grid)
                candidate_world = (
                    free_center[0] + dx * outward_margin_m,
                    free_center[1] + dz * outward_margin_m,
                )
                candidate_grid = self.world_to_grid(candidate_world[0], candidate_world[1])
                if self.is_obstacle(*candidate_grid):
                    continue

                yaw_dx = target_world_pos[0] - candidate_world[0]
                yaw_dz = target_world_pos[1] - candidate_world[1]
                target_distance = math.sqrt(yaw_dx * yaw_dx + yaw_dz * yaw_dz)
                ideal_yaw = math.degrees(math.atan2(yaw_dx, yaw_dz)) % 360
                obstacle_distance = math.sqrt(
                    (target_world_pos[0] - obstacle_center[0]) ** 2
                    + (target_world_pos[1] - obstacle_center[1]) ** 2
                )
                lateral_offset = abs(
                    (candidate_world[0] - target_world_pos[0]) * dz
                    - (candidate_world[1] - target_world_pos[1]) * dx
                )
                score = target_distance + lateral_offset * 0.25 + obstacle_distance * 0.05
                info = {
                    "edge": edge_name,
                    "obstacle_grid": [int(gx), int(gz)],
                    "free_grid": [int(free_grid[0]), int(free_grid[1])],
                    "candidate_world_pos": [round(float(candidate_world[0]), 4), round(float(candidate_world[1]), 4)],
                    "candidate_grid": [int(candidate_grid[0]), int(candidate_grid[1])],
                    "candidate_grid_center_world": [
                        round(float(free_center[0]), 4),
                        round(float(free_center[1]), 4),
                    ],
                    "target_distance_m": round(float(target_distance), 4),
                    "obstacle_distance_m": round(float(obstacle_distance), 4),
                    "lateral_offset_m": round(float(lateral_offset), 4),
                    "score": round(float(score), 4),
                    **self._build_grid_debug_info(candidate_world, candidate_grid),
                }
                candidate_entries.append((score, target_distance, candidate_world, ideal_yaw, info))
                candidate_debug.append(info)

        selection_mode = "local_boundary"
        if not candidate_entries:
            max_axis_steps = 250
            for dx, dz, edge_name in axis_dirs:
                for step in range(1, max_axis_steps + 1):
                    free_grid = (target_grid[0] + dx * step, target_grid[1] + dz * step)
                    if self.is_hard_obstacle(*free_grid):
                        continue

                    free_center = self.grid_to_world(*free_grid)
                    candidate_world = (
                        free_center[0] + dx * outward_margin_m,
                        free_center[1] + dz * outward_margin_m,
                    )
                    candidate_grid = self.world_to_grid(candidate_world[0], candidate_world[1])
                    if self.is_obstacle(*candidate_grid):
                        continue

                    yaw_dx = target_world_pos[0] - candidate_world[0]
                    yaw_dz = target_world_pos[1] - candidate_world[1]
                    target_distance = math.sqrt(yaw_dx * yaw_dx + yaw_dz * yaw_dz)
                    ideal_yaw = math.degrees(math.atan2(yaw_dx, yaw_dz)) % 360
                    score = target_distance
                    info = {
                        "edge": edge_name,
                        "selection_mode": "axis_ray_exit",
                        "exit_step": int(step),
                        "free_grid": [int(free_grid[0]), int(free_grid[1])],
                        "candidate_world_pos": [round(float(candidate_world[0]), 4), round(float(candidate_world[1]), 4)],
                        "candidate_grid": [int(candidate_grid[0]), int(candidate_grid[1])],
                        "target_distance_m": round(float(target_distance), 4),
                        "score": round(float(score), 4),
                        **self._build_grid_debug_info(candidate_world, candidate_grid),
                    }
                    candidate_entries.append((score, target_distance, candidate_world, ideal_yaw, info))
                    candidate_debug.append(info)
                    break

            if not candidate_entries:
                print("[位姿计算v2] 容器 bbox 特判未找到边界外侧可站位，回退通用逻辑")
                return None
            selection_mode = "axis_ray_exit"

        candidate_entries.sort(key=lambda item: (item[0], item[1]))
        best_score, best_target_distance, best_world, best_yaw, best_info = candidate_entries[0]

        print(
            f"[位姿计算v2] 容器 bbox 特判: mode={selection_mode} "
            f"局部障碍格={len(local_obstacles)} 连通簇={len(component)} 候选站位={len(candidate_entries)}"
        )
        print(
            f"[位姿计算v2] 容器 bbox 最近边站位: edge={best_info['edge']} "
            f"pos=({best_world[0]:.3f}, {best_world[1]:.3f}) "
            f"target_distance={best_target_distance:.3f}m outward_margin={outward_margin_m:.3f}m"
        )

        self.last_docking_debug = {
            "mode": "container_bbox_edge_special_case",
            "start_world_pos": (
                [round(float(start_world_pos[0]), 4), round(float(start_world_pos[1]), 4)]
                if start_world_pos is not None
                else None
            ),
            "target_world_pos": [round(float(target_world_pos[0]), 4), round(float(target_world_pos[1]), 4)],
            "container_world_bounds": {
                "xmin": round(float(xmin), 4),
                "xmax": round(float(xmax), 4),
                "zmin": round(float(zmin), 4),
                "zmax": round(float(zmax), 4),
            },
            "outward_margin_m": round(float(outward_margin_m), 4),
            "selection_mode": selection_mode,
            "local_obstacle_count": int(len(local_obstacles)),
            "component_cell_count": int(len(component)),
            "candidate_count": int(len(candidate_entries)),
            "candidate_directions": candidate_debug[:16],
            "chosen": {
                "edge": best_info["edge"],
                "ideal_world_pos": [round(float(best_world[0]), 4), round(float(best_world[1]), 4)],
                "ideal_yaw": round(float(best_yaw), 4),
                "target_distance_m": round(float(best_target_distance), 4),
                "score": round(float(best_score), 4),
            },
        }
        return (best_world, best_yaw)

    def world_to_grid(self, x: float, z: float) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        grid_x = int(x / self.grid_size)
        grid_z = int(z / self.grid_size)
        return (grid_x, grid_z)

    def grid_to_world(self, grid_x: int, grid_z: int) -> Tuple[float, float]:
        """栅格坐标转世界坐标（格子中心）"""
        x = (grid_x + 0.5) * self.grid_size
        z = (grid_z + 0.5) * self.grid_size
        return (x, z)

    def is_obstacle(self, grid_x: int, grid_z: int) -> bool:
        """
        检查栅格是否是障碍物

        Args:
            grid_x, grid_z: 栅格坐标

        Returns:
            是否是障碍物（考虑玩家50cm清除半径）
        """
        self._is_obstacle_calls += 1
        if self._is_obstacle_calls == 1 or (self._is_obstacle_calls % 20000 == 0):
            print(
                f"[A*][SNAPSHOT] is_obstacle calls={self._is_obstacle_calls} "
                f"query=({grid_x},{grid_z}) obstacle_grid_type={type(self.obstacle_grid).__name__} "
                f"obstacle_grid_len={len(self.obstacle_grid) if hasattr(self.obstacle_grid, '__len__') else 'n/a'} "
                f"clearance_active={self.player_clearance_pos is not None} grid_size={self.grid_size}"
            )

        # 检查是否在障碍物网格中
        if (grid_x, grid_z) not in self.obstacle_grid:
            return False

        if self._is_obstacle_calls <= 3 or (self._is_obstacle_calls % 20000 == 0):
            print(
                f"[A*][SNAPSHOT] is_obstacle_hit calls={self._is_obstacle_calls} "
                f"grid=({grid_x},{grid_z}) clearance_pos={self.player_clearance_pos}"
            )

        # 如果设置了玩家位置，检查是否在50cm清除半径内
        if self.player_clearance_pos is not None:
            grid_world = self.grid_to_world(grid_x, grid_z)
            dx = grid_world[0] - self.player_clearance_pos[0]
            dz = grid_world[1] - self.player_clearance_pos[1]
            distance = math.sqrt(dx * dx + dz * dz)

            # 50cm内的障碍物视为可通行（补偿手绘地图误差）
            if distance < 0.5:
                return False

        return True

    def is_hard_obstacle(self, grid_x: int, grid_z: int) -> bool:
        """Check raw obstacle membership without player-clearance compensation."""
        return (grid_x, grid_z) in self.obstacle_grid

    def _nearest_obstacle_distance_m(
        self,
        grid_pos: Tuple[int, int],
        world_pos: Tuple[float, float],
        max_radius: int = 250,
    ) -> Optional[float]:
        """Approximate distance from a world position to the nearest obstacle cell center."""
        if not self.obstacle_grid:
            return None

        if self.is_hard_obstacle(*grid_pos):
            obstacle_world = self.grid_to_world(*grid_pos)
            return math.sqrt((obstacle_world[0] - world_pos[0]) ** 2 + (obstacle_world[1] - world_pos[1]) ** 2)

        gx, gz = grid_pos
        best_distance: Optional[float] = None
        for radius in range(1, max_radius + 1):
            found_in_ring = False
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if max(abs(dx), abs(dz)) != radius:
                        continue
                    test_grid = (gx + dx, gz + dz)
                    if not self.is_hard_obstacle(*test_grid):
                        continue
                    obstacle_world = self.grid_to_world(*test_grid)
                    distance = math.sqrt((obstacle_world[0] - world_pos[0]) ** 2 + (obstacle_world[1] - world_pos[1]) ** 2)
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                    found_in_ring = True
            if found_in_ring:
                return best_distance
        return None

    def _build_grid_debug_info(
        self,
        world_pos: Tuple[float, float],
        grid_pos: Tuple[int, int],
    ) -> dict[str, Any]:
        nearest = self._nearest_obstacle_distance_m(grid_pos, world_pos)
        grid_center = self.grid_to_world(*grid_pos)
        return {
            "world_pos": [round(world_pos[0], 4), round(world_pos[1], 4)],
            "grid": [int(grid_pos[0]), int(grid_pos[1])],
            "grid_center_world": [round(grid_center[0], 4), round(grid_center[1], 4)],
            "hard_obstacle": bool(self.is_hard_obstacle(*grid_pos)),
            "effective_obstacle": bool(self.is_obstacle(*grid_pos)),
            "nearest_obstacle_distance_m": (None if nearest is None else round(nearest, 4)),
        }

    def _emit_path_failure_debug(
        self,
        *,
        reason: str,
        start_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        start_grid_original: Tuple[int, int],
        goal_grid_original: Tuple[int, int],
        start_grid_used: Tuple[int, int],
        goal_grid_used: Tuple[int, int],
        explored_count: Optional[int] = None,
    ) -> None:
        payload = {
            "reason": reason,
            "clearance_active": self.player_clearance_pos is not None,
            "clearance_pos": (
                None
                if self.player_clearance_pos is None
                else [round(self.player_clearance_pos[0], 4), round(self.player_clearance_pos[1], 4)]
            ),
            "start_original": self._build_grid_debug_info(start_pos, start_grid_original),
            "start_used": self._build_grid_debug_info(start_pos, start_grid_used),
            "goal_original": self._build_grid_debug_info(goal_pos, goal_grid_original),
            "goal_used": self._build_grid_debug_info(goal_pos, goal_grid_used),
        }
        if explored_count is not None:
            payload["explored_count"] = int(explored_count)
        self.last_path_debug = payload
        print(f"[A*][DEBUG] path_failure_diagnostics={json.dumps(payload, ensure_ascii=False, sort_keys=True)}")

    def heuristic(self, from_grid: Tuple[int, int], to_grid: Tuple[int, int]) -> float:
        """
        启发式函数（欧几里得距离）

        Args:
            from_grid: 起点栅格坐标
            to_grid: 终点栅格坐标

        Returns:
            估算距离
        """
        dx = abs(to_grid[0] - from_grid[0])
        dz = abs(to_grid[1] - from_grid[1])
        # 欧几里得距离
        return math.sqrt(dx * dx + dz * dz) * self.grid_size

    def get_neighbors(self, grid_x: int, grid_z: int) -> List[Tuple[int, int, float]]:
        """
        获取相邻可通行的栅格

        Args:
            grid_x, grid_z: 当前栅格坐标

        Returns:
            [(邻居x, 邻居z, 移动代价), ...]
        """
        self._assert_valid_instance(caller="get_neighbors")
        self._get_neighbors_calls += 1
        snapshot_this_call = self._get_neighbors_calls == 1 or (self._get_neighbors_calls % 2000 == 0)
        if snapshot_this_call:
            print(
                f"[A*][SNAPSHOT] get_neighbors calls={self._get_neighbors_calls} "
                f"grid=({grid_x},{grid_z}) obstacle_grid_len={len(self.obstacle_grid)} "
                f"clearance_active={self.player_clearance_pos is not None} grid_size={self.grid_size}"
            )
        neighbors = []

        # 8个方向：上下左右 + 4个对角
        directions = [
            (-1, 0, 1.0),    # 左
            (1, 0, 1.0),     # 右
            (0, -1, 1.0),    # 下
            (0, 1, 1.0),     # 上
            (-1, -1, 1.414), # 左下
            (-1, 1, 1.414),  # 左上
            (1, -1, 1.414),  # 右下
            (1, 1, 1.414),   # 右上
        ]

        for dx, dz, cost in directions:
            nx, nz = grid_x + dx, grid_z + dz

            # 检查是否是障碍物
            if self.is_obstacle(nx, nz):
                continue

            # 对角移动时，检查两个相邻格子是否都不是障碍物（避免穿墙）
            if abs(dx) == 1 and abs(dz) == 1:
                if self.is_obstacle(grid_x + dx, grid_z) or \
                   self.is_obstacle(grid_x, grid_z + dz):
                    continue

            neighbors.append((nx, nz, cost * self.grid_size))

        if snapshot_this_call:
            print(
                f"[A*][SNAPSHOT] get_neighbors_result calls={self._get_neighbors_calls} "
                f"grid=({grid_x},{grid_z}) neighbor_count={len(neighbors)} sample={neighbors[:4]}"
            )

        return neighbors

    def find_path(
        self,
        start_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        safe_margin: float = 0.3
    ) -> Optional[List[Tuple[float, float]]]:
        """
        使用A*算法查找路径

        Args:
            start_pos: 起点世界坐标 (x, z)
            goal_pos: 终点世界坐标 (x, z)
            safe_margin: 安全边距（米），会膨胀障碍物

        Returns:
            路径点列表 [(x, z), ...] 或 None（无路径）
        """
        self._assert_valid_instance(caller="find_path")
        if not self.map_loaded:
            print("[A*] 地图未加载")
            return None

        # 转换为栅格坐标
        start_grid = self.world_to_grid(start_pos[0], start_pos[1])
        goal_grid = self.world_to_grid(goal_pos[0], goal_pos[1])
        start_grid_original = start_grid
        goal_grid_original = goal_grid
        self.last_path_debug = {}

        # 检查起点是否在障碍物中，如果是则启用清除半径
        # 临时启用清除半径检查起点
        self.player_clearance_pos = start_pos
        start_in_obstacle = self.is_hard_obstacle(*start_grid)

        if start_in_obstacle:
            print(f"[A*] 起点在障碍物中，启用50cm清除半径直到脱困")
            # 保持清除半径，稍后在搜索中动态禁用
        else:
            # 起点不在障碍物中，立即禁用清除半径
            self.player_clearance_pos = None
            print(f"[A*] 起点无障碍，不使用清除半径")

        # 检查起点和终点是否可达
        self.player_clearance_pos = start_pos if start_in_obstacle else None
        if self.is_obstacle(*start_grid):
            print(f"[A*] 起点在障碍物中: {start_grid}")
            # 尝试找最近的可通行点
            start_grid = self._find_nearest_free(start_grid)
            if start_grid is None:
                self._emit_path_failure_debug(
                    reason="start_grid_no_nearest_free",
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    start_grid_original=start_grid_original,
                    goal_grid_original=goal_grid_original,
                    start_grid_used=start_grid_original,
                    goal_grid_used=goal_grid,
                )
                self.player_clearance_pos = None  # reset temporary clearance override
                return None


        if self.is_obstacle(*goal_grid):
            print(f"[A*] 终点在障碍物中: {goal_grid}")
            # 尝试找最近的可通行点
            goal_grid = self._find_nearest_free(goal_grid)
            if goal_grid is None:
                self._emit_path_failure_debug(
                    reason="goal_grid_no_nearest_free",
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    start_grid_original=start_grid_original,
                    goal_grid_original=goal_grid_original,
                    start_grid_used=start_grid,
                    goal_grid_used=goal_grid_original,
                )
                self.player_clearance_pos = None  # reset temporary clearance override
                return None


        print(f"\n[A*] 开始寻路:")
        print(f"  起点: {start_pos} -> 栅格 {start_grid}")
        print(f"  终点: {goal_pos} -> 栅格 {goal_grid}")

        # A*算法
        open_set = []
        start_node = PathNode(start_grid[0], start_grid[1], g_cost=0.0)
        start_node.h_cost = self.heuristic(start_grid, goal_grid)
        start_node.f_cost = start_node.h_cost

        self._safe_heappush_open_set(open_set, start_node, context="start_node")

        closed_set: Set[Tuple[int, int]] = set()
        open_dict = {start_grid: start_node}

        explored_count = 0
        max_iterations = 100000  # Increased from 10000 to handle longer paths

        while open_set and explored_count < max_iterations:
            self._repair_open_heap(open_set, open_dict, context=f"heappop iter={explored_count}")
            if not open_set:
                break
            # 取出f_cost最小的节点
            current = self._safe_heappop_open_set(
                open_set,
                context=f"iter={explored_count}",
                open_dict=open_dict,
                current_grid=start_grid,
                goal_grid=goal_grid,
            )
            current_grid = (current.grid_x, current.grid_z)

            # 从open_dict中移除
            if current_grid in open_dict:
                del open_dict[current_grid]

            # 到达终点
            if current_grid == goal_grid:
                path = self._reconstruct_path(current)
                print(f"[A*] 找到路径！")
                print(f"  探索节点: {explored_count}")
                print(f"  路径长度: {len(path)} 个点")
                self.player_clearance_pos = None  # 清除临时设置
                return path

            closed_set.add(current_grid)
            explored_count += 1

            # 动态禁用清除半径：如果起点在障碍物中，当找到第一个无障碍格子时禁用
            if start_in_obstacle and self.player_clearance_pos is not None:
                # 临时禁用清除半径来检查当前格子
                temp_clearance = self.player_clearance_pos
                self.player_clearance_pos = None

                if not self.is_obstacle(current_grid[0], current_grid[1]):
                    # 当前格子无障碍，玩家已脱困，永久禁用清除半径
                    print(f"[A*] 已脱困至无障碍区域 {current_grid}，禁用清除半径")
                    start_in_obstacle = False  # 标记已脱困
                else:
                    # 还在障碍区，恢复清除半径
                    self.player_clearance_pos = temp_clearance

            # 检查所有邻居
            # Call through the class explicitly to avoid any accidental instance-level
            # method shadowing/corruption on `self.get_neighbors`.
            for nx, nz, move_cost in AStarPathfinder.get_neighbors(self, current.grid_x, current.grid_z):
                neighbor_grid = (nx, nz)

                if neighbor_grid in closed_set:
                    continue

                tentative_g = current.g_cost + move_cost

                # 如果邻居已在open_set中，检查是否找到更好的路径
                if neighbor_grid in open_dict:
                    neighbor_node = open_dict[neighbor_grid]
                    if not isinstance(neighbor_node, PathNode):
                        print(
                            f"[A*][WARN] open_dict corrupted at neighbor={neighbor_grid}: "
                            f"{neighbor_node!r} ({type(neighbor_node).__name__}); rebuilding node"
                        )
                        neighbor_node = PathNode(nx, nz, g_cost=tentative_g)
                        neighbor_node.h_cost = self.heuristic(neighbor_grid, goal_grid)
                        neighbor_node.f_cost = tentative_g + neighbor_node.h_cost
                        neighbor_node.parent = current
                        open_dict[neighbor_grid] = neighbor_node
                        self._safe_heappush_open_set(
                            open_set,
                            neighbor_node,
                            context=f"rebuild_corrupted_neighbor neighbor={neighbor_grid}",
                        )
                        continue
                    if tentative_g < neighbor_node.g_cost:
                        neighbor_node.g_cost = tentative_g
                        neighbor_node.f_cost = tentative_g + neighbor_node.h_cost
                        neighbor_node.parent = current
                        self._safe_heapify_open_set(open_set, open_dict, context=f"heapify neighbor={neighbor_grid}")
                else:
                    # 新节点
                    neighbor_node = PathNode(nx, nz, g_cost=tentative_g)
                    neighbor_node.h_cost = self.heuristic(neighbor_grid, goal_grid)
                    neighbor_node.f_cost = tentative_g + neighbor_node.h_cost
                    neighbor_node.parent = current

                    self._safe_heappush_open_set(
                        open_set,
                        neighbor_node,
                        context=f"new_neighbor neighbor={neighbor_grid}",
                    )
                    open_dict[neighbor_grid] = neighbor_node

        print(f"[A*] Search exhausted after exploring {explored_count} nodes")
        self._emit_path_failure_debug(
            reason="path_not_found_after_search",
            start_pos=start_pos,
            goal_pos=goal_pos,
            start_grid_original=start_grid_original,
            goal_grid_original=goal_grid_original,
            start_grid_used=start_grid,
            goal_grid_used=goal_grid,
            explored_count=explored_count,
        )
        self.player_clearance_pos = None  # reset temporary clearance override
        return None

    def _find_nearest_free(self, grid_pos: Tuple[int, int], max_radius: int = 100) -> Optional[Tuple[int, int]]:
        """找到最近的可通行栅格（max_radius=100栅格 = 2米）"""
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    test_pos = (grid_pos[0] + dx, grid_pos[1] + dz)
                    if not self.is_obstacle(*test_pos):
                        actual_dist = ((dx**2 + dz**2)**0.5) * self.grid_size
                        print(f"[A*] 找到最近可通行点: {test_pos}, 距离目标 {actual_dist:.2f}m")
                        return test_pos
        print(f"[A*] 在{max_radius * self.grid_size:.1f}m范围内未找到可通行点")
        return None

    def calculate_target_facing_angle(self, target_world_pos: Tuple[float, float], search_radius: float = 1.0) -> Optional[float]:
        """
        计算面向目标物品时的理想朝向角度

        Args:
            target_world_pos: 目标世界坐标 (x, z)
            search_radius: 搜索半径（米）

        Returns:
            朝向角度（度），0度=正北，90度=正东，180度=正南，270度=正西
            如果无法确定则返回None
        """
        self._assert_valid_instance(caller="calculate_ideal_docking_pose_v2")
        target_gx, target_gz = self.world_to_grid(target_world_pos[0], target_world_pos[1])
        search_radius_grids = int(search_radius / self.grid_size)

        # 收集目标周围的障碍物点
        obstacle_points = []
        for dx in range(-search_radius_grids, search_radius_grids + 1):
            for dz in range(-search_radius_grids, search_radius_grids + 1):
                test_gx = target_gx + dx
                test_gz = target_gz + dz
                if self.is_obstacle(test_gx, test_gz):
                    obstacle_points.append((test_gx, test_gz))

        if len(obstacle_points) < 3:
            print("[朝向计算] 周围障碍物太少，无法确定朝向")
            return None

        # 按X和Z方向分别统计障碍物边缘
        # 方法：找到最长的连续障碍物线段（水平或垂直）
        from collections import defaultdict

        # 统计每一行（相同gz）的障碍物
        rows = defaultdict(list)
        for gx, gz in obstacle_points:
            rows[gz].append(gx)

        # 统计每一列（相同gx）的障碍物
        cols = defaultdict(list)
        for gx, gz in obstacle_points:
            cols[gx].append(gz)

        # 找最长的连续行
        longest_row_length = 0
        longest_row_gz = None
        for gz, gx_list in rows.items():
            gx_list_sorted = sorted(gx_list)
            # 计算连续长度
            if len(gx_list_sorted) > longest_row_length:
                longest_row_length = len(gx_list_sorted)
                longest_row_gz = gz

        # 找最长的连续列
        longest_col_length = 0
        longest_col_gx = None
        for gx, gz_list in cols.items():
            gz_list_sorted = sorted(gz_list)
            if len(gz_list_sorted) > longest_col_length:
                longest_col_length = len(gz_list_sorted)
                longest_col_gx = gx

        # 选择更长的边作为障碍物主边缘
        if longest_row_length > longest_col_length:
            # 水平边缘（沿X轴）→ 垂直朝向应该是沿Z轴
            edge_type = "水平"
            edge_gz = longest_row_gz
            # 判断目标在边缘的哪一侧
            if target_gz > edge_gz:
                # 目标在边缘上方，应该朝下（180度 = 正南）
                target_angle = 180.0
            else:
                # 目标在边缘下方，应该朝上（0度 = 正北）
                target_angle = 0.0
            print(f"[朝向计算] 检测到{edge_type}边缘（长度{longest_row_length}格），建议朝向{target_angle:.0f}°")
        else:
            # 垂直边缘（沿Z轴）→ 垂直朝向应该是沿X轴
            edge_type = "垂直"
            edge_gx = longest_col_gx
            # 判断目标在边缘的哪一侧
            if target_gx > edge_gx:
                # 目标在边缘右侧，应该朝左（270度 = 正西）
                target_angle = 270.0
            else:
                # 目标在边缘左侧，应该朝右（90度 = 正东）
                target_angle = 90.0
            print(f"[朝向计算] 检测到{edge_type}边缘（长度{longest_col_length}格），建议朝向{target_angle:.0f}°")

        return target_angle

    def calculate_ideal_docking_pose(self, target_world_pos: Tuple[float, float], safe_distance: float = 0.8, search_radius: float = 1.0) -> Optional[Tuple[Tuple[float, float], float]]:
        """
        计算面向目标物品时的理想停靠位姿（位置+朝向）

        Args:
            target_world_pos: 目标世界坐标 (x, z)
            safe_distance: 距离障碍物边缘的安全距离（米）
            search_radius: 搜索半径（米）

        Returns:
            ((ideal_x, ideal_z), ideal_yaw) - 理想位置和朝向
            如果无法确定则返回None
        """
        target_gx, target_gz = self.world_to_grid(target_world_pos[0], target_world_pos[1])

        rect = self._find_inner_obstacle_rect_for_target(target_world_pos)
        if rect is not None:
            special_pose = self._calculate_rect_internal_target_docking_pose(
                target_world_pos=target_world_pos,
                rect=rect,
                safe_distance=safe_distance,
                strategy="nearest_edge",
            )
            if special_pose is not None:
                return special_pose

        search_radius_grids = int(search_radius / self.grid_size)

        # 收集目标周围的障碍物点
        obstacle_points = []
        for dx in range(-search_radius_grids, search_radius_grids + 1):
            for dz in range(-search_radius_grids, search_radius_grids + 1):
                test_gx = target_gx + dx
                test_gz = target_gz + dz
                if self.is_obstacle(test_gx, test_gz):
                    obstacle_points.append((test_gx, test_gz))

        if len(obstacle_points) < 3:
            print("[位姿计算] 周围障碍物太少，无法确定理想位姿")
            return None

        # 统计水平和垂直边缘
        from collections import defaultdict

        rows = defaultdict(list)
        cols = defaultdict(list)
        for gx, gz in obstacle_points:
            rows[gz].append(gx)
            cols[gx].append(gz)

        # 找最长的水平边
        longest_row_length = 0
        longest_row_gz = None
        for gz, gx_list in rows.items():
            if len(gx_list) > longest_row_length:
                longest_row_length = len(gx_list)
                longest_row_gz = gz

        # 找最长的垂直边
        longest_col_length = 0
        longest_col_gx = None
        for gx, gz_list in cols.items():
            if len(gz_list) > longest_col_length:
                longest_col_length = len(gz_list)
                longest_col_gx = gx

        # 根据最长边计算理想停靠位姿
        # 关键：玩家→目标的连线 ⊥ 墙体方向，玩家距离目标=safe_distance
        if longest_row_length > longest_col_length:
            # 水平边缘（沿X轴方向延伸）
            # 墙体方向: X轴，法向量: ±Z轴
            edge_world_z = self.grid_to_world(0, longest_row_gz)[1]

            # X坐标与目标对齐（保证连线垂直于墙体）
            ideal_x = target_world_pos[0]

            # 确定玩家位置：距离目标=safe_distance，站在墙的外侧
            target_to_edge_z = target_world_pos[1] - edge_world_z

            # 玩家应该在目标的某一侧（垂直于墙），距离目标=safe_distance
            if target_to_edge_z >= 0:
                # 目标在墙北侧或墙上 → 玩家站在目标南侧（更远离墙）
                ideal_z = target_world_pos[1] - safe_distance
            else:
                # 目标在墙南侧 → 玩家站在目标北侧（更远离墙）
                ideal_z = target_world_pos[1] + safe_distance

            # 朝向：从玩家指向目标
            dx = target_world_pos[0] - ideal_x
            dz = target_world_pos[1] - ideal_z
            ideal_yaw = math.degrees(math.atan2(dx, dz)) % 360

            # 计算实际距离目标的距离
            dist_to_target = math.sqrt(dx**2 + dz**2)

            print(f"[位姿计算] 水平边缘 at Z={edge_world_z:.2f}")
            print(f"           目标: ({target_world_pos[0]:.2f}, {target_world_pos[1]:.2f})")
            print(f"           理想位置: ({ideal_x:.2f}, {ideal_z:.2f})")
            print(f"           理想朝向: {ideal_yaw:.0f}°")
            print(f"           距离目标: {dist_to_target*100:.0f}cm (目标={safe_distance*100:.0f}cm)")

        else:
            # 垂直边缘（沿Z轴方向延伸）
            # 墙体方向: Z轴，法向量: ±X轴
            edge_world_x = self.grid_to_world(longest_col_gx, 0)[0]

            # Z坐标与目标对齐（保证连线垂直于墙体）
            ideal_z = target_world_pos[1]

            # 确定玩家位置：距离目标=safe_distance，站在墙的外侧
            target_to_edge_x = target_world_pos[0] - edge_world_x

            # 玩家应该在目标的某一侧（垂直于墙），距离目标=safe_distance
            if target_to_edge_x >= 0:
                # 目标在墙东侧或墙上 → 玩家站在目标西侧（更远离墙）
                ideal_x = target_world_pos[0] - safe_distance
            else:
                # 目标在墙西侧 → 玩家站在目标东侧（更远离墙）
                ideal_x = target_world_pos[0] + safe_distance

            # 朝向：从玩家指向目标
            dx = target_world_pos[0] - ideal_x
            dz = target_world_pos[1] - ideal_z
            ideal_yaw = math.degrees(math.atan2(dx, dz)) % 360

            # 计算实际距离目标的距离
            dist_to_target = math.sqrt(dx**2 + dz**2)

            print(f"[位姿计算] 垂直边缘 at X={edge_world_x:.2f}")
            print(f"           目标: ({target_world_pos[0]:.2f}, {target_world_pos[1]:.2f})")
            print(f"           理想位置: ({ideal_x:.2f}, {ideal_z:.2f})")
            print(f"           理想朝向: {ideal_yaw:.0f}°")
            print(f"           距离目标: {dist_to_target*100:.0f}cm (目标={safe_distance*100:.0f}cm)")

        return ((ideal_x, ideal_z), ideal_yaw)

    def calculate_ideal_docking_pose_v2(self, target_world_pos: Tuple[float, float],
                                          safe_distance: float = 0.6,
                                          search_radius: float = 0.1,
                                          prefer_closer_to: Optional[Tuple[float, float]] = None,
                                          internal_rect_strategy: str = "normalized_ratio",
                                          start_world_pos: Optional[Tuple[float, float]] = None,
                                          container_bounds_min: Optional[dict] = None,
                                          container_bounds_max: Optional[dict] = None) -> Optional[Tuple[Tuple[float, float], float]]:
        """
        计算理想停靠位姿 v2 - 基于多点采样的平均方向

        策略：
        1. 找到距离物品最近的13个障碍物点
        2. 计算每个点到物品的方向向量，求平均（对称方向抵消）
        3. 将平均方向量化到横平竖直（厨房墙壁特性）
        4. 理想位置距离物品 safe_distance
        5. 如果指定了 prefer_closer_to，则从可行位置中选择离它最近的

        Args:
            target_world_pos: 目标世界坐标 (x, z)
            safe_distance: 理想停靠距离（米），默认60cm
            search_radius: 搜索最近障碍物的半径（米），默认10cm
            prefer_closer_to: 优先选择靠近此位置的候选点 (x, z)，用于冰箱等容器内物品

        Returns:
            ((ideal_x, ideal_z), ideal_yaw) - 理想位置和朝向
            如果无法确定则返回None
        """
        target_gx, target_gz = self.world_to_grid(target_world_pos[0], target_world_pos[1])
        if isinstance(container_bounds_min, dict) and isinstance(container_bounds_max, dict):
            container_special_pose = self._calculate_container_bbox_edge_docking_pose(
                target_world_pos=target_world_pos,
                container_bounds_min=container_bounds_min,
                container_bounds_max=container_bounds_max,
                start_world_pos=start_world_pos,
                outward_margin_m=0.05,
            )
            if container_special_pose is not None:
                return container_special_pose
        rect = self._find_inner_obstacle_rect_for_target(target_world_pos)
        if rect is not None:
            special_pose = self._calculate_rect_internal_target_docking_pose(
                target_world_pos=target_world_pos,
                rect=rect,
                safe_distance=safe_distance,
                strategy=internal_rect_strategy,
                start_world_pos=start_world_pos,
            )
            if special_pose is not None:
                return special_pose

        # 步骤1: 找到距离物品最近的13个障碍物点
        NUM_SAMPLE_POINTS = 13
        obstacle_candidates = []  # [(distance_sq, grid_x, grid_z), ...]
        radius_schedule_m = [float(search_radius), 0.16, 0.2, 0.25, 0.3]
        radius_schedule_m = sorted({round(max(self.grid_size, r), 4) for r in radius_schedule_m})
        used_search_radius = radius_schedule_m[0]

        for radius_m in radius_schedule_m:
            search_radius_grids = int(radius_m / self.grid_size)
            obstacle_candidates = []

            for dx in range(-search_radius_grids, search_radius_grids + 1):
                for dz in range(-search_radius_grids, search_radius_grids + 1):
                    test_gx = target_gx + dx
                    test_gz = target_gz + dz

                    if self.is_obstacle(test_gx, test_gz):
                        dist_sq = dx * dx + dz * dz
                        obstacle_candidates.append((dist_sq, test_gx, test_gz))

            if obstacle_candidates:
                used_search_radius = radius_m
                if radius_m > float(search_radius) + 1e-9:
                    print(
                        f"[位姿计算v2] 初始搜索半径{float(search_radius):.2f}m未命中障碍，"
                        f"扩展到{radius_m:.2f}m"
                    )
                break

        if len(obstacle_candidates) == 0:
            print("[位姿计算v2] 扩展搜索范围后仍未找到障碍物")
            return None

        # 按距离排序，取最近的13个点
        obstacle_candidates.sort(key=lambda x: x[0])
        nearest_obstacles = obstacle_candidates[:NUM_SAMPLE_POINTS]

        print(f"[位姿计算v2] 目标位置: ({target_world_pos[0]:.3f}, {target_world_pos[1]:.3f})")
        print(f"[位姿计算v2] 实际障碍采样半径: {used_search_radius:.2f}m")
        print(f"[位姿计算v2] 采样障碍物点数: {len(nearest_obstacles)}")

        # 步骤2: 计算每个障碍物点到物品的方向向量，求平均
        sum_dir_x = 0.0
        sum_dir_z = 0.0

        for dist_sq, obs_gx, obs_gz in nearest_obstacles:
            obs_world_x, obs_world_z = self.grid_to_world(obs_gx, obs_gz)

            # 从障碍物指向物品的方向
            dx = target_world_pos[0] - obs_world_x
            dz = target_world_pos[1] - obs_world_z

            # 归一化后累加
            length = math.sqrt(dx * dx + dz * dz)
            if length > 0.001:
                sum_dir_x += dx / length
                sum_dir_z += dz / length

        # 求平均方向
        num_points = len(nearest_obstacles)
        avg_dir_x = sum_dir_x / num_points
        avg_dir_z = sum_dir_z / num_points

        print(f"[位姿计算v2] 平均方向向量: ({avg_dir_x:.3f}, {avg_dir_z:.3f})")

        # 归一化平均方向
        avg_length = math.sqrt(avg_dir_x * avg_dir_x + avg_dir_z * avg_dir_z)
        if avg_length < 0.001:
            print("[位姿计算v2] 平均方向向量太小，无法确定方向")
            return None

        dir_x = avg_dir_x / avg_length
        dir_z = avg_dir_z / avg_length

        # 步骤3: 方向量化 + 优先级降级策略
        # 厨房墙壁都是横平竖直的，将方向量化到四个轴向之一
        # 如果首选方向不可行，自动降级到次选方向

        # 确定主方向和次方向（基于分量大小）
        if abs(dir_x) > abs(dir_z):
            # X方向主导
            primary_dir = (1.0 if dir_x > 0 else -1.0, 0.0)
            secondary_dir = (0.0, 1.0 if dir_z > 0 else -1.0)
        else:
            # Z方向主导
            primary_dir = (0.0, 1.0 if dir_z > 0 else -1.0)
            secondary_dir = (1.0 if dir_x > 0 else -1.0, 0.0)

        diag_scale = 1.0 / math.sqrt(2.0)

        orthogonal_candidate_directions = [
            (primary_dir[0], primary_dir[1], "主方向"),
            (secondary_dir[0], secondary_dir[1], "次方向"),
            (-primary_dir[0], -primary_dir[1], "主方向反向"),
            (-secondary_dir[0], -secondary_dir[1], "次方向反向"),
        ]
        diagonal_candidate_directions = [
            ((primary_dir[0] + secondary_dir[0]) * diag_scale, (primary_dir[1] + secondary_dir[1]) * diag_scale, "主+次方向"),
            ((primary_dir[0] - secondary_dir[0]) * diag_scale, (primary_dir[1] - secondary_dir[1]) * diag_scale, "主-次方向"),
            ((-primary_dir[0] + secondary_dir[0]) * diag_scale, (-primary_dir[1] + secondary_dir[1]) * diag_scale, "主反向+次方向"),
            ((-primary_dir[0] - secondary_dir[0]) * diag_scale, (-primary_dir[1] - secondary_dir[1]) * diag_scale, "主反向-次方向"),
        ]

        print(f"[位姿计算v2] 主方向: ({primary_dir[0]:.0f}, {primary_dir[1]:.0f}), 次方向: ({secondary_dir[0]:.0f}, {secondary_dir[1]:.0f})")
        if prefer_closer_to:
            print(f"[位姿计算v2] 优先靠近位置: ({prefer_closer_to[0]:.3f}, {prefer_closer_to[1]:.3f})")

        candidate_debug: list[dict[str, Any]] = []

        def _evaluate_candidate_directions(
            directions: list[tuple[float, float, str]],
        ) -> list[tuple[float, float, float, float, str, dict[str, Any]]]:
            valid_candidates_local = []
            for cand_x, cand_z, cand_name in directions:
                test_x = target_world_pos[0] + cand_x * safe_distance
                test_z = target_world_pos[1] + cand_z * safe_distance
                test_gx, test_gz = self.world_to_grid(test_x, test_z)
                candidate_info = {
                    "name": str(cand_name),
                    "direction": [float(cand_x), float(cand_z)],
                    "distance_m": round(float(safe_distance), 4),
                    **self._build_grid_debug_info((test_x, test_z), (test_gx, test_gz)),
                }
                previous_path_debug = dict(self.last_path_debug)
                if start_world_pos is not None and not self.is_obstacle(test_gx, test_gz):
                    probe_stdout = io.StringIO()
                    with redirect_stdout(probe_stdout):
                        # Use the class method explicitly so candidate probing does not
                        # depend on a possibly shadowed instance attribute.
                        probe_path = AStarPathfinder.find_path(self, start_world_pos, (test_x, test_z))
                    candidate_info["path_reachable"] = bool(probe_path)
                    candidate_info["path_length_points"] = len(probe_path) if probe_path else 0
                    candidate_info["path_probe_stdout_tail"] = self._tail_text(probe_stdout.getvalue())
                    candidate_info["path_failure_debug"] = dict(self.last_path_debug) if not probe_path else {}
                    self.last_path_debug = previous_path_debug
                else:
                    candidate_info["path_reachable"] = None

                if not self.is_obstacle(test_gx, test_gz):
                    valid_candidates_local.append((test_x, test_z, cand_x, cand_z, cand_name, candidate_info))
                    print(
                        f"[位姿计算v2] candidate={cand_name} dir=({cand_x:.0f}, {cand_z:.0f}) "
                        f"world=({test_x:.3f}, {test_z:.3f}) grid=({test_gx}, {test_gz}) "
                        f"hard_obstacle={candidate_info['hard_obstacle']} effective_obstacle={candidate_info['effective_obstacle']} "
                        f"nearest_obstacle_distance_m={candidate_info['nearest_obstacle_distance_m']} "
                        f"path_reachable={candidate_info['path_reachable']}"
                    )
                else:
                    print(
                        f"[位姿计算v2] candidate={cand_name} dir=({cand_x:.0f}, {cand_z:.0f}) "
                        f"world=({test_x:.3f}, {test_z:.3f}) grid=({test_gx}, {test_gz}) "
                        f"hard_obstacle={candidate_info['hard_obstacle']} effective_obstacle={candidate_info['effective_obstacle']} "
                        f"nearest_obstacle_distance_m={candidate_info['nearest_obstacle_distance_m']} "
                        "path_reachable=skipped(obstacle)"
                    )
                candidate_debug.append(candidate_info)
            return valid_candidates_local

        def _choose_from_candidates(
            valid_candidates_local: list[tuple[float, float, float, float, str, dict[str, Any]]],
        ) -> tuple[tuple[float, float] | None, float, float]:
            chosen_dir_local = None
            ideal_x_local, ideal_z_local = 0.0, 0.0
            if len(valid_candidates_local) > 0:
                reachable_candidates = [
                    cand
                    for cand in valid_candidates_local
                    if cand[5].get("path_reachable") is True
                ]
                probed_candidates = [
                    cand
                    for cand in valid_candidates_local
                    if cand[5].get("path_reachable") is not None
                ]
                if reachable_candidates:
                    selection_pool = reachable_candidates
                    print(f"[位姿计算v2] 优先从 {len(reachable_candidates)} 个可达候选中选择")
                elif probed_candidates:
                    selection_pool = []
                    print("[位姿计算v2] 候选位置均已探测但当前不可达")
                else:
                    selection_pool = valid_candidates_local
                    print("[位姿计算v2] 没有可达候选，退回到非障碍物候选选择")

                if not selection_pool:
                    return None, ideal_x_local, ideal_z_local
                if prefer_closer_to and len(selection_pool) > 1:
                    def dist_to_prefer(cand):
                        return (cand[0] - prefer_closer_to[0])**2 + (cand[1] - prefer_closer_to[1])**2
                    selection_pool.sort(key=dist_to_prefer)
                    best = selection_pool[0]
                    ideal_x_local, ideal_z_local, cand_x, cand_z, cand_name, candidate_info = best
                    chosen_dir_local = (cand_x, cand_z)
                    dist = math.sqrt(dist_to_prefer(best))
                    print(
                        f"[位姿计算v2] 选择离优先位置最近的: {cand_name} ({cand_x:.0f}, {cand_z:.0f})，"
                        f"距离优先位置: {dist:.2f}m path_reachable={candidate_info.get('path_reachable')}"
                    )
                else:
                    best = selection_pool[0]
                    ideal_x_local, ideal_z_local, cand_x, cand_z, cand_name, candidate_info = best
                    chosen_dir_local = (cand_x, cand_z)
                    print(
                        f"[位姿计算v2] 选择{cand_name}: ({cand_x:.0f}, {cand_z:.0f}) → "
                        f"理想位置可行 path_reachable={candidate_info.get('path_reachable')}"
                    )
            return chosen_dir_local, ideal_x_local, ideal_z_local

        # 步骤4: 先检查正交4方向，只有失败时才补充45°斜向4方向
        candidate_directions = list(orthogonal_candidate_directions)
        valid_candidates = _evaluate_candidate_directions(candidate_directions)

        # 从可行候选中选择最佳位置
        chosen_dir = None
        ideal_x, ideal_z = 0.0, 0.0
        chosen_dir, ideal_x, ideal_z = _choose_from_candidates(valid_candidates)

        if chosen_dir is None:
            print("[位姿计算v2] 正交4方向未找到可用停靠位，启用45°斜向补充检查")
            candidate_directions = list(orthogonal_candidate_directions) + list(diagonal_candidate_directions)
            diagonal_valid_candidates = _evaluate_candidate_directions(diagonal_candidate_directions)
            valid_candidates.extend(diagonal_valid_candidates)
            chosen_dir, ideal_x, ideal_z = _choose_from_candidates(valid_candidates)
            if chosen_dir is None:
                print("[位姿计算v2] 8方向候选仍未找到可用停靠位，继续尝试其他停靠距离")

        # 如果所有方向都不可行，尝试增大距离
        if chosen_dir is None:
            print(f"[位姿计算v2] 所有方向在{safe_distance*100:.0f}cm都不可行，尝试增大距离...")
            expand_distance_schedule = sorted(
                {
                    round(float(d), 4)
                    for d in (
                        0.2,
                        0.25,
                        0.3,
                        0.4,
                        0.5,
                        0.8,
                        1.0,
                        1.2,
                        1.4,
                        1.5,
                        1.8,
                        2.0,
                    )
                    if float(d) > float(safe_distance) + 1e-6
                },
            )
            for cand_x, cand_z, cand_name in candidate_directions:
                for adjust_dist in expand_distance_schedule:
                    test_x = target_world_pos[0] + cand_x * adjust_dist
                    test_z = target_world_pos[1] + cand_z * adjust_dist
                    test_gx, test_gz = self.world_to_grid(test_x, test_z)

                    if not self.is_obstacle(test_gx, test_gz):
                        path_reachable = None
                        if start_world_pos is not None:
                            probe_stdout = io.StringIO()
                            previous_path_debug = dict(self.last_path_debug)
                            with redirect_stdout(probe_stdout):
                                probe_path = AStarPathfinder.find_path(self, start_world_pos, (test_x, test_z))
                            path_reachable = bool(probe_path)
                            self.last_path_debug = previous_path_debug
                            if not path_reachable:
                                print(
                                    f"[位姿计算v2] {cand_name} 在 {adjust_dist*100:.0f}cm 处非障碍但不可达，继续尝试更远距离"
                                )
                                continue
                        chosen_dir = (cand_x, cand_z)
                        ideal_x, ideal_z = test_x, test_z
                        if path_reachable is None:
                            print(f"[位姿计算v2] 使用{cand_name}增大距离: ({ideal_x:.3f}, {ideal_z:.3f})，距离目标: {adjust_dist*100:.0f}cm")
                        else:
                            print(
                                f"[位姿计算v2] 使用{cand_name}增大距离: ({ideal_x:.3f}, {ideal_z:.3f})，"
                                f"距离目标: {adjust_dist*100:.0f}cm path_reachable=True"
                            )
                        break
                if chosen_dir is not None:
                    break

        if chosen_dir is None:
            print("[位姿计算v2] 无法找到可行的理想位置")
            self.last_docking_debug = {
                "target_world_pos": [round(target_world_pos[0], 4), round(target_world_pos[1], 4)],
                "safe_distance": round(float(safe_distance), 4),
                "used_search_radius_m": round(float(used_search_radius), 4),
                "candidate_directions": candidate_debug,
                "chosen": None,
            }
            return None

        dir_x, dir_z = chosen_dir

        print(f"[位姿计算v2] 最终方向: ({dir_x:.0f}, {dir_z:.0f})")

        # 步骤5: 计算朝向（从理想位置指向目标）
        yaw_dx = target_world_pos[0] - ideal_x
        yaw_dz = target_world_pos[1] - ideal_z
        ideal_yaw = math.degrees(math.atan2(yaw_dx, yaw_dz)) % 360

        # 计算实际距离
        actual_distance = math.sqrt(yaw_dx**2 + yaw_dz**2)

        print(f"[位姿计算v2] ========== 结果 ==========")
        print(f"[位姿计算v2] 理想位置: ({ideal_x:.3f}, {ideal_z:.3f})")
        print(f"[位姿计算v2] 理想朝向: {ideal_yaw:.1f}°")
        print(f"[位姿计算v2] 距离目标: {actual_distance*100:.1f}cm (目标: {safe_distance*100:.0f}cm)")
        print(f"[位姿计算v2] 方向向量: ({dir_x:.3f}, {dir_z:.3f})")
        self.last_docking_debug = {
            "target_world_pos": [round(target_world_pos[0], 4), round(target_world_pos[1], 4)],
            "safe_distance": round(float(safe_distance), 4),
            "used_search_radius_m": round(float(used_search_radius), 4),
            "candidate_directions": candidate_debug,
            "chosen": {
                "direction": [round(float(dir_x), 4), round(float(dir_z), 4)],
                "ideal_world_pos": [round(ideal_x, 4), round(ideal_z, 4)],
                "ideal_yaw": round(float(ideal_yaw), 4),
                "target_distance_m": round(float(actual_distance), 4),
            },
        }

        return ((ideal_x, ideal_z), ideal_yaw)

    def _tail_text(self, text: str, *, max_lines: int = 8, max_chars: int = 1200) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        lines = raw.splitlines()
        tail = "\n".join(lines[-int(max_lines):]).strip()
        if len(tail) <= int(max_chars):
            return tail
        return tail[-int(max_chars):]

    def _reconstruct_path(self, end_node: PathNode) -> List[Tuple[float, float]]:
        """重建路径"""
        path = []
        current = end_node

        while current is not None:
            world_pos = self.grid_to_world(current.grid_x, current.grid_z)
            path.append(world_pos)
            current = current.parent

        path.reverse()
        return path

    def _has_line_of_sight(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """
        检查两点之间是否有视线（是否被障碍物阻挡）
        使用Bresenham算法检查线段上的所有栅格

        Args:
            start: 起点世界坐标 (x, z)
            end: 终点世界坐标 (x, z)

        Returns:
            True表示视线畅通，False表示被障碍物阻挡
        """
        # 转换为栅格坐标
        gx0, gz0 = self.world_to_grid(start[0], start[1])
        gx1, gz1 = self.world_to_grid(end[0], end[1])

        # Bresenham直线算法
        dx = abs(gx1 - gx0)
        dz = abs(gz1 - gz0)
        sx = 1 if gx0 < gx1 else -1
        sz = 1 if gz0 < gz1 else -1
        err = dx - dz

        x, z = gx0, gz0

        while True:
            # 检查当前栅格是否是障碍物
            if self.is_obstacle(x, z):
                return False

            if x == gx1 and z == gz1:
                break

            e2 = 2 * err
            if e2 > -dz:
                err -= dz
                x += sx
            if e2 < dx:
                err += dx
                z += sz

        return True

    def smooth_path(self, path: List[Tuple[float, float]], max_angle: float = 30.0, max_dist: float = 0.3) -> List[Tuple[float, float]]:
        """
        路径平滑 - 移除不必要的中间点（带障碍物检查和距离限制）

        Args:
            path: 原始路径
            max_angle: 最大转角（度），超过此角度保留拐点
            max_dist: 相邻路径点最大距离（米），超过此距离保留中间点

        Returns:
            平滑后的路径
        """
        if len(path) <= 2:
            return path

        smoothed = [path[0]]

        for i in range(1, len(path) - 1):
            prev = smoothed[-1]
            curr = path[i]
            next_p = path[i + 1]

            # 计算两个向量的夹角
            v1 = (curr[0] - prev[0], curr[1] - prev[1])
            v2 = (next_p[0] - curr[0], next_p[1] - curr[1])

            # 向量长度
            len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

            if len_v1 < 0.01 or len_v2 < 0.01:
                continue

            # 计算夹角
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            cos_angle = dot / (len_v1 * len_v2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.degrees(math.acos(cos_angle))

            # 计算如果跳过当前点，从prev到next_p的距离
            skip_dist = math.sqrt((next_p[0] - prev[0])**2 + (next_p[1] - prev[1])**2)

            # 保留这个点的条件：
            # 1. 转角较大（需要拐弯）
            # 2. 跳过后距离太远（超过max_dist）
            # 3. 从prev到next_p没有视线（会穿过障碍物）
            should_keep = False

            if angle > max_angle:
                should_keep = True
            elif skip_dist > max_dist:
                should_keep = True
            elif not self._has_line_of_sight(prev, next_p):
                should_keep = True

            if should_keep:
                smoothed.append(curr)

        smoothed.append(path[-1])

        print(f"[路径平滑] {len(path)} 点 -> {len(smoothed)} 点")
        return smoothed


def test_pathfinder():
    """测试路径规划器"""
    print("=" * 70)
    print("  A*路径规划器测试")
    print("=" * 70)

    # 查找最新的地图文件
    data_dir = os.path.join(os.path.dirname(__file__), "exploration_data")
    if not os.path.exists(data_dir):
        print("\n[错误] exploration_data 目录不存在")
        return

    map_files = [f for f in os.listdir(data_dir) if f.startswith('obstacle_map_') and f.endswith('.json')]
    if not map_files:
        print("\n[错误] 未找到地图文件")
        return

    latest_map = os.path.join(data_dir, sorted(map_files)[-1])
    print(f"\n[加载] 使用地图: {latest_map}")

    # 创建路径规划器
    pathfinder = AStarPathfinder(latest_map)

    # 测试寻路
    start = (2.0, 2.0)
    goal = (5.0, -1.0)

    path = pathfinder.find_path(start, goal)

    if path:
        print(f"\n[路径] 共 {len(path)} 个点:")
        for i, (x, z) in enumerate(path):
            if i < 5 or i >= len(path) - 5:
                print(f"  {i+1}. ({x:.2f}, {z:.2f})")
            elif i == 5:
                print(f"  ...")

        # 路径平滑
        smoothed = pathfinder.smooth_path(path)
        print(f"\n[平滑路径] 共 {len(smoothed)} 个点:")
        for i, (x, z) in enumerate(smoothed):
            print(f"  {i+1}. ({x:.2f}, {z:.2f})")

        # 计算总长度
        total_length = 0.0
        for i in range(len(smoothed) - 1):
            dx = smoothed[i+1][0] - smoothed[i][0]
            dz = smoothed[i+1][1] - smoothed[i][1]
            total_length += math.sqrt(dx**2 + dz**2)

        print(f"\n[统计] 路径总长: {total_length:.2f}m")
    else:
        print("\n[失败] 未找到路径")


if __name__ == "__main__":
    test_pathfinder()

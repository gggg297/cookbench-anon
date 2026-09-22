"""
数据加载模块
负责读取相机位姿信息、物品位置数据库、场景对象信息等
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraInfo:
    """相机信息数据类"""
    position: Tuple[float, float, float]  # 位置 (x, y, z)
    rotation: Tuple[float, float, float]  # 旋转 (pitch, yaw, roll)
    forward: Tuple[float, float, float]   # 前方向向量
    timestamp: str                         # 时间戳


@dataclass
class ObjectInfo:
    """物品/对象信息数据类"""
    id: int                                      # ID
    name_cn: str                                 # 中文名
    name_en: str                                 # 英文名
    position: Tuple[float, float, float]         # 位置
    bounding_box_size: Optional[Tuple[float, float, float]]  # 边界框大小
    bounding_box_center: Optional[Tuple[float, float, float]]  # 边界框中心
    layer: str                                   # Unity Layer
    is_active: bool                              # 是否激活
    instance_id: int                             # 实例ID
    object_type: str                             # 对象类型


class DataLoader:
    """数据加载器类"""

    def __init__(self, data_root: str | None = None, *, realtime_products_file: str | None = None, camera_info_file: str | None = None):
        """
        初始化数据加载器

        Args:
            data_root: 数据根目录，如果为None则使用config中的默认路径
            realtime_products_file: Optional override for realtime_products.json
            camera_info_file: Optional override for realtime_camera_info.txt
        """
        if data_root is None:
            from .config import DATA_ROOT  # type: ignore
            data_root = DATA_ROOT

        self.data_root = data_root
        self.objects_cache: Dict[str, ObjectInfo] = {}
        self.scene_objects_cache: Dict[str, ObjectInfo] = {}  # 场景对象缓存
        self.realtime_objects_cache: Dict[str, ObjectInfo] = {}  # 实时物品缓存
        self.latest_camera_info: Optional[CameraInfo] = None
        self.aliases: Dict[str, str] = {}  # 物品名称别名映射

        # Optional explicit file paths (prefer config-driven absolute paths).
        self.realtime_products_file = realtime_products_file or os.path.join(data_root, "realtime_products.json")
        self.camera_info_file = camera_info_file or os.path.join(data_root, "realtime_camera_info.txt")

        self._load_aliases()  # 加载别名映射

    def _load_aliases(self):
        """
        加载物品名称别名映射文件
        """
        # 查找别名文件（与data_loader.py同目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        aliases_file = os.path.join(script_dir, "item_aliases.json")

        if not os.path.exists(aliases_file):
            # print(f"别名文件不存在: {aliases_file}")
            return

        try:
            with open(aliases_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "aliases" in data:
                self.aliases = data["aliases"]
                # print(f"成功加载 {len(self.aliases)} 个物品别名")
        except Exception as e:
            print(f"加载别名文件失败: {e}")

    def resolve_alias(self, name: str) -> str:
        """
        解析物品名称别名，返回实际的游戏内名称

        Args:
            name: 输入的名称（可能是别名）

        Returns:
            str: 解析后的实际名称（如果有别名映射则返回映射后的名称，否则返回原名称）
        """
        # 尝试精确匹配
        if name in self.aliases:
            resolved = self.aliases[name]
            print(f"别名映射: '{name}' -> '{resolved}'")
            return resolved

        # 尝试小写匹配
        name_lower = name.lower()
        if name_lower in self.aliases:
            resolved = self.aliases[name_lower]
            print(f"别名映射(小写): '{name}' -> '{resolved}'")
            return resolved

        # 尝试替换下划线为空格后匹配
        name_no_underscore = name.replace('_', ' ').lower()
        for alias, target in self.aliases.items():
            if alias.replace('_', ' ').lower() == name_no_underscore:
                print(f"别名映射(模糊): '{name}' -> '{target}'")
                return target

        return name

    def find_latest_scan_dir(self, scan_type: str = "StandardProductsScan") -> Optional[str]:
        """
        查找最新的扫描目录

        Args:
            scan_type: 扫描类型，"StandardProductsScan" 或 "SceneObjectsScan"

        Returns:
            str: 最新扫描目录的完整路径，找不到返回None
        """
        if not os.path.exists(self.data_root):
            print(f"错误：数据根目录不存在: {self.data_root}")
            return None

        # 查找所有匹配的扫描目录
        scan_dirs = []
        for item in os.listdir(self.data_root):
            if item.startswith(scan_type):
                full_path = os.path.join(self.data_root, item)
                if os.path.isdir(full_path):
                    scan_dirs.append(full_path)

        if not scan_dirs:
            print(f"警告：未找到 {scan_type} 扫描目录")
            return None

        # 返回最新的目录（按修改时间排序）
        latest = max(scan_dirs, key=os.path.getmtime)
        print(f"找到最新扫描目录: {latest}")
        return latest

    def load_camera_info(self, camera_file: str = None) -> Optional[CameraInfo]:
        """
        加载相机信息

        Args:
            camera_file: 相机信息文件路径，如果为None则自动查找

        Returns:
            CameraInfo: 相机信息对象，加载失败返回None
        """
        # 如果没有指定文件，自动查找
        if camera_file is None:
            # Prefer explicit realtime camera file when present.
            if self.camera_info_file and os.path.exists(self.camera_info_file):
                camera_file = self.camera_info_file
            else:
                scan_dir = self.find_latest_scan_dir()
                if scan_dir is None:
                    return None
                camera_file = os.path.join(scan_dir, "camera_info.txt")

        if not os.path.exists(camera_file):
            print(f"错误：相机信息文件不存在: {camera_file}")
            return None

        try:
            with open(camera_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析位置信息
            pos_match = re.search(r'Position: \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)', content)
            if not pos_match:
                print("错误：无法解析Position信息")
                return None

            position = (float(pos_match.group(1)),
                       float(pos_match.group(2)),
                       float(pos_match.group(3)))

            # 解析旋转信息
            rot_match = re.search(r'Rotation: \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)', content)
            if not rot_match:
                print("错误：无法解析Rotation信息")
                return None

            rotation = (float(rot_match.group(1)),
                       float(rot_match.group(2)),
                       float(rot_match.group(3)))

            # 解析前方向向量
            fwd_match = re.search(r'Forward: \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)', content)
            if not fwd_match:
                print("错误：无法解析Forward信息")
                return None

            forward = (float(fwd_match.group(1)),
                      float(fwd_match.group(2)),
                      float(fwd_match.group(3)))

            # 解析时间戳
            time_match = re.search(r'Timestamp: (.+)', content)
            timestamp = time_match.group(1) if time_match else "Unknown"

            camera_info = CameraInfo(
                position=position,
                rotation=rotation,
                forward=forward,
                timestamp=timestamp
            )

            self.latest_camera_info = camera_info
            print(f"成功加载相机信息: 位置={position}, 旋转={rotation}")
            return camera_info

        except Exception as e:
            print(f"加载相机信息失败: {e}")
            return None

    def load_objects_from_products_list(self, products_file: str = None) -> Dict[str, ObjectInfo]:
        """
        从standard_products_list.txt加载物品信息

        Args:
            products_file: 产品列表文件路径，如果为None则自动查找最新的

        Returns:
            Dict[str, ObjectInfo]: 物品信息字典，键为物品名称（中文）
        """
        if products_file is None:
            scan_dir = self.find_latest_scan_dir("StandardProductsScan")
            if scan_dir is None:
                return {}
            products_file = os.path.join(scan_dir, "standard_products_list.txt")

        if not os.path.exists(products_file):
            print(f"错误：产品列表文件不存在: {products_file}")
            return {}

        objects = {}

        try:
            with open(products_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 使用正则表达式匹配每个产品块
            pattern = r'###\s+(\d+)\.\s+(.+?)\s+\((.+?)\)\s+分类:\s+ProductType\.(\w+).+?位置信息:\s*\n-\s+世界位置:\s+\(([-\d.]+),\s+([-\d.]+),\s+([-\d.]+)\).+?InstanceID:\s+(\d+)'

            matches = re.finditer(pattern, content, re.DOTALL)

            for match in matches:
                product_id = int(match.group(1))
                name_cn = match.group(2).strip()
                name_en = match.group(3).strip()
                object_type = match.group(4)
                pos_x = float(match.group(5))
                pos_y = float(match.group(6))
                pos_z = float(match.group(7))
                instance_id = int(match.group(8))

                # 尝试提取边界框信息（可选）
                bbox_match = re.search(
                    r'边界框大小:\s+\(([-\d.]+),\s+([-\d.]+),\s+([-\d.]+)\)',
                    match.group(0)
                )
                bbox_size = None
                if bbox_match:
                    bbox_size = (float(bbox_match.group(1)),
                               float(bbox_match.group(2)),
                               float(bbox_match.group(3)))

                # 提取Layer信息
                layer_match = re.search(r'Layer:\s+(\w+)', match.group(0))
                layer = layer_match.group(1) if layer_match else "Default"

                obj_info = ObjectInfo(
                    id=product_id,
                    name_cn=name_cn,
                    name_en=name_en,
                    position=(pos_x, pos_y, pos_z),
                    bounding_box_size=bbox_size,
                    bounding_box_center=(pos_x, pos_y, pos_z),  # 简化处理
                    layer=layer,
                    is_active=True,
                    instance_id=instance_id,
                    object_type=object_type
                )

                objects[name_cn] = obj_info

            self.objects_cache = objects
            print(f"成功加载 {len(objects)} 个物品信息")
            return objects

        except Exception as e:
            print(f"加载物品列表失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def load_objects_from_realtime_products(self, products_file: str = None) -> Dict[str, ObjectInfo]:
        """
        从 realtime_products.json 加载实时物品位置信息

        Args:
            products_file: 实时产品JSON文件路径，如果为None则使用默认路径

        Returns:
            Dict[str, ObjectInfo]: 物品信息字典，键为物品名称（中文）
        """
        if products_file is None:
            products_file = self.realtime_products_file

        if not os.path.exists(products_file):
            print(f"实时产品文件不存在: {products_file}")
            return {}

        objects = {}

        try:
            with open(products_file, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)

            products = data.get("products", [])

            for product in products:
                name_cn = product.get("name_cn", "")
                name_en = product.get("name_en", "")
                position = product.get("position", {})
                pos_x = position.get("x", 0)
                pos_y = position.get("y", 0)
                pos_z = position.get("z", 0)
                product_id = product.get("id", -1)
                instance_id = product.get("instance_id", 0)
                container = product.get("container", "Scene")

                obj_info = ObjectInfo(
                    id=product_id,
                    name_cn=name_cn,
                    name_en=name_en,
                    position=(pos_x, pos_y, pos_z),
                    bounding_box_size=None,
                    bounding_box_center=(pos_x, pos_y, pos_z),
                    layer=container,
                    is_active=True,
                    instance_id=instance_id,
                    object_type="RealtimeProduct"
                )

                # 使用中文名作为键，如果有重复则添加后缀
                key = name_cn
                if key in objects:
                    # 找一个不重复的键
                    suffix = 2
                    while f"{name_cn}_{suffix}" in objects:
                        suffix += 1
                    key = f"{name_cn}_{suffix}"

                objects[key] = obj_info

                # 确保主名称也指向第一个实例
                if name_cn not in objects:
                    objects[name_cn] = obj_info

            self.realtime_objects_cache = objects
            # print(f"成功加载 {len(products)} 个实时物品信息")
            return objects

        except Exception as e:
            print(f"加载实时产品列表失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def load_objects_from_scene_objects_list(self, scene_file: str = None) -> Dict[str, ObjectInfo]:
        """
        从scene_objects_list.txt加载场景物品信息（工具、设备等）

        Args:
            scene_file: 场景对象文件路径，如果为None则自动查找最新的

        Returns:
            Dict[str, ObjectInfo]: 场景物品信息字典，键为物品名称（中文）
        """
        if scene_file is None:
            scan_dir = self.find_latest_scan_dir("SceneObjectsScan")
            if scan_dir is None:
                return {}
            scene_file = os.path.join(scan_dir, "scene_objects_list.txt")

        if not os.path.exists(scene_file):
            print(f"错误：场景对象文件不存在: {scene_file}")
            return {}

        objects = {}

        try:
            with open(scene_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 匹配场景物品块 - 格式: [001] 刀\n      实例数: 2\n      组件类型: KnifeBetter\n...
            # 然后匹配每个实例的详情
            block_pattern = r'\[(\d+)\]\s+(.+?)\n\s+实例数:\s+(\d+)\n\s+组件类型:\s+(\w+)\n\s+分类:\s+(\w+)'

            # 匹配实例详情
            instance_pattern = r'ObjectID:\s+([-\d]+).+?EnglishName:\s+(.+?)\s+//.*?位置:\s+\(([-\d.]+),\s+([-\d.]+),\s+([-\d.]+)\)'

            blocks = re.finditer(block_pattern, content)

            for block in blocks:
                block_id = int(block.group(1))
                name_cn = block.group(2).strip()
                instance_count = int(block.group(3))
                component_type = block.group(4)
                category = block.group(5)

                # 找到这个块在内容中的位置，然后搜索后面的实例详情
                block_start = block.end()
                # 找下一个块的开始位置
                next_block = re.search(r'\n\[(\d+)\]', content[block_start:])
                if next_block:
                    block_end = block_start + next_block.start()
                else:
                    block_end = len(content)

                block_content = content[block_start:block_end]

                # 匹配实例
                instances = re.finditer(instance_pattern, block_content, re.DOTALL)
                instance_list = list(instances)

                for i, inst in enumerate(instance_list):
                    object_id = int(inst.group(1))
                    english_name = inst.group(2).strip()
                    pos_x = float(inst.group(3))
                    pos_y = float(inst.group(4))
                    pos_z = float(inst.group(5))

                    # 使用中文名+序号作为键（避免多实例覆盖）
                    if instance_count > 1:
                        key = f"{name_cn}_{i+1}"
                    else:
                        key = name_cn

                    obj_info = ObjectInfo(
                        id=block_id,
                        name_cn=name_cn,
                        name_en=english_name,
                        position=(pos_x, pos_y, pos_z),
                        bounding_box_size=None,
                        bounding_box_center=(pos_x, pos_y, pos_z),
                        layer="IngredientArea",
                        is_active=True,
                        instance_id=object_id,
                        object_type=category
                    )

                    objects[key] = obj_info

                    # 同时添加以中文名为键的条目（取第一个实例）
                    if name_cn not in objects:
                        objects[name_cn] = obj_info

            self.scene_objects_cache = objects
            print(f"成功加载 {len(objects)} 个场景物品信息")
            return objects

        except Exception as e:
            print(f"加载场景对象列表失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def find_object_by_name(self, name: str) -> Optional[ObjectInfo]:
        """
        根据名称查找物品（优先搜索实时产品，其次搜索静态产品，最后搜索场景对象）

        Args:
            name: 物品名称（中文或英文，也支持别名）

        Returns:
            ObjectInfo: 物品信息，找不到返回None
        """
        # 0. 先尝试解析别名
        resolved_name = self.resolve_alias(name)

        # ========== 优先从实时产品数据查找 ==========
        # 每次都重新加载实时数据，确保获取最新位置
        self.load_objects_from_realtime_products()

        if self.realtime_objects_cache:
            # 1. 精确匹配实时产品中文名
            if resolved_name in self.realtime_objects_cache:
                # print(f"在实时产品中找到: {resolved_name}")
                return self.realtime_objects_cache[resolved_name]

            if name in self.realtime_objects_cache:
                # print(f"在实时产品中找到: {name}")
                return self.realtime_objects_cache[name]

            # 2. 模糊匹配实时产品
            for obj_name, obj_info in self.realtime_objects_cache.items():
                if resolved_name.lower() in obj_name.lower() or resolved_name.lower() in obj_info.name_en.lower():
                    # print(f"在实时产品中模糊匹配找到: {obj_name}")
                    return obj_info
                if name.lower() in obj_name.lower() or name.lower() in obj_info.name_en.lower():
                    # print(f"在实时产品中模糊匹配找到: {obj_name}")
                    return obj_info

        # ========== 注释掉静态产品和场景对象的回退逻辑，只使用实时产品数据 ==========
        # # ========== 如果实时产品中找不到，尝试静态产品缓存 ==========
        # if not self.objects_cache:
        #     self.load_objects_from_products_list()
        #
        # # 精确匹配产品中文名（用解析后的名称）
        # if resolved_name in self.objects_cache:
        #     return self.objects_cache[resolved_name]
        #
        # # 也尝试原始名称精确匹配
        # if name in self.objects_cache:
        #     return self.objects_cache[name]
        #
        # # 尝试模糊匹配产品（同时使用原名称和解析后的名称）
        # for obj_name, obj_info in self.objects_cache.items():
        #     if resolved_name.lower() in obj_name.lower() or resolved_name.lower() in obj_info.name_en.lower():
        #         print(f"在产品中模糊匹配找到: {obj_name}")
        #         return obj_info
        #     if name.lower() in obj_name.lower() or name.lower() in obj_info.name_en.lower():
        #         print(f"在产品中模糊匹配找到: {obj_name}")
        #         return obj_info
        #

        return None

    def find_object_by_name_and_instance_id(self, name: str, instance_id: int) -> Optional[ObjectInfo]:
        """
        根据（名称 + instance_id）精确定位物品。

        约束：必须同时满足 name（支持别名解析）与 instance_id 匹配，否则返回 None。
        """
        resolved_name = self.resolve_alias(name)
        want = (resolved_name or "").strip().lower()
        alt = (name or "").strip().lower()
        try:
            inst = int(instance_id)
        except Exception:
            return None

        self.load_objects_from_realtime_products()
        if not self.realtime_objects_cache:
            return None

        for key, obj in self.realtime_objects_cache.items():
            if not obj:
                continue
            try:
                if int(getattr(obj, "instance_id", 0)) != inst:
                    continue
            except Exception:
                continue

            name_en = (getattr(obj, "name_en", "") or "").strip().lower()
            name_cn = (getattr(obj, "name_cn", "") or "").strip().lower()
            key_l = (key or "").strip().lower()

            # Must match by name too (exact or substring, mirroring `find_object_by_name` behavior).
            if want and (want == key_l or want == name_en or want == name_cn or want in key_l or want in name_en or want in name_cn):
                return obj
            if alt and (alt == key_l or alt == name_en or alt == name_cn or alt in key_l or alt in name_en or alt in name_cn):
                return obj

        return None
        # # ========== 如果产品中找不到，尝试搜索场景对象（工具、设备等） ==========
        # if not self.scene_objects_cache:
        #     self.load_objects_from_scene_objects_list()
        #
        # # 精确匹配场景对象中文名（同时尝试原名称和解析后的名称）
        # if resolved_name in self.scene_objects_cache:
        #     print(f"在场景对象中找到: {resolved_name}")
        #     return self.scene_objects_cache[resolved_name]
        # if name in self.scene_objects_cache:
        #     print(f"在场景对象中找到: {name}")
        #     return self.scene_objects_cache[name]
        #
        # # 模糊匹配场景对象
        # for obj_name, obj_info in self.scene_objects_cache.items():
        #     if resolved_name.lower() in obj_name.lower() or resolved_name.lower() in obj_info.name_en.lower():
        #         print(f"在场景对象中模糊匹配找到: {obj_name}")
        #         return obj_info
        #     if name.lower() in obj_name.lower() or name.lower() in obj_info.name_en.lower():
        #         print(f"在场景对象中模糊匹配找到: {obj_name}")
        #         return obj_info

        print(f"未找到物品: {name} (别名解析后: {resolved_name})")
        return None

    def find_all_objects_by_name(self, name: str) -> List[ObjectInfo]:
        """
        根据名称查找所有匹配的物品（用于验证视野中心是否对准任意一个同类物品）
        优先搜索实时产品，其次搜索静态产品和场景对象

        Args:
            name: 物品名称（中文或英文，也支持别名）

        Returns:
            List[ObjectInfo]: 所有匹配的物品列表
        """
        # 先解析别名
        resolved_name = self.resolve_alias(name)

        matches = []
        name_lower = name.lower()
        resolved_lower = resolved_name.lower()

        # ========== 优先搜索实时产品 ==========
        # 每次都重新加载实时数据，确保获取最新位置
        self.load_objects_from_realtime_products()

        if self.realtime_objects_cache:
            for obj_name, obj_info in self.realtime_objects_cache.items():
                # 精确匹配中文名
                if name == obj_name or resolved_name == obj_name:
                    if obj_info not in matches:
                        matches.append(obj_info)
                # 模糊匹配中文名或英文名
                elif (name_lower in obj_name.lower() or name_lower in obj_info.name_en.lower() or
                      resolved_lower in obj_name.lower() or resolved_lower in obj_info.name_en.lower()):
                    if obj_info not in matches:
                        matches.append(obj_info)

        # 如果实时产品中找到了，直接返回（实时数据优先）
        if matches:
            # print(f"找到 {len(matches)} 个匹配 '{name}' 的实时物品")
            return matches

        # ========== 注释掉静态产品和场景对象的回退逻辑，只使用实时产品数据 ==========
        # # ========== 如果实时产品中没找到，搜索静态产品 ==========
        # if not self.objects_cache:
        #     self.load_objects_from_products_list()
        # if not self.scene_objects_cache:
        #     self.load_objects_from_scene_objects_list()
        #
        # # 搜索静态产品
        # for obj_name, obj_info in self.objects_cache.items():
        #     # 精确匹配中文名
        #     if name == obj_name or resolved_name == obj_name:
        #         matches.append(obj_info)
        #     # 模糊匹配中文名或英文名（同时使用原名称和解析后的名称）
        #     elif (name_lower in obj_name.lower() or name_lower in obj_info.name_en.lower() or
        #           resolved_lower in obj_name.lower() or resolved_lower in obj_info.name_en.lower()):
        #         matches.append(obj_info)
        #
        # # 搜索场景对象
        # for obj_name, obj_info in self.scene_objects_cache.items():
        #     # 精确匹配中文名
        #     if name == obj_name or resolved_name == obj_name:
        #         if obj_info not in matches:  # 避免重复
        #             matches.append(obj_info)
        #     # 模糊匹配中文名或英文名（同时使用原名称和解析后的名称）
        #     elif (name_lower in obj_name.lower() or name_lower in obj_info.name_en.lower() or
        #           resolved_lower in obj_name.lower() or resolved_lower in obj_info.name_en.lower()):
        #         if obj_info not in matches:
        #             matches.append(obj_info)

        if matches:
            print(f"找到 {len(matches)} 个匹配 '{name}' 的物品 (别名解析后: '{resolved_name}')")
        else:
            print(f"未找到匹配 '{name}' 的物品 (别名解析后: '{resolved_name}')")

        return matches

    def get_all_objects(self) -> Dict[str, ObjectInfo]:
        """
        获取所有物品信息

        Returns:
            Dict[str, ObjectInfo]: 所有物品信息字典
        """
        if not self.objects_cache:
            self.load_objects_from_products_list()
        return self.objects_cache


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("数据加载器测试")
    print("=" * 60)

    loader = DataLoader()

    # 测试加载相机信息
    print("\n1. 测试加载相机信息:")
    camera = loader.load_camera_info()
    if camera:
        print(f"  位置: {camera.position}")
        print(f"  旋转: {camera.rotation}")
        print(f"  前方向: {camera.forward}")

    # 测试加载物品列表
    print("\n2. 测试加载物品列表:")
    objects = loader.load_objects_from_products_list()
    print(f"  总共加载: {len(objects)} 个物品")

    # 测试查找物品
    print("\n3. 测试查找物品:")
    test_names = ["黄瓜", "盐", "水"]
    for name in test_names:
        obj = loader.find_object_by_name(name)
        if obj:
            print(f"  {name}: 位置={obj.position}, ID={obj.instance_id}")
        else:
            print(f"  {name}: 未找到")

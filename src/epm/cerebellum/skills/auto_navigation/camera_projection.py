# -*- coding: utf-8 -*-
"""
相机投影工具 - 计算3D世界坐标到2D图像坐标的投影

v2.1: 修复FOV解释 - Unity使用垂直FOV，不是水平FOV
v2.0: 改用Forward向量构建相机坐标系，更可靠，不依赖Pitch/Yaw角度约定
"""
import math
from typing import Tuple, Optional


class CameraProjection:
    """
    相机投影计算器

    使用针孔相机模型将3D世界坐标投影到2D图像平面
    """

    def __init__(self, fx: float = None, fy: float = None, cx: float = None, cy: float = None,
                 screen_width: int = 1600, screen_height: int = 900):
        """
        初始化相机内参

        Args:
            fx: X方向焦距（像素），默认根据60度垂直FOV计算
            fy: Y方向焦距（像素），默认根据60度垂直FOV计算
            cx: 主点X坐标（像素），默认屏幕中心
            cy: 主点Y坐标（像素），默认屏幕中心
            screen_width: 屏幕宽度（像素）
            screen_height: 屏幕高度（像素）
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

        # v2.1: Unity使用垂直FOV，从垂直FOV计算焦距
        # 对于正方形像素，fx = fy
        if fy is None:
            fov_vertical = 60.0  # 垂直FOV（度）- Unity默认
            fy = screen_height / (2 * math.tan(math.radians(fov_vertical / 2)))

        if fx is None:
            fx = fy  # 正方形像素，fx = fy

        if cx is None:
            cx = screen_width / 2.0

        if cy is None:
            cy = screen_height / 2.0

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def _normalize(self, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """归一化向量"""
        length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if length < 1e-6:
            return (0, 0, 1)
        return (v[0]/length, v[1]/length, v[2]/length)

    def _cross(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """向量叉乘"""
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        )

    def _dot(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """向量点乘"""
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _rotate_around_axis(
        self,
        v: Tuple[float, float, float],
        axis: Tuple[float, float, float],
        angle_rad: float,
    ) -> Tuple[float, float, float]:
        """Use Rodrigues' formula to rotate a vector around an axis."""
        axis_n = self._normalize(axis)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        cross = self._cross(axis_n, v)
        dot = self._dot(axis_n, v)
        return (
            v[0] * cos_a + cross[0] * sin_a + axis_n[0] * dot * (1.0 - cos_a),
            v[1] * cos_a + cross[1] * sin_a + axis_n[1] * dot * (1.0 - cos_a),
            v[2] * cos_a + cross[2] * sin_a + axis_n[2] * dot * (1.0 - cos_a),
        )

    def basis_from_rotation(
        self,
        camera_yaw: float,
        camera_pitch: float = 0.0,
        camera_roll: float = 0.0,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Build a stable camera basis from Euler angles.

        Returns:
            (right, down, forward)
        """
        normalized_pitch = float(camera_pitch)
        if normalized_pitch > 180.0:
            normalized_pitch -= 360.0
        elif normalized_pitch < -180.0:
            normalized_pitch += 360.0

        yaw_rad = math.radians(float(camera_yaw))
        pitch_rad = math.radians(normalized_pitch)
        roll_rad = math.radians(float(camera_roll))

        cos_pitch = math.cos(pitch_rad)
        sin_pitch = math.sin(pitch_rad)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        # Game convention:
        # - yaw=0 -> +Z
        # - yaw=90 -> +X
        # - pitch>0 means looking downward
        forward = self._normalize(
            (
                sin_yaw * cos_pitch,
                -sin_pitch,
                cos_yaw * cos_pitch,
            )
        )

        # Keep the local right axis tied to yaw so it remains well-defined near vertical pitch.
        right = self._normalize((cos_yaw, 0.0, -sin_yaw))
        up = self._normalize(self._cross(forward, right))

        if abs(roll_rad) > 1e-6:
            right = self._normalize(self._rotate_around_axis(right, forward, roll_rad))
            up = self._normalize(self._rotate_around_axis(up, forward, roll_rad))

        down = (-up[0], -up[1], -up[2])
        return right, down, forward

    def project_with_rotation(
        self,
        target_world_pos: Tuple[float, float, float],
        camera_world_pos: Tuple[float, float, float],
        camera_yaw: float,
        camera_pitch: float = 0.0,
        camera_roll: float = 0.0,
    ) -> Optional[Tuple[float, float]]:
        """Project a world-space target using a full rotation-derived camera basis."""
        x_cam, y_cam, z_cam = self.basis_from_rotation(camera_yaw, camera_pitch, camera_roll)
        dx = target_world_pos[0] - camera_world_pos[0]
        dy = target_world_pos[1] - camera_world_pos[1]
        dz = target_world_pos[2] - camera_world_pos[2]
        offset = (dx, dy, dz)

        x_in_cam = self._dot(offset, x_cam)
        y_in_cam = self._dot(offset, y_cam)
        z_in_cam = self._dot(offset, z_cam)
        if z_in_cam <= 0.01:
            return None

        u = self.fx * (x_in_cam / z_in_cam) + self.cx
        v = self.fy * (y_in_cam / z_in_cam) + self.cy
        return (u, v)

    def get_camera_space_offset_with_rotation(
        self,
        target_world_pos: Tuple[float, float, float],
        camera_world_pos: Tuple[float, float, float],
        camera_yaw: float,
        camera_pitch: float = 0.0,
        camera_roll: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Return camera-space offset using the rotation-derived camera basis."""
        x_cam, y_cam, z_cam = self.basis_from_rotation(camera_yaw, camera_pitch, camera_roll)
        dx = target_world_pos[0] - camera_world_pos[0]
        dy = target_world_pos[1] - camera_world_pos[1]
        dz = target_world_pos[2] - camera_world_pos[2]
        offset = (dx, dy, dz)
        return (
            self._dot(offset, x_cam),
            self._dot(offset, y_cam),
            self._dot(offset, z_cam),
        )

    def project_with_forward(
        self,
        target_world_pos: Tuple[float, float, float],
        camera_world_pos: Tuple[float, float, float],
        camera_forward: Tuple[float, float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        使用Forward向量将3D世界坐标投影到2D图像坐标（推荐方法）

        原理:
        1. 用Forward向量构建相机坐标系（不依赖Pitch/Yaw角度）
        2. 将目标位置转换到相机坐标系
        3. 使用针孔相机模型投影到屏幕

        Args:
            target_world_pos: 目标世界坐标 (x, y, z)
            camera_world_pos: 相机世界坐标 (x, y, z)
            camera_forward: 相机Forward向量 (x, y, z)，指向相机前方

        Returns:
            图像坐标 (u, v) 或 None（目标在相机后方）
        """
        # 1. 计算目标相对于相机的世界坐标偏移
        dx = target_world_pos[0] - camera_world_pos[0]
        dy = target_world_pos[1] - camera_world_pos[1]
        dz = target_world_pos[2] - camera_world_pos[2]
        offset = (dx, dy, dz)

        # 2. 从Forward向量构建相机坐标系
        # Z_cam = Forward (相机前方)
        z_cam = self._normalize(camera_forward)

        # X_cam = WorldUp × Forward (相机右方)
        # 注意：必须是 WorldUp × Forward，不是 Forward × WorldUp
        world_up = (0, 1, 0)
        x_cam = self._normalize(self._cross(world_up, z_cam))

        # 处理Forward接近垂直的情况
        x_len = math.sqrt(x_cam[0]**2 + x_cam[1]**2 + x_cam[2]**2)
        if x_len < 0.1:
            # Forward几乎垂直，使用备用方向
            world_forward = (0, 0, 1)
            x_cam_raw = self._cross(world_forward, z_cam)
            # 垂直向上时(Forward.y > 0)，叉乘结果方向相反，需要取反
            if camera_forward[1] > 0:
                x_cam_raw = (-x_cam_raw[0], -x_cam_raw[1], -x_cam_raw[2])
            x_cam = self._normalize(x_cam_raw)

        # Y_cam = X_cam × Z_cam (相机下方，屏幕Y轴向下)
        # X_cam × Z_cam 直接得到向下的向量，不需要取反
        y_cam = self._cross(x_cam, z_cam)

        # 3. 将目标偏移转换到相机坐标系（点乘）
        x_in_cam = self._dot(offset, x_cam)
        y_in_cam = self._dot(offset, y_cam)
        z_in_cam = self._dot(offset, z_cam)

        # 4. 检查目标是否在相机前方
        if z_in_cam <= 0.01:
            return None  # 目标在相机后方或太近

        # 5. 投影到图像平面（针孔相机模型）
        u = self.fx * (x_in_cam / z_in_cam) + self.cx
        v = self.fy * (y_in_cam / z_in_cam) + self.cy

        return (u, v)

    def get_camera_space_offset(
        self,
        target_world_pos: Tuple[float, float, float],
        camera_world_pos: Tuple[float, float, float],
        camera_forward: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        获取目标在相机坐标系中的偏移量（用于判断调整方向）

        即使目标在视野外或后方，也能正确判断目标相对于当前视野的方向。

        Args:
            target_world_pos: 目标世界坐标 (x, y, z)
            camera_world_pos: 相机世界坐标 (x, y, z)
            camera_forward: 相机Forward向量

        Returns:
            (x_in_cam, y_in_cam, z_in_cam):
            - x_in_cam > 0: 目标在视野右侧
            - x_in_cam < 0: 目标在视野左侧
            - y_in_cam > 0: 目标在视野下方
            - y_in_cam < 0: 目标在视野上方
            - z_in_cam > 0: 目标在相机前方
            - z_in_cam < 0: 目标在相机后方
        """
        # 计算偏移
        dx = target_world_pos[0] - camera_world_pos[0]
        dy = target_world_pos[1] - camera_world_pos[1]
        dz = target_world_pos[2] - camera_world_pos[2]
        offset = (dx, dy, dz)

        # 构建相机坐标系
        z_cam = self._normalize(camera_forward)
        world_up = (0, 1, 0)
        x_cam_raw = self._cross(world_up, z_cam)

        # 处理Forward接近垂直的情况（必须在normalize之前检查！）
        x_len = math.sqrt(x_cam_raw[0]**2 + x_cam_raw[1]**2 + x_cam_raw[2]**2)
        if x_len < 0.1:
            world_forward = (0, 0, 1)
            x_cam_raw = self._cross(world_forward, z_cam)
            # 垂直向上时(Forward.y > 0)，叉乘结果方向相反，需要取反
            if camera_forward[1] > 0:
                x_cam_raw = (-x_cam_raw[0], -x_cam_raw[1], -x_cam_raw[2])

        x_cam = self._normalize(x_cam_raw)

        y_cam = self._cross(x_cam, z_cam)

        # 转换到相机坐标系
        x_in_cam = self._dot(offset, x_cam)
        y_in_cam = self._dot(offset, y_cam)
        z_in_cam = self._dot(offset, z_cam)

        return (x_in_cam, y_in_cam, z_in_cam)

    def project_to_image(
        self,
        target_world_pos: Tuple[float, float, float],
        camera_world_pos: Tuple[float, float, float],
        camera_yaw: float,
        camera_pitch: float = 0.0,
        camera_forward: Tuple[float, float, float] = None
    ) -> Optional[Tuple[float, float]]:
        """
        将3D世界坐标投影到2D图像坐标

        推荐使用camera_forward参数，更可靠。
        如果提供了camera_forward，将忽略camera_yaw和camera_pitch。

        Args:
            target_world_pos: 目标世界坐标 (x, y, z)
            camera_world_pos: 相机世界坐标 (x, y, z)
            camera_yaw: 相机水平朝向（度）- 仅在未提供forward时使用
            camera_pitch: 相机俯仰角（度）- 仅在未提供forward时使用
            camera_forward: 相机Forward向量（推荐）

        Returns:
            图像坐标 (u, v) 或 None（目标在相机后方）
        """
        # 如果提供了Forward向量，使用更可靠的方法
        if camera_forward is not None:
            return self.project_with_forward(target_world_pos, camera_world_pos, camera_forward)

        # 否则从Pitch/Yaw计算Forward向量（兼容旧代码）
        # 归一化pitch到-180~180
        normalized_pitch = camera_pitch
        if normalized_pitch > 180:
            normalized_pitch = normalized_pitch - 360

        # 计算Forward向量
        # 游戏中: pitch>0表示向下看, yaw=0表示朝向Z+
        pitch_rad = math.radians(normalized_pitch)
        yaw_rad = math.radians(camera_yaw)

        # Forward向量计算（根据游戏约定）
        cos_pitch = math.cos(pitch_rad)
        sin_pitch = math.sin(pitch_rad)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        forward = (
            sin_yaw * cos_pitch,   # X分量
            -sin_pitch,             # Y分量（pitch>0向下看，Y为负）
            cos_yaw * cos_pitch    # Z分量
        )

        return self.project_with_forward(target_world_pos, camera_world_pos, forward)

    def is_in_center(
        self,
        target_world_pos: Tuple[float, float, float],
        camera_world_pos: Tuple[float, float, float],
        camera_yaw: float = 0.0,
        camera_pitch: float = 0.0,
        tolerance_pixels: float = 50.0,
        camera_forward: Tuple[float, float, float] = None
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """
        检查目标是否在视野中央

        Args:
            target_world_pos: 目标世界坐标 (x, y, z)
            camera_world_pos: 相机世界坐标 (x, y, z)
            camera_yaw: 相机水平朝向（度）
            camera_pitch: 相机俯仰角（度）
            tolerance_pixels: 容忍度（像素，距离中心点的最大距离）
            camera_forward: 相机Forward向量（推荐）

        Returns:
            (是否在中央, 投影坐标(u,v))
        """
        projection = self.project_to_image(
            target_world_pos,
            camera_world_pos,
            camera_yaw,
            camera_pitch,
            camera_forward
        )

        if projection is None:
            return (False, None)

        u, v = projection

        # 计算距离中心点的距离
        du = u - self.cx
        dv = v - self.cy
        distance = math.sqrt(du * du + dv * dv)

        is_centered = distance <= tolerance_pixels

        return (is_centered, projection)


def test_projection():
    """测试投影功能"""
    print("=" * 70)
    print("  相机投影测试 (Forward向量方法)")
    print("=" * 70)

    proj = CameraProjection()

    # 测试场景：用户对准洋葱时的实际数据
    camera_pos = (6.497, 1.750, -0.989)
    forward = (0.169, -0.429, 0.887)
    onion_pos = (6.611, 1.415, -0.400)

    print(f"\n[场景] 用户对准洋葱")
    print(f"  相机位置: {camera_pos}")
    print(f"  Forward: {forward}")
    print(f"  洋葱位置: {onion_pos}")

    # 使用Forward向量投影
    result = proj.project_with_forward(onion_pos, camera_pos, forward)

    if result:
        u, v = result
        print(f"\n[投影结果]")
        print(f"  屏幕坐标: ({u:.1f}, {v:.1f})")
        print(f"  屏幕中心: ({proj.cx}, {proj.cy})")
        du = u - proj.cx
        dv = v - proj.cy
        dist = math.sqrt(du*du + dv*dv)
        print(f"  偏差: X={du:.1f}px, Y={dv:.1f}px")
        print(f"  距中心: {dist:.1f}px")

        if dist < 20:
            print("\n  [OK] 投影接近屏幕中心")
        else:
            print(f"\n  [注意] Y偏差{dv:.0f}px，可能是洋葱Y坐标数据偏低")
    else:
        print("\n[投影失败] 目标在相机后方")


if __name__ == "__main__":
    test_projection()

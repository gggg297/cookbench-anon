# -*- coding: utf-8 -*-
"""
截图并标注物品位置
在完成导航后截取游戏画面，并在图上标注目标物品的理论位置

v1.2: 统一使用垂直FOV=60度，与camera_projection.py保持一致
v1.1: 使用CameraProjection类统一投影逻辑
"""
from PIL import Image, ImageDraw, ImageFont, ImageGrab
import win32gui
import win32ui
import win32con
from ctypes import windll
import sys
import os
import math
from datetime import datetime
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'deploy_agent'))
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import DataLoader
from camera_projection import CameraProjection

def capture_game_window(window_title="CookingSimulator"):
    """截取游戏窗口（只截取游戏画面，不包括其他窗口）"""
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd == 0:
        print(f"   ERROR: 找不到窗口 '{window_title}'")
        return None

    # 获取窗口客户区大小（不包括标题栏和边框）
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    width = right - left
    height = bottom - top

    # 获取窗口DC
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)

    # 首先尝试PrintWindow（对DirectX/OpenGL游戏更有效）
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)  # 3 = PW_RENDERFULLCONTENT

    # 如果PrintWindow失败，尝试BitBlt
    if result == 0:
        print("   PrintWindow失败，尝试BitBlt...")
        result = saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

    # 转换为PIL Image
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    img = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1
    )

    # 释放资源
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    return img

def project_world_to_screen(
    obj_world_pos,
    camera_world_pos,
    camera_yaw,
    camera_pitch,
    screen_width=1920,
    screen_height=1080,
    fx=1066.67,
    fy=1066.67,
    cx=960.0,
    cy=540.0
):
    """
    将世界坐标投影到屏幕坐标（使用CameraProjection类统一投影逻辑）

    参数:
        obj_world_pos: (x, y, z) 物品世界坐标
        camera_world_pos: (x, y, z) 相机世界坐标
        camera_yaw: 相机yaw角度（0=北，90=东）
        camera_pitch: 相机pitch角度
        screen_width, screen_height: 屏幕分辨率
        fx, fy: 相机焦距（像素单位）- 传递给CameraProjection
        cx, cy: 相机光心（像素单位）- 传递给CameraProjection

    返回:
        (screen_x, screen_y) 屏幕坐标，如果在视野外则返回None
    """
    # 使用CameraProjection类进行投影（统一坐标变换逻辑）
    proj = CameraProjection(fx=fx, fy=fy, cx=cx, cy=cy)
    result = proj.project_to_image(obj_world_pos, camera_world_pos, camera_yaw, camera_pitch)

    if result is None:
        return None

    screen_x, screen_y = result

    # 检查是否在屏幕范围内（允许超出一点边缘）
    if screen_x < -screen_width * 0.5 or screen_x > screen_width * 1.5:
        return None
    if screen_y < -screen_height * 0.5 or screen_y > screen_height * 1.5:
        return None

    # 转换为整数像素坐标
    return (int(screen_x), int(screen_y))

def annotate_target_on_screenshot(target_name="tomato", fov_horizontal=60.0):
    """
    截图并标注目标物品位置

    参数:
        target_name: 目标物品名称
        fov_horizontal: 水平FOV（度）
    """
    print("\n" + "="*70)
    print("  截图并标注目标物品位置")
    print("="*70)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    data_loader = DataLoader()

    # 加载目标物品
    try:
        target_obj = data_loader.find_object_by_name(target_name)
        target_pos = target_obj.position
        obj_name = getattr(target_obj, 'object_name', getattr(target_obj, 'object_name_zh', target_name))
        print(f"   目标物品: {obj_name}")
        print(f"   世界坐标: X={target_pos[0]:.3f}, Y={target_pos[1]:.3f}, Z={target_pos[2]:.3f}")
    except Exception as e:
        print(f"   ERROR: 无法加载目标物品 - {e}")
        return None

    # 加载相机信息
    try:
        camera_info = data_loader.load_camera_info()
        camera_pos = camera_info.position
        camera_rot = camera_info.rotation
        camera_forward = camera_info.forward  # v1.2: 使用Forward向量（更准确）
        print(f"   相机位置: X={camera_pos[0]:.3f}, Y={camera_pos[1]:.3f}, Z={camera_pos[2]:.3f}")
        print(f"   相机朝向: Pitch={camera_rot[0]:.1f}°, Yaw={camera_rot[1]:.1f}°, Roll={camera_rot[2]:.1f}°")
    except Exception as e:
        print(f"   ERROR: 无法加载相机信息 - {e}")
        return None

    # 2. 计算距离和方向
    print("\n[2/4] 计算物品相对位置...")
    dx = target_pos[0] - camera_pos[0]
    dy = target_pos[1] - camera_pos[1]
    dz = target_pos[2] - camera_pos[2]
    distance_3d = math.sqrt(dx**2 + dy**2 + dz**2)
    distance_2d = math.sqrt(dx**2 + dz**2)

    # 计算方位角（相对于相机朝向）
    target_angle = math.degrees(math.atan2(dx, dz)) % 360
    angle_diff = target_angle - camera_rot[1]
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360

    print(f"   3D距离: {distance_3d:.3f}m")
    print(f"   2D距离: {distance_2d:.3f}m")
    print(f"   高度差: {dy:.3f}m")
    print(f"   方位偏差: {angle_diff:.1f}° (相对相机朝向)")

    # 3. 截图
    print("\n[3/4] 截取游戏画面...")
    screenshot = capture_game_window("CookingSimulator")
    if screenshot is None:
        print("   ERROR: 截图失败")
        return None
    width, height = screenshot.size
    print(f"   游戏窗口尺寸: {width}x{height}")

    # 4. 投影物品位置到屏幕
    print("\n[4/4] 标注物品位置...")
    # v1.2: 使用Forward向量直接投影（与验证代码一致，更准确）
    # 根据实际窗口大小计算焦距（基于垂直FOV=60度）
    fov_vertical = 60.0
    fy = height / (2 * math.tan(math.radians(fov_vertical / 2)))
    fx = fy  # 正方形像素
    cx = width / 2.0
    cy = height / 2.0

    # v1.2: Y坐标补偿（与simple_astar_radar_navigator验证逻辑一致）
    # 设为0直接对准数据中的position坐标
    Y_OFFSET_COMPENSATION = 0.00
    target_pos_compensated = (target_pos[0], target_pos[1] + Y_OFFSET_COMPENSATION, target_pos[2])

    # 使用Forward向量投影（与simple_astar_radar_navigator验证逻辑一致）
    proj = CameraProjection(fx=fx, fy=fy, cx=cx, cy=cy, screen_width=width, screen_height=height)
    result = proj.project_with_forward(target_pos_compensated, camera_pos, camera_forward)

    if result is not None:
        screen_pos = (int(result[0]), int(result[1]))
    else:
        screen_pos = None

    # 绘制标注
    draw = ImageDraw.Draw(screenshot)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # 绘制游戏窗口中心十字（红色）
    center_x, center_y = width // 2, height // 2
    cross_size = 40
    draw.line([(center_x - cross_size, center_y), (center_x + cross_size, center_y)], fill=(255, 0, 0), width=3)
    draw.line([(center_x, center_y - cross_size), (center_x, center_y + cross_size)], fill=(255, 0, 0), width=3)
    draw.text((center_x + 30, center_y - 30), "Game View Center", fill=(255, 0, 0), font=font)

    # 绘制物品位置标注
    if screen_pos is not None:
        sx, sy = screen_pos
        print(f"   物品在游戏视野中的坐标: ({sx}, {sy})")
        print(f"   与视野中心偏差: X={sx - center_x}px, Y={sy - center_y}px")

        # 绘制物品位置（黄色圆圈+十字）
        radius = 30
        draw.ellipse([(sx - radius, sy - radius), (sx + radius, sy + radius)], outline=(255, 255, 0), width=3)
        cross_size2 = 60
        draw.line([(sx - cross_size2, sy), (sx + cross_size2, sy)], fill=(255, 255, 0), width=3)
        draw.line([(sx, sy - cross_size2), (sx, sy + cross_size2)], fill=(255, 255, 0), width=3)

        # 标注文字
        label = f"Target: {obj_name}"
        draw.text((sx + 40, sy - 60), label, fill=(255, 255, 0), font=font)
        draw.text((sx + 40, sy - 30), f"Dist: {distance_3d:.2f}m", fill=(255, 255, 0), font=font_small)
        draw.text((sx + 40, sy), f"Angle: {angle_diff:.1f}deg", fill=(255, 255, 0), font=font_small)

        # 绘制从中心指向物品的箭头
        draw.line([(center_x, center_y), (sx, sy)], fill=(255, 0, 255), width=2)
    else:
        print("   物品不在当前视野内")
        # 在屏幕上显示提示
        draw.text((50, 50), "Target NOT in view", fill=(255, 0, 0), font=font)
        draw.text((50, 100), f"Angle offset: {angle_diff:.1f}deg", fill=(255, 0, 0), font=font_small)

    # 保存截图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"navigation_result_{target_name}_{timestamp}.png"
    screenshot.save(output_path)

    print(f"\n   截图已保存: {output_path}")

    print("\n" + "="*70)
    print("  标注完成！")
    print("="*70 + "\n")

    return output_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="截图并标注目标物品位置")
    parser.add_argument("target", type=str, help="目标物品名称")
    parser.add_argument("--fov", type=float, default=60.0, help="水平FOV（度），默认60")

    args = parser.parse_args()

    annotate_target_on_screenshot(args.target, args.fov)

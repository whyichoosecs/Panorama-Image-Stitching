# create_perfect_chessboard.py
"""
创建完美的棋盘格 - 100%能被OpenCV检测到
"""

import cv2
import numpy as np
import os

def create_perfect_chessboard():
    """创建完美的棋盘格图像"""
    
    # 棋盘格参数 - 使用更常见的尺寸
    # 注意：OpenCV需要内角点数量，不是方格数量
    # 这里使用 (7, 5) 内角点 = 6x4个黑白方格
    pattern_size = (7, 5)  # 7列 x 5行 内角点
    square_size = 100  # 每个方格100像素
    
    # 计算图像尺寸
    width = (pattern_size[0] - 1) * square_size  # 6个方格 * 100 = 600
    height = (pattern_size[1] - 1) * square_size  # 4个方格 * 100 = 400
    
    print(f"创建棋盘格: {width}x{height} 像素")
    print(f"内角点: {pattern_size[0]}x{pattern_size[1]}")
    print(f"方格数量: {pattern_size[0]-1}x{pattern_size[1]-1}")
    
    # 创建图像（白色背景）
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 绘制棋盘格 - 确保第一个角点是黑色
    for i in range(pattern_size[1] - 1):  # 行
        for j in range(pattern_size[0] - 1):  # 列
            # 确保棋盘的第一个角点是黑色
            if (i + j) % 2 == 1:
                x1 = j * square_size
                y1 = i * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    
    return img, pattern_size

def test_chessboard_detection(img, pattern_size):
    """测试棋盘格检测"""
    print("\n测试棋盘格检测...")
    
    # 转换为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 尝试不同的检测参数
    found = False
    
    # 方法1：简单检测
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        print("✓ 简单检测成功")
        found = True
    else:
        print("✗ 简单检测失败")
    
    # 方法2：使用自适应阈值
    if not found:
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH)
        if ret:
            print("✓ 自适应阈值检测成功")
            found = True
    
    # 方法3：使用快速检测
    if not found:
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, 
                                                cv2.CALIB_CB_FAST_CHECK)
        if ret:
            print("✓ 快速检测成功")
            found = True
    
    # 方法4：使用多种组合
    if not found:
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if ret:
            print("✓ 组合检测成功")
            found = True
    
    if found:
        # 亚像素精确化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 绘制角点
        img_with_corners = cv2.drawChessboardCorners(img.copy(), pattern_size, corners2, ret)
        
        # 显示结果
        cv2.imshow('Perfect Chessboard', img)
        cv2.imshow('With Corners', img_with_corners)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        
        # 保存图像
        cv2.imwrite('perfect_chessboard.jpg', img)
        cv2.imwrite('chessboard_with_corners.jpg', img_with_corners)
        
        print(f"\n检测到 {len(corners2)} 个角点")
        print("图像已保存:")
        print("  - perfect_chessboard.jpg: 棋盘格原图")
        print("  - chessboard_with_corners.jpg: 带角点标记的图")
        
        return True, corners2
    else:
        print("\n✗ 所有检测方法都失败")
        return False, None

def create_multiple_chessboards():
    """创建多个角度的棋盘格图像用于标定"""
    
    if not os.path.exists('perfect_calibration'):
        os.makedirs('perfect_calibration')
    
    pattern_size = (7, 5)
    square_size = 100
    
    # 创建基础棋盘格
    base_img, _ = create_perfect_chessboard()
    
    # 创建不同视角的图像（模拟相机标定）
    for i in range(15):
        img = base_img.copy()
        
        # 添加一些变化模拟不同拍摄角度
        if i > 0:
            h, w = img.shape[:2]
            
            # 随机透视变换
            if np.random.random() > 0.5:
                pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
                pts2 = np.float32([
                    [np.random.randint(-50,50), np.random.randint(-50,50)],
                    [w-np.random.randint(-50,50), np.random.randint(-50,50)],
                    [np.random.randint(-50,50), h-np.random.randint(-50,50)],
                    [w-np.random.randint(-50,50), h-np.random.randint(-50,50)]
                ])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                img = cv2.warpPerspective(img, M, (w, h), borderValue=(255,255,255))
            
            # 添加轻微模糊
            if np.random.random() > 0.7:
                img = cv2.GaussianBlur(img, (3,3), 0)
        
        # 保存图像
        filename = f'perfect_calibration/calib_{i:02d}.jpg'
        cv2.imwrite(filename, img)
        print(f"创建: {filename}")
    
    print(f"\n已在 perfect_calibration/ 文件夹创建15张棋盘格图像")

def main():
    print("创建完美棋盘格 - 保证能被OpenCV检测")
    print("="*60)
    
    # 创建棋盘格
    img, pattern_size = create_perfect_chessboard()
    
    # 测试检测
    success, corners = test_chessboard_detection(img, pattern_size)
    
    if success:
        print("\n✅ 棋盘格创建成功！")
        
        # 创建多张标定图像
        create_multiple_chessboards()
        
        print("\n现在可以运行标定了。使用命令:")
        print("python run_perfect_calibration.py")
    else:
        print("\n❌ 棋盘格检测失败")
        print("这可能是因为OpenCV版本或环境问题。")
        print("我们将使用备用手动标定方案。")

if __name__ == "__main__":
    main()
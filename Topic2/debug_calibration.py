"""
调试棋盘格检测问题
"""

import cv2
import numpy as np
import glob
import os

def debug_chessboard_detection():
    """详细调试棋盘格检测"""
    
    images = glob.glob('data/calib_images/*.jpg')
    
    if not images:
        print("错误：没有找到图像")
        return
    
    print(f"找到 {len(images)} 张图像")
    
    # 尝试多种棋盘格尺寸
    patterns_to_try = [
        (9, 6),  # 默认
        (10, 7),
        (8, 5),
        (7, 5),
        (6, 4),
        (5, 4),
        (11, 8),
        (12, 9)
    ]
    
    successful_images = 0
    
    for img_path in images:
        print(f"\n处理图像: {os.path.basename(img_path)}")
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print("  无法读取图像")
            continue
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 保存原始图像用于显示
        display_img = img.copy()
        
        # 尝试不同的图像预处理方法
        preprocessing_methods = [
            ("原始图像", gray),
            ("高斯模糊", cv2.GaussianBlur(gray, (5, 5), 0)),
            ("直方图均衡化", cv2.equalizeHist(gray)),
            ("自适应阈值", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2))
        ]
        
        found = False
        for method_name, processed in preprocessing_methods:
            if found:
                break
                
            for pattern in patterns_to_try:
                ret, corners = cv2.findChessboardCorners(processed, pattern, None)
                
                if ret:
                    print(f"  成功！方法: {method_name}, 尺寸: {pattern}")
                    
                    # 提高精度
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # 在图像上绘制
                    display_img = cv2.drawChessboardCorners(display_img, pattern, corners_refined, ret)
                    
                    # 显示图像
                    cv2.imshow(f'Found: {method_name}, {pattern}', 
                             cv2.resize(display_img, (600, 400)))
                    cv2.waitKey(500)
                    
                    successful_images += 1
                    found = True
                    break
        
        if not found:
            print("  未找到棋盘格")
            
            # 显示图像和边缘检测结果
            edges = cv2.Canny(gray, 50, 150)
            
            # 创建复合图像用于显示
            h, w = gray.shape
            composite = np.zeros((h, w*2, 3), dtype=np.uint8)
            composite[:, :w] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            composite[:, w:] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            cv2.imshow(f'Failed: {os.path.basename(img_path)}', 
                      cv2.resize(composite, (800, 300)))
            cv2.waitKey(1000)
    
    cv2.destroyAllWindows()
    
    print(f"\n成功检测到棋盘格的图像数量: {successful_images}/{len(images)}")
    
    if successful_images == 0:
        print("\n建议：")
        print("1. 确保棋盘格清晰可见，无反光")
        print("2. 尝试不同的棋盘格图案（可以重新生成）")
        print("3. 调整相机角度，确保棋盘格完全在画面中")
        print("4. 使用更好的光照条件")
    
    return successful_images > 0

def create_simple_chessboard():
    """创建一个简单的棋盘格图像用于测试"""
    
    # 确保目录存在
    os.makedirs('data/calib_images', exist_ok=True)
    
    # 创建一个非常清晰的棋盘格
    square_size = 60  # 像素
    rows, cols = 6, 9  # 内角点数量
    
    # 计算图像尺寸（10x7个方格，因为6行内角点需要7行方格）
    img_width = (cols + 1) * square_size
    img_height = (rows + 1) * square_size
    
    # 创建图像
    chessboard = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # 绘制棋盘格
    for i in range(rows + 1):  # 方格行数 = 内角点行数 + 1
        for j in range(cols + 1):  # 方格列数 = 内角点列数 + 1
            if (i + j) % 2 == 0:
                color = (0, 0, 0)  # 黑色
            else:
                color = (255, 255, 255)  # 白色
            
            x1 = j * square_size
            y1 = i * square_size
            x2 = (j + 1) * square_size
            y2 = (i + 1) * square_size
            
            chessboard[y1:y2, x1:x2] = color
    
    # 保存
    output_path = 'data/calib_images/simple_chessboard.jpg'
    cv2.imwrite(output_path, chessboard)
    print(f"已创建清晰的棋盘格图像: {output_path}")
    
    # 显示
    cv2.imshow('Simple Chessboard', chessboard)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=" * 50)
    print("棋盘格检测调试工具")
    print("=" * 50)
    
    # 创建一个简单的棋盘格用于测试
    create_simple_chessboard()
    
    # 运行调试
    debug_chessboard_detection()
    
    print("\n如果仍然无法检测，请尝试：")
    print("1. 用手机或相机拍摄棋盘格，确保图像清晰")
    print("2. 打印棋盘格图像，确保每个方格至少2厘米")
    print("3. 调整光照，避免反光")
    print("4. 确保棋盘格完全在画面中，没有裁剪")
    
    input("\n按 Enter 键退出...")
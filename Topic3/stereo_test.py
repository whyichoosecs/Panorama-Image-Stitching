"""
立体视觉测试程序
使用示例图像测试立体视觉流程
"""

import cv2
import numpy as np
import os
import sys

def create_sample_stereo_images():
    """创建示例立体图像对"""
    
    print("创建示例立体图像对...")
    
    # 创建目录
    os.makedirs('data/stereo_pairs', exist_ok=True)
    os.makedirs('data/calibration', exist_ok=True)
    
    # 1. 创建简单的3D场景
    img_size = (640, 480)
    
    # 左图像：原始场景
    left_img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 200
    
    # 在左图像上绘制一些物体
    # 红色矩形（近处物体）
    cv2.rectangle(left_img, (150, 150), (250, 250), (0, 0, 255), -1)
    cv2.putText(left_img, "Near", (160, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 绿色圆形（中间距离）
    cv2.circle(left_img, (400, 200), 50, (0, 255, 0), -1)
    cv2.putText(left_img, "Mid", (380, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 蓝色三角形（远处）
    pts = np.array([[500, 350], [450, 450], [550, 450]], np.int32)
    cv2.fillPoly(left_img, [pts], (255, 0, 0))
    cv2.putText(left_img, "Far", (520, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 添加一些纹理
    for i in range(0, img_size[0], 50):
        cv2.line(left_img, (i, 0), (i, img_size[1]), (100, 100, 100), 1)
    
    for i in range(0, img_size[1], 50):
        cv2.line(left_img, (0, i), (img_size[0], i), (100, 100, 100), 1)
    
    # 2. 右图像：模拟水平偏移（视差）
    # 近处物体偏移大，远处物体偏移小
    right_img = left_img.copy()
    
    # 模拟视差效果
    # 红色矩形（近处）向右偏移20像素
    cv2.rectangle(right_img, (170, 150), (270, 250), (0, 0, 255), -1)
    cv2.putText(right_img, "Near", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 绿色圆形（中间）向右偏移10像素
    cv2.circle(right_img, (410, 200), 50, (0, 255, 0), -1)
    cv2.putText(right_img, "Mid", (390, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 蓝色三角形（远处）向右偏移5像素
    pts = np.array([[505, 350], [455, 450], [555, 450]], np.int32)
    cv2.fillPoly(right_img, [pts], (255, 0, 0))
    cv2.putText(right_img, "Far", (525, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 保存示例图像
    cv2.imwrite('data/stereo_pairs/left_sample.jpg', left_img)
    cv2.imwrite('data/stereo_pairs/right_sample.jpg', right_img)
    
    print("示例立体图像对已保存:")
    print("  data/stereo_pairs/left_sample.jpg")
    print("  data/stereo_pairs/right_sample.jpg")
    
    # 3. 创建示例标定图像（棋盘格）
    print("\n创建示例标定图像...")
    chessboard_size = (9, 6)
    square_size = 60
    
    for i in range(5):
        # 左标定图像
        left_calib = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # 绘制棋盘格
        for row in range(chessboard_size[1]):
            for col in range(chessboard_size[0]):
                x = col * square_size + 50
                y = row * square_size + 50
                
                if (row + col) % 2 == 0:
                    color = (0, 0, 0)
                else:
                    color = (255, 255, 255)
                
                cv2.rectangle(left_calib, (x, y), (x+square_size, y+square_size), color, -1)
        
        # 添加透视变换
        if i > 0:
            pts1 = np.float32([[50, 50], [590, 50], [50, 410], [590, 410]])
            shift = 30 * i
            pts2 = np.float32([[50+shift, 50], [590-shift, 50+shift], 
                              [50+shift, 410-shift], [590-shift, 410]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            left_calib = cv2.warpPerspective(left_calib, M, (640, 480))
        
        # 右标定图像（类似但稍有不同）
        right_calib = left_calib.copy()
        
        # 添加微小差异模拟立体
        if i > 0:
            right_calib = cv2.warpAffine(right_calib, 
                                        np.float32([[1, 0, 5], [0, 1, 0]]), 
                                        (640, 480))
        
        # 保存标定图像
        cv2.imwrite(f'data/calibration/left_{i+1:02d}.jpg', left_calib)
        cv2.imwrite(f'data/calibration/right_{i+1:02d}.jpg', right_calib)
    
    print("示例标定图像已保存到 data/calibration/ 目录")
    
    return True

def quick_test_without_calibration():
    """快速测试（无需标定）"""
    
    print("\n" + "="*50)
    print("快速立体视觉测试（无需标定）")
    print("="*50)
    
    # 创建示例图像
    if not os.path.exists('data/stereo_pairs/left_sample.jpg'):
        create_sample_stereo_images()
    
    # 读取示例图像
    left_img = cv2.imread('data/stereo_pairs/left_sample.jpg')
    right_img = cv2.imread('data/stereo_pairs/right_sample.jpg')
    
    # 转换为灰度图
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # 使用SGBM计算视差
    print("计算视差图...")
    
    # 创建SGBM对象
    window_size = 5
    min_disp = 0
    num_disp = 16 * 4
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # 计算视差
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    
    # 归一化用于显示
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, 
                                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 应用颜色映射
    disparity_colored = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    
    # 显示结果
    cv2.imshow('Left Image', left_img)
    cv2.imshow('Right Image', right_img)
    cv2.imshow('Disparity Map', disparity_colored)
    
    print("\n显示结果:")
    print("  左图像 - 原始场景")
    print("  右图像 - 模拟水平偏移")
    print("  视差图 - 颜色表示深度（红色近，蓝色远）")
    print("\n按任意键继续...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    cv2.imwrite('results/quick_test_left.jpg', left_img)
    cv2.imwrite('results/quick_test_right.jpg', right_img)
    cv2.imwrite('results/quick_test_disparity.jpg', disparity_colored)
    
    print("\n结果已保存到 results/ 目录")
    
    # 简单深度分析
    print("\n简单深度分析:")
    print("  红色物体（近处）: 视差大")
    print("  绿色物体（中间）: 视差中等")
    print("  蓝色物体（远处）: 视差小")
    print("\n注意：这是简化示例，实际需要相机标定才能获得真实深度")

def main():
    """主函数"""
    
    print("=" * 50)
    print("立体视觉测试程序")
    print("=" * 50)
    
    print("\n选择测试模式:")
    print("1. 快速测试（无需标定，使用示例图像）")
    print("2. 完整流程测试（需要标定图像）")
    print("3. 生成示例图像")
    
    choice = input("\n请输入选择 (1/2/3): ")
    
    if choice == '1':
        quick_test_without_calibration()
    elif choice == '2':
        print("\n完整流程测试:")
        print("1. 确保在 data/calibration/ 目录中有立体标定图像对")
        print("2. 在 data/stereo_pairs/ 目录中有立体图像对")
        print("3. 运行以下命令:")
        print("   python stereo_calibration.py")
        print("   python stereo_depth.py")
        print("   python stereo_3d_reconstruction.py")
    elif choice == '3':
        create_sample_stereo_images()
        print("\n示例图像已生成，现在可以运行快速测试:")
        print("python stereo_test.py")
        print("然后选择选项 1")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
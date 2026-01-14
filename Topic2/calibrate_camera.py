"""
修复版的相机标定程序
自动检测并适应不同的棋盘格尺寸
"""

import cv2
import numpy as np
import glob
import pickle
import os

def find_chessboard_pattern(gray, patterns_to_try=None):
    """
    自动检测棋盘格尺寸
    返回: (成功标志, 角点, 检测到的尺寸)
    """
    if patterns_to_try is None:
        patterns_to_try = [
            (9, 6), (8, 5), (7, 5), (6, 4), (5, 4),
            (10, 7), (11, 8), (12, 9), (7, 6), (8, 6)
        ]
    
    for pattern in patterns_to_try:
        ret, corners = cv2.findChessboardCorners(gray, pattern, None)
        if ret:
            return True, corners, pattern
    
    return False, None, None

def calibrate_camera_adaptive(square_size=0.025, min_images=5):
    """
    自适应棋盘格尺寸的相机标定
    
    参数:
        square_size: 每个棋盘格方格的边长（单位：米）
        min_images: 最少需要多少张成功图像
    """
    
    # 设置角点优化标准
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 获取所有标定图像
    images = glob.glob('data/calib_images/*.jpg')
    
    if not images:
        print("错误：在 data/calib_images/ 目录中没有找到标定图像！")
        return None
    
    print(f"找到 {len(images)} 张标定图像")
    
    # 存储不同尺寸的标定数据
    calibration_data = {}
    
    for i, fname in enumerate(images):
        print(f"\n处理图像 {i+1}/{len(images)}: {os.path.basename(fname)}")
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 尝试检测棋盘格
        ret, corners, pattern = find_chessboard_pattern(gray)
        
        if ret:
            print(f"  检测到棋盘格尺寸: {pattern}")
            
            # 提高角点检测精度
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 准备3D点
            objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
            objp = objp * square_size
            
            # 按尺寸分组存储
            if pattern not in calibration_data:
                calibration_data[pattern] = {
                    'objpoints': [],
                    'imgpoints': [],
                    'images': []
                }
            
            calibration_data[pattern]['objpoints'].append(objp)
            calibration_data[pattern]['imgpoints'].append(corners_refined)
            calibration_data[pattern]['images'].append(fname)
            
            # 可视化
            img_display = cv2.drawChessboardCorners(img.copy(), pattern, corners_refined, ret)
            cv2.putText(img_display, f"Pattern: {pattern}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Chessboard Detected', cv2.resize(img_display, (600, 400)))
            cv2.waitKey(300)
        else:
            print("  警告：未找到棋盘格")
            cv2.imshow('Failed', cv2.resize(img, (600, 400)))
            cv2.waitKey(300)
    
    cv2.destroyAllWindows()
    
    # 分析检测结果
    print(f"\n{'='*50}")
    print("检测结果汇总:")
    for pattern, data in calibration_data.items():
        print(f"  尺寸 {pattern}: {len(data['objpoints'])} 张图像")
    
    # 选择最常用的棋盘格尺寸
    best_pattern = None
    best_count = 0
    for pattern, data in calibration_data.items():
        if len(data['objpoints']) > best_count:
            best_count = len(data['objpoints'])
            best_pattern = pattern
    
    if best_pattern is None or best_count < min_images:
        print(f"\n错误：没有找到足够的棋盘格图像！")
        print(f"至少需要 {min_images} 张图像，最佳尺寸 {best_pattern} 只有 {best_count} 张")
        return None
    
    print(f"\n选择尺寸 {best_pattern} 进行标定（有 {best_count} 张图像）")
    
    # 使用最佳尺寸的图像进行标定
    objpoints = calibration_data[best_pattern]['objpoints']
    imgpoints = calibration_data[best_pattern]['imgpoints']
    
    print(f"使用 {len(objpoints)} 张图像进行标定...")
    
    # 相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # 计算重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"\n相机标定完成！")
    print(f"重投影误差: {mean_error/len(objpoints):.6f} (越小越好)")
    print(f"棋盘格尺寸: {best_pattern}")
    print(f"使用图像数: {len(objpoints)}")
    print(f"\n相机内参矩阵:")
    print(mtx)
    print(f"\n畸变系数:")
    print(dist)
    
    # 保存相机参数
    if not os.path.exists('models'):
        os.makedirs('models')
    
    camera_params = {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'reprojection_error': mean_error/len(objpoints),
        'image_size': gray.shape[::-1],
        'chessboard_size': best_pattern,
        'square_size': square_size,
        'used_images': calibration_data[best_pattern]['images']
    }
    
    with open('models/camera_params.pkl', 'wb') as f:
        pickle.dump(camera_params, f)
    
    print(f"\n相机参数已保存到 models/camera_params.pkl")
    
    # 显示用于标定的图像
    print(f"\n用于标定的图像:")
    for i, img_path in enumerate(camera_params['used_images']):
        print(f"  {i+1}. {os.path.basename(img_path)}")
    
    return camera_params

def undistort_and_test(camera_params):
    """测试畸变校正"""
    print(f"\n{'='*50}")
    print("测试畸变校正...")
    
    # 使用第一张用于标定的图像进行测试
    if camera_params['used_images']:
        test_image = camera_params['used_images'][0]
        img = cv2.imread(test_image)
        
        h, w = img.shape[:2]
        
        # 获取最优的新相机矩阵
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_params['camera_matrix'], 
            camera_params['dist_coeffs'], 
            (w, h), 1, (w, h))
        
        # 校正畸变
        dst = cv2.undistort(img, camera_params['camera_matrix'], 
                           camera_params['dist_coeffs'], None, newcameramtx)
        
        # 裁剪图像
        x, y, w, h = roi
        if w > 0 and h > 0:
            dst = dst[y:y+h, x:x+w]
        
        # 显示对比
        comparison = np.hstack([cv2.resize(img, (400, 300)), 
                               cv2.resize(dst, (400, 300))])
        
        cv2.putText(comparison, "Original", (50, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison, "Undistorted", (450, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Undistortion Test', comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        cv2.imwrite('results/undistorted_test.jpg', dst)
        print("畸变校正测试图像已保存到 results/undistorted_test.jpg")

if __name__ == "__main__":
    print("=" * 50)
    print("自适应相机标定程序")
    print("=" * 50)
    
    # 获取用户输入
    try:
        square_size = float(input("输入棋盘格方格边长（米）[默认0.025]: ") or "0.025")
        min_images = int(input("最少需要多少张成功图像？[默认5]: ") or "5")
    except ValueError:
        print("输入无效，使用默认值")
        square_size = 0.025
        min_images = 5
    
    # 执行标定
    params = calibrate_camera_adaptive(square_size, min_images)
    
    if params:
        # 测试畸变校正
        undistort_and_test(params)
        
        print(f"\n{'='*50}")
        print("标定成功完成！")
        print(f"棋盘格尺寸: {params['chessboard_size']}")
        print(f"重投影误差: {params['reprojection_error']:.6f}")
        print("python augment_reality_image.py")
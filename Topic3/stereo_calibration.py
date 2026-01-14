"""
立体相机标定
标定左右相机并计算立体校正参数
"""

import cv2
import numpy as np
import glob
import pickle
import os

def calibrate_stereo_camera(chessboard_size=(9, 6), square_size=0.025):
    """
    立体相机标定主函数
    
    参数:
        chessboard_size: 棋盘格内角点数量 (列数, 行数)
        square_size: 每个棋盘格方格的边长（单位：米）
    """
    
    print("=" * 50)
    print("立体相机标定")
    print("=" * 50)
    
    # 准备棋盘格3D坐标
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 准备存储数据
    objpoints = []  # 3D点
    left_imgpoints = []  # 左相机2D点
    right_imgpoints = []  # 右相机2D点
    
    # 获取左右相机图像
    left_images = sorted(glob.glob('data/calibration/left_*.jpg'))
    right_images = sorted(glob.glob('data/calibration/right_*.jpg'))
    
    if len(left_images) != len(right_images) or len(left_images) < 5:
        print(f"错误：需要至少5对立体图像！")
        print(f"左图像: {len(left_images)} 张，右图像: {len(right_images)} 张")
        return None
    
    print(f"找到 {len(left_images)} 对立体图像")
    
    image_size = None
    successful_pairs = 0
    
    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        print(f"\n处理图像对 {i+1}/{len(left_images)}:")
        print(f"  左: {os.path.basename(left_path)}")
        print(f"  右: {os.path.basename(right_path)}")
        
        # 读取图像
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        if left_img is None or right_img is None:
            print("  错误：无法读取图像")
            continue
        
        # 转换为灰度图
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = left_gray.shape[::-1]
        
        # 检测左右图像的棋盘格角点
        left_ret, left_corners = cv2.findChessboardCorners(left_gray, chessboard_size, None)
        right_ret, right_corners = cv2.findChessboardCorners(right_gray, chessboard_size, None)
        
        if left_ret and right_ret:
            # 提高角点检测精度
            left_corners2 = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1), criteria)
            right_corners2 = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), criteria)
            
            # 存储数据
            objpoints.append(objp)
            left_imgpoints.append(left_corners2)
            right_imgpoints.append(right_corners2)
            
            successful_pairs += 1
            
            # 可视化
            left_display = cv2.drawChessboardCorners(left_img.copy(), chessboard_size, left_corners2, left_ret)
            right_display = cv2.drawChessboardCorners(right_img.copy(), chessboard_size, right_corners2, right_ret)
            
            # 并排显示
            combined = np.hstack([left_display, right_display])
            cv2.imshow('Stereo Chessboard Detection', cv2.resize(combined, (1200, 400)))
            cv2.waitKey(300)
        else:
            print("  警告：未能在两张图像中都找到棋盘格")
    
    cv2.destroyAllWindows()
    
    if successful_pairs < 5:
        print(f"\n错误：只有 {successful_pairs} 对图像成功检测，需要至少5对！")
        return None
    
    print(f"\n使用 {successful_pairs} 对图像进行立体标定...")
    
    # 1. 标定左相机
    print("\n标定左相机...")
    left_ret, left_mtx, left_dist, left_rvecs, left_tvecs = cv2.calibrateCamera(
        objpoints, left_imgpoints, image_size, None, None)
    
    # 2. 标定右相机
    print("标定右相机...")
    right_ret, right_mtx, right_dist, right_rvecs, right_tvecs = cv2.calibrateCamera(
        objpoints, right_imgpoints, image_size, None, None)
    
    # 3. 立体标定
    print("进行立体标定...")
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC  # 我们已经标定了内参
    
    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, left_imgpoints, right_imgpoints,
        left_mtx, left_dist, right_mtx, right_dist,
        image_size, criteria=stereocalib_criteria, flags=flags)
    
    print(f"\n立体标定完成！")
    print(f"立体标定误差: {ret:.6f}")
    print(f"旋转矩阵 R:\n{R}")
    print(f"平移向量 T:\n{T.ravel()}")
    print(f"基线距离: {np.linalg.norm(T):.4f} 米")
    
    # 4. 立体校正
    print("\n计算立体校正参数...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_mtx, left_dist, right_mtx, right_dist,
        image_size, R, T, alpha=0.9)
    
    # 计算校正映射
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_mtx, left_dist, R1, P1, image_size, cv2.CV_32FC1)
    
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_mtx, right_dist, R2, P2, image_size, cv2.CV_32FC1)
    
    # 保存标定参数
    stereo_params = {
        'image_size': image_size,
        'left_camera_matrix': left_mtx,
        'left_dist_coeffs': left_dist,
        'right_camera_matrix': right_mtx,
        'right_dist_coeffs': right_dist,
        'rotation_matrix': R,
        'translation_vector': T,
        'essential_matrix': E,
        'fundamental_matrix': F,
        'rectify_left_R': R1,
        'rectify_right_R': R2,
        'rectify_left_P': P1,
        'rectify_right_P': P2,
        'disparity_to_depth_matrix': Q,
        'left_map1': left_map1,
        'left_map2': left_map2,
        'right_map1': right_map1,
        'right_map2': right_map2,
        'roi_left': roi1,
        'roi_right': roi2,
        'baseline': float(np.linalg.norm(T)),
        'focal_length': float(left_mtx[0, 0]),  # 假设左右相机焦距相同
        'stereo_error': float(ret)
    }
    
    # 创建模型目录
    if not os.path.exists('models'):
        os.makedirs('models')
    
    with open('models/stereo_params.pkl', 'wb') as f:
        pickle.dump(stereo_params, f)
    
    print(f"\n立体相机参数已保存到 models/stereo_params.pkl")
    
    # 测试校正效果
    test_stereo_rectification(stereo_params)
    
    return stereo_params

def test_stereo_rectification(stereo_params):
    """测试立体校正效果"""
    
    print("\n测试立体校正效果...")
    
    # 读取一对测试图像
    left_test = 'data/calibration/left_01.jpg'
    right_test = 'data/calibration/right_01.jpg'
    
    if not os.path.exists(left_test) or not os.path.exists(right_test):
        print("警告：未找到测试图像，跳过校正测试")
        return
    
    left_img = cv2.imread(left_test)
    right_img = cv2.imread(right_test)
    
    # 校正图像
    left_rectified = cv2.remap(left_img, 
                               stereo_params['left_map1'], 
                               stereo_params['left_map2'], 
                               cv2.INTER_LINEAR)
    
    right_rectified = cv2.remap(right_img, 
                                stereo_params['right_map1'], 
                                stereo_params['right_map2'], 
                                cv2.INTER_LINEAR)
    
    # 绘制水平线以检查行对齐
    left_display = left_rectified.copy()
    right_display = right_rectified.copy()
    
    h, w = left_display.shape[:2]
    for i in range(0, h, 50):
        cv2.line(left_display, (0, i), (w, i), (0, 255, 0), 1)
        cv2.line(right_display, (0, i), (w, i), (0, 255, 0), 1)
    
    # 并排显示
    combined_original = np.hstack([left_img, right_img])
    combined_rectified = np.hstack([left_display, right_display])
    
    # 垂直堆叠
    final_display = np.vstack([
        cv2.resize(combined_original, (1200, 300)),
        cv2.resize(combined_rectified, (1200, 300))
    ])
    
    # 添加标签
    cv2.putText(final_display, "Original (Left + Right)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(final_display, "Rectified (Left + Right) with horizontal lines", (10, 330),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Stereo Rectification Test', final_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存校正后的图像
    cv2.imwrite('results/left_rectified.jpg', left_rectified)
    cv2.imwrite('results/right_rectified.jpg', right_rectified)
    print("校正测试图像已保存到 results/ 目录")

def main():
    """主函数"""
    
    # 检查标定图像
    if not os.path.exists('data/calibration'):
        print("错误：未找到标定图像目录！")
        print("请在 data/calibration/ 目录中放置立体图像对：")
        print("  left_01.jpg, right_01.jpg, left_02.jpg, right_02.jpg, ...")
        return
    
    # 执行立体标定
    stereo_params = calibrate_stereo_camera(chessboard_size=(9, 6))
    
    if stereo_params:
        print("\n" + "="*50)
        print("立体标定成功完成！")
        print("="*50)
        print(f"基线距离: {stereo_params['baseline']:.4f} 米")
        print(f"焦距: {stereo_params['focal_length']:.2f} 像素")
        print(f"立体标定误差: {stereo_params['stereo_error']:.6f}")
        print("\n下一步：运行立体深度计算")
        print("python stereo_depth.py")

if __name__ == "__main__":
    main()
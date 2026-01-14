# run_calibration.py - 专门运行相机标定
import cv2
import numpy as np
import glob
import os

print("相机标定工具")
print("="*60)

# 棋盘格参数
chessboard_size = (9, 6)  # 内角点数量
square_size = 0.025  # 实际棋盘格方格尺寸（米）

# 准备3D世界坐标
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size

# 存储3D点和2D点
objpoints = []  # 3D点
imgpoints = []  # 2D点

# 查找所有标定图像
images = glob.glob('calibration_images/*.jpg')
if len(images) == 0:
    print("错误: calibration_images 文件夹中没有找到图像")
    print("请先运行 create_calibration_images.py 生成棋盘格图像")
    exit()

print(f"找到 {len(images)} 张标定图像")

# 处理每张图像
valid_images = 0
for i, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        valid_images += 1
        objpoints.append(objp)
        
        # 亚像素精确化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # 显示检测结果
        img_with_corners = cv2.drawChessboardCorners(img.copy(), chessboard_size, corners2, ret)
        
        # 调整显示尺寸
        display_img = cv2.resize(img_with_corners, (800, 600))
        cv2.imshow(f'Chessboard Detection ({valid_images}/{len(images)})', display_img)
        cv2.waitKey(500)
        
        print(f"  图像 {i+1}: 检测成功")
    else:
        print(f"  图像 {i+1}: 未检测到棋盘格")

cv2.destroyAllWindows()

if valid_images < 5:
    print(f"\n错误: 只有 {valid_images} 张图像检测到棋盘格，至少需要5张")
    exit()

print(f"\n使用 {valid_images} 张有效图像进行标定...")

# 相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("\n相机标定成功！")
    print(f"相机内参矩阵:\n{camera_matrix}")
    print(f"畸变系数:\n{dist_coeffs.ravel()}")
    
    # 计算平均重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                         camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"平均重投影误差: {mean_error/len(objpoints):.3f} 像素")
    
    # 保存相机参数
    np.savez('camera_params.npz', 
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)
    print("相机参数已保存到 camera_params.npz")
    
    # 测试去畸变
    print("\n测试去畸变效果...")
    test_img = cv2.imread(images[0])
    if test_img is not None:
        h, w = test_img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        
        # 去畸变
        undistorted = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # 显示对比
        combined = np.hstack([cv2.resize(test_img, (400, 300)), 
                             cv2.resize(undistorted, (400, 300))])
        cv2.imshow('原始图像 vs 去畸变图像', combined)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
else:
    print("相机标定失败")

print("\n标定完成！")
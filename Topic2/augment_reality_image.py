"""
增强现实主程序 - 图像版
在图像中检测棋盘格并投影3D虚拟物体
"""

import cv2
import numpy as np
import pickle
import os

def load_camera_params():
    """加载相机参数"""
    if not os.path.exists('models/camera_params.pkl'):
        print("错误：请先运行相机标定程序！")
        return None
    
    with open('models/camera_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    return params

def draw_cube(img, corners, imgpts):
    """在图像上绘制立方体"""
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # 绘制底部（绿色）
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 3)
    
    # 绘制垂直边（蓝色）
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)
    
    # 绘制顶部（红色）
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    
    return img

def draw_coordinate_axes(img, corners, imgpts):
    """绘制3D坐标轴"""
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # 原点
    origin = tuple(corners[0].ravel().astype(int))
    
    # 绘制坐标轴
    img = cv2.line(img, origin, tuple(imgpts[0].ravel()), (255, 0, 0), 5)  # X轴 - 蓝色
    img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 255, 0), 5)  # Y轴 - 绿色
    img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, 0, 255), 5)  # Z轴 - 红色
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, 'X', tuple(imgpts[0].ravel()), font, 0.5, (255, 0, 0), 2)
    img = cv2.putText(img, 'Y', tuple(imgpts[1].ravel()), font, 0.5, (0, 255, 0), 2)
    img = cv2.putText(img, 'Z', tuple(imgpts[2].ravel()), font, 0.5, (0, 0, 255), 2)
    
    return img

def augment_reality_image(image_path, camera_params, mode='cube'):
    """
    增强现实主函数 - 图像版
    
    参数:
        image_path: 输入图像路径
        camera_params: 相机参数
        mode: 绘制模式 ('cube' 或 'axes')
    """
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 棋盘格参数
    chessboard_size = camera_params['chessboard_size']
    
    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if not ret:
        print("错误：在图像中未找到棋盘格！")
        print("请确保图像中包含完整的棋盘格，并且大小与标定时一致。")
        return None
    
    # 提高角点检测精度
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # 准备3D点
    square_size = camera_params['square_size']
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size
    
    # 计算相机姿态（外参）
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, 
                                    camera_params['camera_matrix'], 
                                    camera_params['dist_coeffs'])
    
    if mode == 'cube':
        # 定义立方体3D坐标（相对于棋盘格原点）
        cube_size = 0.02  # 立方体边长（米）
        cube_3d = np.float32([
            [0, 0, 0],             # 0: 底部-左后
            [cube_size, 0, 0],     # 1: 底部-右后
            [cube_size, cube_size, 0],  # 2: 底部-右前
            [0, cube_size, 0],     # 3: 底部-左前
            [0, 0, -cube_size],    # 4: 顶部-左后
            [cube_size, 0, -cube_size], # 5: 顶部-右后
            [cube_size, cube_size, -cube_size], # 6: 顶部-右前
            [0, cube_size, -cube_size]  # 7: 顶部-左前
        ])
        
        # 投影3D点到2D图像
        imgpts, _ = cv2.projectPoints(cube_3d, rvecs, tvecs, 
                                     camera_params['camera_matrix'], 
                                     camera_params['dist_coeffs'])
        
        # 绘制立方体
        result = draw_cube(img.copy(), corners2, imgpts)
        
    elif mode == 'axes':
        # 定义坐标轴3D点
        axes_3d = np.float32([
            [0.03, 0, 0],     # X轴
            [0, 0.03, 0],     # Y轴
            [0, 0, -0.03]     # Z轴
        ])
        
        # 投影坐标轴
        imgpts, _ = cv2.projectPoints(axes_3d, rvecs, tvecs,
                                     camera_params['camera_matrix'],
                                     camera_params['dist_coeffs'])
        
        # 绘制坐标轴
        result = draw_coordinate_axes(img.copy(), corners2, imgpts)
    
    # 绘制棋盘格角点（可选）
    result = cv2.drawChessboardCorners(result, chessboard_size, corners2, ret)
    
    return result

def main():
    """主函数"""
    # 加载相机参数
    camera_params = load_camera_params()
    if camera_params is None:
        return
    
    print("相机参数加载成功！")
    print(f"图像尺寸: {camera_params['image_size']}")
    print(f"重投影误差: {camera_params['reprojection_error']:.6f}")
    
    # 测试图像路径
    test_images = [
        'data/test_images/test1.jpg',
        'data/test_images/test2.jpg',
        'data/test_images/chessboard_test.jpg'
    ]
    
    for i, img_path in enumerate(test_images):
        if os.path.exists(img_path):
            print(f"\n处理图像: {img_path}")
            
            # 两种模式都试一下
            for mode in ['cube', 'axes']:
                print(f"  模式: {mode}")
                
                # 执行增强现实
                result = augment_reality_image(img_path, camera_params, mode)
                
                if result is not None:
                    # 保存结果
                    output_dir = 'results'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    filename = os.path.basename(img_path).split('.')[0]
                    output_path = f'{output_dir}/{filename}_{mode}.jpg'
                    cv2.imwrite(output_path, result)
                    print(f"  结果已保存: {output_path}")
                    
                    # 显示结果
                    cv2.imshow(f'AR Result - {mode}', result)
                    cv2.waitKey(1000)
            
    cv2.destroyAllWindows()
    print("\n增强现实处理完成！")

if __name__ == "__main__":
    main()
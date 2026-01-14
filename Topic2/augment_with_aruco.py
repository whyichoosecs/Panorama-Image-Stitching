"""
使用ArUco标记进行增强现实
更稳定的标记检测方法
"""

import cv2
import numpy as np
import pickle

def create_aruco_markers():
    """创建ArUco标记"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # 创建并保存标记
    for i in range(5):
        marker_size = 200  # 像素
        marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size, marker_image, 1)
        
        # 保存标记
        cv2.imwrite(f'data/aruco_marker_{i}.png', marker_image)
        print(f"ArUco标记 {i} 已保存")
        
        # 显示标记
        cv2.imshow(f'ArUco Marker {i}', marker_image)
        cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    return aruco_dict

def detect_aruco_markers(frame, aruco_dict):
    """检测ArUco标记"""
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # 检测标记
    corners, ids, rejected = detector.detectMarkers(frame)
    
    return corners, ids

def draw_aruco_cube(frame, corners, camera_matrix, dist_coeffs):
    """在ArUco标记上绘制立方体"""
    
    # 标记大小（米）
    marker_length = 0.05
    
    # 定义立方体3D点（相对于标记中心）
    cube_points = np.float32([
        [-marker_length/2, -marker_length/2, 0],
        [marker_length/2, -marker_length/2, 0],
        [marker_length/2, marker_length/2, 0],
        [-marker_length/2, marker_length/2, 0],
        [-marker_length/2, -marker_length/2, marker_length],
        [marker_length/2, -marker_length/2, marker_length],
        [marker_length/2, marker_length/2, marker_length],
        [-marker_length/2, marker_length/2, marker_length]
    ])
    
    # 估计标记姿态
    if len(corners) > 0:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs)
        
        # 为每个检测到的标记绘制立方体
        for i in range(len(ids)):
            # 投影立方体点
            imgpts, _ = cv2.projectPoints(cube_points, rvecs[i], tvecs[i],
                                         camera_matrix, dist_coeffs)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            
            # 绘制底部
            frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 3)
            
            # 绘制边
            for j in range(4):
                frame = cv2.line(frame, tuple(imgpts[j]), tuple(imgpts[j+4]), (255, 0, 0), 3)
            
            # 绘制顶部
            frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 3)
    
    return frame

def main():
    """主函数"""
    print("ArUco增强现实演示")
    
    # 创建ArUco标记（如果需要）
    create_markers = input("创建新的ArUco标记？ (y/n): ").lower() == 'y'
    if create_markers:
        create_aruco_markers()
    
    # 加载相机参数
    with open('models/camera_params.pkl', 'rb') as f:
        camera_params = pickle.load(f)
    
    camera_matrix = camera_params['camera_matrix']
    dist_coeffs = camera_params['dist_coeffs']
    
    # 使用摄像头
    cap = cv2.VideoCapture(0)
    
    # ArUco字典
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    print("\n按 'q' 退出程序")
    print("在摄像头前展示ArUco标记（使用data/目录下生成的标记）")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测ArUco标记
        corners, ids = detect_aruco_markers(frame, aruco_dict)
        
        # 绘制检测到的标记
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 在标记上绘制立方体
            frame = draw_aruco_cube(frame, corners, camera_matrix, dist_coeffs)
            
            # 显示检测到的ID
            cv2.putText(frame, f"Markers detected: {len(ids)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No markers detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 显示帧
        cv2.imshow('ArUco Augmented Reality', frame)
        
        # 退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
"""
增强现实主程序 - 视频版
实时视频中投影3D虚拟物体
"""

import cv2
import numpy as np
import pickle
import time

def load_camera_params():
    """加载相机参数"""
    with open('models/camera_params.pkl', 'rb') as f:
        params = pickle.load(f)
    return params

def draw_pyramid(img, corners, imgpts):
    """绘制金字塔"""
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # 绘制底部（四边形）
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 255), 3)
    
    # 绘制边
    for i in range(4):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[4]), (255, 255, 0), 3)
    
    return img

def process_frame(frame, camera_params, chessboard_size=(9, 6)):
    """处理单帧图像"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 查找棋盘格
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 准备3D点
        square_size = camera_params['square_size']
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp = objp * square_size
        
        # 计算相机姿态
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2,
                                        camera_params['camera_matrix'],
                                        camera_params['dist_coeffs'])
        
        # 定义金字塔3D点
        pyramid_size = 0.015
        pyramid_3d = np.float32([
            [0, 0, 0],                    # 0: 底部-左后
            [pyramid_size, 0, 0],         # 1: 底部-右后
            [pyramid_size, pyramid_size, 0],  # 2: 底部-右前
            [0, pyramid_size, 0],         # 3: 底部-左前
            [pyramid_size/2, pyramid_size/2, -pyramid_size]  # 4: 顶部
        ])
        
        # 投影到2D
        imgpts, _ = cv2.projectPoints(pyramid_3d, rvecs, tvecs,
                                     camera_params['camera_matrix'],
                                     camera_params['dist_coeffs'])
        
        # 绘制金字塔
        frame = draw_pyramid(frame, corners2, imgpts)
        
        # 显示棋盘格角点
        frame = cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)
        
        # 显示状态
        cv2.putText(frame, "Tracking: ON", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking: OFF - Show Chessboard", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame, ret

def main():
    """主函数"""
    # 加载相机参数
    camera_params = load_camera_params()
    
    # 选择视频源
    print("选择视频源:")
    print("1. 摄像头")
    print("2. 视频文件")
    
    choice = input("请输入选择 (1/2): ")
    
    if choice == '1':
        # 使用摄像头
        cap = cv2.VideoCapture(0)
        print("正在打开摄像头...")
    elif choice == '2':
        # 使用视频文件
        video_path = input("请输入视频文件路径: ")
        cap = cv2.VideoCapture(video_path)
    else:
        print("无效选择，使用默认摄像头")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开视频源")
        return
    
    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频参数: {width}x{height}, {fps:.2f} FPS")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('results/ar_output.avi', fourcc, 20.0, (width, height))
    
    print("\n按 'q' 退出")
    print("按 's' 保存当前帧")
    
    frame_count = 0
    tracking_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 处理帧
        processed_frame, is_tracking = process_frame(frame, camera_params)
        
        if is_tracking:
            tracking_count += 1
        
        # 计算并显示FPS
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 计算并显示跟踪率
        tracking_rate = (tracking_count / frame_count * 100) if frame_count > 0 else 0
        cv2.putText(processed_frame, f"Tracking: {tracking_rate:.1f}%", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示结果
        cv2.imshow('Augmented Reality', processed_frame)
        
        # 写入视频
        out.write(processed_frame)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'results/frame_{frame_count}.jpg', processed_frame)
            print(f"帧 {frame_count} 已保存")
    
    # 计算最终统计
    total_time = time.time() - start_time
    print(f"\n处理完成！")
    print(f"总帧数: {frame_count}")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均FPS: {frame_count/total_time:.2f}")
    print(f"跟踪率: {tracking_rate:.1f}%")
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
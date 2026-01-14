"""
3D点云生成
从深度图生成3D点云
"""

import cv2
import numpy as np
import pickle
import os
from plyfile import PlyData, PlyElement

def depth_to_point_cloud(depth_map, color_img, stereo_params, downsample_factor=4):
    """
    从深度图生成3D点云
    
    参数:
        depth_map: 深度图（单位：米）
        color_img: 彩色图像（用于点云颜色）
        stereo_params: 立体相机参数
        downsample_factor: 下采样因子，减少点数
    """
    
    # 获取相机内参
    if 'disparity_to_depth_matrix' in stereo_params:
        # 使用Q矩阵进行投影
        Q = stereo_params['disparity_to_depth_matrix']
        points_3d = cv2.reprojectImageTo3D(depth_map, Q)
    else:
        # 手动计算3D点
        h, w = depth_map.shape
        
        # 创建像素坐标网格
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # 相机内参（假设主点在图像中心）
        fx = stereo_params['focal_length']
        fy = stereo_params['focal_length']
        cx = w / 2
        cy = h / 2
        
        # 计算3D坐标
        # Z = depth
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        
        Z = depth_map
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        points_3d = np.stack([X, Y, Z], axis=-1)
    
    # 下采样以减少点数
    if downsample_factor > 1:
        points_3d = points_3d[::downsample_factor, ::downsample_factor]
        color_img = color_img[::downsample_factor, ::downsample_factor]
        depth_map = depth_map[::downsample_factor, ::downsample_factor]
    
    # 展平数组
    h, w = points_3d.shape[:2]
    points_3d = points_3d.reshape(-1, 3)
    
    # 获取颜色（从BGR转换为RGB）
    colors = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    colors = colors.reshape(-1, 3)
    
    # 移除无效点（深度为0或无穷大）
    valid_mask = (depth_map.flatten() > 0) & (depth_map.flatten() < 10)  # 假设有效深度在0-10米之间
    points_3d = points_3d[valid_mask]
    colors = colors[valid_mask]
    
    print(f"生成 {len(points_3d)} 个3D点")
    
    return points_3d, colors

def save_point_cloud_ply(points, colors, filename='point_cloud.ply'):
    """保存点云为PLY格式"""
    
    # 确保输出目录存在
    os.makedirs('results', exist_ok=True)
    
    # 创建顶点数据
    vertices = np.empty(len(points), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    vertices['x'] = points[:, 0].astype('f4')
    vertices['y'] = points[:, 1].astype('f4')
    vertices['z'] = points[:, 2].astype('f4')
    vertices['red'] = colors[:, 0].astype('u1')
    vertices['green'] = colors[:, 1].astype('u1')
    vertices['blue'] = colors[:, 2].astype('u1')
    
    # 创建PLY元素
    vertex_element = PlyElement.describe(vertices, 'vertex')
    
    # 保存PLY文件
    ply_filename = os.path.join('results', filename)
    PlyData([vertex_element], text=True).write(ply_filename)
    
    print(f"点云已保存到 {ply_filename}")
    
    return ply_filename

def save_point_cloud_txt(points, colors, filename='point_cloud.txt'):
    """保存点云为文本格式（兼容性更好）"""
    
    txt_filename = os.path.join('results', filename)
    
    with open(txt_filename, 'w') as f:
        f.write("# 3D Point Cloud Data\n")
        f.write("# X Y Z R G B\n")
        
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    
    print(f"点云文本文件已保存到 {txt_filename}")
    
    return txt_filename

def visualize_point_cloud_3d(points, colors):
    """使用Matplotlib可视化3D点云"""
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # 下采样以加快显示速度
        if len(points) > 10000:
            step = len(points) // 10000
            points = points[::step]
            colors = colors[::step]
        
        # 将颜色归一化到0-1范围
        colors_normalized = colors / 255.0
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制点云
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors_normalized, s=1, alpha=0.6, marker='.')
        
        # 设置坐标轴标签
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Point Cloud Reconstruction')
        
        # 设置坐标轴范围
        max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                             points[:, 1].max()-points[:, 1].min(),
                             points[:, 2].max()-points[:, 2].min()]).max() / 2.0
        
        mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 设置视角
        ax.view_init(elev=20, azim=-60)
        
        plt.tight_layout()
        plt.savefig('results/3d_point_cloud.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("3D点云可视化图已保存到 results/3d_point_cloud.png")
        
    except ImportError as e:
        print(f"警告：无法导入Matplotlib 3D模块: {e}")
        print("请安装: pip install matplotlib")

def generate_point_cloud_from_depth():
    """从深度图生成点云主函数"""
    
    print("=" * 50)
    print("3D点云生成")
    print("=" * 50)
    
    # 加载立体相机参数
    if not os.path.exists('models/stereo_params.pkl'):
        print("错误：请先运行立体相机标定和深度计算！")
        print("运行顺序:")
        print("1. python stereo_calibration.py")
        print("2. python stereo_depth.py")
        return None
    
    with open('models/stereo_params.pkl', 'rb') as f:
        stereo_params = pickle.load(f)
    
    # 查找深度图和彩色图像
    depth_map_path = 'results/depth_map_gray.png'
    color_image_path = 'results/left_rectified.jpg'  # 或使用原始左图像
    
    if not os.path.exists(depth_map_path):
        print(f"错误：未找到深度图 {depth_map_path}")
        print("请先运行 stereo_depth.py 生成深度图")
        return None
    
    if not os.path.exists(color_image_path):
        # 尝试查找其他彩色图像
        color_images = glob.glob('data/stereo_pairs/left_*.jpg')
        if color_images:
            color_image_path = color_images[0]
        else:
            print("错误：未找到彩色图像")
            return None
    
    print(f"使用深度图: {depth_map_path}")
    print(f"使用彩色图像: {color_image_path}")
    
    # 读取深度图（假设是8位或16位图像）
    depth_img = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    
    # 如果深度图是8位，需要转换回实际深度值
    if depth_img.dtype == np.uint8:
        # 假设深度图已经归一化到0-255
        depth_map = depth_img.astype(np.float32) / 255.0 * 10.0  # 假设最大深度10米
    elif depth_img.dtype == np.uint16:
        depth_map = depth_img.astype(np.float32) / 65535.0 * 10.0  # 假设最大深度10米
    else:
        depth_map = depth_img.astype(np.float32)
    
    # 读取彩色图像
    color_img = cv2.imread(color_image_path)
    
    if depth_map is None or color_img is None:
        print("错误：无法读取图像")
        return None
    
    # 调整彩色图像大小以匹配深度图
    if color_img.shape[:2] != depth_map.shape:
        color_img = cv2.resize(color_img, (depth_map.shape[1], depth_map.shape[0]))
    
    # 生成点云
    print("\n生成3D点云...")
    points_3d, colors = depth_to_point_cloud(depth_map, color_img, stereo_params, downsample_factor=4)
    
    if len(points_3d) == 0:
        print("错误：未能生成有效的点云")
        return None
    
    # 保存点云
    print("\n保存点云文件...")
    ply_file = save_point_cloud_ply(points_3d, colors, 'stereo_point_cloud.ply')
    txt_file = save_point_cloud_txt(points_3d, colors, 'stereo_point_cloud.txt')
    
    # 可视化点云
    print("\n生成3D可视化...")
    visualize_point_cloud_3d(points_3d, colors)
    
    print("\n" + "="*50)
    print("3D点云生成完成！")
    print("="*50)
    print(f"生成点数: {len(points_3d)}")
    print(f"PLY文件: {ply_file}")
    print(f"文本文件: {txt_file}")
    print("\n可以使用MeshLab、CloudCompare或Blender打开PLY文件查看3D点云")
    
    return points_3d, colors

def main():
    """主函数"""
    
    # 导入glob
    import glob
    
    result = generate_point_cloud_from_depth()
    
    if result:
        points_3d, colors = result
        
        # 显示点云统计信息
        print("\n点云统计:")
        print(f"  X范围: [{points_3d[:, 0].min():.3f}, {points_3d[:, 0].max():.3f}] 米")
        print(f"  Y范围: [{points_3d[:, 1].min():.3f}, {points_3d[:, 1].max():.3f}] 米")
        print(f"  Z范围（深度）: [{points_3d[:, 2].min():.3f}, {points_3d[:, 2].max():.3f}] 米")
        print(f"  平均深度: {points_3d[:, 2].mean():.3f} 米")

if __name__ == "__main__":
    main()
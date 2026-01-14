"""
立体深度计算
从立体图像对计算深度图
"""

import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import cm

def load_stereo_params():
    """加载立体相机参数"""
    if not os.path.exists('models/stereo_params.pkl'):
        print("错误：请先运行立体相机标定程序！")
        print("运行: python stereo_calibration.py")
        return None
    
    with open('models/stereo_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    print("立体相机参数加载成功")
    print(f"图像尺寸: {params['image_size']}")
    print(f"基线: {params['baseline']:.4f} 米")
    print(f"焦距: {params['focal_length']:.2f} 像素")
    
    return params

def preprocess_images(left_img, right_img):
    """预处理图像以提高匹配效果"""
    
    # 转换为灰度图
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # 直方图均衡化（增强对比度）
    left_eq = cv2.equalizeHist(left_gray)
    right_eq = cv2.equalizeHist(right_gray)
    
    return left_eq, right_eq

def compute_disparity(left_img, right_img, method='SGBM'):
    """
    计算视差图
    
    参数:
        method: 'BM' (Block Matching) 或 'SGBM' (Semi-Global Block Matching)
    """
    
    # 预处理图像
    left_processed, right_processed = preprocess_images(left_img, right_img)
    
    if method == 'BM':
        # 使用Block Matching算法
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        
        # 调整参数
        stereo.setPreFilterType(1)
        stereo.setPreFilterSize(5)
        stereo.setPreFilterCap(31)
        stereo.setTextureThreshold(10)
        stereo.setUniquenessRatio(15)
        stereo.setSpeckleRange(32)
        stereo.setSpeckleWindowSize(100)
        
    elif method == 'SGBM':
        # 使用Semi-Global Block Matching算法（通常效果更好）
        window_size = 5
        min_disp = 0
        num_disp = 16 * 6  # 必须是16的倍数
        
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    
    else:
        raise ValueError(f"未知的方法: {method}")
    
    print(f"使用 {method} 算法计算视差图...")
    disparity = stereo.compute(left_processed, right_processed).astype(np.float32)
    
    # 归一化视差图用于显示
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, 
                                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return disparity, disparity_normalized

def disparity_to_depth(disparity, stereo_params):
    """
    将视差图转换为深度图
    
    深度 = (焦距 × 基线) / 视差
    """
    
    # 避免除以零
    disparity[disparity == 0] = 0.1
    
    # 提取参数
    focal_length = stereo_params['focal_length']
    baseline = stereo_params['baseline']
    
    # 计算深度
    depth = (focal_length * baseline) / disparity
    
    # 限制深度范围（可选）
    max_depth = 10.0  # 10米
    depth[depth > max_depth] = max_depth
    
    return depth

def postprocess_depth(depth_map):
    """后处理深度图"""
    
    # 应用中值滤波去除噪声
    depth_filtered = cv2.medianBlur(depth_map.astype(np.float32), 5)
    
    # 可选：使用双边滤波保持边缘
    # depth_filtered = cv2.bilateralFilter(depth_map.astype(np.float32), 9, 75, 75)
    
    return depth_filtered

def visualize_results(left_img, disparity_norm, depth_map):
    """可视化结果"""
    
    # 创建Matplotlib图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 左图像
    axes[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Left Image')
    axes[0, 0].axis('off')
    
    # 2. 归一化视差图
    axes[0, 1].imshow(disparity_norm, cmap='jet')
    axes[0, 1].set_title('Disparity Map (Normalized)')
    axes[0, 1].axis('off')
    
    # 3. 深度图（彩色）
    depth_normalized = cv2.normalize(depth_map, None, alpha=0, beta=1, 
                                     norm_type=cv2.NORM_MINMAX)
    axes[1, 0].imshow(depth_normalized, cmap='jet')
    axes[1, 0].set_title('Depth Map (Colored)')
    axes[1, 0].axis('off')
    
    # 4. 深度图（灰度）
    axes[1, 1].imshow(depth_map, cmap='gray')
    axes[1, 1].set_title('Depth Map (Grayscale)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/stereo_depth_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 保存各个图像
    cv2.imwrite('results/disparity_map.png', disparity_norm)
    
    depth_colored = (depth_normalized * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)
    cv2.imwrite('results/depth_map_colored.png', depth_colored)
    
    depth_gray = (depth_map / depth_map.max() * 255).astype(np.uint8)
    cv2.imwrite('results/depth_map_gray.png', depth_gray)
    
    print("结果已保存到 results/ 目录")

def process_stereo_pair(left_path, right_path, stereo_params, method='SGBM'):
    """处理单对立体图像"""
    
    print(f"\n处理立体图像对:")
    print(f"  左图像: {os.path.basename(left_path)}")
    print(f"  右图像: {os.path.basename(right_path)}")
    
    # 读取图像
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        print("错误：无法读取图像")
        return None
    
    # 如果需要，进行立体校正
    if 'left_map1' in stereo_params:
        print("应用立体校正...")
        left_img = cv2.remap(left_img, 
                            stereo_params['left_map1'], 
                            stereo_params['left_map2'], 
                            cv2.INTER_LINEAR)
        right_img = cv2.remap(right_img, 
                             stereo_params['right_map1'], 
                             stereo_params['right_map2'], 
                             cv2.INTER_LINEAR)
    
    # 计算视差图
    disparity, disparity_norm = compute_disparity(left_img, right_img, method)
    
    # 转换为深度图
    depth_map = disparity_to_depth(disparity, stereo_params)
    
    # 后处理深度图
    depth_processed = postprocess_depth(depth_map)
    
    return left_img, disparity_norm, depth_processed

def main():
    """主函数"""
    
    print("=" * 50)
    print("立体视觉深度计算")
    print("=" * 50)
    
    # 加载立体相机参数
    stereo_params = load_stereo_params()
    if stereo_params is None:
        return
    
    # 选择处理方法
    print("\n选择立体匹配算法:")
    print("1. SGBM (Semi-Global Block Matching) - 推荐")
    print("2. BM (Block Matching) - 快速但精度较低")
    
    choice = input("请输入选择 (1/2): ")
    method = 'SGBM' if choice == '1' else 'BM'
    
    # 查找立体图像对
    stereo_pairs_dir = 'data/stereo_pairs'
    if not os.path.exists(stereo_pairs_dir):
        print(f"错误：未找到立体图像对目录 {stereo_pairs_dir}")
        print("请将立体图像对放置在此目录中，命名为:")
        print("  left_01.jpg, right_01.jpg, left_02.jpg, right_02.jpg, ...")
        return
    
    # 获取所有左图像
    left_images = sorted(glob.glob(os.path.join(stereo_pairs_dir, 'left_*.jpg')))
    
    if not left_images:
        print("错误：未找到左图像文件")
        return
    
    print(f"\n找到 {len(left_images)} 对立体图像")
    
    for left_path in left_images:
        # 构建对应的右图像路径
        filename = os.path.basename(left_path)
        right_filename = filename.replace('left_', 'right_')
        right_path = os.path.join(stereo_pairs_dir, right_filename)
        
        if not os.path.exists(right_path):
            print(f"警告：未找到对应的右图像 {right_filename}")
            continue
        
        # 处理立体图像对
        results = process_stereo_pair(left_path, right_path, stereo_params, method)
        
        if results:
            left_img, disparity_norm, depth_map = results
            
            # 创建输出目录
            os.makedirs('results', exist_ok=True)
            
            # 保存中间结果
            base_name = os.path.splitext(filename)[0]
            cv2.imwrite(f'results/{base_name}_disparity.png', disparity_norm)
            
            # 可视化结果
            visualize_results(left_img, disparity_norm, depth_map)
            
            # 询问是否继续处理下一对
            if len(left_images) > 1:
                continue_processing = input(f"\n处理下一对图像？(y/n): ")
                if continue_processing.lower() != 'y':
                    break
    
    print("\n" + "="*50)
    print("立体深度计算完成！")
    print("="*50)
    print("结果已保存到 results/ 目录")
    print("\n下一步：运行3D点云生成")
    print("python stereo_3d_reconstruction.py")

if __name__ == "__main__":
    # 导入glob用于文件查找
    import glob
    main()
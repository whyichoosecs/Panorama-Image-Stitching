"""
Panorama Image Stitching - Complete Implementation
Computer Vision Project 1: CS460/EIE460/SE460
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import argparse
import os
import time
from scipy.ndimage import distance_transform_edt

class PanoramaStitcher:
    """
    全景图像拼接类
    实现多张重叠图像的拼接
    """
    
    def __init__(self, feature_type='sift', match_ratio=0.75, ransac_thresh=5.0, 
                 blending_method='simple', display_steps=False):
        """
        初始化拼接器
        
        参数:
        feature_type: 特征检测类型 ('sift', 'orb', 'surf', 'akaze')
        match_ratio: Lowe's ratio test阈值
        ransac_thresh: RANSAC重投影误差阈值
        blending_method: 融合方法 ('simple', 'linear', 'multiband')
        display_steps: 是否显示中间步骤
        """
        self.feature_type = feature_type
        self.match_ratio = match_ratio
        self.ransac_thresh = ransac_thresh
        self.blending_method = blending_method
        self.display_steps = display_steps
        
        # 存储中间结果
        self.images = []
        self.keypoints = []
        self.descriptors = []
        self.panorama = None
        self.homographies = []
        
        # 创建特征检测器
        self._init_detector()
        
        print(f"Panorama Stitcher initialized with: {self.feature_type.upper()}")
    
    def _init_detector(self):
        """初始化特征检测器"""
        if self.feature_type == 'sift':
            self.detector = cv2.SIFT_create(
                nfeatures=0,          # 不限特征点数量
                nOctaveLayers=3,      # 每层的尺度数
                contrastThreshold=0.04,  # 对比度阈值
                edgeThreshold=10,     # 边缘阈值
                sigma=1.6            # 高斯模糊sigma
            )
        elif self.feature_type == 'orb':
            self.detector = cv2.ORB_create(
                nfeatures=5000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=20
            )
        elif self.feature_type == 'surf':
            try:
                self.detector = cv2.xfeatures2d.SURF_create(
                    hessianThreshold=100,
                    nOctaves=4,
                    nOctaveLayers=3,
                    extended=False,
                    upright=False
                )
            except:
                print("SURF not available, using SIFT instead")
                self.detector = cv2.SIFT_create()
                self.feature_type = 'sift'
        elif self.feature_type == 'akaze':
            self.detector = cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                descriptor_size=0,
                descriptor_channels=3,
                threshold=0.001,
                nOctaves=4,
                nOctaveLayers=4,
                diffusivity=cv2.KAZE_DIFF_PM_G2
            )
        else:
            print(f"Unknown feature type: {self.feature_type}, using SIFT")
            self.detector = cv2.SIFT_create()
            self.feature_type = 'sift'
    
    def load_images(self, image_paths):
        """
        加载图像
        
        参数:
        image_paths: 图像路径列表
        
        返回:
        加载的图像列表
        """
        self.images = []
        for i, path in enumerate(image_paths):
            if not os.path.exists(path):
                print(f"Warning: Image {path} not found!")
                continue
                
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Failed to load image {path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
            self.images.append(img)
            print(f"Loaded image {i+1}: {path}, size: {img.shape}")
        
        if len(self.images) < 2:
            raise ValueError("Need at least 2 images for stitching")
        
        return self.images
    
    def detect_features(self):
        """
        检测所有图像的特征点和描述子
        """
        self.keypoints = []
        self.descriptors = []
        
        print("Detecting features...")
        start_time = time.time()
        
        for i, img in enumerate(self.images):
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # 检测关键点和计算描述子
            kp, des = self.detector.detectAndCompute(gray, None)
            
            self.keypoints.append(kp)
            self.descriptors.append(des)
            
            print(f"  Image {i+1}: {len(kp)} keypoints")
        
        elapsed_time = time.time() - start_time
        print(f"Feature detection completed in {elapsed_time:.2f} seconds")
        
        # 可视化特征点
        if self.display_steps:
            self.visualize_keypoints()
    
    def visualize_keypoints(self, max_images=2):
        """可视化特征点"""
        fig, axes = plt.subplots(1, min(max_images, len(self.images)), figsize=(15, 5))
        
        if len(self.images) == 1:
            axes = [axes]
        
        for i in range(min(max_images, len(self.images))):
            img_with_kp = cv2.drawKeypoints(
                cv2.cvtColor(self.images[i], cv2.COLOR_RGB2GRAY),
                self.keypoints[i],
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            axes[i].imshow(img_with_kp)
            axes[i].set_title(f'Image {i+1}: {len(self.keypoints[i])} keypoints')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def match_features(self, desc1, desc2):
        """
        匹配两幅图像的特征描述子
        
        参数:
        desc1: 第一幅图像的描述子
        desc2: 第二幅图像的描述子
        
        返回:
        匹配点列表
        """
        # 对于ORB特征，使用汉明距离
        if self.feature_type == 'orb':
            # 暴力匹配器
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)
            return matches
        else:
            # 对于SIFT/SURF/AKAZE，使用FLANN匹配器
            # FLANN参数
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            # KNN匹配 (k=2)
            matches = flann.knnMatch(desc1, desc2, k=2)
            
            # 应用Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
            
            return good_matches
    
    def compute_homography(self, kp1, kp2, matches):
        """
        使用RANSAC计算单应性矩阵
        
        参数:
        kp1: 第一幅图像的关键点
        kp2: 第二幅图像的关键点
        matches: 匹配点
        
        返回:
        单应性矩阵和内点掩码
        """
        if len(matches) < 4:
            print(f"Warning: Only {len(matches)} matches, need at least 4")
            return None, None
        
        # 提取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC估计单应性矩阵
        H, mask = cv2.findHomography(
            src_pts, 
            dst_pts, 
            cv2.RANSAC, 
            self.ransac_thresh
        )
        
        # 统计内点（inliers）
        if mask is not None:
            inliers = mask.ravel().tolist()
            num_inliers = sum(inliers)
            inlier_ratio = num_inliers / len(matches)
            
            print(f"  Matches: {len(matches)}, Inliers: {num_inliers}, Ratio: {inlier_ratio:.3f}")
        else:
            num_inliers = 0
            inlier_ratio = 0
            print(f"  Matches: {len(matches)}, Homography estimation failed")
        
        return H, mask
    
    def visualize_matches(self, img1, img2, kp1, kp2, matches, title="Feature Matches"):
        """可视化特征匹配"""
        match_img = cv2.drawMatches(
            cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
            kp1,
            cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY),
            kp2,
            matches[:100],  # 只显示前100个匹配点
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        plt.figure(figsize=(15, 6))
        plt.imshow(match_img)
        plt.title(f'{title} ({len(matches)} matches)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def stitch_pair(self, img1, img2, is_left=True):
        """
        拼接两张图像
        
        参数:
        img1: 第一幅图像（参考图像）
        img2: 第二幅图像（待变换图像）
        is_left: 是否将img2拼接到img1的左侧
        
        返回:
        拼接结果
        """
        print(f"Stitching pair (is_left={is_left})...")
        
        # 检测特征
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        # 匹配特征
        matches = self.match_features(des1, des2)
        
        if len(matches) < 10:
            print(f"Warning: Only {len(matches)} matches, stitching may fail")
            return img1
        
        # 可视化匹配结果
        if self.display_steps:
            self.visualize_matches(img1, img2, kp1, kp2, matches, 
                                  f"Feature Matching ({self.feature_type.upper()})")
        
        # 计算单应性矩阵
        if is_left:
            # 将img2拼接到img1的左侧
            H, mask = self.compute_homography(kp2, kp1, matches)
        else:
            # 将img2拼接到img1的右侧
            H, mask = self.compute_homography(kp1, kp2, matches)
        
        if H is None:
            print("Homography estimation failed, returning original image")
            return img1
        
        # 保存单应性矩阵
        self.homographies.append(H)
        
        # 执行拼接
        if is_left:
            result = self._warp_and_stitch_left(img1, img2, H)
        else:
            result = self._warp_and_stitch_right(img1, img2, H)
        
        return result
    
    def _warp_and_stitch_left(self, img1, img2, H):
        """
        将img2拼接到img1的左侧
        
        参数:
        img1: 右侧图像（参考图像）
        img2: 左侧图像（待变换图像）
        H: 从img2到img1的单应性矩阵
        
        返回:
        拼接结果
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 获取img2的四个角点
        corners_img2 = np.array([
            [0, 0, 1],
            [0, h2-1, 1],
            [w2-1, h2-1, 1],
            [w2-1, 0, 1]
        ])
        
        # 将角点变换到img1的坐标系
        transformed_corners = np.dot(H, corners_img2.T).T
        transformed_corners = transformed_corners / transformed_corners[:, 2].reshape(-1, 1)
        
        # 计算画布大小
        x_min = min(0, transformed_corners[:, 0].min())
        x_max = max(w1, transformed_corners[:, 0].max())
        y_min = min(0, transformed_corners[:, 1].min())
        y_max = max(h1, transformed_corners[:, 1].max())
        
        # 添加平移变换，确保所有像素都在正坐标区域
        translation_x = -x_min if x_min < 0 else 0
        translation_y = -y_min if y_min < 0 else 0
        
        # 新的画布尺寸
        canvas_width = int(x_max - x_min)
        canvas_height = int(y_max - y_min)
        
        # 创建平移矩阵
        translation_matrix = np.array([
            [1, 0, translation_x],
            [0, 1, translation_y],
            [0, 0, 1]
        ])
        
        # 应用变换矩阵将img2扭曲到画布上
        warped_img2 = cv2.warpPerspective(
            img2,
            translation_matrix.dot(H),  # 先H变换，再平移
            (canvas_width, canvas_height)
        )
        
        # 创建img1在画布上的掩码
        img1_on_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        img1_on_canvas[translation_y:translation_y+h1, translation_x:translation_x+w1] = img1
        
        # 融合图像
        result = self._blend_images(img1_on_canvas, warped_img2)
        
        return result
    
    def _warp_and_stitch_right(self, img1, img2, H):
        """
        将img2拼接到img1的右侧
        
        参数:
        img1: 左侧图像（参考图像）
        img2: 右侧图像（待变换图像）
        H: 从img1到img2的单应性矩阵
        
        返回:
        拼接结果
        """
        # 反转单应性矩阵
        H_inv = np.linalg.inv(H)
        return self._warp_and_stitch_left(img2, img1, H_inv)
    
    def _blend_images(self, img1, img2):
        """
        融合两幅图像
        
        参数:
        img1: 第一幅图像
        img2: 第二幅图像
        
        返回:
        融合后的图像
        """
        # 创建掩码
        mask1 = (img1 > 0).any(axis=2).astype(np.float32)
        mask2 = (img2 > 0).any(axis=2).astype(np.float32)
        
        # 重叠区域掩码
        overlap = (mask1 > 0) & (mask2 > 0)
        
        # 初始化结果
        result = img1.copy()
        
        # 处理非重叠区域
        result[mask2 > 0] = img2[mask2 > 0]
        
        # 处理重叠区域
        if np.any(overlap):
            if self.blending_method == 'simple':
                # 简单平均
                result[overlap] = (img1[overlap].astype(np.float32) * 0.5 + 
                                  img2[overlap].astype(np.float32) * 0.5).astype(np.uint8)
            
            elif self.blending_method == 'linear':
                # 线性加权融合
                # 计算权重：距离图像边界越近权重越小
                weights = np.zeros(overlap.shape, dtype=np.float32)
                
                # 计算到非重叠区域的距离
                dist_to_non_overlap = distance_transform_edt(~overlap)
                
                # 归一化权重
                max_dist = dist_to_non_overlap.max()
                if max_dist > 0:
                    weights = dist_to_non_overlap / max_dist
                
                # 应用加权融合
                for c in range(3):
                    result[overlap, c] = (
                        (1 - weights[overlap]) * img1[overlap, c].astype(np.float32) + 
                        weights[overlap] * img2[overlap, c].astype(np.float32)
                    ).astype(np.uint8)
            
            elif self.blending_method == 'multiband':
                # 多频段融合（简化版）
                result[overlap] = self._multiband_blend(img1, img2, overlap)
        
        return result
    
    def _multiband_blend(self, img1, img2, overlap_mask, num_bands=5):
        """
        简化的多频段融合
        
        参数:
        img1: 第一幅图像
        img2: 第二幅图像
        overlap_mask: 重叠区域掩码
        num_bands: 频段数量
        
        返回:
        融合后的重叠区域
        """
        # 仅处理重叠区域
        overlap_region = np.where(overlap_mask)
        if len(overlap_region[0]) == 0:
            return img1
        
        y_min, y_max = overlap_region[0].min(), overlap_region[0].max()
        x_min, x_max = overlap_region[1].min(), overlap_region[1].max()
        
        # 提取重叠区域
        overlap1 = img1[y_min:y_max+1, x_min:x_max+1]
        overlap2 = img2[y_min:y_max+1, x_min:x_max+1]
        
        # 创建拉普拉斯金字塔
        gaussian1 = [overlap1.astype(np.float32)]
        gaussian2 = [overlap2.astype(np.float32)]
        
        for i in range(num_bands-1):
            gaussian1.append(cv2.pyrDown(gaussian1[-1]))
            gaussian2.append(cv2.pyrDown(gaussian2[-1]))
        
        # 创建拉普拉斯金字塔
        laplacian1 = [gaussian1[-1]]
        laplacian2 = [gaussian2[-1]]
        
        for i in range(num_bands-2, -1, -1):
            size = (gaussian1[i].shape[1], gaussian1[i].shape[0])
            expanded1 = cv2.pyrUp(gaussian1[i+1], dstsize=size)
            expanded2 = cv2.pyrUp(gaussian2[i+1], dstsize=size)
            
            laplacian1.append(gaussian1[i] - expanded1)
            laplacian2.append(gaussian2[i] - expanded2)
        
        laplacian1.reverse()
        laplacian2.reverse()
        
        # 创建权重掩码
        h, w = overlap1.shape[:2]
        weights = np.zeros((h, w), dtype=np.float32)
        
        # 创建渐变权重（从左到右）
        for i in range(w):
            weights[:, i] = i / (w-1) if w > 1 else 0.5
        
        # 创建权重金字塔
        gaussian_weights = [weights]
        for i in range(num_bands-1):
            gaussian_weights.append(cv2.pyrDown(gaussian_weights[-1]))
        
        gaussian_weights.reverse()
        
        # 融合金字塔
        blended_pyramid = []
        for i in range(num_bands):
            blended = (gaussian_weights[i][:, :, None] * laplacian1[i] + 
                      (1 - gaussian_weights[i][:, :, None]) * laplacian2[i])
            blended_pyramid.append(blended)
        
        # 重建图像
        result = blended_pyramid[0]
        for i in range(1, num_bands):
            size = (blended_pyramid[i-1].shape[1], blended_pyramid[i-1].shape[0])
            result = cv2.pyrUp(result, dstsize=size) + blended_pyramid[i]
        
        # 限制像素值范围
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 将融合结果放回原图
        blended_overlap = img1[y_min:y_max+1, x_min:x_max+1].copy()
        blended_overlap[:] = result
        
        return blended_overlap
    
    def stitch_all(self, center_index=None):
        """
        拼接所有图像
        
        参数:
        center_index: 中心图像的索引，如果为None则自动选择中间图像
        
        返回:
        全景图像
        """
        if len(self.images) < 2:
            raise ValueError("Need at least 2 images for stitching")
        
        print(f"\nStarting panorama stitching for {len(self.images)} images...")
        start_time = time.time()
        
        # 检测所有图像的特征
        self.detect_features()
        
        # 确定中心图像
        if center_index is None:
            center_index = len(self.images) // 2
        
        print(f"Using image {center_index+1} as center reference")
        
        # 初始化全景图为中间图像
        self.panorama = self.images[center_index].copy()
        
        # 向左拼接（从中心向左）
        print("\nStitching left side...")
        for i in range(center_index-1, -1, -1):
            print(f"  Adding image {i+1} to the left")
            self.panorama = self.stitch_pair(self.images[i], self.panorama, is_left=True)
        
        # 向右拼接（从中心向右）
        print("\nStitching right side...")
        for i in range(center_index+1, len(self.images)):
            print(f"  Adding image {i+1} to the right")
            self.panorama = self.stitch_pair(self.panorama, self.images[i], is_left=False)
        
        elapsed_time = time.time() - start_time
        print(f"\nPanorama stitching completed in {elapsed_time:.2f} seconds")
        
        return self.panorama
    
    def evaluate_stitching(self, img1, img2, H):
        """
        评估拼接质量
        
        参数:
        img1: 第一幅图像
        img2: 第二幅图像
        H: 单应性矩阵
        
        返回:
        重投影误差
        """
        # 检测特征
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        # 匹配特征
        matches = self.match_features(des1, des2)
        
        if len(matches) < 4:
            print("Not enough matches for evaluation")
            return None
        
        # 提取匹配点
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # 将src_pts转换为齐次坐标
        src_pts_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
        
        # 应用单应性变换
        transformed_pts = np.dot(H, src_pts_h.T).T
        transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2].reshape(-1, 1)
        
        # 计算重投影误差
        errors = np.sqrt(np.sum((transformed_pts - dst_pts) ** 2, axis=1))
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        median_error = np.median(errors)
        
        print(f"Reprojection Error:")
        print(f"  Mean: {mean_error:.2f}px")
        print(f"  Std: {std_error:.2f}px")
        print(f"  Median: {median_error:.2f}px")
        
        return {
            'mean_error': mean_error,
            'std_error': std_error,
            'median_error': median_error,
            'errors': errors
        }
    
    def save_panorama(self, output_path):
        """
        保存全景图
        
        参数:
        output_path: 输出文件路径
        """
        if self.panorama is not None:
            # 转换为BGR格式保存
            panorama_bgr = cv2.cvtColor(self.panorama, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, panorama_bgr)
            print(f"Panorama saved to {output_path}")
        else:
            print("No panorama to save. Run stitch_all() first.")
    
    def visualize_panorama(self, figsize=(15, 8)):
        """可视化全景图"""
        if self.panorama is not None:
            plt.figure(figsize=figsize)
            plt.imshow(self.panorama)
            plt.title(f'Final Panorama ({len(self.images)} images, {self.feature_type.upper()})')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            print("No panorama to display. Run stitch_all() first.")


def compare_detectors(image_paths):
    """
    比较不同特征检测器的性能
    
    参数:
    image_paths: 图像路径列表
    """
    detectors = ['sift', 'orb', 'akaze']  # SURF可能不可用
    
    results = {}
    
    for det in detectors:
        print(f"\n{'='*50}")
        print(f"Testing {det.upper()} detector")
        print('='*50)
        
        try:
            # 创建拼接器
            stitcher = PanoramaStitcher(
                feature_type=det,
                match_ratio=0.75,
                ransac_thresh=5.0,
                blending_method='simple',
                display_steps=False
            )
            
            # 加载图像
            stitcher.load_images(image_paths[:2])  # 只测试前两张图像
            
            # 拼接图像
            start_time = time.time()
            panorama = stitcher.stitch_all()
            elapsed_time = time.time() - start_time
            
            # 评估质量
            if len(stitcher.homographies) > 0:
                evaluation = stitcher.evaluate_stitching(
                    stitcher.images[0], 
                    stitcher.images[1], 
                    stitcher.homographies[0]
                )
            else:
                evaluation = None
            
            # 存储结果
            results[det] = {
                'panorama': panorama,
                'time': elapsed_time,
                'evaluation': evaluation,
                'stitcher': stitcher
            }
            
            print(f"{det.upper()} completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error with {det}: {e}")
            results[det] = None
    
    # 可视化比较结果
    fig, axes = plt.subplots(1, len(detectors), figsize=(15, 5))
    
    for i, det in enumerate(detectors):
        if results[det] is not None and results[det]['panorama'] is not None:
            axes[i].imshow(results[det]['panorama'])
            eval_info = results[det]['evaluation']
            if eval_info:
                title = f"{det.upper()}\nTime: {results[det]['time']:.2f}s\nError: {eval_info['mean_error']:.2f}px"
            else:
                title = f"{det.upper()}\nTime: {results[det]['time']:.2f}s"
            axes[i].set_title(title)
        else:
            axes[i].text(0.5, 0.5, f"{det.upper()}\nFailed", 
                        ha='center', va='center', fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Panorama Image Stitching')
    parser.add_argument('--images', nargs='+', help='Input image paths', required=False)
    parser.add_argument('--output', type=str, default='panorama_result.jpg', 
                       help='Output panorama path')
    parser.add_argument('--feature', type=str, default='sift', 
                       choices=['sift', 'orb', 'surf', 'akaze'],
                       help='Feature detector type')
    parser.add_argument('--blending', type=str, default='linear',
                       choices=['simple', 'linear', 'multiband'],
                       help='Blending method')
    parser.add_argument('--display', action='store_true',
                       help='Display intermediate steps')
    parser.add_argument('--compare', action='store_true',
                       help='Compare different feature detectors')
    
    args = parser.parse_args()
    
    # 如果没有提供图像，使用示例图像
    if args.images is None:
        print("No images provided. Please provide image paths.")
        print("Example usage: python panorama_stitching.py --images img1.jpg img2.jpg img3.jpg")
        return
    
    # 检查图像文件是否存在
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Error: Image {img_path} not found!")
            return
    
    print(f"Panorama Image Stitching")
    print(f"  Images: {args.images}")
    print(f"  Feature detector: {args.feature}")
    print(f"  Blending method: {args.blending}")
    print(f"  Output: {args.output}")
    print()
    
    # 比较不同检测器
    if args.compare:
        print("Comparing different feature detectors...")
        compare_detectors(args.images)
        return
    
    # 创建全景拼接器
    stitcher = PanoramaStitcher(
        feature_type=args.feature,
        match_ratio=0.75,
        ransac_thresh=5.0,
        blending_method=args.blending,
        display_steps=args.display
    )
    
    # 加载图像
    images = stitcher.load_images(args.images)
    
    # 显示输入图像
    if args.display:
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        for i, img in enumerate(images):
            axes[i].imshow(img)
            axes[i].set_title(f'Input Image {i+1}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
    
    # 拼接图像
    panorama = stitcher.stitch_all()
    
    # 显示结果
    stitcher.visualize_panorama()
    
    # 保存结果
    stitcher.save_panorama(args.output)
    
    # 如果只有两张图像，评估拼接质量
    if len(args.images) == 2 and len(stitcher.homographies) > 0:
        print("\nEvaluating stitching quality...")
        stitcher.evaluate_stitching(images[0], images[1], stitcher.homographies[0])
    
    print("\nDone!")


if __name__ == "__main__":
    main()
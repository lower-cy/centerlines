import numpy as np
import cv2

class CorrectedLaserStereoMapper:
    """修正版的激光映射器，正确处理矫正坐标系"""
    
    def __init__(self, P1, P2, Q):
        """
        使用矫正后的投影矩阵初始化
        
        Parameters:
        -----------
        P1, P2 : 矫正后的投影矩阵 (3x4)
        Q : 重投影矩阵
        """
        self.P1 = P1
        self.P2 = P2
        self.Q = Q
        
        # 提取参数
        self.fx = P1[0, 0]
        self.fy = P1[1, 1]
        self.cx1 = P1[0, 2]
        self.cy1 = P1[1, 2]
        self.cx2 = P2[0, 2]
        self.cy2 = P2[1, 2]
        self.Tx = -P2[0, 3] / self.fx  # 基线距离
        
        print("=" * 60)
        print("映射器参数:")
        print(f"焦距: fx={self.fx:.2f}, fy={self.fy:.2f}")
        print(f"左图主点: ({self.cx1:.2f}, {self.cy1:.2f})")
        print(f"右图主点: ({self.cx2:.2f}, {self.cy2:.2f})")
        print(f"主点差: {self.cx1 - self.cx2:.2f} 像素")
        print(f"基线: {self.Tx:.2f} mm")
        print("=" * 60)
    
    def project_left_rectified_to_right(self, points_3d):
        """
        将左矫正相机坐标系下的3D点投影到右图
        
        Parameters:
        -----------
        points_3d : numpy.ndarray, shape=(N, 3)
            左矫正相机坐标系下的3D点
            
        Returns:
        --------
        points_right : 右图像素坐标
        valid_mask : 有效点掩码
        """
        points_3d = np.asarray(points_3d, dtype=np.float64)
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, -1)
        
        N = points_3d.shape[0]
        X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        
        # 正确的投影公式（考虑基线偏移）
        # 右矫正相机坐标系下的坐标：P_right = (X - Tx, Y, Z)
        X_right = X - self.Tx
        
        # 投影到右图
        u_right = self.fx * X_right / Z + self.cx2
        v_right = self.fy * Y / Z + self.cy2
        
        # 有效性检查
        valid_depth = Z > 0
        points_right = np.full((N, 2), np.nan, dtype=np.float64)
        points_right[valid_depth, 0] = u_right[valid_depth]
        points_right[valid_depth, 1] = v_right[valid_depth]
        
        return points_right, valid_depth
    
    def project_left_rectified_to_left(self, points_3d):
        """
        将左矫正相机坐标系下的3D点投影到左图
        
        Parameters:
        -----------
        points_3d : numpy.ndarray, shape=(N, 3)
            左矫正相机坐标系下的3D点
            
        Returns:
        --------
        points_left : 左图像素坐标
        valid_mask : 有效点掩码
        """
        points_3d = np.asarray(points_3d, dtype=np.float64)
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, -1)
        
        N = points_3d.shape[0]
        X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        
        # 投影到左图
        u_left = self.fx * X / Z + self.cx1
        v_left = self.fy * Y / Z + self.cy1
        
        valid_depth = Z > 0
        points_left = np.full((N, 2), np.nan, dtype=np.float64)
        points_left[valid_depth, 0] = u_left[valid_depth]
        points_left[valid_depth, 1] = v_left[valid_depth]
        
        return points_left, valid_depth
    
    def verify_projection(self, points_3d, left_img, right_img):
        """
        验证投影是否正确
        
        Parameters:
        -----------
        points_3d : 左矫正坐标系下的3D点
        left_img, right_img : 左右矫正图像
        """
        # 投影到左右图
        pts_left, valid_left = self.project_left_rectified_to_left(points_3d)
        pts_right, valid_right = self.project_left_rectified_to_right(points_3d)
        
        # 创建可视化
        vis_left = left_img.copy()
        vis_right = right_img.copy()
        
        # 绘制左图投影
        for u, v in pts_left[valid_left][:1000]:
            if not np.isnan(u) and not np.isnan(v):
                cv2.circle(vis_left, (int(u), int(v)), 2, (0, 255, 0), -1)
        
        # 绘制右图投影
        for u, v in pts_right[valid_right][:1000]:
            if not np.isnan(u) and not np.isnan(v):
                cv2.circle(vis_right, (int(u), int(v)), 2, (0, 0, 255), -1)
        
        # 并排显示
        h, w = left_img.shape[:2]
        combined = np.zeros((h, w*2, 3), dtype=np.uint8)
        combined[:, :w] = vis_left
        combined[:, w:] = vis_right
        
        cv2.putText(combined, "Left Image (Green: Projected Points)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Right Image (Red: Projected Points)", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return combined, pts_left, pts_right


class LaserStereoMapper:
    def __init__(self, K_left, K_right, R, T, world_to_left=None):
        """
        初始化激光双目映射器
        
        Parameters:
        -----------
        K_left, K_right : 左右相机内参矩阵 (3x3)
        R, T : 右相机到左相机的旋转矩阵(3x3)和平移向量(3x1)
               即: P_left = R * P_right + T
        light_plane_coeffs : 光平面方程系数 (a,b,c,d)，满足 a*x + b*y + c*z + d = 0
        world_to_left : 世界坐标系到左相机坐标系的变换矩阵 (4x4)，可选
        """
        self.K_left = np.asarray(K_left, dtype=np.float64)
        self.K_right = np.asarray(K_right, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.T = np.asarray(T, dtype=np.float64).reshape(3, 1)

        
        # 世界坐标系到左相机坐标系的变换（如果没有提供，则假设世界坐标系与左相机坐标系重合）
        if world_to_left is not None:
            self.world_to_left = np.asarray(world_to_left, dtype=np.float64)
        else:
            # 默认世界坐标系 = 左相机坐标系
            self.world_to_left = np.eye(4, dtype=np.float64)
        
        # 预计算逆矩阵
        self.inv_K_left = np.linalg.inv(self.K_left)
        self.R_inv = self.R.T
        self.inv_K_right_T = np.linalg.inv(self.K_right.T)
        
        # 计算从世界坐标系到右相机坐标系的变换矩阵
        self._compute_world_to_right()
        
        # 计算投影矩阵
        self._compute_projection_matrices()
    
    def _compute_world_to_right(self):
        """计算世界坐标系到右相机坐标系的变换矩阵"""
        # 世界坐标系 -> 左相机坐标系
        # 左相机坐标系 -> 右相机坐标系
        # P_right = R_inv @ (P_left - T)
        
        # 构建从世界到右相机的变换矩阵 (4x4)
        self.world_to_right = np.eye(4, dtype=np.float64)
        
        # 世界到左相机的变换
        R_world_to_left = self.world_to_left[:3, :3]
        T_world_to_left = self.world_to_left[:3, 3:4]
        
        # 左相机到右相机的变换
        R_left_to_right = self.R_inv
        T_left_to_right = -R_left_to_right @ self.T
        
        # 组合变换：世界 -> 右相机
        R_world_to_right = R_left_to_right @ R_world_to_left
        T_world_to_right = R_left_to_right @ T_world_to_left + T_left_to_right
        
        self.world_to_right[:3, :3] = R_world_to_right
        self.world_to_right[:3, 3:4] = T_world_to_right
    
    def _compute_projection_matrices(self):
        """计算投影矩阵"""
        # 右相机投影矩阵 P_right = K_right * [R_world_to_right | T_world_to_right]
        self.P_right = self.K_right @ self.world_to_right[:3, :]
        
        # 左相机投影矩阵（如果需要）
        P_left_world = self.K_left @ self.world_to_left[:3, :]
        self.P_left = P_left_world
    
    def project_world_points_to_right(self, points_world, check_bounds=True, 
                                      image_shape=None, return_depth=False):
        """
        将世界坐标系下的点云投影到右图
        
        Parameters:
        -----------
        points_world : numpy.ndarray, shape=(N, 3)
            世界坐标系下的三维点云
        check_bounds : bool
            是否检查图像边界
        image_shape : tuple
            图像尺寸 (height, width)，用于边界检查
        return_depth : bool
            是否返回深度值（右相机坐标系下的Z坐标）
        
        Returns:
        --------
        points_right : numpy.ndarray, shape=(N, 2)
            右图像素坐标，无效点用NaN填充
        valid_mask : numpy.ndarray, shape=(N,)
            有效点掩码（投影成功且在图像内）
        depths : numpy.ndarray, shape=(N,), optional
            右相机坐标系下的深度值（如果return_depth=True）
        """
        points_world = np.asarray(points_world, dtype=np.float64)
        if points_world.ndim == 1:
            points_world = points_world.reshape(1, -1)
        
        N = points_world.shape[0]
        
        # 添加齐次坐标
        points_world_homo = np.column_stack([points_world, np.ones(N)])
        
        # 投影到右图像
        points_proj = (self.P_right @ points_world_homo.T).T

        print(points_proj)#打印投影后像素点
        
        # 提取深度值（右相机坐标系下的Z坐标）
        depths = points_proj[:, 2]
        
        # 检查深度值是否为正
        valid_depth = depths > 0
        
        # 计算像素坐标
        points_right = np.full((N, 2), np.nan, dtype=np.float64)
        points_right[valid_depth, 0] = points_proj[valid_depth, 0] / depths[valid_depth]
        points_right[valid_depth, 1] = points_proj[valid_depth, 1] / depths[valid_depth]

        print(points_right)
        
        # 边界检查
        valid_mask = valid_depth.copy()
        if check_bounds and image_shape is not None:
            h, w = image_shape[:2]
            in_bounds = ((points_right[:, 0] >= 0) & (points_right[:, 0] < w) &
                        (points_right[:, 1] >= 0) & (points_right[:, 1] < h))
            valid_mask = valid_mask & in_bounds
            
            # 将超出边界的点设为NaN
            points_right[~in_bounds] = np.nan
        
        if return_depth:
            return points_right, valid_mask, depths
        
        return points_right, valid_mask
    
    def project_world_points_to_left(self, points_world, check_bounds=True, 
                                     image_shape=None, return_depth=False):
        """
        将世界坐标系下的点云投影到左图
        
        Parameters:
        -----------
        points_world : numpy.ndarray, shape=(N, 3)
            世界坐标系下的三维点云
        check_bounds : bool
            是否检查图像边界
        image_shape : tuple
            图像尺寸 (height, width)，用于边界检查
        return_depth : bool
            是否返回深度值（左相机坐标系下的Z坐标）
        
        Returns:
        --------
        points_left : numpy.ndarray, shape=(N, 2)
            左图像素坐标，无效点用NaN填充
        valid_mask : numpy.ndarray, shape=(N,)
            有效点掩码
        depths : numpy.ndarray, shape=(N,), optional
            左相机坐标系下的深度值（如果return_depth=True）
        """
        points_world = np.asarray(points_world, dtype=np.float64)
        if points_world.ndim == 1:
            points_world = points_world.reshape(1, -1)
        
        N = points_world.shape[0]
        
        # 添加齐次坐标
        points_world_homo = np.column_stack([points_world, np.ones(N)])
        
        # 投影到左图像
        points_proj = (self.P_left @ points_world_homo.T).T
        
        # 提取深度值
        depths = points_proj[:, 2]
        valid_depth = depths > 0
        
        # 计算像素坐标
        points_left = np.full((N, 2), np.nan, dtype=np.float64)
        points_left[valid_depth, 0] = points_proj[valid_depth, 0] / depths[valid_depth]
        points_left[valid_depth, 1] = points_proj[valid_depth, 1] / depths[valid_depth]
        
        # 边界检查
        valid_mask = valid_depth.copy()
        if check_bounds and image_shape is not None:
            h, w = image_shape[:2]
            in_bounds = ((points_left[:, 0] >= 0) & (points_left[:, 0] < w) &
                        (points_left[:, 1] >= 0) & (points_left[:, 1] < h))
            valid_mask = valid_mask & in_bounds
            points_left[~in_bounds] = np.nan
        
        if return_depth:
            return points_left, valid_mask, depths
        
        return points_left, valid_mask
    
    def draw_points_on_img(self, points_world, right_image, color=(0, 255, 0),
                            radius=2,return_proj_points=False):
        """
        将二维点云绘制

        Returns:
        --------
        result_image : numpy.ndarray
            绘制后的图像
        points_right : numpy.ndarray, optional
            投影点坐标（如果return_proj_points=True）
        """
        
        # 复制图像
        result_image = right_image.copy()
        
        # 绘制有效点
        valid_points = points_world
        
        for i, (u, v) in enumerate(valid_points):
            if not np.isnan(u) and not np.isnan(v):
                cv2.circle(result_image, (int(u), int(v)), radius, color, -1)
        
        if return_proj_points:
            return result_image, points_right
        
        return result_image
    
    def draw_points_on_both(self, points_world, left_image, right_image,
                           left_color=(0, 255, 0), right_color=(0, 0, 255),
                           radius=2):
        """
        将世界坐标系点云同时投影到左右图并绘制
        
        Parameters:
        -----------
        points_world : numpy.ndarray, shape=(N, 3)
            世界坐标系下的三维点云
        left_image, right_image : numpy.ndarray
            左右图像
        left_color, right_color : tuple
            左右图点的颜色
        radius : int
            点的半径
        
        Returns:
        --------
        left_result : numpy.ndarray
            绘制后的左图
        right_result : numpy.ndarray
            绘制后的右图
        """
        # 投影到左图
        points_left, valid_mask_left = self.project_world_points_to_left(
            points_world, check_bounds=True, image_shape=left_image.shape[:2]
        )
        
        # 投影到右图
        points_right, valid_mask_right = self.project_world_points_to_right(
            points_world, check_bounds=True, image_shape=right_image.shape[:2]
        )
        
        # 绘制左图
        left_result = left_image.copy()
        valid_left = points_left[valid_mask_left]
        for u, v in valid_left:
            if not np.isnan(u) and not np.isnan(v):
                cv2.circle(left_result, (int(u), int(v)), radius, left_color, -1)
        
        # 绘制右图
        right_result = right_image.copy()
        valid_right = points_right[valid_mask_right]
        for u, v in valid_right:
            if not np.isnan(u) and not np.isnan(v):
                cv2.circle(right_result, (int(u), int(v)), radius, right_color, -1)
        
        return left_result, right_result

from 匹配对接 import *
from 光平面标定 import read_ply_points
import yaml

if __name__ == "__main__":
    """世界坐标系点云投影示例"""
    # 1. 准备相机参数
    K_left = CAMERA_MATRIX_LEFT
    K_right = CAMERA_MATRIX_RIGHT

    # 提取点云
    points_world = read_ply_points("output_with_planes.ply")
    print(f"Z range: {points_world[:,2].min()} ~ {points_world[:,2].max()}")

    print(f"世界坐标系点云形状: {points_world.shape}")  # (400, 3)
    

     # 1. 读取原始图像
    left_img = cv2.imread('31.1.bmp')
    right_img = cv2.imread('31.0.bmp')
    
    # 2. 进行极线矫正（与生成点云时一致）
    left_rectified, right_rectified, P1, P2, Q = stereo_rectify(left_img, right_img)

    print(Q)
    
    # 3. 准备相机参数（使用矫正后的投影矩阵）
    # 矫正后的相机内参（从P1和P2提取）
    K_left_rect = P1[:3, :3]
    K_right_rect = P2[:3, :3]
    
    # 矫正后的旋转和平移（矫正后左右相机平行）
    # 实际上在矫正坐标系下，R_rect = I, T_rect = [Tx, 0, 0]
    R_rect = np.eye(3)
    T_rect = np.array([-P2[0, 3] / P2[0, 0], 0, 0])  # 从投影矩阵提取基线
    print(R_rect)
    print(T_rect)
    
    pixel_size_mm = 0.00345  # 3.45 µm
    delta_cx=P1[0][2]-P2[0][2]
    # 主点差的物理距离
    cx1_cx2_mm = delta_cx * pixel_size_mm
    print(f"主点差物理距离: {cx1_cx2_mm:.2f} mm")


    # 5. 世界坐标系 = 左矫正相机坐标系
    world_to_left = np.eye(4)
    
    # 6. 创建映射器（使用矫正后的参数）
    mapper = LaserStereoMapper(K_left_rect, K_right_rect, R_rect, T_rect, 
                               world_to_left)


    # 4. 创建修正版映射器
    mapper_corrected = CorrectedLaserStereoMapper(P1, P2, Q)
    
    # 5. 投影点云到左右图
    combined, pts_left, pts_right = mapper_corrected.verify_projection(
        points_world, left_rectified, right_rectified
    )
      
    # 8. 投影到矫正后的右图
    points_right, valid_mask = mapper.project_world_points_to_right(
        points_world, check_bounds=True, 
        image_shape=right_rectified.shape[:2]
    )
    points_left, valid_mask = mapper.project_world_points_to_left(
        points_world, check_bounds=True, 
        image_shape=right_rectified.shape[:2]
    )

    
    # 9. 绘制结果
    right_result = mapper.draw_points_on_img(
        points_right, right_rectified,
        color=(0, 255, 0), radius=2
    )
    left_result = mapper.draw_points_on_img(
        points_left, left_rectified,
        color=(0, 255, 0), radius=2
    )
    
    # 10. 显示
    cv2.imshow('Right Rectified with Points', cv2.resize(right_result,(1024,1224)))
    cv2.imshow('Left Rectified with Points', cv2.resize(left_result,(1024,1224)))
    cv2.waitKey(0)
import numpy as np
import cv2


# ====================== 相机参数配置 ======================
CAMERA_MATRIX_LEFT = np.array([
    [4703.3840666469305, 0.0, 1133.8966264844476],
    [0.0, 4657.770006641158, 983.7755276735744],
    [0.0, 0.0, 1.0]
])

CAMERA_MATRIX_RIGHT = np.array([
    [4409.199175099535, 0.0, 1531.0013908252736],
    [0.0, 4384.905205883512, 1013.4751888939345],
    [0.0, 0.0, 1.0]
])

DIST_COEFF_LEFT = np.array([-0.19060368249367288, -6.827044122904246, 0.015377030028687984, -0.00750634791176898, 107.39588017569562])
DIST_COEFF_RIGHT = np.array([-0.42270673798875497, 1.378263372731151, 0.009909410979026863, -0.008593483642757997, -1.0961258361436514])

R = np.array([
    [0.9867230542685737, 0.007483211056180142, 0.1622393778562597],
    [-0.005753664364150946, 0.9999215317777955, -0.011127696685821956],
    [-0.16230991812357692, 0.010046483933974946, 0.9866886837494805]
])

T = np.array([-65.930698300496, 0.7317230319931822, -12.020455702540955])


class LaserStereoMapper:
    def __init__(self, K_left, K_right, R, T, light_plane_coeffs):
        """
        初始化激光双目映射器
        
        Parameters:
        -----------
        K_left, K_right : 左右相机内参矩阵 (3x3)
        R, T : 右相机到左相机的旋转矩阵(3x3)和平移向量(3x1)
               即: P_left = R * P_right + T
        light_plane_coeffs : 光平面方程系数 (a,b,c,d)，满足 a*x + b*y + c*z + d = 0
        """
        self.K_left = K_left
        self.K_right = K_right
        self.R = R
        self.T = T.reshape(3, 1)
        
        # 光平面方程系数
        self.a, self.b, self.c, self.d = light_plane_coeffs
        
        # 计算基础矩阵（用于极线约束）
        self._compute_fundamental_matrix()
        
    def _compute_fundamental_matrix(self):
        """计算基础矩阵"""
        # 构造本质矩阵 E = [T]_x * R
        T_cross = np.array([
            [0, -self.T[2, 0], self.T[1, 0]],
            [self.T[2, 0], 0, -self.T[0, 0]],
            [-self.T[1, 0], self.T[0, 0], 0]
        ])
        E = T_cross @ self.R
        
        # 基础矩阵 F = K_right^(-T) * E * K_left^(-1)
        self.F = np.linalg.inv(self.K_right.T) @ E @ np.linalg.inv(self.K_left)
    
    def point_to_3d(self, u_left, v_left):
        """
        通过光平面方程将左图点转换为三维点
        
        返回: (x, y, z) 在左相机坐标系下
        """
        # 从像素坐标到归一化相机坐标
        inv_K_left = np.linalg.inv(self.K_left)
        p_normalized = inv_K_left @ np.array([u_left, v_left, 1.0])
        
        # 射线方向
        direction = p_normalized / np.linalg.norm(p_normalized)
        
        # 求解射线与光平面的交点
        # 射线: P = t * direction
        # 光平面: a*x + b*y + c*z + d = 0
        denominator = self.a * direction[0] + self.b * direction[1] + self.c * direction[2]
        
        if abs(denominator) < 1e-6:
            return None  # 射线与平面平行
        
        t = -self.d / denominator
        if t < 0:
            return None  # 交点在相机后方
        
        point_3d = t * direction
        return point_3d
    
    def project_to_right(self, point_3d):
        """将三维点投影到右图像"""
        # 将左相机坐标系下的点转换到右相机坐标系
        # P_right = R^(-1) * (P_left - T)
        R_inv = self.R.T
        P_left = point_3d.reshape(3, 1)
        P_right = R_inv @ (P_left - self.T)
        
        # 投影到右图像
        p_right = self.K_right @ P_right
        u_right = p_right[0, 0] / p_right[2, 0]
        v_right = p_right[1, 0] / p_right[2, 0]
        
        return u_right, v_right
    def map_points_to_right_vectorized(self, points_left):
        """
        向量化版本，速度更快（适用于大量点）
        
        Parameters:
        -----------
        points_left : numpy.ndarray, shape=(N, 2)
            左图中的点坐标，每行为(u, v)
        
        Returns:
        --------
        points_right : numpy.ndarray, shape=(N, 2)
            右图中对应的点坐标，无效点用NaN填充
        valid_mask : numpy.ndarray, shape=(N,)
            有效点掩码
        """
        points_left = np.asarray(points_left)
        if points_left.ndim == 1:
            points_left = points_left.reshape(1, -1)
        
        N = points_left.shape[0]
        
        # 构建齐次坐标
        points_homo = np.column_stack([points_left, np.ones(N)])
        
        # 1. 归一化坐标
        inv_K_left = np.linalg.inv(self.K_left)
        points_norm = (inv_K_left @ points_homo.T).T
        
        # 2. 射线方向（归一化）
        norms = np.linalg.norm(points_norm, axis=1, keepdims=True)
        directions = points_norm / norms
        
        # 3. 计算与光平面的交点参数t
        denominator = (self.a * directions[:, 0] + 
                    self.b * directions[:, 1] + 
                    self.c * directions[:, 2])
        
        # 避免除零
        valid_denom = np.abs(denominator) > 1e-6
        t = np.full(N, -1.0)
        t[valid_denom] = -self.d / denominator[valid_denom]
        
        # 检查t > 0
        valid_t = t > 0
        
        # 4. 计算三维点
        points_3d = np.zeros((N, 3))
        valid_points = valid_denom & valid_t
        points_3d[valid_points] = (t[valid_points, np.newaxis] * 
                                    directions[valid_points])
        
        # 5. 转换到右相机坐标系
        R_inv = self.R.T
        T = self.T.flatten()
        
        points_right_3d = np.zeros((N, 3))
        points_right_3d[valid_points] = (R_inv @ (points_3d[valid_points].T - T[:, np.newaxis])).T
        
        # 6. 投影到右图像
        points_right = np.full((N, 2), np.nan)
        valid_proj = np.zeros(N, dtype=bool)
        
        if np.any(valid_points):
            # 投影
            points_right_homo = (self.K_right @ points_right_3d[valid_points].T).T
            z_valid = points_right_homo[:, 2] > 0
            
            if np.any(z_valid):
                valid_subset = valid_points.copy()
                valid_subset[valid_points] = z_valid
                
                u_right = points_right_homo[z_valid, 0] / points_right_homo[z_valid, 2]
                v_right = points_right_homo[z_valid, 1] / points_right_homo[z_valid, 2]
                
                # 映射回原索引
                valid_indices = np.where(valid_points)[0][z_valid]
                points_right[valid_indices] = np.column_stack([u_right, v_right])
                valid_proj[valid_indices] = True
        
        return points_right, valid_proj


    def paste_to_right_image(self, points_left, right_image, color=(0, 255, 0), 
                            radius=2, thickness=1, draw_line=False, 
                            original_segment=None):
        """
        将左图点映射到右图并绘制
        
        Parameters:
        -----------
        points_left : numpy.ndarray, shape=(N, 2)
            左图中的点坐标
        right_image : numpy.ndarray
            右图图像（会被修改）
        color : tuple
            绘制颜色 (B, G, R)
        radius : int
            点的半径
        thickness : int
            线条粗细
        draw_line : bool
            是否连接点绘制线段
        original_segment : tuple or list, optional
            原始线段端点，用于绘制连线
        
        Returns:
        --------
        right_image : numpy.ndarray
            绘制后的右图
        points_right : numpy.ndarray
            映射后的右图点坐标
        valid_mask : numpy.ndarray
            有效点掩码
        """
        # 映射点
        points_right, valid_mask = self.map_points_to_right(points_left)
        
        # 复制图像以避免修改原图
        result_image = right_image.copy()
        
        # 获取有效点
        valid_points = points_right[valid_mask]
        
        if len(valid_points) == 0:
            print("警告：没有有效的映射点")
            return result_image, points_right, valid_mask
        
        # 绘制点
        for (u, v) in valid_points:
            # 检查是否在图像范围内
            if 0 <= u < result_image.shape[1] and 0 <= v < result_image.shape[0]:
                cv2.circle(result_image, (int(u), int(v)), radius, color, -1)
        
        # 如果需要绘制线段
        if draw_line and len(valid_points) > 1:
            # 方法1：按顺序连接所有点
            points_for_line = valid_points.astype(np.int32)
            cv2.polylines(result_image, [points_for_line], False, color, thickness)
            
            # 方法2：如果提供了原始线段，按照原始线段的顺序连接
            if original_segment is not None:
                # 计算原始线段的方向
                if len(original_segment) == 4:
                    u1, v1, u2, v2 = original_segment
                else:
                    (u1, v1), (u2, v2) = original_segment
                
                # 根据参数t对映射点进行排序
                # 计算每个点对应的原始参数t
                total_length = np.sqrt((u2-u1)**2 + (v2-v1)**2)
                if total_length > 0:
                    t_values = []
                    for i, (u, v) in enumerate(points_left[valid_mask]):
                        # 计算投影参数
                        dx = u - u1
                        dy = v - v1
                        t = (dx*(u2-u1) + dy*(v2-v1)) / (total_length**2)
                        t_values.append(t)
                    
                    # 根据t值排序
                    sorted_indices = np.argsort(t_values)
                    sorted_points = valid_points[sorted_indices].astype(np.int32)
                    cv2.polylines(result_image, [sorted_points], False, color, thickness)
        
        return result_image, points_right, valid_mask
    def map_segment_to_right(self, segment_left, num_points=50):
        """
        将左图的线段映射到右图
        
        Parameters:
        -----------
        segment_left : 线段端点 [(u1, v1), (u2, v2)] 或 [x1, y1, x2, y2]
        num_points : 采样点数
        
        Returns:
        --------
        segment_right : 右图中的线段端点
        epipolar_lines : 极线方程参数（用于可视化）
        """
        # 解析线段端点
        if len(segment_left) == 4:
            u1, v1, u2, v2 = segment_left
        else:
            (u1, v1), (u2, v2) = segment_left
        
        # 在左图线段上采样
        points_left = []
        for i in range(num_points + 1):
            t = i / num_points
            u = u1 + t * (u2 - u1)
            v = v1 + t * (v2 - v1)
            points_left.append((u, v))
        
        # 计算每个点对应的右图点
        points_right = []
        points_3d = []
        
        for u, v in points_left:
            # 三维重建
            point_3d = self.point_to_3d(u, v)
            if point_3d is not None:
                points_3d.append(point_3d)
                # 投影到右图
                u_right, v_right = self.project_to_right(point_3d)
                points_right.append((u_right, v_right))
        
        if len(points_right) < 2:
            return None, None
        
        # 拟合右图线段
        points_right = np.array(points_right)
        
        # 使用最小二乘法拟合直线
        if len(points_right) > 2:
            # 计算主成分分析(PCA)得到直线方向
            mean = np.mean(points_right, axis=0)
            centered = points_right - mean
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eig(cov)
            direction = eigvecs[:, np.argmax(eigvals)]
            
            # 投影到直线上得到端点
            t_vals = centered @ direction
            t_min, t_max = t_vals.min(), t_vals.max()
            
            endpoint1 = mean + t_min * direction
            endpoint2 = mean + t_max * direction
            segment_right = (tuple(endpoint1), tuple(endpoint2))
        else:
            segment_right = (points_right[0], points_right[1])
        
        # 计算极线（用于验证）
        epipolar_lines = []
        for u, v in points_left[::max(1, num_points//5)]:  # 采样5条极线
            p_left = np.array([u, v, 1.0])
            line = self.F @ p_left  # 极线方程系数 (a,b,c)，满足 a*u_right + b*v_right + c = 0
            epipolar_lines.append(line)
        
        return segment_right, epipolar_lines
    


    def map_points_to_right(self, points_left):
        """
        将左图点集映射到右图
        
        Parameters:
        -----------
        points_left : numpy.ndarray, shape=(N, 2)
            左图中的点坐标，每行为(u, v)
        
        Returns:
        --------
        points_right : numpy.ndarray, shape=(N, 2)
            右图中对应的点坐标，无效点用NaN填充
        valid_mask : numpy.ndarray, shape=(N,)
            有效点掩码，True表示映射成功
        """
        # 转换为numpy数组
        points_left = np.asarray(points_left)
        if points_left.ndim == 1:
            points_left = points_left.reshape(1, -1)
        
        N = points_left.shape[0]
        
        # 初始化结果数组
        points_right = np.full((N, 2), np.nan, dtype=np.float32)
        valid_mask = np.zeros(N, dtype=bool)
        
        # 预计算归一化矩阵（相机内参逆矩阵）
        inv_K_left = np.linalg.inv(self.K_left)
        R_inv = self.R.T  # 右相机到左相机的旋转矩阵逆
        
        # 批量处理
        for i in range(N):
            u, v = points_left[i]
            
            # 1. 归一化坐标
            p_normalized = inv_K_left @ np.array([u, v, 1.0])
            
            # 2. 射线方向
            direction = p_normalized / np.linalg.norm(p_normalized)
            
            # 3. 求解射线与光平面的交点
            denominator = self.a * direction[0] + self.b * direction[1] + self.c * direction[2]
            
            if abs(denominator) < 1e-6:
                continue
            
            t = -self.d / denominator
            if t < 0:
                continue
            
            # 4. 三维点（左相机坐标系）
            point_3d = t * direction
            
            # 5. 转换到右相机坐标系并投影
            P_right = R_inv @ (point_3d.reshape(3, 1) - self.T)
            
            # 6. 投影到右图像
            p_right = self.K_right @ P_right
            if p_right[2, 0] > 0:
                u_right = p_right[0, 0] / p_right[2, 0]
                v_right = p_right[1, 0] / p_right[2, 0]
                
                points_right[i] = [u_right, v_right]
                valid_mask[i] = True
        
        return points_right, valid_mask
    

    def map_and_blend(self, points_left, left_image, right_image, 
                  left_color=(0, 255, 0), right_color=(0, 255, 255),
                  alpha=0.5):
        # 绘制左图
        left_result = left_image.copy()
        for (u, v) in points_left:
            if 0 <= u < left_result.shape[1] and 0 <= v < left_result.shape[0]:
                cv2.circle(left_result, (int(u), int(v)), 2, left_color, -1)
        
        # 映射到右图并绘制
        points_right, valid_mask = self.map_points_to_right(points_left)
        right_result = right_image.copy()
        print(points_right)
        
        for (u, v) in points_right[valid_mask]:
            if 0 <= u < right_result.shape[1] and 0 <= v < right_result.shape[0]:
                cv2.circle(right_result, (int(u), int(v)), 2, right_color, -1)
        
        # 创建融合图像
        if left_image.shape == right_image.shape:
            # 图像大小相同，可以并排显示
            h, w = left_image.shape[:2]
            blended = np.zeros((h, w*2, 3), dtype=np.uint8)
            blended[:, :w] = left_result
            blended[:, w:] = right_result
            
            # 添加标题文字
            cv2.putText(blended, "Left Image", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(blended, "Right Image", (w + 10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # 图像大小不同，使用加权融合
            left_resized = cv2.resize(left_result, (right_result.shape[1], right_result.shape[0]))
            blended = cv2.addWeighted(left_resized, alpha, right_result, 1-alpha, 0)
        
        return left_result, right_result, blended


    def verify_mapping(self, segment_left, segment_right, image_left=None, image_right=None):
        """
        验证映射结果的准确性
        
        Returns:
        --------
        reprojection_error : 重投影误差
        """
        # 在左图线段上采样
        if len(segment_left) == 4:
            u1, v1, u2, v2 = segment_left
        else:
            (u1, v1), (u2, v2) = segment_left
        
        errors = []
        num_samples = 20
        
        for i in range(num_samples + 1):
            t = i / num_samples
            u = u1 + t * (u2 - u1)
            v = v1 + t * (v2 - v1)
            
            # 重建三维点
            point_3d = self.point_to_3d(u, v)
            if point_3d is not None:
                # 投影到右图
                u_right_pred, v_right_pred = self.project_to_right(point_3d)
                
                # 计算到右图线段的最短距离
                if segment_right is not None:
                    p1, p2 = segment_right
                    # 点到线段的距离
                    p1 = np.array(p1)
                    p2 = np.array(p2)
                    p_pred = np.array([u_right_pred, v_right_pred])
                    
                    # 计算投影点
                    v_vec = p2 - p1
                    w_vec = p_pred - p1
                    t_proj = np.dot(w_vec, v_vec) / np.dot(v_vec, v_vec)
                    t_proj = np.clip(t_proj, 0, 1)
                    closest = p1 + t_proj * v_vec
                    
                    error = np.linalg.norm(p_pred - closest)
                    errors.append(error)
        
        if errors:
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            return {'mean': mean_error, 'max': max_error}
        return None


from 匹配对接 import *
import yaml


def interactive_tune_light_plane():
    """
    交互式微调光平面参数（逐线调节），实时显示映射结果，最后保存到 YAML

    每条检测到的激光线对应独立的光平面方程。

    按键控制:
      [ / ]  : 切换上一条/下一条激光线
      1/2    : 当前线 A -=/+ 0.0005
      3/4    : 当前线 B -=/+ 0.0005
      5/6    : 当前线 C -=/+ 0.0005
      7/8    : 当前线 D -=/+ 0.05
      0      : 重置当前线为初始值
      s      : 保存所有线的光平面到 YAML
      q      : 退出
    """
    # ===== 1. 加载光平面（用第一条作为所有线的初始值）=====
    with open("plane_equations.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    default_plane = cfg["plane_equations"][-1]
    orig = (default_plane["A"], default_plane["B"],
            default_plane["C"], default_plane["D"])
    print(f"初始光平面: {orig[0]:.6f}x + {orig[1]:.6f}y + {orig[2]:.6f}z + {orig[3]:.6f} = 0")

    # ===== 2. 读取并矫正图像 =====
    left_img = cv2.imread("31.1.bmp")
    right_img = cv2.imread("31.0.bmp")
    if left_img is None or right_img is None:
        print("错误: 无法读取图像")
        return

    left_rect, right_rect, P1, P2, Q = stereo_rectify(left_img, right_img)
    print(f"重投影矩阵 Q:\n{Q}")

    # ===== 3. 检测激光线 =====
    print("检测激光线...")
    left_lines = detect_laser_lines(left_rect, LEFT_CONFIG)
    right_lines = detect_laser_lines(right_rect, RIGHT_CONFIG)
    num_lines = len(left_lines)
    print(f"左图: {num_lines} 条, 右图: {len(right_lines)} 条")

    if num_lines == 0:
        print("错误: 未检测到激光线")
        return

    # ===== 4. 每条线独立的光平面系数 =====
    # line_planes[i] = [A, B, C, D]  for left_lines[i]
    line_planes = [list(orig) for _ in range(num_lines)]
    current_idx = 0  # 当前正在调节的线号

    # ===== 5. 矫正后相机参数 =====
    K_left_rect = P1[:3, :3]
    K_right_rect = P2[:3, :3]
    Tx = -P2[0, 3] / P2[0, 0]
    R_rect = np.eye(3)
    T_rect = np.array([Tx, 0, 0])

    # ===== 6. 创建映射器（用第一条线的系数，交互中会替换）=====
    mapper = LaserStereoMapper(K_left_rect, K_right_rect, R_rect, T_rect, line_planes[0])

    # 预计算右图激光线（用于背景绘制）
    right_mask = np.zeros(right_rect.shape[:2], dtype=np.uint8)
    for line in right_lines:
        cv2.polylines(right_mask, [line.astype(int)], False, 255, 1)

    # ===== 7. 为每条线预计算归一化射线方向 =====
    line_data = []  # 每个元素: { n, ray_dir, all_pts }
    for line in left_lines:
        pts = np.asarray(line, dtype=np.float64)
        n = len(pts)
        inv_K = np.linalg.inv(K_left_rect)
        homo = np.column_stack([pts, np.ones(n)])
        norm = (inv_K @ homo.T).T
        ray_dir = norm / np.linalg.norm(norm, axis=1, keepdims=True)
        line_data.append({"n": n, "ray_dir": ray_dir, "pts": pts})

    def compute_projection(ray_dir, n, a, b, c, d):
        """给定光平面系数，批量计算该线所有点的右图投影"""
        denom = a * ray_dir[:, 0] + b * ray_dir[:, 1] + c * ray_dir[:, 2]
        valid = np.abs(denom) > 1e-6
        t = np.full(n, -1.0)
        t[valid] = -d / denom[valid]
        valid = valid & (t > 0)

        pts_3d = np.zeros((n, 3))
        pts_3d[valid] = t[valid, None] * ray_dir[valid]

        pts_right = np.full((n, 2), np.nan)
        valid_z = valid & (pts_3d[:, 2] > 0)
        if np.any(valid_z):
            xr = pts_3d[valid_z, 0] - Tx
            yr = pts_3d[valid_z, 1]
            zr = pts_3d[valid_z, 2]
            pts_right[valid_z, 0] = K_right_rect[0, 0] * xr / zr + K_right_rect[0, 2]
            pts_right[valid_z, 1] = K_right_rect[1, 1] * yr / zr + K_right_rect[1, 2]

        return pts_right, valid_z, pts_3d

    # ===== 8. 显示准备 =====
    def to_bgr(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()

    left_disp_base = to_bgr(left_rect)
    right_disp_base = to_bgr(right_rect)

    def line_color(idx, highlight=False):
        if highlight:
            return (0, 255, 255)  # 黄色高亮当前线
        # 用 tab10 风格区分各线
        palette = [(0, 255, 0), (255, 128, 0), (0, 200, 255),
                   (255, 0, 255), (128, 255, 0), (0, 128, 255)]
        return palette[idx % len(palette)]

    def render(idx):
        """绘制第 idx 条线的映射结果"""
        a, b, c, d = line_planes[idx]
        data = line_data[idx]
        pts_right, valid_mask, _ = compute_projection(
            data["ray_dir"], data["n"], a, b, c, d)

        # 左图: 所有线灰色绘制，当前线高亮
        left = left_disp_base.copy()
        for i, ln in enumerate(left_lines):
            clr = line_color(i, highlight=(i == idx))
            cv2.polylines(left, [ln.astype(int)], False, clr, 2 if i == idx else 1)

        # 右图: 实际激光线(蓝) + 当前线的投影点(绿)
        right = right_disp_base.copy()
        for ln in right_lines:
            cv2.polylines(right, [ln.astype(int)], False, (255, 0, 0), 1)

        valid_pts = pts_right[valid_mask]
        if len(valid_pts):
            h, w = right.shape[:2]
            on = (valid_pts[:, 0] >= 0) & (valid_pts[:, 0] < w) & \
                 (valid_pts[:, 1] >= 0) & (valid_pts[:, 1] < h)
            for pt in valid_pts[on]:
                cv2.circle(right, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

        # 并排
        h, w = left.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = left
        combined[:, w:] = right

        info = [
            f"Line [{idx}/{num_lines - 1}]  ( [ ] 切换)",
            f"A: {a:.6f}  [1/2: +/-0.0005]",
            f"B: {b:.6f}  [3/4: +/-0.0005]",
            f"C: {c:.6f}  [5/6: +/-0.0005]",
            f"D: {d:.4f}  [7/8: +/-0.05]",
            f"有效投影: {np.sum(valid_mask)} / {data['n']}",
            "[S]保存  [0]重置  [Q]退出",
        ]
        for i, txt in enumerate(info):
            cv2.putText(combined, txt, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(combined, f"LEFT (yellow=line {idx})", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(combined, f"RIGHT (blue=actual, green=projected line {idx})",
                    (w + 10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return combined

    # ===== 9. 交互循环 =====
    cv2.namedWindow("Light Plane Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Light Plane Tuner", 1850, 900)
    saved = False

    while True:
        frame = render(current_idx)
        cv2.imshow("Light Plane Tuner", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            # 只保存被调节过的线（系数与初始值不同）
            with open("plane_equations.yaml", "r", encoding="utf-8") as f:
                yml = yaml.safe_load(f) or {}
            yml.setdefault("plane_equations", [])
            saved_count = 0
            for li in range(num_lines):
                a, b, c, d = line_planes[li]
                # 跳过未被调节的线（仍为初始默认值）
                if abs(a - orig[0]) < 1e-6 and abs(b - orig[1]) < 1e-6 \
                   and abs(c - orig[2]) < 1e-6 and abs(d - orig[3]) < 1e-6:
                    continue
                new_plane = {
                    "line_index": li,
                    "A": a, "B": b, "C": c, "D": d,
                    "equation": f"{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0",
                }
                yml["plane_equations"].append(new_plane)
                saved_count += 1
            with open("plane_equations.yaml", "w", encoding="utf-8") as f:
                yaml.dump(yml, f, default_flow_style=False, sort_keys=False)
            print(f"已保存 {saved_count}/{num_lines} 条线的光平面到 plane_equations.yaml")
            saved = True
        elif key == ord('0'):
            line_planes[current_idx] = list(orig)
            print(f"线 {current_idx} 已重置为初始值")
        elif key == ord(']') or key == ord('.'):
            current_idx = (current_idx + 1) % num_lines
            print(f"切换到线 {current_idx}")
        elif key == ord('[') or key == ord(','):
            current_idx = (current_idx - 1) % num_lines
            print(f"切换到线 {current_idx}")
        elif key == ord('1'): line_planes[current_idx][0] -= 0.0005
        elif key == ord('2'): line_planes[current_idx][0] += 0.0005
        elif key == ord('3'): line_planes[current_idx][1] -= 0.0005
        elif key == ord('4'): line_planes[current_idx][1] += 0.0005
        elif key == ord('5'): line_planes[current_idx][2] -= 0.0005
        elif key == ord('6'): line_planes[current_idx][2] += 0.0005
        elif key == ord('7'): line_planes[current_idx][3] -= 0.05
        elif key == ord('8'): line_planes[current_idx][3] += 0.05

    cv2.destroyAllWindows()

    # 打印汇总
    print("\n所有线的光平面:")
    for li in range(num_lines):
        a, b, c, d = line_planes[li]
        print(f"  线 {li}: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
    if not saved:
        print("提示: 未保存，按 s 可保存到 YAML")
    return mapper


if __name__ == "__main__":
    mapper = interactive_tune_light_plane()
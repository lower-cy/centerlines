import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

# ==========================================================
#                         相机标定参数
# ==========================================================
# 左、右相机的内参矩阵和畸变系数，以及外参(R:旋转矩阵  T:平移向量)
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
DIST_COEFF_LEFT = np.array([-0.190603, -6.827044, 0.015377, -0.007506, 107.39588])
DIST_COEFF_RIGHT = np.array([-0.422707, 1.378263, 0.009909, -0.008593, -1.096125])
R = np.array([
    [0.98672305, 0.00748321, 0.16223937],
    [-0.00575366, 0.99992153, -0.01112769],
    [-0.16230991, 0.01004648, 0.98668868]
])
T = np.array([-65.9306983, 0.73172303, -12.0204557])  

# ==========================================================
#                   激光线提取参数
# ==========================================================
LEFT_CONFIG = {
    'laser_color': 'gray',          # 激光颜色（左相机灰度）
    'min_laser_intensity': 75,      # 最低有效激光强度
    'clahe_clip': 3.5,               # CLAHE对比度增强系数
    'blur_kernel': (3, 3),           # 高斯模糊内核大小
    'gamma_correct': 1.0,            # 伽马校正系数
    'specular_thresh': 200,          # 高光阈值
    'local_enhance_region': (0, 1),  # 局部增强区域比例
    'clahe_clip_local': 1.5,         # 局部增强CLAHE系数
    'blend_weights': (0.2, 0.8),     # 原图与增强图混合权重
    'morph_kernel': (5, 11),         # 形态学核
    'morph_iterations': 4,           # 形态学迭代次数
    'dynamic_thresh_ratio': 0.6,     # 动态二值化系数
    'min_line_width': 1,             # 最小有效线宽
    'max_line_gap': 200,             # 最大线间gap
    'roi_padding': 10,               # ROI边距
    'cluster_eps': 6,                # DBSCAN聚类空间半径
    'min_samples': 6,                # DBSCAN聚类最小点数
    'min_line_length': 80,           # 最小线段长度
    'smooth_sigma': 2.5,              # 高斯平滑系数
    'max_end_curvature': 0.08,       # 端点曲率过滤
    'smooth_degree': 3.0             # Spline平滑度
}
RIGHT_CONFIG = {
    # 基础参数
    'laser_color': 'red',        # 激光颜色类型
    'min_laser_intensity': 75,   # 最低有效激光强度

    # 预处理参数
    'clahe_clip': 2.0,           # 对比度增强上限
    'blur_kernel': (3, 3),       # 高斯模糊核大小
    'gamma_correct': 0.75,       # 高光抑制
    'specular_thresh': 180,      # 高光检测阈值

    # 局部增强参数
    'local_enhance_region': (0, 1),  # 右侧1/4区域增强
    'clahe_clip_local': 5.0,
    'blend_weights': (0.2, 0.8),

    # 形态学参数
    'morph_kernel': (5, 11),     # 竖向特征检测
    'morph_iterations': 4,

    # 质心检测
    'dynamic_thresh_ratio': 0.25,# 抗噪阈值
    'min_line_width': 1,         # 激光线宽度
    'max_line_gap': 200,          # 断裂容忍度

    # 几何约束
    'roi_padding': 15,           # 边缘裁剪
    'cluster_eps': 6,            # 更小聚类半径
    'min_samples': 6,           # 更小样本数
    'min_line_length': 100,      # 有效线段长度

    # 后处理
    'smooth_sigma': 2.0,         # 平滑强度
    'max_end_curvature': 0.15,   # 端点曲率限制
    'smooth_degree': 2.5,        # 插值平滑度
    
    # 断线匹配参数
    'max_gap_for_matching': 500,   # 最大匹配间隙
    'direction_similarity': 0.2, # 方向相似度阈值
    'intensity_similarity': 0.75, # 强度相似度阈值
    'position_tolerance': 20,     # 位置容忍度
    'min_extension_length': 40,   # 最小延长线长度
    'max_extension_angle': 60,    # 最大延长角度(度),
    
    # 立体匹配参数
    'min_depth': -100,           # 最小深度(mm)
    'max_depth': 100,          # 最大深度(mm)
    'disparity_tolerance': 5,   # 视差容差(像素)
    'width_diff_threshold': 0.3, # 宽度差异阈值
    'min_matched_lines': 0      # 最小匹配线段数
}

# ==========================================================
#                      极线矫正
# ==========================================================
def stereo_rectify(left_img, right_img):
    """
    双目极线校正，将左右图对齐到同一水平视线
    输出：
        - left_rectified / right_rectified: 矫正后的图像
        - Q: 重投影矩阵（用于三维重建）
    """
    h, w = left_img.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        CAMERA_MATRIX_LEFT, DIST_COEFF_LEFT,
        CAMERA_MATRIX_RIGHT, DIST_COEFF_RIGHT,
        (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
    )
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX_LEFT, DIST_COEFF_LEFT, R1, P1, (w, h), cv2.CV_32FC1)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX_RIGHT, DIST_COEFF_RIGHT, R2, P2, (w, h), cv2.CV_32FC1)
    left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)
    return left_rectified, right_rectified, Q

# ==========================================================
#                        中心线提取
# ==========================================================
def local_contrast_enhancement(gray, config):
    """局部对比度增强"""
    h, w = gray.shape
    x_start = int(w * config['local_enhance_region'][0])
    x_end = int(w * config['local_enhance_region'][1])
    region = gray[:, x_start:x_end]
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip_local'], tileGridSize=(8,8))
    enhanced = clahe.apply(region)
    alpha, beta = config['blend_weights']
    blended = cv2.addWeighted(region, alpha, enhanced, beta, 0)
    result = gray.copy()
    result[:, x_start:x_end] = blended
    return result

def enhance_laser_channel(img, config):
    """激光通道增强"""
    if config['laser_color'] == 'gray':
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    b, g, r = cv2.split(img)
    if config['laser_color'] == 'red':
        enhanced = cv2.addWeighted(r, 2.2, cv2.add(b, g), -1.0, 0)
    elif config['laser_color'] == 'green':
        enhanced = cv2.addWeighted(g, 2.2, cv2.add(r, b), -1.0, 0)
    else:
        enhanced = cv2.addWeighted(b, 2.2, cv2.add(r, g), -1.0, 0)
    return cv2.merge([enhanced, enhanced, enhanced])

def multi_scale_preprocess(img, config):
    """多尺度预处理：灰度化→CLAHE增强→高斯模糊→激光通道增强"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(8,8))
    l = clahe.apply(gray)
    blur1 = cv2.GaussianBlur(l, config['blur_kernel'], 0)
    enhanced = enhance_laser_channel(blur1, config)
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    enhanced_gray = local_contrast_enhancement(enhanced_gray, config)
    return enhanced_gray

def dynamic_centroid_detection(row, config):
    """动态阈值质心检测"""
    max_val = np.max(row)
    if max_val < config['min_laser_intensity']:
        return []
    thresh = max_val * config['dynamic_thresh_ratio']
    binary = np.where(row > thresh, 255, 0).astype(np.uint8)
    centers = [i for i, val in enumerate(binary) if val == 255]
    return centers

def detect_laser_lines(img, config):
    """激光线检测主流程"""
    preprocessed = multi_scale_preprocess(img, config)
    points = []
    for y in range(preprocessed.shape[0]):
        centers = dynamic_centroid_detection(preprocessed[y, :], config)
        points.extend([[x, y] for x in centers])
    if not points:
        return []
    return [{'label': 1, 'points': np.array(points)}]

def visualize_results(img, lines, title):
    """结果可视化"""
    vis = img.copy()
    for line in lines:
        pts = line['points'].astype(int)
        for p in pts:
            cv2.circle(vis, tuple(p), 1, (0,0,255), -1)
    cv2.imshow(title, vis)

# ==========================================================
#       （5）CUDA 加速 SGBM 稠密匹配
# ==========================================================
def cuda_sgbm_match(left_rectified, right_rectified):
    """
    使用 CUDA StereoSGM 进行稠密视差计算，加速匹配
    返回：
        - pts_left / pts_right：左右匹配点
        - disparity：视差图
    """
    grayL = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    gpu_L = cv2.cuda_GpuMat()
    gpu_R = cv2.cuda_GpuMat()
    gpu_L.upload(grayL)
    gpu_R.upload(grayR)
    num_disp = 16 * 12
    min_disp = 0
    sgbm = cv2.cuda.createStereoSGM(minDisparity=min_disp,
                                    numDisparities=num_disp,
                                    P1=10*3*3,
                                    P2=120*3*3)
    gpu_disp = sgbm.compute(gpu_L, gpu_R)
    disparity = gpu_disp.download().astype(np.float32) / 16.0
    mask = disparity > min_disp
    y, x = np.where(mask)
    d = disparity[mask]
    pts_left = np.column_stack((x, y)).astype(np.float32)
    pts_right = np.column_stack((x - d, y)).astype(np.float32)
    return pts_left, pts_right, disparity

# ==========================================================
#       （6）SIFT 特征匹配（稀疏，高精度）
# ==========================================================
def sift_match(left_rectified, right_rectified, delta_cx=0):
    """
    使用 SIFT+FLANN 提取特征点并匹配
    适合纹理丰富的场景
    """
    grayL = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(grayL, None)
    kp2, des2 = sift.detectAndCompute(grayR, None)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    pts_left = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts_right = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    pts_right[:, 0] += delta_cx
    return pts_left, pts_right, good_matches, kp1, kp2

# ==========================================================
#       （7）主程序：单帧/旋转平台多帧
# ==========================================================
if __name__ == "__main__":
    image_pairs = [("31.1.bmp", "31.0.bmp")]  # 可扩展为旋转平台批量采集的图片列表
    all_points = []
    for left_path, right_path in image_pairs:
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        if left_img is None or right_img is None:
            print(f"无法读取: {left_path}, {right_path}")
            continue
        # 极线校正
        left_rec, right_rec, Q = stereo_rectify(left_img, right_img)
        # 中心线提取
        left_lines = detect_laser_lines(left_rec, LEFT_CONFIG)
        right_lines = detect_laser_lines(right_rec, RIGHT_CONFIG)
        visualize_results(left_rec, left_lines, "Left Laser Lines")
        visualize_results(right_rec, right_lines, "Right Laser Lines")
        # 匹配方法选择
        use_method = "SIFT"  # 可改成 "SIFT"
        if use_method == "CUDA_SGBM":
            pts_left, pts_right, disparity = cuda_sgbm_match(left_rec, right_rec)
        else:
            pts_left, pts_right, good_matches, kp1, kp2 = sift_match(left_rec, right_rec)
        # 三维重建（调用你的 3drebuild.py）
        '''
        from _3drebuild import reconstruct_3d
        points_3d = reconstruct_3d(pts_left, pts_right, Q)
        all_points.extend(points_3d)
    print(f"累计点云总数: {len(all_points)}")
    '''
    cv2.waitKey(0)

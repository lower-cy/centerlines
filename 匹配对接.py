import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm

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

# ====================== 左图参数配置（针对灰度图优化） ======================
LEFT_CONFIG = {
    # 基础参数
    'laser_color': 'gray',       # 激光颜色类型
    'min_laser_intensity': 75,  # [30-100] 最低有效激光强度
    
    # 预处理参数
    'clahe_clip': 3.5,          # [1.0-5.0] 对比度增强上限
    'blur_kernel': (3, 3),      # 高斯模糊核大小
    'gamma_correct': 1.0,       # 伽马校正系数
    'specular_thresh': 200,     # 高光检测阈值
    
    # 局部增强参数
    'local_enhance_region': (0, 1),  # 右侧1/3区域增强
    'clahe_clip_local': 1.5,
    'blend_weights': (0.2, 0.8),

    # 形态学参数
    'morph_kernel': (5, 11),    # 竖向特征检测
    'morph_iterations': 4,
    
    # 质心检测
    'dynamic_thresh_ratio':0.6, # 动态阈值比例
    'min_line_width': 1,        # 最小有效线宽
    'max_line_gap': 200,         # 断裂容忍度

    # 几何约束
    'roi_padding': 10,          # 边缘裁剪
    'cluster_eps': 6,          # 更小聚类半径（适应结构光连续性）
    'min_samples': 6,          # 最小样本数
    'min_line_length': 80,      # 有效线段长度

    # 后处理
    'smooth_sigma': 2.5,        # 平滑强度
    'max_end_curvature': 0.08, # 更严格的端点曲率限制
    'smooth_degree': 3.0,       # 插值平滑度
}

# ====================== 右图参数配置（针对彩色图优化） ====================== 
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
}

# ====================== SGBM参数配置 ======================
SGBM_CONFIG = {
    'minDisparity': 15,
    'numDisparities': 16*20,     # 视差范围
    'blockSize': 5,              # 匹配窗口大小
    'P1': 8 * 3 * 5 * 5,         # 控制视差平滑度的参数
    'P2': 32 * 3 * 5 * 5,        # 控制视差平滑度的参数
    'disp12MaxDiff': 1,
    'uniquenessRatio': 15,
    'speckleWindowSize': 100,
    'speckleRange': 32,
    'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
}

# ====================== 极线矫正相关函数 ======================

def stereo_rectify(left_img, right_img):
    """
    双目极线矫正（畸变问题难以解决）
    Args:
        left_img: 左图
        right_img: 右图
    Returns:
        rectified_left: 矫正后的左图
        rectified_right: 矫正后的右图
        Q: 重投影矩阵
    """
    # 获取图像尺寸
    h, w = left_img.shape[:2]
    
    # 计算立体矫正映射
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        CAMERA_MATRIX_LEFT, DIST_COEFF_LEFT,
        CAMERA_MATRIX_RIGHT, DIST_COEFF_RIGHT,
        (w, h), R, T, alpha=-1,flags=0|cv2.CALIB_USE_INTRINSIC_GUESS
    )
    
    # 计算矫正映射
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX_LEFT, DIST_COEFF_LEFT, R1, P1, (w, h), cv2.CV_32FC1
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX_RIGHT, DIST_COEFF_RIGHT, R2, P2, (w, h), cv2.CV_32FC1
    )
    
    # 应用矫正
    left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)
    
    return left_rectified, right_rectified,P1, P2, Q

# ====================== 图像预处理函数 ======================

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
    """
    激光通道增强
    Args:
        img: 输入图像（单通道或三通道）
        config: 配置参数字典
    Returns:
        增强后的三通道图像（BGR格式）
    """
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

def adaptive_gamma_correction(img, config):
    """
    自适应伽马校正（局部抑制高光）
    Args:
        img: 输入BGR图像
        config: 配置参数
    Returns:
        校正后的图像（BGR格式）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, config['specular_thresh'], 255, cv2.THRESH_BINARY)
    inv_gamma = 1.0 / config['gamma_correct']
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(img, table)
    return cv2.bitwise_and(corrected, corrected, mask=mask) + cv2.bitwise_and(img, img, mask=~mask)

def multi_scale_preprocess(img, config):
    """
    多尺度预处理
    Args:
        img: 原始输入图像
        config: 配置参数
    Returns:
        预处理后的单通道灰度图
    """
    # 1：伽马校正抑制高光
    corrected = adaptive_gamma_correction(img, config)

    # 2：转换到LAB颜色空间
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 3：自适应直方图均衡化（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(8,8))
    l = clahe.apply(l)

    # 4：混合模糊
    blur1 = cv2.GaussianBlur(l, config['blur_kernel'], 0)
    blur2 = cv2.medianBlur(l, 5)
    merged = cv2.addWeighted(blur1, 0.6, blur2, 0.4, 0)

    # 5：激光通道增强
    enhanced = enhance_laser_channel(merged, config)

    # 转换为灰度图后进行局部增强
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    enhanced_gray = local_contrast_enhancement(gray, config)
    
    return enhanced_gray

# ====================== 激光线检测函数 ======================

def dynamic_centroid_detection(row, config):
    """
    动态阈值质心检测算法（逐行处理）
    Args:
        row: 单行像素值数组
        config: 配置参数
    Returns:
        该行的质心坐标列表
    """
    max_val = np.max(row)
    if max_val < config['min_laser_intensity']:
        return []

    thresh = max_val * config['dynamic_thresh_ratio']
    binary = np.where(row > thresh, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config['morph_kernel'])  
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    segments, start = [], -1
    for i, val in enumerate(closed):
        if val == 255 and start == -1:
            start = i
        elif val == 0 and start != -1:
            if i - start >= config['min_line_width']:
                segments.append((start, i-1))
            start = -1
    if start != -1 and len(closed)-start >= config['min_line_width']:
        segments.append((start, len(closed)-1))

    centers = []
    for s, e in segments:
        x = np.arange(s, e+1)
        weights = row[s:e+1]
        if np.sum(weights) == 0:
            continue
        centroid = np.sum(x * weights) / np.sum(weights)
        centers.append(int(round(centroid)))
    
    return centers

def filter_endpoints_curvature(line, config):
    """
    端点曲率过滤（消除毛刺）
    Args:
        line: 输入线段点集
        config: 配置参数
    Returns:
        过滤后的线段点集
    """
    if len(line) < 10:
        return line

    epsilon = 1e-6
    head, tail = line[:10], line[-10:]
    
    def calculate_curvature(segment):
        dx = np.gradient(segment[:,0])
        dy = np.gradient(segment[:,1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        return np.abs(d2x * dy - dx * d2y) / ((dx**2 + dy**2)**1.5 + epsilon)

    if np.mean(calculate_curvature(head)) > config['max_end_curvature']:
        line = line[5:]
    if np.mean(calculate_curvature(tail)) > config['max_end_curvature']:
        line = line[:-5]

    return line

def geometry_based_clustering(points, img_size, config):
    """
    基于几何约束的聚类优化
    Args:
        points: 原始点集
        img_size: 图像尺寸
        config: 配置参数
    Returns:
        线段列表
    """
    h, w = img_size
    mask = (points[:,0] > config['roi_padding']) & (points[:,0] < w - config['roi_padding'])
    points = points[mask]

    db = DBSCAN(eps=config['cluster_eps'], min_samples=config['min_samples']).fit(points)

    valid_lines = []
    for label in set(db.labels_):
        if label == -1:
            continue

        cluster = points[db.labels_ == label]
        if len(cluster) < config['min_line_length']/2:
            continue

        sorted_cluster = cluster[cluster[:,1].argsort()]
        
        try:
            tck, u = splprep(sorted_cluster.T, s=config['smooth_degree'])
            new_u = np.linspace(u.min(), u.max(), int(len(u)*2))
            new_points = np.column_stack(splev(new_u, tck))
        except:
            new_points = sorted_cluster

        new_points[:,0] = gaussian_filter1d(new_points[:,0], config['smooth_sigma'])
        new_points[:,1] = gaussian_filter1d(new_points[:,1], config['smooth_sigma'])

        filtered_line = filter_endpoints_curvature(new_points, config)
        valid_lines.append(filtered_line)
    
    return valid_lines

def detect_laser_lines(img, config):
    """激光线检测主流程"""
    preprocessed = multi_scale_preprocess(img, config)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config['morph_kernel'])
    closed = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, kernel, iterations=config['morph_iterations'])

    enhanced = local_contrast_enhancement(closed, {
        'local_enhance_region': config['local_enhance_region'],
        'clahe_clip_local': config['clahe_clip_local'],
        'blend_weights': config['blend_weights']
    })

    points = []
    for y in range(enhanced.shape[0]):
        centers = dynamic_centroid_detection(enhanced[y, :], config)
        points.extend([[x, y] for x in centers])

    if not points:
        return []

    lines = geometry_based_clustering(np.array(points), enhanced.shape, config)
    return lines

# ====================== SGBM视差计算函数 ======================

def compute_disparity_sgbm(left_img, right_img, config):
    """
    使用SGBM算法计算视差图
    Args:
        left_img: 左图（灰度）
        right_img: 右图（灰度）
        config: SGBM配置参数
    Returns:
        disparity: 视差图
    """
    # 创建SGBM匹配器
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=config['minDisparity'],
        numDisparities=config['numDisparities'],
        blockSize=config['blockSize'],
        P1=config['P1'],
        P2=config['P2'],
        disp12MaxDiff=config['disp12MaxDiff'],
        uniquenessRatio=config['uniquenessRatio'],
        speckleWindowSize=config['speckleWindowSize'],
        speckleRange=config['speckleRange'],
        mode=config['mode']
    )
    
    # 计算视差
    disparity = left_matcher.compute(left_img, right_img).astype(np.float32) / 16.0
    
    return disparity


def sift_disparity(left_img, right_img,offset):
    # # 转换为灰度图
    # gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    # gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # 计算SIFT特征检测和匹配的时间
    start = cv2.getTickCount()
    # 创建SIFT对象
    sift = cv2.SIFT_create()

    # 检测关键点和计算描述子
    keypoints1, descriptors1 = sift.detectAndCompute(left_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(right_img, None)

    # 创建FLANN匹配器
    flann = cv2.FlannBasedMatcher()

    # 进行特征匹配
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 筛选匹配结果
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    end = cv2.getTickCount()
    print('SIFT匹配时间:', (end - start) / cv2.getTickFrequency(), 's')
    # 提取匹配点坐标
    pts_left = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts_right = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    
    # 计算视差（右图x坐标 - 左图x坐标）
    # disparities = pts_left[:, 0] - pts_right[:, 0] +63.977317810058594
    disparities = pts_left[:, 0] - pts_right[:, 0] - offset
    # print(disparities)
    # 创建稀疏视差图
    h, w = left_img.shape
    sparse_disp = np.zeros((h, w), dtype=np.float32)
    for (x, y), d in zip(pts_left.astype(int), disparities):
        if 0 <= x < w and 0 <= y < h:
            sparse_disp[y, x] = d
    
    # 使用Inpaint进行空洞填充（简单示例）
    mask = (sparse_disp == 0).astype(np.uint8)
    dense_disp = cv2.inpaint(sparse_disp, mask, 3, cv2.INPAINT_TELEA)

    # 绘制匹配结果
    result = cv2.drawMatches(left_img, keypoints1,right_img, keypoints2, good_matches, None)

    # 保存结果
    cv2.imwrite('SIFT_Matches.jpg', result)

    
    return sparse_disp, dense_disp


def extract_matched_points(disparity, left_lines, right_img_shape):
    """
    从视差图中提取匹配点
    Args:
        disparity: 视差图
        left_lines: 左图检测到的激光线
        right_img_shape: 右图尺寸
    Returns:
        pts_left: 左图匹配点坐标
        pts_right: 右图匹配点坐标
    """
    pts_left = []
    pts_right = []
    
    h, w = right_img_shape
    
    for line in left_lines:
        for point in line:
            x, y = int(round(point[0])), int(round(point[1]))
            
            # 确保坐标在图像范围内
            if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
                d = disparity[y, x]
                
                # 过滤无效视差
                if d > 0:
                    x_right = x - d
                    
                    # 确保右图坐标有效
                    if 0 <= x_right < w and 0 <= y < h:
                        pts_left.append([x, y])
                        pts_right.append([x_right, y])
    
    return np.float32(pts_left), np.float32(pts_right)


import numpy as np

def generate_disparity_map_from_matches(pts_left, pts_right, img_shape):
    """
    根据匹配点生成视差图
    Args:
        pts_left: 左图匹配点坐标，shape=(N, 2)，float32或int
        pts_right: 右图匹配点坐标，shape=(N, 2)，float32或int
        img_shape: 图像尺寸 (H, W)
    Returns:
        disparity_map: 视差图，shape=(H, W)，无匹配点处为-1
    """
    H, W = img_shape[:2]
    disparity_map = np.full((H, W), -1, dtype=np.float32)  # 初始化为-1表示无效
    
    for (xL, yL), (xR, yR) in zip(pts_left, pts_right):
        xL_int, yL_int = int(round(xL)), int(round(yL))
        disp = xL - xR  # 视差
        
        if 0 <= xL_int < W and 0 <= yL_int < H:
            disparity_map[yL_int, xL_int] = disp
    
    return disparity_map

# ====================== 可视化函数 ======================

def visualize_results(img, lines, title):
    """可视化检测结果"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f'{title} Original')

    preprocessed = multi_scale_preprocess(img, LEFT_CONFIG if 'L' in title else RIGHT_CONFIG)
    ax[1].imshow(preprocessed, cmap='gray')
    ax[1].set_title('Preprocessed')

    vis = img.copy()
    try:
        cmap = plt.colormaps['tab10']
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10')
    
    for i, line in enumerate(lines):
        color_rgba = cmap(i % 10)
        color = tuple(int(c * 255) for c in color_rgba[:3])
        pts = line.astype(int)
        cv2.polylines(vis, [pts], False, color, 2)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected {len(lines)} Laser Lines')
    plt.show()


from scipy.interpolate import interp1d

def resample_line(pts, num_points):
    # 按y坐标排序
    pts = pts[np.argsort(pts[:, 1])]
    y = pts[:, 1]
    x = pts[:, 0]
    f = interp1d(y, x, kind='linear', fill_value='extrapolate')
    y_new = np.linspace(y.min(), y.max(), num_points)
    x_new = f(y_new)
    return np.stack([x_new, y_new], axis=1)

def compute_disparity_with_resample(lineL, lineR):
    disparities = []
    for ptsL, ptsR in zip(lineL, lineR):
        num_points = max(ptsL.shape[0], ptsR.shape[0])
        ptsL_resampled = resample_line(ptsL, num_points)
        ptsR_resampled = resample_line(ptsR, num_points)
        disp = ptsL_resampled[:, 0] - ptsR_resampled[:, 0]
        disparities.append(disp)
    return disparities


def disparity_lines_to_map(img_shape, lineL, disparities):
    H, W = img_shape[:2]
    print(f"H:{H},W:{W}")
    disp_map = np.full((H, W), -1, dtype=np.float32)  # -1表示无效视差
    
    for pts, disp in zip(lineL, disparities):
        pts_int = pts.astype(int)
        for (x, y), d in zip(pts_int, disp):
            if 0 <= x < W and 0 <= y < H:
                disp_map[y, x] = d
    
    return disp_map




def line_results(img, lines):
    """可视化检测结果"""
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # ax[0].set_title(f'{title} Original')

    # preprocessed = multi_scale_preprocess(img, LEFT_CONFIG if 'L' in title else RIGHT_CONFIG)
    # ax[1].imshow(preprocessed, cmap='gray')
    # ax[1].set_title('Preprocessed')
    
    vis = img.copy()
    try:
        cmap = plt.colormaps['tab10']
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10')
    
    for i, line in enumerate(lines):
        color_rgba = cmap(i % 10)
        color = tuple(int(c * 255) for c in color_rgba[:3])
        pts = line.astype(int)
        cv2.polylines(vis, [pts], False, color, 2)
    return vis


def visualize_disparity(disparity):
    """可视化视差图"""
    plt.figure(figsize=(10, 8))
    plt.imshow(disparity, cmap='viridis')
    plt.colorbar(label='Disparity (pixels)')
    plt.title('Disparity Map')
    plt.show()

def visualize_matched_points(left_img, right_img, pts_left, pts_right):
    """可视化匹配点"""
    # 创建合成图像
    h, w = left_img.shape[:2]
    composite = np.zeros((h, w*2, 3), dtype=np.uint8)
    composite[:, :w] = left_img
    composite[:, w:] = right_img
    
    # 绘制匹配点
    for pt_left, pt_right in zip(pts_left, pts_right):
        x1, y1 = int(pt_left[0]), int(pt_left[1])
        x2, y2 = int(pt_right[0] + w), int(pt_right[1])
        
        cv2.circle(composite, (x1, y1), 3, (0, 255, 0), -1)
        cv2.circle(composite, (x2, y2), 3, (0, 255, 0), -1)
        cv2.line(composite, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    plt.title(f'Matched Points ({len(pts_left)} points)')
    plt.show()

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 读取图像
    left_img = cv2.imread('31.1.bmp')
    right_img = cv2.imread('31.0.bmp')

    if left_img is None or right_img is None:
        print("错误：无法读取图像文件！请检查文件路径和名称是否正确。")
        print(f"左图路径: {'31.1.bmp'}")
        print(f"右图路径: {'31.0.bmp'}")
        exit()

    print(f"图像尺寸 - 左图: {left_img.shape}, 右图: {right_img.shape}")





    # 极线矫正
    print("进行极线矫正...")
    left_rectified, right_rectified,P1 ,P2, Q = stereo_rectify(left_img, right_img)
    print(f"重投影矩阵 Q: \n{Q}")

    # visualize_disparity(left_rectified)
        # 检测激光线（在原始图像上）
    print("\n处理左图...")
    left_lines = detect_laser_lines(left_rectified, LEFT_CONFIG)
    print(f"左图提取到 {len(left_lines)} 条激光线")

    print("\n处理右图...")
    right_lines = detect_laser_lines(right_rectified, RIGHT_CONFIG)
    print(f"右图提取到 {len(right_lines)} 条激光线")

    # # 可视化原始检测结果
    # visualize_results(left_rectified, left_lines, 'Left Image')
    # visualize_results(right_rectified, right_lines, 'Right Image')



    # imgL_rectified= left_rectified
    # imgR_rectified =right_rectified
    # imgL_rectified.astype(np.float32)
    # imgR_rectified.astype(np.float32)
    # # 在校正后图像上绘制水平参考线
    # h, w = imgL_rectified.shape[:2]
    # for y in range(0, h, 50):
    #     cv2.line(imgL_rectified, (0, y), (w, y), (255,0,0), 2)
    #     cv2.line(imgR_rectified, (0, y), (w, y), (255,0,0), 2)
    # combined = np.hstack((imgL_rectified, imgR_rectified))
    # cv2.imwrite('rectified_check.jpg', combined)






    vis = np.zeros(left_rectified.shape[:2], np.uint8)
    line_L=line_results(left_rectified,left_lines)
    line_R=line_results(right_rectified,right_lines)
    cv2.imwrite("img_line/left_line.bmp",line_L)
    cv2.imwrite("img_line/right_line.bmp",line_R)

    line_dis=compute_disparity_with_resample(left_lines, right_lines)
    simple_disp=disparity_lines_to_map(left_rectified.shape,left_lines,line_dis)

    # 转换为灰度图用于SGBM
    left_gray = cv2.cvtColor(line_L, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(line_R, cv2.COLOR_BGR2GRAY)
    # 计算视差图
    print("计算视差图...")
    sbgm_disp = compute_disparity_sgbm(line_L, line_R, SGBM_CONFIG)



    # delta_cx=P1[0][2]-P2[0][2]
    # sparse_disp, dense_disp = sift_disparity(line_L, line_R,delta_cx)





    # 提取匹配点
    print("提取匹配点...")
    pts_left, pts_right = extract_matched_points(sbgm_disp, left_lines, right_gray.shape)
    print(f"提取到 {len(pts_left)} 对匹配点")

    remap_disp=generate_disparity_map_from_matches(pts_left, pts_right,left_rectified.shape)


    disparity=remap_disp
    visualize_disparity(disparity)
    # 可视化匹配点
    # visualize_matched_points(left_rectified, right_rectified, pts_left, pts_right)



    colorR_rectified = line_L
    # 生成三维点云
    print(f"Gererating point clouds")
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    # 创建点云颜色映射（使用右图颜色）
    colors =cv2.cvtColor(colorR_rectified,cv2.COLOR_BGR2RGB)

    # 设定最小视差阈值（视差小于此值的点视为太远）
    min_disp = 0  # 根据场景调整，值越小，保留的深度越远
    mask = disparity > min_disp
    out_points = points_3D[mask]
    out_colors = colors[mask]

    # 保存为PLY文件
    def write_ply(filename, verts, colors):
        with open(filename, 'w') as f:
            header = '''ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''.format(len(verts))
            f.write(header)
            for (x, y, z), (r, g, b) in zip(verts, colors):
                f.write('{} {} {} {} {} {}\n'.format(x, y, z, r, g, b))

    write_ply('output.ply', out_points, out_colors)

    # 保存匹配点结果
    print("保存匹配点结果...")
    np.savetxt('matched_points_left.txt', pts_left, fmt='%.2f')
    np.savetxt('matched_points_right.txt', pts_right, fmt='%.2f')
    print("匹配点已保存为 matched_points_left.txt 和 matched_points_right.txt")

    # 输出用于三维重建的关键信息
    print("\n=== 三维重建接口数据 ===")
    print(f"左图匹配点坐标形状: {pts_left.shape}")
    print(f"右图匹配点坐标形状: {pts_right.shape}")
    print(f"重投影矩阵 Q: \n{Q}")
    print("\n后续三维重建代码可以使用以下数据:")
    # print("1. 左图匹配点坐标: pts_left (N, 2)")
    # print("2. 右图匹配点坐标: pts_right (N, 2)") 
    # print("3. 重投影矩阵 Q: 用于将视差转换为3D坐标")
    # print("\n三维重建示例代码:")
    # print("points_3D = cv2.reprojectImageTo3D(disparity, Q)")
    # print("# 或者使用:")
    # print("points = cv2.triangulatePoints(P1, P2, pts_left.T, pts_right.T)")
    # print("points /= points[3]  # 齐次坐标归一化")
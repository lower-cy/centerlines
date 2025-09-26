import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

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

E = np.array([
    [-0.1879275730285138, 12.026863722436927, 0.5882228502232973],
    [-22.562067007495063, 0.5724200942181044, 63.102882670071544],
    [-0.34266487569182286, -65.930698300496, 0.6149425234986867]
])

F = np.array([
    [1.7127838396910227e-08, -1.1068705822074972e-06, 0.0008173378496220493],
    [2.0677138875277873e-06, -5.297352029018583e-08, -0.029492603715315258],
    [-0.0019840971148673923, 0.02850262291237524, 1.0]
])

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
    
    # 断线匹配参数
    'max_gap_for_matching': 500,  # 最大匹配间隙（像素）
    'direction_similarity': 0.2,  # 方向相似度阈值（cosθ）
    'intensity_similarity': 0.8,  # 强度相似度阈值
    'position_tolerance': 30,     # 位置容忍度（像素）
    'min_extension_length': 50,   # 最小延长线长度
    'max_extension_angle': 60,    # 最大延长角度(度)
    
    # 立体匹配参数
    'min_depth': 100,           # 最小深度(mm)
    'max_depth': 3000,          # 最大深度(mm)
    'disparity_tolerance': 5,   # 视差容差(像素)
    'width_diff_threshold': 0.3 # 宽度差异阈值
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
    
    # 断线匹配参数
    'max_gap_for_matching': 500,   # 最大匹配间隙
    'direction_similarity': 0.2, # 方向相似度阈值
    'intensity_similarity': 0.75, # 强度相似度阈值
    'position_tolerance': 20,     # 位置容忍度
    'min_extension_length': 40,   # 最小延长线长度
    'max_extension_angle': 60,    # 最大延长角度(度),
    
    # 立体匹配参数
    'min_depth': 100,           # 最小深度(mm)
    'max_depth': 3000,          # 最大深度(mm)
    'disparity_tolerance': 5,   # 视差容差(像素)
    'width_diff_threshold': 0.3 # 宽度差异阈值
}

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
    激光通道增强核心算法
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
    多尺度预处理流水线（核心预处理流程）
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

def extract_line_features(line, img):
    """
    提取线段特征向量
    Args:
        line: 线段点集 (N,2)
        img: 原始图像（用于提取强度特征）
    Returns:
        特征向量字典
    """
    if len(line) < 2:
        return None
    
    # 基本几何特征
    start_pt = line[0]
    end_pt = line[-1]
    length = np.linalg.norm(end_pt - start_pt)
    direction = (end_pt - start_pt) / (length + 1e-6)
    
    # 强度特征（从原始图像获取）
    intensities = []
    for pt in line:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            if len(img.shape) == 3:
                intensities.append(np.mean(img[y, x]))
            else:
                intensities.append(img[y, x])
    
    if not intensities:
        return None
    
    # 曲率特征
    if len(line) >= 3:
        dx = np.gradient(line[:,0])
        dy = np.gradient(line[:,1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        curvature = np.abs(d2x * dy - dx * d2y) / ((dx**2 + dy**2)**1.5 + 1e-6)
        avg_curvature = np.mean(curvature)
    else:
        avg_curvature = 0
    
    return {
        'start_point': start_pt,
        'end_point': end_pt,
        'direction': direction,
        'length': length,
        'mean_intensity': np.mean(intensities),
        'intensity_std': np.std(intensities),
        'curvature': avg_curvature,
        'points': line
    }

def compute_similarity_matrix(features, img):
    """
    构建包含几何约束和上下的相似度矩阵
    """
    n = len(features)
    similarity = np.zeros((n, n))
    
    # 计算基本几何相似度
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity[i,j] = 1.0
            else:
                # 计算各维度相似度
                dir_sim = abs(features[i]['direction'] @ features[j]['direction'])  # 方向相似度
                
                # 位置相似度（端点距离）
                end_dist = min(
                    np.linalg.norm(features[i]['end_point'] - features[j]['start_point']),
                    np.linalg.norm(features[i]['start_point'] - features[j]['end_point']),
                    np.linalg.norm(features[i]['end_point'] - features[j]['end_point']),
                    np.linalg.norm(features[i]['start_point'] - features[j]['start_point'])
                )
                pos_sim = max(0, 1 - end_dist / 100)  # 距离相似度
                
                # 曲率相似度
                curv_sim = 1 - abs(features[i]['curvature'] - features[j]['curvature']) / max(features[i]['curvature'], features[j]['curvature'], 0.01)
                
                # 强度相似度
                inten_sim = 1 - abs(features[i]['mean_intensity'] - features[j]['mean_intensity']) / 255
                
                # 综合相似度
                similarity[i,j] = (
                    0.4 * dir_sim +
                    0.2 * pos_sim +
                    0.2 * curv_sim +
                    0.2 * inten_sim
                )
    
    # 应用回转体约束
    center = np.array([img.shape[1]/2, img.shape[0]/2])
    for i in range(n):
        for j in range(n):
            # 计算径向方向差异
            i_center = features[i]['start_point'] + (features[i]['end_point'] - features[i]['start_point'])/2
            j_center = features[j]['start_point'] + (features[j]['end_point'] - features[j]['start_point'])/2
            
            radial_i = i_center - center
            radial_j = j_center - center
            
            # 计算径向方向相似度
            radial_dir_sim = abs(radial_i @ radial_j) / (np.linalg.norm(radial_i) * np.linalg.norm(radial_j) + 1e-6)
            
            # 应用约束
            similarity[i,j] *= radial_dir_sim ** 2  # 强化径向一致性约束
    
    return similarity

def match_broken_lines(lines, img, config):
    """
    基于几何特征和回转体约束的断线匹配
    Args:
        lines: 检测到的线段列表
        img: 原始图像（用于特征提取）
        config: 配置参数
    Returns:
        带标签的线段列表，相同标签表示属于同一条线
    """
    if not lines:
        return []
    
    # 提取所有线段的特征
    features = []
    valid_lines = []
    for line in lines:
        feat = extract_line_features(line, img)
        if feat and feat['length'] > config['min_line_length']/2:
            features.append(feat)
            valid_lines.append(line)
    
    if not features:
        return []
    
    # 构建相似度矩阵
    similarity_matrix = compute_similarity_matrix(features, img)
    
    # 使用谱聚类进行分组
    n_clusters = max(3, len(features) // 4)  # 自适应簇数
    clusterer = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42
    )
    
    # 构建亲和矩阵（相似度转换为距离）
    affinity_matrix = similarity_matrix / similarity_matrix.max()
    labels = clusterer.fit_predict(affinity_matrix)
    
    # 构建带标签的结果
    labeled_lines = []
    for i, label in enumerate(labels):
        labeled_lines.append({
            'label': label + 1,  # 1-based标签
            'points': valid_lines[i],
            'features': features[i]
        })
    
    return labeled_lines

def geometry_based_clustering(points, img_size, config, original_img):
    """
    基于几何约束的聚类优化（核心逻辑 + 断线标记）
    Args:
        points: 原始点集
        img_size: 图像尺寸
        config: 配置参数
        original_img: 原始图像（用于特征提取）
    Returns:
        带标签的线段列表
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
        if len(cluster) < config['min_line_length']/2:  # 降低初始长度要求
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

    # 断线标记（从左到右排序）
    labeled_lines = match_broken_lines(valid_lines, original_img, config)
    
    # 重新从左到右编号
    if labeled_lines:
        # 按线段中心x坐标排序
        sorted_lines = sorted(labeled_lines, key=lambda x: np.mean(x['points'][:,0]))
        for new_label, line in enumerate(sorted_lines, 1):
            line['label'] = new_label
        return sorted_lines
    
    return []

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

    lines = geometry_based_clustering(np.array(points), enhanced.shape, config, img)
    return lines

def visualize_results(img, lines, title):
    """增强可视化（显示标签）"""
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f'{title} Original')

    preprocessed = multi_scale_preprocess(img, LEFT_CONFIG if 'L' in title else RIGHT_CONFIG)
    ax[1].imshow(preprocessed, cmap='gray')
    ax[1].set_title('Preprocessed')

    vis = img.copy()
    colors = plt.cm.get_cmap('tab20', 20)  # 使用更多颜色区分标签
    
    # 统计不同标签数量
    unique_labels = set(line['label'] for line in lines) if lines else set()
    
    for line in lines:
        color = colors(line['label'] % 20)
        color_rgb = (np.array(color[:3]) * 255).astype(int).tolist()
        pts = line['points'].astype(int)
        cv2.polylines(vis, [pts], False, color_rgb, 2)
        
        # 在起点显示标签
        if len(pts) > 0:
            cv2.putText(vis, str(line['label']), tuple(pts[0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    ax[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax[2].set_title(f'Detected {len(unique_labels)} Lines with {len(lines)} Segments')
    plt.tight_layout()
    plt.show()

def save_labeled_lines(lines, filename):
    """保存带标签的线段数据"""
    with open(filename, 'w') as f:
        for line in lines:
            f.write(f"# Label: {line['label']}\n")
            np.savetxt(f, line['points'], fmt='%.2f', delimiter=',')
            f.write("\n")


# ====================== 立体匹配相关函数 ======================

def calculate_disparity_range(config, baseline, focal_length):
    """
    计算视差范围基于深度限制
    Args:
        config: 配置参数
        baseline: 基线长度(mm)
        focal_length: 焦距(像素)
    Returns:
        (min_disparity, max_disparity)
    """
    min_disparity = baseline * focal_length / config['max_depth']
    max_disparity = baseline * focal_length / config['min_depth']
    return min_disparity, max_disparity

def project_line_to_right(left_line, right_lines, config, baseline, focal_length):
    """
    将左图线段投影到右图，寻找最佳匹配线段
    Args:
        left_line: 左图线段数据
        right_lines: 右图所有线段列表
        config: 配置参数
        baseline: 基线长度(mm)
        focal_length: 焦距(像素)
    Returns:
        最佳匹配的右图线段索引，匹配质量评分
    """
    # 计算视差范围
    min_disp, max_disp = calculate_disparity_range(config, baseline, focal_length)
    
    # 获取左图线段中点
    left_points = left_line['points']
    left_mid_y = np.mean(left_points[:, 1])
    left_mid_x = np.mean(left_points[:, 0])
    
    best_match_idx = -1
    best_score = -1
    candidate_indices = []
    
    # 在右图中寻找候选线段
    for idx, right_line in enumerate(right_lines):
        right_points = right_line['points']
        right_mid_y = np.mean(right_points[:, 1])
        
        # 初步筛选：y坐标相近的线段
        if abs(right_mid_y - left_mid_y) > config['position_tolerance']:
            continue
            
        right_mid_x = np.mean(right_points[:, 0])
        disparity = left_mid_x - right_mid_x
        
        # 检查视差是否在合理范围内
        if min_disp <= disparity <= max_disp:
            candidate_indices.append(idx)
    
    # 如果没有候选线段，返回无匹配
    if not candidate_indices:
        return -1, 0
    
    # 在候选线段中寻找最佳匹配
    for idx in candidate_indices:
        right_line = right_lines[idx]
        right_points = right_line['points']
        
        # 计算重叠比例
        left_y_range = (np.min(left_points[:, 1]), np.max(left_points[:, 1]))
        right_y_range = (np.min(right_points[:, 1]), np.max(right_points[:, 1]))
        
        overlap_start = max(left_y_range[0], right_y_range[0])
        overlap_end = min(left_y_range[1], right_y_range[1])
        
        if overlap_end <= overlap_start:
            continue
            
        overlap_ratio = (overlap_end - overlap_start) / (left_y_range[1] - left_y_range[0])
        
        # 计算方向相似度
        left_dir = left_line['features']['direction']
        right_dir = right_line['features']['direction']
        dir_similarity = abs(left_dir @ right_dir)
        
        # 计算综合评分
        score = overlap_ratio * 0.6 + dir_similarity * 0.4
        
        if score > best_score:
            best_score = score
            best_match_idx = idx
    
    return best_match_idx, best_score

def calculate_line_width(line_points, img):
    """
    计算线段的平均宽度
    Args:
        line_points: 线段点集
        img: 原始图像
    Returns:
        平均宽度(像素)
    """
    if len(line_points) < 2:
        return 0
    
    widths = []
    for pt in line_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= y < img.shape[0]:
            row = img[y, :] if len(img.shape) == 2 else img[y, :, 0]
            binary = row > (np.max(row) * 0.5)  # 使用50%强度作为阈值
            edges = np.where(np.diff(binary.astype(int)) != 0)[0]
            if len(edges) >= 2:
                line_width = edges[-1] - edges[0]
                widths.append(line_width)
    
    return np.mean(widths) if widths else 0

def refine_matches(left_lines, right_lines, left_img, right_img, config):
    """
    优化匹配结果，基于宽度一致性进行筛选
    Args:
        left_lines: 左图线段列表
        right_lines: 右图线段列表
        left_img: 左图原始图像
        right_img: 右图原始图像
        config: 配置参数
    Returns:
        匹配对列表，每个元素为(left_idx, right_idx, match_score)
    """
    # 计算基线长度和平均焦距
    baseline = np.linalg.norm(T)  # 使用平移向量的模作为基线长度
    focal_length = (CAMERA_MATRIX_LEFT[0,0] + CAMERA_MATRIX_RIGHT[0,0]) / 2
    
    # 初始匹配
    matches = []
    for left_idx, left_line in enumerate(left_lines):
        right_idx, score = project_line_to_right(left_line, right_lines, config, baseline, focal_length)
        if right_idx != -1:
            matches.append((left_idx, right_idx, score))
    
    # 计算宽度一致性
    width_pairs = []
    for left_idx, right_idx, _ in matches:
        left_width = calculate_line_width(left_lines[left_idx]['points'], left_img)
        right_width = calculate_line_width(right_lines[right_idx]['points'], right_img)
        width_pairs.append((left_width, right_width))
    
    if not width_pairs:
        return []
    
    # 计算宽度比率的平均值
    width_ratios = [left/right for left, right in width_pairs if right > 0]
    if not width_ratios:
        return []
    
    mean_ratio = np.mean(width_ratios)
    
    # 筛选匹配对
    refined_matches = []
    for i, (left_idx, right_idx, score) in enumerate(matches):
        left_width, right_width = width_pairs[i]
        if right_width == 0:
            continue
            
        ratio = left_width / right_width
        if abs(ratio - mean_ratio) / mean_ratio < config['width_diff_threshold']:
            refined_matches.append((left_idx, right_idx, score))
    
    return refined_matches

def stereo_match(left_lines, right_lines, left_img, right_img, config):
    """
    立体匹配主函数
    Args:
        left_lines: 左图检测到的线段
        right_lines: 右图检测到的线段
        left_img: 左图原始图像
        right_img: 右图原始图像
        config: 配置参数
    Returns:
        匹配结果列表，每个元素为字典包含左右线段信息和匹配分数
    """
    # 进行初始匹配
    matches = refine_matches(left_lines, right_lines, left_img, right_img, config)
    
    # 构建结果
    results = []
    for left_idx, right_idx, score in matches:
        left_line = left_lines[left_idx]
        right_line = right_lines[right_idx]
        
        # 计算视差
        left_mid = np.mean(left_line['points'], axis=0)
        right_mid = np.mean(right_line['points'], axis=0)
        disparity = left_mid[0] - right_mid[0]
        
        # 计算深度
        baseline = np.linalg.norm(T)
        depth = (baseline * CAMERA_MATRIX_LEFT[0,0]) / (disparity + 1e-6)
        
        results.append({
            'left_line': left_line,
            'right_line': right_line,
            'match_score': score,
            'disparity': disparity,
            'depth': depth,
            'left_width': calculate_line_width(left_line['points'], left_img),
            'right_width': calculate_line_width(right_line['points'], right_img)
        })
    
    return results

def visualize_matches(left_img, right_img, matches):
    """
    可视化匹配结果
    Args:
        left_img: 左图
        right_img: 右图
        matches: 匹配结果列表
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 使用新的颜色映射方式
    from matplotlib import cm
    
    # 创建颜色映射
    cmap = cm.get_cmap('tab20')
    
    # 绘制左图
    left_vis = left_img.copy()
    for i, match in enumerate(matches):
        # 获取颜色并转换为 OpenCV 格式 (BGR)
        color = tuple(np.round(cmap(i % cmap.N)[:3] * 255).astype(int))
        pts = match['left_line']['points'].astype(int)
        cv2.polylines(left_vis, [pts], False, color, 2)
        cv2.putText(left_vis, str(i+1), tuple(pts[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    ax1.imshow(cv2.cvtColor(left_vis, cv2.COLOR_BGR2RGB))
    ax1.set_title('Left Image with Matched Lines')
    
    # 绘制右图
    right_vis = right_img.copy()
    for i, match in enumerate(matches):
        # 获取颜色并转换为 OpenCV 格式 (BGR)
        color = tuple(np.round(cmap(i % cmap.N)[:3] * 255).astype(int))
        pts = match['right_line']['points'].astype(int)
        cv2.polylines(right_vis, [pts], False, color, 2)
        cv2.putText(right_vis, str(i+1), tuple(pts[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    ax2.imshow(cv2.cvtColor(right_vis, cv2.COLOR_BGR2RGB))
    ax2.set_title('Right Image with Matched Lines')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 读取图像
    left_img = cv2.imread('31.1.bmp')
    right_img = cv2.imread('31.0.bmp')

    # 检测激光线
    print("处理左图...")
    left_lines = detect_laser_lines(left_img, LEFT_CONFIG)
    unique_left_labels = set(line['label'] for line in left_lines) if left_lines else set()
    print(f"左图提取到 {len(unique_left_labels)} 条中心线（共 {len(left_lines)} 个线段）")

    print("\n处理右图...")
    right_lines = detect_laser_lines(right_img, RIGHT_CONFIG)
    unique_right_labels = set(line['label'] for line in right_lines) if right_lines else set()
    print(f"右图提取到 {len(unique_right_labels)} 条中心线（共 {len(right_lines)} 个线段）")

    # 可视化原始检测结果
    visualize_results(left_img, left_lines, 'Left Image')
    visualize_results(right_img, right_lines, 'Right Image')

    # 进行立体匹配
    print("\n进行立体匹配...")
    matches = stereo_match(left_lines, right_lines, left_img, right_img, LEFT_CONFIG)
    print(f"找到 {len(matches)} 对匹配线段")
    
    # 可视化匹配结果
    visualize_matches(left_img, right_img, matches)
    
    # 保存匹配结果
    print("\n匹配结果:")
    for i, match in enumerate(matches):
        print(f"匹配对 {i+1}:")
        print(f"  左图线段 {match['left_line']['label']} <-> 右图线段 {match['right_line']['label']}")
        print(f"  匹配分数: {match['match_score']:.2f}, 视差: {match['disparity']:.2f}px, 深度: {match['depth']:.2f}mm")
        print(f"  左图宽度: {match['left_width']:.2f}px, 右图宽度: {match['right_width']:.2f}px")
    
    # 保存带标签的线段数据
    save_labeled_lines(left_lines, 'left_labeled_lines.csv')
    save_labeled_lines(right_lines, 'right_labeled_lines.csv')
    print("\n结果已保存为 left_labeled_lines.csv 和 right_labeled_lines.csv")
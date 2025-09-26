import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

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
    
    # 形态学参数
    'morph_kernel': (5, 11),    # 竖向特征检测
    'morph_iterations': 4,
    
    # 质心检测
    'dynamic_thresh_ratio':0.6, # 动态阈值比例
    'min_line_width': 1,        # 最小有效线宽
    'max_line_gap': 200,        # 断裂容忍度

    # 几何约束
    'roi_padding': 10,          # 边缘裁剪
    'cluster_eps': 6,           # 更小聚类半径（适应结构光连续性）
    'min_samples': 6,           # 最小样本数
    'min_line_length': 80,      # 有效线段长度

    # 后处理
    'smooth_sigma': 2.5,        # 平滑强度
    'max_end_curvature': 0.08,  # 更严格的端点曲率限制
    'smooth_degree': 3.0,       # 插值平滑度
    
    # 亚像素拟合参数
    'gaussian_window_size': 7,  # 高斯拟合窗口大小(奇数)
    'min_r2_for_subpixel': 0.8, # 最小R²值接受拟合
    'subpixel_refinement': True # 是否启用亚像素优化
}

# ====================== 右图参数配置（针对彩色图优化） ====================== 
RIGHT_CONFIG = {
    # 基础参数
    'laser_color': 'red',       # 激光颜色类型
    'min_laser_intensity': 75,  # 最低有效激光强度

    # 预处理参数
    'clahe_clip': 2.0,         # 对比度增强上限
    'blur_kernel': (3, 3),     # 高斯模糊核大小
    'gamma_correct': 0.75,     # 高光抑制
    'specular_thresh': 180,    # 高光检测阈值

    # 形态学参数
    'morph_kernel': (5, 11),   # 竖向特征检测
    'morph_iterations': 4,

    # 质心检测
    'dynamic_thresh_ratio': 0.25, # 抗噪阈值
    'min_line_width': 1,       # 激光线宽度
    'max_line_gap': 200,       # 断裂容忍度

    # 几何约束
    'roi_padding': 15,         # 边缘裁剪
    'cluster_eps': 6,          # 更小聚类半径
    'min_samples': 6,          # 更小样本数
    'min_line_length': 100,    # 有效线段长度

    # 后处理
    'smooth_sigma': 2.0,       # 平滑强度
    'max_end_curvature': 0.15, # 端点曲率限制
    'smooth_degree': 2.5,      # 插值平滑度

    # 亚像素拟合参数
    'gaussian_window_size': 7, # 高斯拟合窗口大小
    'min_r2_for_subpixel': 0.8,# 最小R²值接受拟合
    'subpixel_refinement': True # 是否启用亚像素优化
}

# ====================== 大津阈值法预处理 ======================
def otsu_preprocess(img, config):
    """
    大津阈值法预处理流程
    Args:
        img: 输入图像
        config: 配置参数
    Returns:
        预处理后的单通道二值图像
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, config['blur_kernel'], 0)
    
    # 大津阈值法
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config['morph_kernel'])
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=config['morph_iterations'])
    
    return closed

# ====================== 亚像素处理函数 ======================
def gaussian_1d(x, amp, mu, sigma, baseline):
    """1D高斯函数定义"""
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2)) + baseline

def subpixel_gaussian_fit(x_data, y_data, init_guess, config):
    """亚像素级高斯拟合"""
    try:
        popt, pcov = curve_fit(
            f=gaussian_1d,
            xdata=x_data,
            ydata=y_data,
            p0=init_guess,
            maxfev=1000,
            bounds=([
                0,                      # 幅度下限
                min(x_data),            # 均值下限
                0.5,                    # σ下限
                0                       # 基线下限
            ], [
                np.inf,                 # 幅度上限
                max(x_data),            # 均值上限
                5.0,                    # σ上限
                max(y_data)             # 基线上限
            ])
        )
        
        y_pred = gaussian_1d(x_data, *popt)
        r2 = r2_score(y_data, y_pred)
        
        if r2 < config['min_r2_for_subpixel']:
            return init_guess[1], r2  # 返回原始质心
            
        return popt[1], r2  # 返回亚像素优化结果
    except:
        return init_guess[1], 0  # 拟合失败时返回原值

# ====================== 质心检测 ======================
def dynamic_centroid_detection(row, config):
    """动态阈值质心检测算法（逐行处理）"""
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
    half_win = config['gaussian_window_size'] // 2
    
    for s, e in segments:
        # 确保窗口大小足够
        start_win = max(s - half_win, 0)
        end_win = min(e + half_win, len(row) - 1)
        
        x_data = np.arange(start_win, end_win + 1)
        y_data = row[start_win:end_win + 1]
        
        # 初始质心估计
        x_segment = np.arange(s, e+1)
        weights = row[s:e+1]
        centroid = np.sum(x_segment * weights) / np.sum(weights)
        init_guess = [np.max(y_data), centroid, 1.0, np.min(y_data)]
        
        # 亚像素拟合
        if config['subpixel_refinement']:
            subpixel_centroid, r2 = subpixel_gaussian_fit(x_data, y_data, init_guess, config)
            centers.append(float(subpixel_centroid))
        else:
            centers.append(int(round(centroid)))
    
    return centers

# ====================== 线段处理函数 ======================
def filter_endpoints_curvature(line, config):
    """端点曲率过滤（消除毛刺）"""
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
    """提取线段特征向量"""
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
    """构建相似度矩阵"""
    n = len(features)
    similarity = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity[i,j] = 1.0
            else:
                dir_sim = abs(features[i]['direction'] @ features[j]['direction'])
                end_dist = min(
                    np.linalg.norm(features[i]['end_point'] - features[j]['start_point']),
                    np.linalg.norm(features[i]['start_point'] - features[j]['end_point']),
                    np.linalg.norm(features[i]['end_point'] - features[j]['end_point']),
                    np.linalg.norm(features[i]['start_point'] - features[j]['start_point'])
                )
                pos_sim = max(0, 1 - end_dist / 100)
                curv_sim = 1 - abs(features[i]['curvature'] - features[j]['curvature']) / max(features[i]['curvature'], features[j]['curvature'], 0.01)
                inten_sim = 1 - abs(features[i]['mean_intensity'] - features[j]['mean_intensity']) / 255
                
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
            i_center = features[i]['start_point'] + (features[i]['end_point'] - features[i]['start_point'])/2
            j_center = features[j]['start_point'] + (features[j]['end_point'] - features[j]['start_point'])/2
            
            radial_i = i_center - center
            radial_j = j_center - center
            
            radial_dir_sim = abs(radial_i @ radial_j) / (np.linalg.norm(radial_i) * np.linalg.norm(radial_j) + 1e-6)
            similarity[i,j] *= radial_dir_sim ** 2
    
    return similarity

def match_broken_lines(lines, img, config):
    """基于几何特征和回转体约束的断线匹配"""
    if not lines:
        return []
    
    features = []
    valid_lines = []
    for line in lines:
        feat = extract_line_features(line, img)
        if feat and feat['length'] > config['min_line_length']/2:
            features.append(feat)
            valid_lines.append(line)
    
    if not features:
        return []
    
    similarity_matrix = compute_similarity_matrix(features, img)
    n_clusters = max(3, len(features) // 4)
    clusterer = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42
    )
    
    affinity_matrix = similarity_matrix / similarity_matrix.max()
    labels = clusterer.fit_predict(affinity_matrix)
    
    labeled_lines = []
    for i, label in enumerate(labels):
        labeled_lines.append({
            'label': label + 1,
            'points': valid_lines[i],
            'features': features[i]
        })
    
    return labeled_lines

def geometry_based_clustering(points, img_size, config, original_img):
    """基于几何约束的聚类优化"""
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

    labeled_lines = match_broken_lines(valid_lines, original_img, config)
    
    if labeled_lines:
        sorted_lines = sorted(labeled_lines, key=lambda x: np.mean(x['points'][:,0]))
        for new_label, line in enumerate(sorted_lines, 1):
            line['label'] = new_label
        return sorted_lines
    
    return []

def detect_laser_lines(img, config):
    """激光线检测主流程"""
    # 使用大津阈值法预处理
    binary = otsu_preprocess(img, config)

    points = []
    for y in range(binary.shape[0]):
        row = binary[y, :]
        centers = dynamic_centroid_detection(row, config)
        points.extend([[x, y] for x in centers])

    if not points:
        return []

    lines = geometry_based_clustering(np.array(points), binary.shape, config, img)
    return lines

def visualize_results(img, lines, title):
    """增强可视化（显示标签）"""
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f'{title} Original')

    # 显示大津阈值处理结果
    binary = otsu_preprocess(img, LEFT_CONFIG if 'L' in title else RIGHT_CONFIG)
    ax[1].imshow(binary, cmap='gray')
    ax[1].set_title('Otsu Thresholding')

    vis = img.copy()
    colors = plt.cm.get_cmap('tab20', 20)
    
    unique_labels = set(line['label'] for line in lines) if lines else set()
    
    for line in lines:
        color = colors(line['label'] % 20)
        color_rgb = (np.array(color[:3]) * 255).astype(int).tolist()
        pts = line['points'].astype(int)
        cv2.polylines(vis, [pts], False, color_rgb, 2)
        
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

if __name__ == "__main__":
    left_img = cv2.imread('30.1.bmp')
    right_img = cv2.imread('30.0.bmp')

    print("处理左图(L.bmp)...")
    left_lines = detect_laser_lines(left_img, LEFT_CONFIG)
    unique_left_labels = set(line['label'] for line in left_lines) if left_lines else set()
    print(f"左图提取到 {len(unique_left_labels)} 条中心线（共 {len(left_lines)} 个线段）")

    print("\n处理右图(R.jpg)...")
    right_lines = detect_laser_lines(right_img, RIGHT_CONFIG)
    unique_right_labels = set(line['label'] for line in right_lines) if right_lines else set()
    print(f"右图提取到 {len(unique_right_labels)} 条中心线（共 {len(right_lines)} 个线段）")

    visualize_results(left_img, left_lines, 'Left Image')
    visualize_results(right_img, right_lines, 'Right Image')

    save_labeled_lines(left_lines, 'left_labeled_lines.csv')
    save_labeled_lines(right_lines, 'right_labeled_lines.csv')
    print("结果已保存为 left_labeled_lines.csv 和 right_labeled_lines.csv")
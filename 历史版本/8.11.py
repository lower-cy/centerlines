# laser_center_extraction_improved.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from scipy.stats import cauchy
import warnings

warnings.filterwarnings("ignore")

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

# ====================== 左/右配置（包含新增参数） ======================
LEFT_CONFIG = {
    'laser_color': 'gray',
    'min_laser_intensity': 75,
    'clahe_clip': 3.5,
    'blur_kernel': (3, 3),
    'gamma_correct': 1.0,        # baseline gamma (used as lower bound)
    'specular_thresh': 200,
    'local_enhance_region': (0, 1),
    'clahe_clip_local': 1.5,
    'blend_weights': (0.2, 0.8),
    'morph_kernel': (5, 11),
    'morph_iterations': 4,
    'dynamic_thresh_ratio':0.6,
    'min_line_width': 1,
    'max_line_gap': 200,
    'roi_padding': 10,
    'cluster_eps': 6,
    'min_samples': 6,
    'min_line_length': 80,
    'smooth_sigma': 2.5,
    'max_end_curvature': 0.08,
    'smooth_degree': 3.0,
    'max_gap_for_matching': 500,
    'direction_similarity': 0.2,
    'intensity_similarity': 0.8,
    'position_tolerance': 30,
    'min_extension_length': 50,
    'max_extension_angle': 60,
    'min_depth': -100,
    'max_depth': 100,
    'disparity_tolerance': 5,
    'width_diff_threshold': 0.3,
    'min_matched_lines': 0,
    # new params
    'composite_filter_weights': (0.6, 0.4),  # (gaussian_weight, median_weight)
    'gaussian_sigma': 1.2,
    'cauchy_fit_width': 11,  # when fitting per-row, sample +- this radius
    'gamma_dynamic_gain': 2.0,  # how strongly gamma reacts to specularness
}

RIGHT_CONFIG = {
    'laser_color': 'red',
    'min_laser_intensity': 75,
    'clahe_clip': 2.0,
    'blur_kernel': (3, 3),
    'gamma_correct': 0.75,
    'specular_thresh': 180,
    'local_enhance_region': (0, 1),
    'clahe_clip_local': 5.0,
    'blend_weights': (0.2, 0.8),
    'morph_kernel': (5, 11),
    'morph_iterations': 4,
    'dynamic_thresh_ratio':0.25,
    'min_line_width': 1,
    'max_line_gap': 200,
    'roi_padding': 15,
    'cluster_eps': 6,
    'min_samples': 6,
    'min_line_length': 100,
    'smooth_sigma': 2.0,
    'max_end_curvature': 0.15,
    'smooth_degree': 2.5,
    'max_gap_for_matching': 500,
    'direction_similarity': 0.2,
    'intensity_similarity': 0.75,
    'position_tolerance': 20,
    'min_extension_length': 40,
    'max_extension_angle': 60,
    'min_depth': -100,
    'max_depth': 100,
    'disparity_tolerance': 5,
    'width_diff_threshold': 0.3,
    'min_matched_lines': 0,
    # new params
    'composite_filter_weights': (0.6, 0.4),
    'gaussian_sigma': 1.0,
    'cauchy_fit_width': 11,
    'gamma_dynamic_gain': 1.5,
}

# ====================== 极线矫正相关函数 ======================

def stereo_rectify(left_img, right_img):
    h, w = left_img.shape[:2]
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        CAMERA_MATRIX_LEFT, DIST_COEFF_LEFT,
        CAMERA_MATRIX_RIGHT, DIST_COEFF_RIGHT,
        (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
    )

    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX_LEFT, DIST_COEFF_LEFT, R1, P1, (w, h), cv2.CV_32FC1
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX_RIGHT, DIST_COEFF_RIGHT, R2, P2, (w, h), cv2.CV_32FC1
    )

    left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)

    return left_rectified, right_rectified, Q, R1, R2

def rectify_lines(lines, Rmat, img_size):
    rectified_lines = []
    for line in lines:
        points = line['points'].astype(np.float32)
        homogeneous = np.column_stack((points, np.ones(len(points))))
        rect_points = (Rmat @ homogeneous.T).T[:, :2]
        rect_points[:, 0] = np.clip(rect_points[:, 0], 0, img_size[0]-1)
        rect_points[:, 1] = np.clip(rect_points[:, 1], 0, img_size[1]-1)
        rect_line = line.copy()
        rect_line['points'] = rect_points
        rectified_lines.append(rect_line)
    return rectified_lines

# ====================== 图像预处理（动态 gamma + 复合滤波） ======================

def adaptive_gamma_correction_dynamic(img, config):
    """
    动态 gamma 校正：根据图像中高光像素的统计特征自适应调整 gamma。
    gamma = base_gamma + gain * (mean_specular - spec_thresh)/255
    仅对高光区域应用校正，非高光区域保持原图
    """
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    spec_mask = gray >= config['specular_thresh']
    if np.sum(spec_mask) == 0:
        # 没有高光，直接返回原图
        return img.copy()

    mean_spec = np.mean(gray[spec_mask])
    base_gamma = float(config.get('gamma_correct', 1.0))
    gain = float(config.get('gamma_dynamic_gain', 1.0))
    gamma = base_gamma + gain * max(0.0, (mean_spec - config['specular_thresh']) / 255.0)
    gamma = np.clip(gamma, 0.3, 3.0)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(img, table)

    # 合成：高光区域用校正后的像素，其他区域保留原图（保证不损失弱条纹）
    if len(img.shape) == 2:
        out = img.copy()
        out[spec_mask] = corrected[spec_mask]
    else:
        out = img.copy()
        for c in range(3):
            ch = out[:,:,c]
            ch[spec_mask] = corrected[:,:,c][spec_mask]
            out[:,:,c] = ch

    return out

def composite_filter(img_gray, config):
    """
    复合滤波：按权重融合 Gaussian 平滑与 Median 滤波，兼顾边缘与脉冲噪声抑制
    """
    gw, mw = config.get('composite_filter_weights', (0.6, 0.4))
    sigma = config.get('gaussian_sigma', 1.2)
    gauss = cv2.GaussianBlur(img_gray, (0,0), sigmaX=sigma, sigmaY=sigma)
    med = cv2.medianBlur(img_gray, 5)
    merged = (gw * gauss + mw * med).astype(np.uint8)
    return merged

def local_contrast_enhancement(gray, config):
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
    """
    多尺度预处理流程：动态 gamma -> LAB CLAHE -> 复合滤波 -> 激光通道增强 -> 局部 CLAHE
    """
    # 1: 动态伽马自适应高光抑制
    corrected = adaptive_gamma_correction_dynamic(img, config)

    # 2: 转到 LAB, 对 L 做 CLAHE
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(8,8))
    l = clahe.apply(l)
    lab_eq = cv2.merge([l, a, b])
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 3: 转为灰度并执行复合滤波
    gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    merged = composite_filter(gray, config)

    # 4: 激光通道增强并再转灰度
    enhanced = enhance_laser_channel(merged, config)
    gray2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # 5: 局部增强（论文方法）
    enhanced_gray = local_contrast_enhancement(gray2, config)

    return enhanced_gray

# ====================== 高斯 + 柯西混合拟合用于行截面质心 ======================

def gauss_lorentz(x, a_g, mu, sigma, a_c, gamma, offset):
    """
    混合模型： 高斯 + Cauchy(Lorentz) (Cauchy location-scale form uses gamma as scale)
    a_g * exp(-0.5*((x-mu)/sigma)^2) + a_c * (gamma^2 / ((x-mu)^2 + gamma^2)) + offset
    """
    g = a_g * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    c = a_c * (gamma**2 / ((x - mu)**2 + gamma**2))
    return g + c + offset

def fit_gauss_cauchy(x, y):
    """
    用 curve_fit 拟合混合模型，返回拟合参数（或抛出异常）
    """
    # 初值估计
    a_g0 = max(np.max(y) - np.min(y), 1.0)
    mu0 = x[np.argmax(y)]
    sigma0 = max(1.0, np.std(x) if np.std(x)>0 else 1.0)
    a_c0 = a_g0 * 0.5
    gamma0 = max(1.0, sigma0 * 0.6)
    offset0 = np.min(y)

    p0 = [a_g0, mu0, sigma0, a_c0, gamma0, offset0]

    bounds_low = [0, x.min(), 0.1, 0, 0.1, 0]
    bounds_high = [np.max(y)*3, x.max(), (x.max()-x.min()), np.max(y)*3, (x.max()-x.min()), np.max(y)]

    popt, _ = curve_fit(gauss_lorentz, x, y, p0=p0, bounds=(bounds_low, bounds_high), maxfev=5000)
    return popt

def robust_centroid_from_profile(profile, config):
    """
    基于混合拟合的稳健质心提取（输入为一维强度 profile）
    若拟合失败或不可靠（比如拟合峰很低），回退到动态阈值加权质心法
    """
    L = len(profile)
    x = np.arange(L)
    maxv = np.max(profile)
    if maxv < config['min_laser_intensity']:
        return []

    # 基于连通段先分割 candidate segments
    thresh = maxv * config['dynamic_thresh_ratio']
    mask = profile > thresh
    segments = []
    start = -1
    for i, m in enumerate(mask):
        if m and start == -1:
            start = i
        elif (not m) and start != -1:
            if i - start >= config['min_line_width']:
                segments.append((start, i-1))
            start = -1
    if start != -1 and L - start >= config['min_line_width']:
        segments.append((start, L-1))

    centers = []
    for s, e in segments:
        seglen = e - s + 1
        # sample profile in an expanded window to fit global shape
        pad = config.get('cauchy_fit_width', 11)
        xs = np.arange(max(0, s-pad), min(L, e+pad+1))
        ys = profile[xs]
        try:
            popt = fit_gauss_cauchy(xs, ys)
            # estimate centroid from fitted Gaussian+Lorentzian weighted mean (use mu)
            mu = popt[1]
            if mu >= s-1 and mu <= e+1:
                centers.append(int(round(mu)))
                continue
            else:
                # fallback to weighted centroid
                pass
        except Exception:
            pass

        # 回退：加权质心（动态阈值区域内）
        weights = profile[s:e+1].astype(np.float64)
        xs_local = np.arange(s, e+1)
        if np.sum(weights) == 0:
            continue
        centroid = np.sum(xs_local * weights) / np.sum(weights)
        centers.append(int(round(centroid)))

    return centers

# ====================== 激光线检测主流程（质心改造） ======================

def dynamic_centroid_detection(row, config):
    """
    新版：每行使用 robust_centroid_from_profile（混合拟合回退机制）
    """
    return robust_centroid_from_profile(row, config)

def filter_endpoints_curvature(line, config):
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
    if len(line) < 2:
        return None

    start_pt = line[0]
    end_pt = line[-1]
    length = np.linalg.norm(end_pt - start_pt)
    direction = (end_pt - start_pt) / (length + 1e-6)

    intensities = []
    for pt in line:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            if len(img.shape) == 3:
                intensities.append(np.mean(img[y, x]))
            else:
                intensities.append(img[y, x])

    if not intensities:
        return None

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
        'mean_intensity': float(np.mean(intensities)),
        'intensity_std': float(np.std(intensities)),
        'curvature': float(avg_curvature),
        'points': line
    }

def compute_similarity_matrix(features, img):
    """
    更鲁棒的相似度：先归一化各特征维度，再加回转体径向一致性约束
    """
    n = len(features)
    similarity = np.zeros((n, n))

    # Collect vectors for normalization
    directions = np.array([f['direction'] for f in features])
    curvs = np.array([f['curvature'] for f in features])
    intensities = np.array([f['mean_intensity'] for f in features])
    centers = np.array([(f['start_point'] + f['end_point']) / 2.0 for f in features])

    # normalization (avoid div by zero)
    def norm_arr(a):
        mn, mx = a.min(), a.max()
        if abs(mx - mn) < 1e-6:
            return np.zeros_like(a) + 0.5
        return (a - mn) / (mx - mn)

    norm_curv = norm_arr(curvs)
    norm_inten = norm_arr(intensities)
    # direction similarity via absolute dot product
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity[i,j] = 1.0
            else:
                dir_sim = abs(np.dot(directions[i], directions[j]))
                # position distance between endpoints (use minimum of endpoint combos)
                endsdist = min(
                    np.linalg.norm(features[i]['end_point'] - features[j]['start_point']),
                    np.linalg.norm(features[i]['start_point'] - features[j]['end_point']),
                    np.linalg.norm(features[i]['end_point'] - features[j]['end_point']),
                    np.linalg.norm(features[i]['start_point'] - features[j]['start_point'])
                )
                pos_sim = max(0, 1 - endsdist / 200.0)  # more tolerant baseline

                curv_sim = 1 - abs(norm_curv[i] - norm_curv[j])
                inten_sim = 1 - abs(norm_inten[i] - norm_inten[j])

                similarity[i,j] = 0.45 * dir_sim + 0.2 * pos_sim + 0.2 * curv_sim + 0.15 * inten_sim

    # 回转体径向一致性：计算向量与图像中心的径向方向，然后增加权重
    center_img = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0])
    for i in range(n):
        for j in range(n):
            ri = centers[i] - center_img
            rj = centers[j] - center_img
            ni = np.linalg.norm(ri)
            nj = np.linalg.norm(rj)
            if ni < 1e-3 or nj < 1e-3:
                radial_dir_sim = 0.5
            else:
                radial_dir_sim = abs(np.dot(ri, rj) / (ni * nj))
            # 强化到 similarity
            similarity[i,j] *= (0.5 + 0.5 * (radial_dir_sim ** 2))  # in [0.5,1.0]
    return similarity

def match_broken_lines(lines, img, config):
    """
    基于特征和谱聚类的断线连接
    """
    if not lines:
        return []

    features = []
    valid_lines = []
    for line in lines:
        # `line` may already be np.array of points or list of pts
        arr = np.array(line)
        feat = extract_line_features(arr, img)
        if feat and feat['length'] > config['min_line_length'] / 2.0:
            features.append(feat)
            valid_lines.append(arr)

    if not features:
        return []

    S = compute_similarity_matrix(features, img)
    # affinity in [0,1] scale
    if np.max(S) > 0:
        affinity = S / np.max(S)
    else:
        affinity = S

    # choose cluster num adaptively
    n_clusters = max(1, min(8, len(features) // 3))
    if n_clusters == 1:
        labels = np.zeros(len(features), dtype=int)
    else:
        clusterer = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        labels = clusterer.fit_predict(affinity)

    labeled_lines = []
    for i, lbl in enumerate(labels):
        labeled_lines.append({
            'label': int(lbl) + 1,
            'points': valid_lines[i],
            'features': features[i]
        })
    return labeled_lines

def geometry_based_clustering(points, img_size, config, original_img):
    h, w = img_size
    if len(points) == 0:
        return []

    mask = (points[:,0] > config['roi_padding']) & (points[:,0] < w - config['roi_padding'])
    points = points[mask]
    if len(points) == 0:
        return []

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
        except Exception:
            new_points = sorted_cluster

        new_points[:,0] = gaussian_filter1d(new_points[:,0], config['smooth_sigma'])
        new_points[:,1] = gaussian_filter1d(new_points[:,1], config['smooth_sigma'])

        filtered_line = filter_endpoints_curvature(new_points, config)
        valid_lines.append(filtered_line)

    labeled_lines = match_broken_lines(valid_lines, original_img, config)

    # 从左到右重新编号
    if labeled_lines:
        sorted_lines = sorted(labeled_lines, key=lambda x: np.mean(x['points'][:,0]))
        for new_label, line in enumerate(sorted_lines, 1):
            line['label'] = new_label
        return sorted_lines
    return []

def detect_laser_lines(img, config):
    preprocessed = multi_scale_preprocess(img, config)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config['morph_kernel'])
    closed = cv2.morphologyEx(preprocessed, cv2.MORPH_CLOSE, kernel, iterations=config['morph_iterations'])

    # 局部增强（论文式）
    enhanced = local_contrast_enhancement(closed, {
        'local_enhance_region': config['local_enhance_region'],
        'clahe_clip_local': config['clahe_clip_local'],
        'blend_weights': config['blend_weights']
    })

    points = []
    # 逐行检测质心
    for y in range(enhanced.shape[0]):
        centers = dynamic_centroid_detection(enhanced[y, :], config)
        points.extend([[x, y] for x in centers])

    if not points:
        return []

    lines = geometry_based_clustering(np.array(points), enhanced.shape, config, img)
    return lines

# ====================== 可视化与保存函数 ======================

def visualize_results(img, lines, title):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f'{title} Original')
    preprocessed = multi_scale_preprocess(img, LEFT_CONFIG if 'L' in title else RIGHT_CONFIG)
    ax[1].imshow(preprocessed, cmap='gray')
    ax[1].set_title('Preprocessed')
    vis = img.copy()
    try:
        cmap = plt.colormaps['tab20']
    except AttributeError:
        cmap = plt.cm.get_cmap('tab20')
    unique_labels = set(line['label'] for line in lines) if lines else set()
    for line in lines:
        color = cmap(line['label'] % 20)
        color_rgb = (np.array(color[:3]) * 255).astype(int).tolist()
        pts = line['points'].astype(int)
        cv2.polylines(vis, [pts], False, color_rgb, 2)
        if len(pts) > 0:
            cv2.putText(vis, str(line['label']), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    ax[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax[2].set_title(f'Detected {len(unique_labels)} Lines with {len(lines)} Segments')
    plt.tight_layout()
    plt.show()

def save_labeled_lines(lines, filename):
    with open(filename, 'w') as f:
        for line in lines:
            f.write(f"# Label: {line['label']}\n")
            np.savetxt(f, line['points'], fmt='%.2f', delimiter=',')
            f.write("\n")

# ====================== 立体匹配（保留你原有逻辑，修正若干计算） ======================

def calculate_depth_range(config, baseline, focal_length):
    return config['min_depth'], config['max_depth']

def find_candidate_matches(left_line, right_lines, config, baseline, focal_length, Q):
    left_points = left_line['points']
    left_mid = np.mean(left_points, axis=0)
    min_depth, max_depth = calculate_depth_range(config, baseline, focal_length)
    # avoid divide by zero / invalid ranges
    if min_depth == 0 or max_depth == 0:
        return list(range(len(right_lines)))

    min_disp = baseline * focal_length / (max_depth if max_depth!=0 else 1.0)
    max_disp = baseline * focal_length / (min_depth if min_depth!=0 else 1.0)
    # ensure min_disp <= max_disp
    lo, hi = min(min_disp, max_disp), max(min_disp, max_disp)

    candidates = []
    for idx, right_line in enumerate(right_lines):
        right_points = right_line['points']
        right_mid = np.mean(right_points, axis=0)
        if abs(right_mid[1] - left_mid[1]) > config['position_tolerance']:
            continue
        disparity = left_mid[0] - right_mid[0]
        if lo - config['disparity_tolerance'] <= disparity <= hi + config['disparity_tolerance']:
            candidates.append(idx)
    return candidates

def calculate_line_width(line_points, img):
    if len(line_points) < 2:
        return 0
    widths = []
    for pt in line_points:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= y < img.shape[0]:
            row = img[y, :] if len(img.shape) == 2 else img[y, :, 0]
            maxv = np.max(row) if np.max(row)>0 else 1
            binary = row > (maxv * 0.5)
            edges = np.where(np.diff(binary.astype(int)) != 0)[0]
            if len(edges) >= 2:
                line_width = edges[-1] - edges[0]
                widths.append(line_width)
    return np.mean(widths) if widths else 0

def evaluate_match_quality(left_line, right_line, left_img, right_img, config):
    left_y_range = (np.min(left_line['points'][:, 1]), np.max(left_line['points'][:, 1]))
    right_y_range = (np.min(right_line['points'][:, 1]), np.max(right_line['points'][:, 1]))
    overlap_start = max(left_y_range[0], right_y_range[0])
    overlap_end = min(left_y_range[1], right_y_range[1])
    if overlap_end <= overlap_start:
        return 0
    overlap_ratio = (overlap_end - overlap_start) / (left_y_range[1] - left_y_range[0] + 1e-6)
    left_dir = left_line['features']['direction']
    right_dir = right_line['features']['direction']
    dir_similarity = abs(left_dir @ right_dir)
    left_width = calculate_line_width(left_line['points'], left_img)
    right_width = calculate_line_width(right_line['points'], right_img)
    if right_width == 0:
        width_similarity = 0
    else:
        width_ratio = left_width / (right_width + 1e-6)
        width_similarity = max(0, 1 - abs(width_ratio - 1) / (config['width_diff_threshold'] + 1e-6))
    score = overlap_ratio * 0.45 + dir_similarity * 0.35 + width_similarity * 0.2
    return score

def stereo_match(left_lines, right_lines, left_img, right_img, config, Q):
    baseline = np.linalg.norm(T)
    focal_length = (CAMERA_MATRIX_LEFT[0,0] + CAMERA_MATRIX_RIGHT[0,0]) / 2.0
    left_lines = [line for line in left_lines if len(line['points']) >= config['min_line_length']]
    right_lines = [line for line in right_lines if len(line['points']) >= config['min_line_length']]
    total_candidates = len(left_lines)
    print(f"深度范围内待匹配线段数量: {total_candidates}")
    matches = []
    matched_right_indices = set()
    left_lines_sorted = sorted(left_lines, key=lambda x: np.mean(x['points'][:,0]))
    if len(left_lines_sorted) > 1:
        x_positions = [np.mean(line['points'][:,0]) for line in left_lines_sorted]
        avg_width_diff = np.mean(np.diff(x_positions))
    else:
        avg_width_diff = 0
    for left_idx, left_line in enumerate(left_lines_sorted):
        candidates = find_candidate_matches(left_line, right_lines, config, baseline, focal_length, Q)
        best_score = -1
        best_right_idx = -1
        for right_idx in candidates:
            if right_idx in matched_right_indices:
                continue
            score = evaluate_match_quality(left_line, right_lines[right_idx], left_img, right_img, config)
            if avg_width_diff > 0 and len(left_lines_sorted)>1:
                expected_x = np.mean(left_line['points'][:,0]) - left_idx * avg_width_diff
                actual_x = np.mean(right_lines[right_idx]['points'][:,0])
                x_diff = abs(expected_x - actual_x)
                width_consistency = max(0, 1 - x_diff / (abs(avg_width_diff)*2 + 1e-6))
                score = score * 0.75 + width_consistency * 0.25
            if score > best_score:
                best_score = score
                best_right_idx = right_idx
        if best_right_idx != -1 and best_score > 0:
            matches.append((left_idx, best_right_idx, best_score))
            matched_right_indices.add(best_right_idx)
    if len(matches) < config['min_matched_lines']:
        print("警告：初始匹配数量不足，尝试放宽条件重新匹配")
        matches = []
        matched_right_indices = set()
        for left_idx, left_line in enumerate(left_lines_sorted):
            candidates = find_candidate_matches(left_line, right_lines, config, baseline, focal_length, Q)
            best_score = -1
            best_right_idx = -1
            for right_idx in candidates:
                if right_idx in matched_right_indices:
                    continue
                score = evaluate_match_quality(left_line, right_lines[right_idx], left_img, right_img, config)
                if score > best_score:
                    best_score = score
                    best_right_idx = right_idx
            if best_right_idx != -1 and best_score > 0:
                matches.append((left_idx, best_right_idx, best_score))
                matched_right_indices.add(best_right_idx)
    results = []
    for left_idx, right_idx, score in matches:
        left_line = left_lines_sorted[left_idx]
        right_line = right_lines[right_idx]
        left_mid = np.mean(left_line['points'], axis=0)
        right_mid = np.mean(right_line['points'], axis=0)
        disparity = left_mid[0] - right_mid[0]
        point_3d = np.array([[[left_mid[0], left_mid[1], disparity]]], dtype=np.float32)
        try:
            point_4d = cv2.perspectiveTransform(point_3d, Q)
            depth = point_4d[0][0][2]
        except Exception:
            focal = CAMERA_MATRIX_LEFT[0, 0]
            depth = (baseline * focal) / (disparity + 1e-6)
        results.append({
            'left_line': left_line,
            'right_line': right_line,
            'match_score': score,
            'disparity': disparity,
            'depth': depth,
            'left_width': calculate_line_width(left_line['points'], left_img),
            'right_width': calculate_line_width(right_line['points'], right_img),
            'left_index': left_idx
        })
    results.sort(key=lambda x: x['left_index'])
    if len(results) > 1:
        x_positions = [np.mean(res['left_line']['points'][:,0]) for res in results]
        width_diffs = np.diff(x_positions)
        width_std = np.std(width_diffs)
        print(f"匹配线段宽度均匀性指标 - 平均宽度差: {np.mean(width_diffs):.2f}px, 标准差: {width_std:.2f}px")
    return results

def calculate_circularity(depths):
    if len(depths) < 3:
        return 0
    angles = np.linspace(0, 2*np.pi, len(depths), endpoint=False)
    radii = (depths - np.min(depths)) / (np.max(depths) - np.min(depths) + 1e-6)
    fft = np.fft.fft(radii * np.exp(1j*angles))
    energy = np.sum(np.abs(fft[1:len(fft)//2])) / max(1, (len(fft)//2 - 1))
    circularity = 1 - np.clip(energy / (np.max(radii) + 1e-6), 0, 1)
    return circularity

def visualize_matches(left_img, right_img, matches):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    try:
        cmap = plt.colormaps['tab20']
    except AttributeError:
        cmap = plt.cm.get_cmap('tab20')
    left_vis = left_img.copy()
    for i, match in enumerate(matches):
        color_rgba = cmap(i % 20)
        color = tuple(int(c * 255) for c in color_rgba[:3])
        pts = match['left_line']['points'].astype(int)
        cv2.polylines(left_vis, [pts], False, color, 2)
        cv2.putText(left_vis, str(i+1), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    ax1.imshow(cv2.cvtColor(left_vis, cv2.COLOR_BGR2RGB))
    ax1.set_title('Left Image with Matched Lines')
    right_vis = right_img.copy()
    for i, match in enumerate(matches):
        color_rgba = cmap(i % 20)
        color = tuple(int(c * 255) for c in color_rgba[:3])
        pts = match['right_line']['points'].astype(int)
        cv2.polylines(right_vis, [pts], False, color, 2)
        cv2.putText(right_vis, str(i+1), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    ax2.imshow(cv2.cvtColor(right_vis, cv2.COLOR_BGR2RGB))
    ax2.set_title('Right Image with Matched Lines')
    plt.tight_layout()
    plt.show()

# ====================== 主程序 ======================

if __name__ == "__main__":
    left_img = cv2.imread('31.1.bmp')
    right_img = cv2.imread('31.0.bmp')

    if left_img is None or right_img is None:
        print("错误：无法读取图像文件！请检查文件路径和名称是否正确。")
        print(f"左图路径: {'31.1.bmp'}")
        print(f"右图路径: {'31.0.bmp'}")
        exit()

    print(f"图像尺寸 - 左图: {left_img.shape}, 右图: {right_img.shape}")

    print("\n处理左图...")
    left_lines = detect_laser_lines(left_img, LEFT_CONFIG)
    unique_left_labels = set(line['label'] for line in left_lines) if left_lines else set()
    print(f"左图提取到 {len(unique_left_labels)} 条中心线（共 {len(left_lines)} 个线段）")

    print("\n处理右图...")
    right_lines = detect_laser_lines(right_img, RIGHT_CONFIG)
    unique_right_labels = set(line['label'] for line in right_lines) if right_lines else set()
    print(f"右图提取到 {len(unique_right_labels)} 条中心线（共 {len(right_lines)} 个线段）")

    visualize_results(left_img, left_lines, 'Left Image')
    visualize_results(right_img, right_lines, 'Right Image')

    print("进行极线矫正...")
    left_rectified, right_rectified, Q, R1, R2 = stereo_rectify(left_img, right_img)

    print(f"重投影矩阵 Q: \n{Q}")

    print("矫正线段坐标...")
    img_size = (left_img.shape[1], left_img.shape[0])
    left_lines_rectified = rectify_lines(left_lines, R1, img_size)
    right_lines_rectified = rectify_lines(right_lines, R2, img_size)

    print("\n进行立体匹配...")
    matches = stereo_match(left_lines_rectified, right_lines_rectified, left_rectified, right_rectified, LEFT_CONFIG, Q)
    print(f"找到 {len(matches)} 对匹配线段")

    if matches:
        depths = [match['depth'] for match in matches]
        circularity = calculate_circularity(depths)
        print(f"深度分布圆度评估分数: {circularity:.2f}")

    visualize_matches(left_rectified, right_rectified, matches)

    print("\n匹配结果:")
    for i, match in enumerate(matches):
        print(f"匹配对 {i+1}:")
        print(f"  左图线段 {match['left_line']['label']} <-> 右图线段 {match['right_line']['label']}")
        print(f"  匹配分数: {match['match_score']:.2f}, 视差: {match['disparity']:.2f}px, 深度: {match['depth']:.2f}mm")
        print(f"  左图宽度: {match['left_width']:.2f}px, 右图宽度: {match['right_width']:.2f}px")

    save_labeled_lines(left_lines, 'left_labeled_lines.csv')
    save_labeled_lines(right_lines, 'right_labeled_lines.csv')
    print("\n结果已保存为 left_labeled_lines.csv 和 right_labeled_lines.csv")

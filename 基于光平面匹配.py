import cv2
import numpy as np
import yaml
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

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

# ====================== 左图参数配置 ======================
LEFT_CONFIG = {
    'laser_color': 'gray',
    'min_laser_intensity': 75,
    'clahe_clip': 3.5,
    'blur_kernel': (3, 3),
    'gamma_correct': 1.0,
    'specular_thresh': 200,
    'local_enhance_region': (0, 1),
    'clahe_clip_local': 1.5,
    'blend_weights': (0.2, 0.8),
    'morph_kernel': (5, 11),
    'morph_iterations': 4,
    'dynamic_thresh_ratio': 0.6,
    'min_line_width': 1,
    'max_line_gap': 200,
    'roi_padding': 10,
    'cluster_eps': 6,
    'min_samples': 6,
    'min_line_length': 80,
    'smooth_sigma': 2.5,
    'max_end_curvature': 0.08,
    'smooth_degree': 3.0,
}

# ====================== 右图参数配置 ======================
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
    'dynamic_thresh_ratio': 0.25,
    'min_line_width': 1,
    'max_line_gap': 200,
    'roi_padding': 15,
    'cluster_eps': 6,
    'min_samples': 6,
    'min_line_length': 100,
    'smooth_sigma': 2.0,
    'max_end_curvature': 0.15,
    'smooth_degree': 2.5,
}

# ====================== 极线矫正 ======================

def stereo_rectify(left_img, right_img):
    h, w = left_img.shape[:2]
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        CAMERA_MATRIX_LEFT, DIST_COEFF_LEFT,
        CAMERA_MATRIX_RIGHT, DIST_COEFF_RIGHT,
        (w, h), R, T, alpha=-1, flags=0 | cv2.CALIB_USE_INTRINSIC_GUESS
    )
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX_LEFT, DIST_COEFF_LEFT, R1, P1, (w, h), cv2.CV_32FC1
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX_RIGHT, DIST_COEFF_RIGHT, R2, P2, (w, h), cv2.CV_32FC1
    )
    left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)
    return left_rectified, right_rectified, P1, P2, Q

# ====================== 图像预处理 ======================

def local_contrast_enhancement(gray, config):
    h, w = gray.shape
    x_start = int(w * config['local_enhance_region'][0])
    x_end = int(w * config['local_enhance_region'][1])
    region = gray[:, x_start:x_end]
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip_local'], tileGridSize=(8, 8))
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

def adaptive_gamma_correction(img, config):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, config['specular_thresh'], 255, cv2.THRESH_BINARY)
    inv_gamma = 1.0 / config['gamma_correct']
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(img, table)
    return cv2.bitwise_and(corrected, corrected, mask=mask) + cv2.bitwise_and(img, img, mask=~mask)

def multi_scale_preprocess(img, config):
    corrected = adaptive_gamma_correction(img, config)
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(8, 8))
    l = clahe.apply(l)
    blur1 = cv2.GaussianBlur(l, config['blur_kernel'], 0)
    blur2 = cv2.medianBlur(l, 5)
    merged = cv2.addWeighted(blur1, 0.6, blur2, 0.4, 0)
    enhanced = enhance_laser_channel(merged, config)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    enhanced_gray = local_contrast_enhancement(gray, config)
    return enhanced_gray

# ====================== 激光线检测 ======================

def dynamic_centroid_detection(row, config):
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
                segments.append((start, i - 1))
            start = -1
    if start != -1 and len(closed) - start >= config['min_line_width']:
        segments.append((start, len(closed) - 1))
    centers = []
    for s, e in segments:
        x = np.arange(s, e + 1)
        weights = row[s:e + 1]
        if np.sum(weights) == 0:
            continue
        centroid = np.sum(x * weights) / np.sum(weights)
        centers.append(int(round(centroid)))
    return centers

def filter_endpoints_curvature(line, config):
    if len(line) < 10:
        return line
    epsilon = 1e-6
    head, tail = line[:10], line[-10:]

    def calculate_curvature(segment):
        dx = np.gradient(segment[:, 0])
        dy = np.gradient(segment[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        return np.abs(d2x * dy - dx * d2y) / ((dx**2 + dy**2)**1.5 + epsilon)

    if np.mean(calculate_curvature(head)) > config['max_end_curvature']:
        line = line[5:]
    if np.mean(calculate_curvature(tail)) > config['max_end_curvature']:
        line = line[:-5]
    return line

def geometry_based_clustering(points, img_size, config):
    h, w = img_size
    mask = (points[:, 0] > config['roi_padding']) & (points[:, 0] < w - config['roi_padding'])
    points = points[mask]
    db = DBSCAN(eps=config['cluster_eps'], min_samples=config['min_samples']).fit(points)
    valid_lines = []
    for label in set(db.labels_):
        if label == -1:
            continue
        cluster = points[db.labels_ == label]
        if len(cluster) < config['min_line_length'] / 2:
            continue
        sorted_cluster = cluster[cluster[:, 1].argsort()]
        try:
            tck, u = splprep(sorted_cluster.T, s=config['smooth_degree'])
            new_u = np.linspace(u.min(), u.max(), int(len(u) * 2))
            new_points = np.column_stack(splev(new_u, tck))
        except:
            new_points = sorted_cluster
        new_points[:, 0] = gaussian_filter1d(new_points[:, 0], config['smooth_sigma'])
        new_points[:, 1] = gaussian_filter1d(new_points[:, 1], config['smooth_sigma'])
        filtered_line = filter_endpoints_curvature(new_points, config)
        valid_lines.append(filtered_line)
    return valid_lines

def detect_laser_lines(img, config):
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

# ====================== 光平面映射器 ======================

class LightPlaneMapper:
    """基于光平面的激光点匹配器（矫正坐标系下）"""

    def __init__(self, K_left_rect, K_right_rect, Tx, light_plane_coeffs):
        """
        Parameters:
        -----------
        K_left_rect, K_right_rect : 矫正后相机内参 (3x3), 来自 P1[:3,:3], P2[:3,:3]
        Tx : 基线距离 (mm), Tx = -P2[0,3] / P2[0,0]
        light_plane_coeffs : (a,b,c,d) 满足 a*x + b*y + c*z + d = 0
        """
        self.K_left = K_left_rect
        self.K_right = K_right_rect
        self.Tx = Tx
        self.a, self.b, self.c, self.d = light_plane_coeffs
        self.inv_K_left = np.linalg.inv(self.K_left)

    def point_to_3d(self, u, v):
        """左图点 -> 光平面交点的3D坐标 (左矫正相机坐标系)"""
        p_norm = self.inv_K_left @ np.array([u, v, 1.0])
        direction = p_norm / np.linalg.norm(p_norm)
        denom = self.a * direction[0] + self.b * direction[1] + self.c * direction[2]
        if abs(denom) < 1e-6:
            return None
        t = -self.d / denom
        if t < 0:
            return None
        return t * direction

    def project_3d_to_right(self, pt3d):
        """3D点 (左矫正坐标系) -> 右图像素坐标 (u, v)"""
        if pt3d[2] <= 0:
            return None
        x_right = pt3d[0] - self.Tx
        u = self.K_right[0, 0] * x_right / pt3d[2] + self.K_right[0, 2]
        v = self.K_right[1, 1] * pt3d[1] / pt3d[2] + self.K_right[1, 2]
        return np.array([u, v])

    def map_points_vectorized(self, pts_left):
        """批量: 左图点 -> 3D -> 右图投影 (向量化)"""
        pts_left = np.asarray(pts_left, dtype=np.float64)
        if pts_left.ndim == 1:
            pts_left = pts_left.reshape(1, -1)
        N = pts_left.shape[0]

        homo = np.column_stack([pts_left, np.ones(N)])
        norm = (self.inv_K_left @ homo.T).T
        norms = np.linalg.norm(norm, axis=1, keepdims=True)
        directions = norm / norms

        denom = self.a * directions[:, 0] + self.b * directions[:, 1] + self.c * directions[:, 2]
        valid = np.abs(denom) > 1e-6
        t = np.full(N, -1.0)
        t[valid] = -self.d / denom[valid]
        valid = valid & (t > 0)

        pts_3d = np.zeros((N, 3))
        pts_3d[valid] = t[valid, np.newaxis] * directions[valid]

        pts_right = np.full((N, 2), np.nan)
        valid_z = valid & (pts_3d[:, 2] > 0)
        if np.any(valid_z):
            xr = pts_3d[valid_z, 0] - self.Tx
            yr = pts_3d[valid_z, 1]
            zr = pts_3d[valid_z, 2]
            pts_right[valid_z, 0] = self.K_right[0, 0] * xr / zr + self.K_right[0, 2]
            pts_right[valid_z, 1] = self.K_right[1, 1] * yr / zr + self.K_right[1, 2]

        return pts_3d, pts_right, valid_z


def build_right_laser_index(right_lines):
    """构建右图激光线的空间索引: dict[y] = [(x, line_idx, point_idx), ...]"""
    index = {}
    for li, line in enumerate(right_lines):
        for pi, pt in enumerate(line):
            v = int(round(pt[1]))
            if v not in index:
                index[v] = []
            index[v].append((pt[0], li, pi))
    return index


def score_plane_for_line(left_line, right_index, K_left_rect, K_right_rect, Tx, plane, search_radius=3):
    """
    评估一个光平面对某条左图激光线的匹配质量。
    将所有左图点投影到右图，统计落在右激光线附近的点数。

    Returns:
        score : 落在右激光线附近的投影点数（越高越好）
        valid_count : 有效投影点数（Z>0 且分母不为零）
    """
    mapper = LightPlaneMapper(K_left_rect, K_right_rect, Tx, plane)
    _, pts_right, valid = mapper.map_points_vectorized(left_line)
    valid_indices = np.where(valid)[0]

    match_count = 0
    for i in valid_indices:
        uR, vR = pts_right[i]
        if np.isnan(uR) or np.isnan(vR):
            continue
        v_int = int(round(vR))
        found = False
        for dv in range(-search_radius, search_radius + 1):
            candidates = right_index.get(v_int + dv)
            if candidates is None:
                continue
            for u_cand, _, _ in candidates:
                if abs(u_cand - uR) <= search_radius:
                    match_count += 1
                    found = True
                    break
            if found:
                break

    return match_count, int(np.sum(valid))


def match_single_line(left_line, right_lines, mapper, search_radius=3):
    """
    用指定的光平面映射器，匹配单条左图激光线到右图。

    Returns:
        pts_left:   该线的左图匹配点 (M, 2)
        pts_right:  该线的右图匹配点 (M, 2)
        points_3d:  该线的 3D 点云 (M, 3)
        errors:     该线的重投影误差 (M,) 或空数组
        matched:    匹配点数
        total:      左线点数
    """
    right_index = build_right_laser_index(right_lines)

    pts_left = []
    pts_right = []
    points_3d = []
    errors = []

    for pt in left_line:
        uL, vL = pt[0], pt[1]

        pt3d = mapper.point_to_3d(uL, vL)
        if pt3d is None:
            continue

        proj = mapper.project_3d_to_right(pt3d)
        if proj is None:
            continue

        uR_est, vR_est = proj

        # 在右图索引中搜索最近邻
        v_int = int(round(vR_est))
        best_u = None
        best_v = None
        best_dist = search_radius

        for dv in range(-search_radius, search_radius + 1):
            candidates = right_index.get(v_int + dv)
            if candidates is None:
                continue
            for u_cand, _, _ in candidates:
                dist = abs(u_cand - uR_est)
                if dist < best_dist:
                    best_dist = dist
                    best_u = u_cand
                    best_v = v_int + dv

        if best_u is not None:
            pts_left.append([uL, vL])
            pts_right.append([best_u, best_v])
            points_3d.append(pt3d)
            errors.append(abs(best_u - uR_est))

    n = len(left_line)
    m = len(pts_left)
    if m == 0:
        return (np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 3), dtype=np.float64),
                np.array([]), 0, n)
    return (np.float32(pts_left), np.float32(pts_right),
            np.array(points_3d, dtype=np.float64),
            np.array(errors), m, n)


def visualize_plane_projections(left_line, right_img, right_lines, right_index,
                                 K_left_rect, K_right_rect, Tx,
                                 calibrated_planes, scored_planes, line_idx,
                                 search_radius=3):
    """
    交互式显示每条光平面对左图线的投影效果，供人工选择最佳平面。

    Parameters:
        scored_planes : list of (score, valid_count, plane) 已排序的评分结果
    Returns:
        int or None : 选中的平面在 calibrated_planes 中的索引，None 表示跳过此线

    按键:
        [a/d]     上一个/下一个平面
        [Space]   接受当前平面
        [Enter]   接受评分最高的平面
        [q]       跳过此线
    """
    n_planes = len(calibrated_planes)
    if n_planes == 0:
        return None

    # 预先计算所有平面的投影
    proj_results = []
    for plane in calibrated_planes:
        mapper = LightPlaneMapper(K_left_rect, K_right_rect, Tx, plane)
        _, pts_right, valid = mapper.map_points_vectorized(left_line)
        proj_results.append(pts_right)

    # 右图底图（画上右激光线，暗色）
    h, w = right_img.shape[:2]
    base = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR) if len(right_img.shape) == 2 else right_img.copy()
    for line in right_lines:
        pts = line.astype(int)
        cv2.polylines(base, [pts], False, (0, 0, 120), 1)

    # 排序后的平面索引（评分降序）
    sorted_indices = [calibrated_planes.index(sp[2]) for sp in scored_planes]

    idx = 0  # 当前查看的平面索引（在 sorted_indices 中的位置）
    window_name = f"光平面评估 - 线 {line_idx}"

    while True:
        vis = base.copy()
        plane_idx = sorted_indices[idx]
        plane = calibrated_planes[plane_idx]
        score, valid_count = scored_planes[idx][0], scored_planes[idx][1]
        pts_right = proj_results[plane_idx]

        # 绘制投影点
        valid_mask = ~np.any(np.isnan(pts_right), axis=1)
        for i in np.where(valid_mask)[0]:
            u, v = pts_right[i]
            if not (0 <= u < w and 0 <= v < h):
                continue
            v_int = int(round(v))
            matched = False
            for dv in range(-search_radius, search_radius + 1):
                candidates = right_index.get(v_int + dv)
                if candidates is None:
                    continue
                for u_cand, _, _ in candidates:
                    if abs(u_cand - u) <= search_radius:
                        matched = True
                        break
                if matched:
                    break
            color = (0, 255, 0) if matched else (0, 0, 255)
            cv2.circle(vis, (int(u), int(v)), 2, color, -1)

        # 信息叠加
        a, b, c, d = plane
        lines_text = [
            f"线 {line_idx}  平面 {idx+1}/{n_planes}  (rank {idx+1})  score={score}  valid={valid_count}",
            f"{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0",
            "[a/d]切换平面  [Space]选用此平面  [Enter]用最高分平面  [q]跳过此线"
        ]
        for i, t in enumerate(lines_text):
            cv2.putText(vis, t, (10, 30 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # 最佳平面的评分标注
        if idx > 0:
            best_score = scored_planes[0][0]
            cv2.putText(vis, f"最高分: {best_score}", (10, 30 + 3 * 24 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        show = cv2.resize(vis, (min(w, 1280), min(h, 960)))
        cv2.imshow(window_name, show)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return None
        elif key == ord(' ') or key == 13:  # Space = 选当前, Enter = 选最高分
            cv2.destroyWindow(window_name)
            if key == 13:  # Enter
                return sorted_indices[0]
            return plane_idx
        elif key == ord('a'):
            idx = (idx - 1) % n_planes
        elif key == ord('d'):
            idx = (idx + 1) % n_planes


def match_all_lines_auto(left_lines, right_lines, K_left_rect, K_right_rect, Tx,
                          calibrated_planes, search_radius=3, visualize=False, right_img=None):
    """
    自动检测每条左图激光线对应的最佳光平面，然后逐线匹配。

    遍历每条左图激光线，对所有已标定的光平面进行评分，
    选择匹配点数最多的光平面进行精确匹配。

    Parameters:
        left_lines : list of (Ni, 2) arrays
        right_lines : list of (Mj, 2) arrays
        calibrated_planes : list of (A, B, C, D) 所有已标定光平面
    Returns:
        pts_left, pts_right, points_3d : 汇总后的 numpy 数组
        mean_error, max_error, rmse : 重投影误差统计
        assignments : dict[int -> (A,B,C,D)] 线编号到光平面的分配
    """
    right_index = build_right_laser_index(right_lines)

    all_pts_left = []
    all_pts_right = []
    all_pts_3d = []
    all_errors = []
    total_pts = 0
    total_matched = 0
    assignments = {}

    for li, left_line in enumerate(left_lines):
        total_pts += len(left_line)

        # 对所有光平面评分
        scored = []
        for plane in calibrated_planes:
            score, valid_count = score_plane_for_line(
                left_line, right_index, K_left_rect, K_right_rect, Tx, plane, search_radius)
            scored.append((score, valid_count, plane))

        scored.sort(key=lambda x: (-x[0], -x[1]))

        best_score, best_valid, best_plane = scored[0]
        if best_score <= 0:
            print(f"  线 {li}: 无有效光平面，跳过 ({len(left_line)} 点)")
            continue

        # 可选交互可视化：人工确认/选择平面
        chosen = None
        if visualize and right_img is not None and len(left_line) > 0:
            chosen = visualize_plane_projections(
                left_line, right_img, right_lines, right_index,
                K_left_rect, K_right_rect, Tx,
                calibrated_planes, scored, li, search_radius)

        if chosen is None:
            # 用户跳过或未启用可视化 → 用自动评分最优平面
            pass
        else:
            best_plane = calibrated_planes[chosen]
            # 重新获取该平面的评分
            for s, v, p in scored:
                if p == best_plane:
                    best_score, best_valid = s, v
                    break

        assignments[li] = best_plane

        # 用最佳平面进行精确匹配
        mapper = LightPlaneMapper(K_left_rect, K_right_rect, Tx, best_plane)
        pts_l, pts_r, pts_3d, errs, matched, n_pts = match_single_line(
            left_line, right_lines, mapper, search_radius)

        a, b, c, d = best_plane
        print(f"  线 {li}: 平面({a:.4f},{b:.4f},{c:.4f},{d:.4f}) "
              f"score={best_score} {matched}/{n_pts} 匹配", end="")
        if len(errs):
            print(f" err={np.mean(errs):.2f}px")
        else:
            print()

        all_pts_left.append(pts_l)
        all_pts_right.append(pts_r)
        all_pts_3d.append(pts_3d)
        all_errors.append(errs)
        total_matched += matched

    # 汇总
    if not all_pts_left:
        print("错误: 无任何匹配点")
        return (np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 3)),
                0, 0, 0, {})

    pts_left = np.concatenate(all_pts_left, axis=0)
    pts_right = np.concatenate(all_pts_right, axis=0)
    points_3d = np.concatenate(all_pts_3d, axis=0)
    all_errors = np.concatenate(all_errors) if all_errors else np.array([])

    mean_err = np.mean(all_errors) if len(all_errors) else 0
    max_err = np.max(all_errors) if len(all_errors) else 0
    rmse = np.sqrt(np.mean(all_errors**2)) if len(all_errors) else 0

    print(f"\n总计: {total_matched}/{total_pts} 点匹配成功 "
          f"({100 * total_matched / max(total_pts, 1):.1f}%)")
    print(f"重投影误差: mean={mean_err:.2f} max={max_err:.2f} RMSE={rmse:.2f} 像素")

    print(f"\n光平面分配 (线 -> 平面):")
    for li, plane in sorted(assignments.items()):
        a, b, c, d = plane
        print(f"  线 {li} -> ({a:.4f}, {b:.4f}, {c:.4f}, {d:.4f})")

    return pts_left, pts_right, points_3d, mean_err, max_err, rmse, assignments


# ====================== 可视化函数 ======================

def line_results(img, lines):
    vis = img.copy()
    for i, line in enumerate(lines):
        color = (0, 255, 0) if i % 2 == 0 else (0, 128, 255)
        pts = line.astype(int)
        cv2.polylines(vis, [pts], False, color, 2)
        head_x, head_y = pts[0][0], pts[0][1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(i)
        text_size, _ = cv2.getTextSize(text, font, 0.8, 2)
        text_x = max(0, head_x - text_size[0] // 2)
        text_y = max(text_size[1], head_y - 10)
        cv2.putText(vis, text, (text_x, text_y), font, 0.8, (0, 0, 0), 3)
        cv2.putText(vis, text, (text_x, text_y), font, 0.8, (255, 255, 255), 2)
    return vis


def visualize_matched_points(left_img, right_img, pts_left, pts_right):
    h, w = left_img.shape[:2]
    composite = np.zeros((h, w * 2, 3), dtype=np.uint8)
    composite[:, :w] = left_img
    composite[:, w:] = right_img
    for pt_left, pt_right in zip(pts_left, pts_right):
        x1, y1 = int(pt_left[0]), int(pt_left[1])
        x2, y2 = int(pt_right[0] + w), int(pt_right[1])
        cv2.circle(composite, (x1, y1), 3, (0, 255, 0), -1)
        cv2.circle(composite, (x2, y2), 3, (0, 255, 0), -1)
        cv2.line(composite, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return composite


def write_ply(filename, verts, colors=None):
    with open(filename, 'w') as f:
        has_color = colors is not None
        header = f"""ply
format ascii 1.0
element vertex {len(verts)}
property float x
property float y
property float z
"""
        if has_color:
            header += """property uchar red
property uchar green
property uchar blue
"""
        header += "end_header\n"
        f.write(header)
        if has_color:
            for (x, y, z), (r, g, b) in zip(verts, colors):
                f.write(f'{x} {y} {z} {r} {g} {b}\n')
        else:
            for x, y, z in verts:
                f.write(f'{x} {y} {z}\n')


# ====================== 主程序 ======================

if __name__ == "__main__":
    # 1. 加载所有已标定的光平面
    with open("plane_equations.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 收集所有含 line_index 的光平面（线与平面的对应关系未知）
    calibrated_planes = []
    for entry in cfg["plane_equations"]:
        if "line_index" in entry:
            calibrated_planes.append((entry["A"], entry["B"], entry["C"], entry["D"]))
    print(f"加载了 {len(calibrated_planes)} 个已标定光平面:")
    for i, (a, b, c, d) in enumerate(calibrated_planes):
        print(f"  平面 {i}: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")

    # 2. 读取图像
    left_img = cv2.imread('30.1.bmp')
    right_img = cv2.imread('30.0.bmp')
    if left_img is None or right_img is None:
        print("错误: 无法读取图像")
        exit()

    # 3. 极线矫正
    print("\n进行极线矫正...")
    left_rect, right_rect, P1, P2, Q = stereo_rectify(left_img, right_img)
    print(f"重投影矩阵 Q:\n{Q}")

    # 4. 检测激光线
    print("\n处理左图...")
    left_lines = detect_laser_lines(left_rect, LEFT_CONFIG)
    print(f"左图提取到 {len(left_lines)} 条激光线")

    print("\n处理右图...")
    right_lines = detect_laser_lines(right_rect, RIGHT_CONFIG)
    print(f"右图提取到 {len(right_lines)} 条激光线")

    # 保存激光线可视化
    cv2.imwrite("img_line/left_line.bmp", line_results(left_rect, left_lines))
    cv2.imwrite("img_line/right_line.bmp", line_results(right_rect, right_lines))

    # 5. 准备矫正后相机参数
    K_left_rect = P1[:3, :3]
    K_right_rect = P2[:3, :3]
    fx = K_left_rect[0, 0]
    Tx = -P2[0, 3] / P2[0, 0]  # 基线 (mm)
    print(f"\n焦距: fx={fx:.2f}")
    print(f"基线: Tx={Tx:.2f} mm")
    print(f"左主点: ({K_left_rect[0,2]:.2f}, {K_left_rect[1,2]:.2f})")
    print(f"右主点: ({K_right_rect[0,2]:.2f}, {K_right_rect[1,2]:.2f})")

    # 6. 自动检测线-平面对应关系并匹配
    print("\n自动检测线-平面对应关系并进行匹配...")
    pts_left, pts_right, points_3d, mean_err, max_err, rmse, assignments = match_all_lines_auto(
        left_lines, right_lines, K_left_rect, K_right_rect, Tx,
        calibrated_planes, search_radius=3,
        visualize=True, right_img=right_rect
    )
    print(f"匹配点总数: {len(pts_left)}")

    # 8. 保存匹配点
    np.savetxt('matched_points_left.txt', pts_left, fmt='%.2f')
    np.savetxt('matched_points_right.txt', pts_right, fmt='%.2f')
    print(f"\n匹配点已保存")

    # 9. 生成并保存点云
    print(f"\n生成点云: {points_3d.shape[0]} 个点")
    colors = np.zeros((points_3d.shape[0], 3), dtype=np.uint8)
    for i, (u, v) in enumerate(pts_right):
        x, y = int(round(u)), int(round(v))
        if 0 <= x < right_rect.shape[1] and 0 <= y < right_rect.shape[0]:
            c = right_rect[y, x]
            if isinstance(c, np.ndarray) and len(c) >= 3:
                colors[i] = c[:3][::-1]  # BGR -> RGB
            else:
                colors[i] = [c, c, c]

    write_ply('output.ply', points_3d, colors)
    print("点云已保存为 output.ply")

    # 10. 可视化匹配结果
    print("\n显示匹配结果...")
    vis = visualize_matched_points(left_rect, right_rect, pts_left, pts_right)
    cv2.imshow('Matched Points (Per-Line Light Plane)', cv2.resize(vis, (1400, 960)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

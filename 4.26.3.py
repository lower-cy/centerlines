import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from collections import defaultdict

# 增强配置参数
ENHANCED_CONFIG = {
    # 预处理参数
    'clahe_clip': 3.0,
    'blur_kernel': (7, 7),
    'closure_kernel_L': (15,15),
    'closure_kernel_R': (25,25),
    'mask_ratio_L': 0.65,
    'mask_ratio_R': 0.4,
    
    # 中心线检测
    'dynamic_thresh_ratio': 0.25,
    'min_line_width': 3,
    'gap_tolerance': 5,
    
    # 聚类参数
    'cluster_eps': 20.0,
    'min_samples': 8,
    
    # 匹配参数
    'smooth_degree': 3,
    'max_disparity': 60,
    'min_disparity': 15,
    'y_tolerance': 5,
    'ncc_threshold': 0.8,
    'match_window': 21
}

def enhance_contrast(img):
    """自适应对比度增强"""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=ENHANCED_CONFIG['clahe_clip'], tileGridSize=(8,8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(cv2.createCLAHE().apply(img), cv2.COLOR_GRAY2BGR)

def morphology_closure(img, kernel_size):
    """形态学闭运算"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def create_adaptive_mask(img, thresh_ratio):
    """动态阈值掩膜"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, int(np.max(gray)*thresh_ratio), 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(img, img, mask=mask)

def robust_centroid_detection(row):
    """鲁棒质心检测"""
    thresh = np.max(row) * ENHANCED_CONFIG['dynamic_thresh_ratio']
    binary = cv2.threshold(row, thresh, 255, cv2.THRESH_BINARY)[1]
    
    # 横向连接
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ENHANCED_CONFIG['gap_tolerance'],1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 分段处理
    segments = []
    start = -1
    for i, val in enumerate(closed):
        if val and start == -1:
            start = i
        elif not val and start != -1:
            if (i - start) >= ENHANCED_CONFIG['min_line_width']:
                segments.append((start, i))
            start = -1
    if start != -1 and (len(closed)-start) >= ENHANCED_CONFIG['min_line_width']:
        segments.append((start, len(closed)))
    
    # 计算质心
    centers = []
    for s, e in segments:
        x = np.arange(s, e)
        weights = row[s:e]
        centroid = np.sum(x * weights) / (np.sum(weights) + 1e-6)
        centers.append(int(centroid))
    return centers

def cluster_lines(points):
    """线特征聚类"""
    if len(points) < 10:
        return []
    
    db = DBSCAN(eps=ENHANCED_CONFIG['cluster_eps'], 
               min_samples=ENHANCED_CONFIG['min_samples']).fit(points)
    
    lines = []
    for label in set(db.labels_):
        if label == -1: continue
        cluster = np.array(points)[db.labels_ == label]
        sorted_cluster = cluster[cluster[:,1].argsort()]
        
        # 样条插值
        if len(sorted_cluster) > 3:
            try:
                tck, _ = splprep(sorted_cluster.T, s=ENHANCED_CONFIG['smooth_degree'])
                u = np.linspace(0, 1, len(sorted_cluster))
                interp_points = np.column_stack(splev(u, tck))
                lines.append(interp_points.astype(int))
            except:
                lines.append(sorted_cluster)
    return lines

def stereo_match(left_lines, right_lines, left_gray, right_gray):
    """改进的立体匹配"""
    h, w = left_gray.shape
    half_win = ENHANCED_CONFIG['match_window'] // 2
    matches = []
    
    # 构建右图索引
    right_dict = defaultdict(list)
    for idx, line in enumerate(right_lines):
        for x, y in line:
            right_dict[int(y)].append( (x, idx) )
    
    used_right = set()
    
    # 遍历左图线条
    for l_idx, l_line in enumerate(left_lines):
        best_score = -1
        best_r_idx = None
        
        # 提取有效点
        valid_points = []
        for x, y in l_line:
            if (half_win <= x < w-half_win) and (half_win <= y < h-half_win):
                valid_points.append( (x, y) )
        if not valid_points: continue
        
        # 动态视差范围
        lx_avg = np.mean([x for x, y in valid_points])
        min_d = max(ENHANCED_CONFIG['min_disparity'], int(lx_avg) - ENHANCED_CONFIG['max_disparity'])
        max_d = int(lx_avg) - ENHANCED_CONFIG['min_disparity']
        
        # 收集候选匹配
        candidates = defaultdict(list)
        for lx, ly in valid_points:
            for y_off in range(-ENHANCED_CONFIG['y_tolerance'], ENHANCED_CONFIG['y_tolerance']+1):
                ry = ly + y_off
                for rx, r_idx in right_dict.get(ry, []):
                    if min_d <= (lx - rx) <= max_d:
                        candidates[r_idx].append( (lx, ly, rx, ry) )
        
        # 评估候选
        for r_idx, pairs in candidates.items():
            if r_idx in used_right: continue
            
            scores = []
            for lx, ly, rx, ry in pairs:
                patch_L = left_gray[ly-half_win:ly+half_win+1, lx-half_win:lx+half_win+1]
                patch_R = right_gray[ry-half_win:ry+half_win+1, rx-half_win:rx+half_win+1]
                
                if patch_L.shape == patch_R.shape:
                    ncc = np.corrcoef(patch_L.flatten(), patch_R.flatten())[0,1]
                    scores.append(ncc)
            
            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score and avg_score > ENHANCED_CONFIG['ncc_threshold']:
                    best_score = avg_score
                    best_r_idx = r_idx
        
        if best_r_idx is not None:
            matches.append( (l_line, right_lines[best_r_idx]) )
            used_right.add(best_r_idx)
    
    return matches

def visualize_matches(left_img, right_img, matches, line_thickness=2, connection_step=5):
    """增强的可视化：带匹配连线"""
    # 创建合成图像
    h, w = left_img.shape[:2]
    composite = np.hstack([left_img, right_img])
    composite = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    
    # 颜色生成
    colors = plt.cm.get_cmap('tab10').colors
    
    for idx, (l_line, r_line) in enumerate(matches):
        color = np.array(colors[idx % len(colors)]) * 255
        
        # 绘制左线条
        l_points = l_line.astype(int)
        for i in range(1, len(l_points)):
            cv2.line(composite, tuple(l_points[i-1]), tuple(l_points[i]), 
                    color.tolist(), line_thickness)
        
        # 绘制右线条（偏移到右侧）
        r_points = r_line.astype(int) + [w, 0]
        for i in range(1, len(r_points)):
            cv2.line(composite, tuple(r_points[i-1]), tuple(r_points[i]),
                    color.tolist(), line_thickness)
        
        # 绘制匹配连线
        for i in range(0, len(l_points), connection_step):
            if i < len(r_points):
                cv2.line(composite, tuple(l_points[i]), tuple(r_points[i]),
                        color.tolist(), 1)

    plt.figure(figsize=(20,10))
    plt.imshow(composite)
    plt.axis('off')
    plt.title("Stereo Matching Results")
    plt.show()

if __name__ == "__main__":
    # 载入图像
    left_img = cv2.imread('L.bmp')
    right_img = cv2.imread('R.jpg')
    assert left_img is not None and right_img is not None, "图像加载失败"
    
    # 预处理流程
    def full_preprocess(img, is_right=False):
        img = enhance_contrast(img)
        kernel = ENHANCED_CONFIG['closure_kernel_R'] if is_right else ENHANCED_CONFIG['closure_kernel_L']
        img = morphology_closure(img, kernel)
        mask_ratio = ENHANCED_CONFIG['mask_ratio_R'] if is_right else ENHANCED_CONFIG['mask_ratio_L']
        return create_adaptive_mask(img, mask_ratio)
    
    # 处理左图
    proc_L = full_preprocess(left_img)
    gray_L = cv2.cvtColor(proc_L, cv2.COLOR_BGR2GRAY)
    left_points = []
    for y in range(gray_L.shape[0]):
        centers = robust_centroid_detection(gray_L[y])
        left_points.extend([[x, y] for x in centers])
    left_lines = cluster_lines(np.array(left_points))
    
    # 处理右图
    proc_R = full_preprocess(right_img, is_right=True)
    gray_R = cv2.cvtColor(proc_R, cv2.COLOR_BGR2GRAY)
    right_points = []
    for y in range(gray_R.shape[0]):
        centers = robust_centroid_detection(gray_R[y])
        right_points.extend([[x, y] for x in centers])
    right_lines = cluster_lines(np.array(right_points))
    
    # 执行匹配
    matched_pairs = stereo_match(left_lines, right_lines, gray_L, gray_R)
    
    # 可视化匹配结果
    visualize_matches(left_img, right_img, matched_pairs)
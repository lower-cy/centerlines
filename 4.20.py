import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev

UPGRADED_SYSTEM = {
    # 通用预处理
    'clahe': {
        'clip_limit': 3.0,
        'grid_size': (8,8)
    },
    'blur': {
        'left': (5,5),
        'right': (7,7)  # 右图更强模糊去噪
    },
    
    # 左右差异化处理
    'morphology': {
        'left': {
            'close': (9,9),
            'open': (3,3)
        },
        'right': {
            'close': (15,15),
            'open': (5,5)
        }
    },
    
    # 质心检测优化
    'centroid': {
        'dynamic_thresh': 0.4,
        'min_width': 3,
        'gap_tolerance': 5,
        'subpixel_refine': True  # 亚像素优化开关
    },
    
    # 聚类增强
    'cluster': {
        'eps': 20.0,
        'min_samples': 5,
        'y_weight': 0.3  # y坐标权重系数
    },
    
    # 样条处理
    'spline': {
        'smoothness': {
            'left': 2.0,
            'right': 3.0  # 右图更需要平滑
        },
        'density': 1.2  # 插值点密度因子
    }
}

def process_image(img, is_right=False):
    """智能预处理（自动区分左右）"""
    # CLAHE自适应增强
    clahe = cv2.createCLAHE(
        clipLimit=UPGRADED_SYSTEM['clahe']['clip_limit'],
        tileGridSize=UPGRADED_SYSTEM['clahe']['grid_size']
    )
    
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        gray = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = clahe.apply(img)
    
    # 差异化模糊
    blur_size = UPGRADED_SYSTEM['blur']['right' if is_right else 'left']
    blurred = cv2.GaussianBlur(gray, blur_size, 0)
    
    # 形态学处理
    morph_cfg = UPGRADED_SYSTEM['morphology']['right' if is_right else 'left']
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_cfg['close'])
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_cfg['open'])
    
    processed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel_close)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_open)
    
    return processed

def find_centroids(row_vector):
    """改进的质心定位算法"""
    max_val = np.max(row_vector)
    if max_val < 10:
        return []
    
    # 动态阈值计算
    thresh = max_val * UPGRADED_SYSTEM['centroid']['dynamic_thresh']
    binary = np.where(row_vector > thresh, 255, 0).astype(np.uint8)
    
    # 连通域分析
    segments = []
    start = -1
    for i, val in enumerate(binary):
        if val == 255 and start == -1:
            start = i
        elif val == 0 and start != -1:
            width = i - start
            if width >= UPGRADED_SYSTEM['centroid']['min_width']:
                segments.append((start, i-1))
            start = -1
    if start != -1:
        segments.append((start, len(binary)-1))
    
    # 亚像素级质心计算
    centers = []
    for s, e in segments:
        region = row_vector[s:e+1]
        if UPGRADED_SYSTEM['centroid']['subpixel_refine'] and (e-s) > 2:
            # 二次多项式拟合
            x = np.arange(s, e+1)
            coeff = np.polyfit(x, region, 2)
            derivative = np.polyder(coeff)
            root = np.roots(derivative)
            if s <= root[0] <= e:
                centers.append(int(round(root[0])))
        else:
            centers.append((s + e) // 2)
    
    return centers

def cluster_points(points_array, img_height):
    """优化聚类算法"""
    if len(points_array) < UPGRADED_SYSTEM['cluster']['min_samples']:
        return []
    
    # 改进的距离度量
    def weighted_metric(a, b):
        dx = a[0] - b[0]
        dy = (a[1] - b[1]) * UPGRADED_SYSTEM['cluster']['y_weight']
        return np.sqrt(dx*dx + dy*dy)
    
    db = DBSCAN(
        eps=UPGRADED_SYSTEM['cluster']['eps'],
        min_samples=UPGRADED_SYSTEM['cluster']['min_samples'],
        metric=weighted_metric
    ).fit(points_array)
    
    clustered_lines = []
    for label in set(db.labels_):
        if label == -1:
            continue
        
        cluster = points_array[db.labels_ == label]
        # 按y坐标排序
        sorted_cluster = cluster[cluster[:,1].argsort()]
        clustered_lines.append(sorted_cluster)
    
    return clustered_lines

def refine_curves(raw_lines, is_right=False):
    """曲线优化"""
    smooth_deg = UPGRADED_SYSTEM['spline']['smoothness']['right' if is_right else 'left']
    interp_num = int(UPGRADED_SYSTEM['spline']['density'] * 100)
    
    refined = []
    for line in raw_lines:
        if len(line) < 5:
            refined.append(line)
            continue
        
        try:
            tck, _ = splprep(line.T, s=smooth_deg)
            u_new = np.linspace(0, 1, interp_num)
            new_curve = np.column_stack(splev(u_new, tck))
            refined.append(new_curve)
        except:
            refined.append(line)
    
    return refined

def extract_laser_lines(img, is_right=False):
    # 1. 智能预处理
    processed = process_image(img, is_right)
    
    # 2. 逐行质心检测
    height, width = processed.shape
    all_points = []
    for y in range(height):
        centroids = find_centroids(processed[y,:])
        all_points.extend([[x, y] for x in centroids])
    
    if not all_points:
        return []
    
    # 3. 聚类处理
    points_array = np.array(all_points)
    clustered = cluster_points(points_array, height)
    
    # 4. 曲线优化
    final_lines = refine_curves(clustered, is_right)
    
    return final_lines

def match_lines(self, left_lines, right_lines):
        """线匹配"""
        matches = []
        used_right = set()
        
        for i, left_line in enumerate(left_lines):
            left_depth = np.mean(left_line[:,1])
            
            best_match = (-1, float('inf'))
            for j, right_line in enumerate(right_lines):
                if j in used_right:
                    continue
                    
                right_depth = np.mean(right_line[:,1])
                depth_diff = abs(left_depth - right_depth)
                
                if depth_diff < self.depth_constraints['line_gap']/2:
                    if depth_diff < best_match[1]:
                        best_match = (j, depth_diff)
            
            if best_match[0] != -1:
                matches.append((i, best_match[0]))
                used_right.add(best_match[0])
        
        return matches

if __name__ == "__main__":
    # 保持原有调用方式不变
    left_img = cv2.imread('L.bmp')
    right_img = cv2.imread('R.jpg')
    
    left_lines = extract_laser_lines(left_img)
    right_lines = extract_laser_lines(right_img, is_right=True)

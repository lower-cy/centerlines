import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev, interp1d
from mpl_toolkits.mplot3d import Axes3D

# ================== 配置参数 ==================
ENHANCED_CONFIG = {
    # 预处理参数
    'clahe_clip': 3.0,
    'blur_kernel': (7, 7),
    # 质心检测参数
    'dynamic_thresh_ratio': 0.25,
    'min_line_width': 3,
    'gap_tolerance': 5,
    # 聚类参数
    'cluster_eps': 20.0,
    'min_samples': 8,
    # 后处理参数
    'smooth_degree': 3,
    'interp_step': 0.5,
    'closure_kernel_L': (15,15),
    'closure_kernel_R': (25,25),
    'mask_ratio_L': 0.65,
    'mask_ratio_R': 0.4,
    # 立体匹配参数
    'focal_length': 1200,     # 相机焦距（像素单位）
    'baseline': 0.12,         # 基线距离（米）
    'min_disparity': 5,       # 最小有效视差
}

# ================== 中心线提取模块 ==================
def enhance_contrast(img):
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=ENHANCED_CONFIG['clahe_clip'], tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
    else:
        return cv2.cvtColor(cv2.createCLAHE().apply(img), cv2.COLOR_GRAY2BGR)

def adaptive_blur(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(img, ENHANCED_CONFIG['blur_kernel'], 0)

def morphology_closure(img, kernel_size):
    if len(img.shape) == 3:
        channels = []
        for i in range(3):
            channel = cv2.morphologyEx(img[:,:,i], cv2.MORPH_CLOSE, 
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size))
            channels.append(channel)
        return cv2.merge(channels)
    else:
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size))

def create_adaptive_mask(img, thresh_ratio, is_right=False):
    working_img = img.copy()
    if len(working_img.shape) == 3:
        gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
        if is_right:
            working_img[:,:,2] = cv2.threshold(working_img[:,:,2], 
                                             int(255*thresh_ratio), 
                                             255, cv2.THRESH_TOZERO)[1]
    else:
        gray = working_img
    
    _, mask = cv2.threshold(gray, int(np.max(gray)*thresh_ratio), 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(working_img, working_img, mask=mask)

def robust_centroid_detection(row):
    thresh = np.max(row) * ENHANCED_CONFIG['dynamic_thresh_ratio']
    binary = cv2.threshold(row, thresh, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ENHANCED_CONFIG['gap_tolerance'],1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    segments = []
    start = -1
    for i, val in enumerate(closed):
        if val and start == -1:
            start = i
        elif not val and start != -1:
            segments.append((start, i))
            start = -1
    if start != -1:
        segments.append((start, len(closed)))
    
    centers = []
    for s, e in segments:
        if e - s >= ENHANCED_CONFIG['min_line_width']:
            x = np.arange(s, e)
            weights = row[s:e]
            centroid = np.sum(x * weights) / np.sum(weights)
            centers.append(int(centroid))
    return centers

def interpolate_points(points):
    if len(points) < 4: 
        return points
    try:
        tck, _ = splprep(points.T, s=ENHANCED_CONFIG['smooth_degree'])
        u = np.arange(0, 1, ENHANCED_CONFIG['interp_step']/len(points))
        return np.column_stack(splev(u, tck))
    except:
        return points

def detect_connected_lines(img):
    enhanced = enhance_contrast(img)
    blurred = adaptive_blur(enhanced)
    
    points = []
    for y in range(blurred.shape[0]):
        row = blurred[y, :]
        centers = robust_centroid_detection(row)
        points.extend([[x, y] for x in centers])
    
    if len(points) < 10:
        return []
    
    db = DBSCAN(eps=ENHANCED_CONFIG['cluster_eps'], 
               min_samples=ENHANCED_CONFIG['min_samples']).fit(points)
    
    lines = []
    for label in set(db.labels_):
        if label == -1:
            continue
        cluster = np.array(points)[db.labels_ == label]
        sorted_cluster = cluster[cluster[:,1].argsort()]
        interpolated = interpolate_points(sorted_cluster)
        lines.append(interpolated)
    return lines

# ================== 立体匹配模块 ==================
def match_lines(left_lines, right_lines, img_shape):
    """基于几何约束的线匹配"""
    h, w = img_shape[:2]
    matched_pairs = []
    
    for l_idx, l_line in enumerate(left_lines):
        # 计算左线特征
        l_centroid = np.mean(l_line, axis=0)
        l_length = l_line[-1,1] - l_line[0,1]
        
        best_match = None
        best_score = -np.inf
        
        for r_idx, r_line in enumerate(right_lines):
            # 计算右线特征
            r_centroid = np.mean(r_line, axis=0)
            r_length = r_line[-1,1] - r_line[0,1]
            
            # 计算匹配得分
            y_diff = abs(l_centroid[1] - r_centroid[1])
            length_diff = abs(l_length - r_length)
            x_offset = (w - r_centroid[0]) - l_centroid[0]  # 期望右线在左侧
            
            score = 1/(y_diff+1) + 1/(length_diff+1) + x_offset/w
            if score > best_score:
                best_score = score
                best_match = r_idx
        
        if best_score > 0.5:  # 匹配阈值
            matched_pairs.append((l_idx, best_match))
    
    return matched_pairs

def compute_disparity(left_line, right_line):
    """基于样条插值的视差计算"""
    # 按y坐标排序
    left_line = left_line[left_line[:,1].argsort()]
    right_line = right_line[right_line[:,1].argsort()]
    
    # 创建插值函数
    try:
        f_left = interp1d(left_line[:,1], left_line[:,0], kind='linear', 
                         bounds_error=False, fill_value=np.nan)
        f_right = interp1d(right_line[:,1], right_line[:,0], kind='linear',
                          bounds_error=False, fill_value=np.nan)
    except:
        return None
    
    # 确定公共y范围
    y_min = max(left_line[0,1], right_line[0,1])
    y_max = min(left_line[-1,1], right_line[-1,1])
    if y_min >= y_max:
        return None
    
    # 生成插值点
    ys = np.arange(y_min, y_max+1)
    xl = f_left(ys)
    xr = f_right(ys)
    
    # 筛选有效点
    valid = ~np.isnan(xl) & ~np.isnan(xr)
    xl = xl[valid]
    xr = xr[valid]
    ys = ys[valid]
    
    if len(xl) < 10:
        return None
    
    # 计算视差并过滤
    disparities = xl - xr
    valid_disp = disparities > ENHANCED_CONFIG['min_disparity']
    return np.column_stack([xl[valid_disp], ys[valid_disp], disparities[valid_disp]])

def calculate_depth(disparity_data):
    """根据视差计算深度"""
    f = ENHANCED_CONFIG['focal_length']
    B = ENHANCED_CONFIG['baseline']
    disparities = disparity_data[:,2]
    Z = (f * B) / disparities
    
    # 转换为三维坐标（相机坐标系）
    cx = disparity_data[:,0] - disparity_data[:,0].mean()  # 简化处理
    cy = disparity_data[:,1] - disparity_data[:,1].mean()
    X = (disparity_data[:,0] - cx) * Z / f
    Y = (disparity_data[:,1] - cy) * Z / f
    
    return np.column_stack([X, Y, Z])

# ================== 主程序 ==================
if __name__ == "__main__":
    # 读取图像
    left_img = cv2.imread('L.bmp')
    right_img = cv2.imread('R.jpg')
    assert left_img is not None and right_img is not None
    
    # 预处理
    processed_L = enhance_contrast(left_img)
    processed_L = morphology_closure(processed_L, ENHANCED_CONFIG['closure_kernel_L'])
    processed_L = create_adaptive_mask(processed_L, ENHANCED_CONFIG['mask_ratio_L'])
    
    processed_R = enhance_contrast(right_img)
    processed_R = morphology_closure(processed_R, ENHANCED_CONFIG['closure_kernel_R'])
    processed_R = create_adaptive_mask(processed_R, ENHANCED_CONFIG['mask_ratio_R'], is_right=True)

    # 中心线检测
    print("Processing left image...")
    left_lines = detect_connected_lines(cv2.cvtColor(processed_L, cv2.COLOR_BGR2GRAY))
    print(f"Detected {len(left_lines)} lines in left image")
    
    print("Processing right image...")
    right_lines = detect_connected_lines(cv2.cvtColor(processed_R, cv2.COLOR_BGR2GRAY))
    print(f"Detected {len(right_lines)} lines in right image")

    # 立体匹配
    matched_pairs = match_lines(left_lines, right_lines, left_img.shape)
    print(f"Matched {len(matched_pairs)} line pairs")
    
    all_depth = []
    for l_idx, r_idx in matched_pairs:
        disparity_data = compute_disparity(left_lines[l_idx], right_lines[r_idx])
        if disparity_data is not None:
            depth_data = calculate_depth(disparity_data)
            all_depth.append(depth_data)
    
    if len(all_depth) == 0:
        print("No valid depth data calculated!")
    else:
        all_depth = np.vstack(all_depth)
        np.savetxt('depth_data.txt', all_depth, fmt='%.3f', 
                  header='X(m) Y(m) Z(m)')
        print(f"Saved {len(all_depth)} depth points to depth_data.txt")

        # 可视化
        fig = plt.figure(figsize=(12,6))
        
        # 原始图像
        ax1 = fig.add_subplot(121)
        ax1.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Left Image with Detected Lines')
        for line in left_lines:
            ax1.plot(line[:,0], line[:,1], 'r-', lw=1)
        
        # 3D可视化
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(all_depth[:,0], all_depth[:,1], all_depth[:,2], 
                   c=all_depth[:,2], cmap='viridis', s=5)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Depth (m)')
        ax2.view_init(elev=30, azim=45)
        plt.title('3D Reconstruction')
        
        plt.tight_layout()
        plt.show()
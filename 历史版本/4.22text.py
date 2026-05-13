import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

# ====================== 左图参数配置 ======================
LEFT_CONFIG = {
    # 基础参数
    'laser_color': 'gray',       
    'min_laser_intensity': 20,   
    
    # 预处理参数
    'clahe_clip': 5.0,           
    'blur_kernel': (7,7),        
    'gamma_correct': 0.7,        
    'specular_thresh': 180,      
    
    # 暗区增强参数
    'dark_enhance': True,        
    'dark_threshold': 50,        
    'enhance_alpha': 2.0,        
    'enhance_beta': 40,          
    
    # 质心检测
    'dynamic_thresh_ratio': 0.25,
    'min_line_width': 4,         
    'max_line_gap': 15,          
    
    # 几何约束
    'roi_padding': 10,           
    'cluster_eps': 35.0,         
    'min_samples': 5,            
    'min_line_length': 80,       # 新增关键参数
    
    # 后处理
    'smooth_sigma': 3.0,         
    'max_end_curvature': 0.15    
}

# ====================== 右图参数配置 ====================== 
RIGHT_CONFIG = {
    # 基础参数
    'laser_color': 'red',        
    'min_laser_intensity': 35,   
    
    # 预处理参数
    'clahe_clip': 4.5,           
    'blur_kernel': (3,3),        
    'gamma_correct': 0.3,        
    'specular_thresh': 210,      
    
    # 质心检测
    'dynamic_thresh_ratio': 0.45,
    'min_line_width': 3,         
    'max_line_gap': 5,           
    
    # 几何约束
    'roi_padding': 30,           
    'cluster_eps': 22.0,         
    'min_samples': 15,           
    'min_line_length': 100,      # 参数存在性修复
    
    # 后处理
    'smooth_sigma': 1.8,         
    'max_end_curvature': 0.2     
}

# ====================== 以下是保持不变的核心算法部分 ======================

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
        enhanced = cv2.addWeighted(r, 2.5, cv2.add(b, g), -1.2, 0)
    elif config['laser_color'] == 'green':
        enhanced = cv2.addWeighted(g, 2.5, cv2.add(r, b), -1.2, 0)
    else: 
        enhanced = cv2.addWeighted(b, 2.5, cv2.add(r, g), -1.2, 0)
    
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([enhanced, enhanced, enhanced])

def adaptive_gamma_correction(img, config):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, config['specular_thresh'], 255, cv2.THRESH_BINARY)
    
    inv_gamma = 1.0 / config['gamma_correct']
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(img, table)
    
    return cv2.bitwise_and(corrected, corrected, mask=mask) + \
           cv2.bitwise_and(img, img, mask=~mask)

def enhance_dark_region(img, config):
    if not config.get('dark_enhance', False):
        return img
    
    h, w = img.shape[:2]
    roi = img[:, int(w*0.4):]  
    
    _, dark_mask = cv2.threshold(roi, config['dark_threshold'], 255, cv2.THRESH_BINARY_INV)
    enhanced = cv2.convertScaleAbs(roi, 
                                 alpha=config['enhance_alpha'],
                                 beta=config['enhance_beta'])
    
    enhanced_roi = cv2.bitwise_and(enhanced, enhanced, mask=dark_mask) + \
                  cv2.bitwise_and(roi, roi, mask=~dark_mask)
    
    result = img.copy()
    result[:, int(w*0.4):] = enhanced_roi
    return result

def multi_scale_preprocess(img, config):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    corrected = adaptive_gamma_correction(img, config)
    
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    if config.get('dark_enhance', False):
        l = enhance_dark_region(l, config)
    
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(16,16))
    l = clahe.apply(l)
    
    blur1 = cv2.GaussianBlur(l, config['blur_kernel'], 0)
    blur2 = cv2.medianBlur(l, 7)
    merged = cv2.addWeighted(blur1, 0.7, blur2, 0.3, 0)
    
    enhanced = enhance_laser_channel(merged, config)
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

def dynamic_centroid_detection(row, config):
    max_val = np.max(row)
    if max_val < config['min_laser_intensity']:
        return []
    
    thresh = max_val * config['dynamic_thresh_ratio']
    binary = np.where(row > thresh, 255, 0).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config['max_line_gap'], 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    segments = []
    start = -1
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
        if 10 < centroid < len(row)-10:  
            centers.append(int(round(centroid)))
    return centers

def filter_endpoints_curvature(line, config):
    if len(line) < 10:
        return line
    
    epsilon = 1e-6
    head = line[:10]
    dx = np.gradient(head[:,0])
    dy = np.gradient(head[:,1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(d2x*dy - dx*d2y) / ((dx**2 + dy**2)**1.5 + epsilon)
    if np.mean(curvature) > config['max_end_curvature']:
        line = line[5:]
    
    tail = line[-10:]
    dx = np.gradient(tail[:,0])
    dy = np.gradient(tail[:,1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(d2x*dy - dx*d2y) / ((dx**2 + dy**2)**1.5 + epsilon)
    if np.mean(curvature) > config['max_end_curvature']:
        line = line[:-5]
    
    return line

def geometry_based_clustering(points, img_size, config):
    h, w = img_size
    mask = (points[:,0] > config['roi_padding']) & \
           (points[:,0] < w - config['roi_padding']) & \
           (points[:,1] > config['roi_padding']) & \
           (points[:,1] < h - config['roi_padding'])
    points = points[mask]
    
    if config.get('dark_enhance', False):
        right_points = points[points[:,0] > w*0.6]
        points = np.vstack([points, right_points])
    
    if len(points) < config['min_samples']:
        return []
    
    db = DBSCAN(eps=config['cluster_eps'], min_samples=config['min_samples']).fit(points)
    
    valid_lines = []
    for label in set(db.labels_):
        if label == -1:
            continue
            
        cluster = points[db.labels_ == label]
        if len(cluster) < config['min_line_length']:  # 修复后的参数调用
            continue
        
        sorted_cluster = cluster[cluster[:,1].argsort()]
        try:
            tck, u = splprep(sorted_cluster.T, s=1.0)
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
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    preprocessed = multi_scale_preprocess(img, config)
    
    points = []
    for y in range(preprocessed.shape[0]):
        centers = dynamic_centroid_detection(preprocessed[y,:], config)
        points.extend([[x, y] for x in centers])
    
    if not points:
        return []
    
    lines = geometry_based_clustering(np.array(points), preprocessed.shape, config)
    return lines

def visualize_results(img, lines, title):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    if len(img.shape) == 2:
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        display_img = img.copy()
    ax[0].imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f'{title} Original')
    
    preprocessed = multi_scale_preprocess(img, LEFT_CONFIG if 'L' in title else RIGHT_CONFIG)
    ax[1].imshow(preprocessed, cmap='gray')
    ax[1].set_title('Preprocessed')
    
    vis = display_img.copy()
    colors = [(0,255,0), (255,0,0), (0,0,255)]
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        pts = line.astype(int)
        cv2.polylines(vis, [pts], False, color, 2)
    ax[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax[2].set_title(f'Detected {len(lines)} Lines')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    left_img = cv2.imread('L.bmp', cv2.IMREAD_ANYCOLOR)
    right_img = cv2.imread('R.jpg', cv2.IMREAD_COLOR)
    
    if len(left_img.shape) == 2:
        left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    
    print("处理左图...")
    left_lines = detect_laser_lines(left_img, LEFT_CONFIG)
    print(f"左图提取到 {len(left_lines)} 条中心线")
    
    print("\n处理右图...")
    right_lines = detect_laser_lines(right_img, RIGHT_CONFIG)
    print(f"右图提取到 {len(right_lines)} 条中心线")
    
    visualize_results(left_img, left_lines, 'Left Image')
    visualize_results(right_img, right_lines, 'Right Image')
    
    def save_lines(lines, filename):
        with open(filename, 'w') as f:
            for i, line in enumerate(lines):
                np.savetxt(f, line, fmt='%.2f',
                          header=f'Line {i+1}', comments='# ',
                          delimiter=',')
    save_lines(left_lines, 'left_lines.csv')
    save_lines(right_lines, 'right_lines.csv')
    print("结果已保存至 left_lines.csv 和 right_lines.csv")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev

ENHANCED_CONFIG = {
    # 预处理参数
    'clahe_clip': 3.0,# 对比度增强参数
    'blur_kernel': (7, 7), # 自适应模糊核
    # 质心检测参数
    'dynamic_thresh_ratio': 0.25,# 动态阈值系数
    'min_line_width': 3,# 最小有效条纹宽度
    'gap_tolerance': 5,# 横向间断容忍度
    #聚类参数
    'cluster_eps': 20.0,# 增大聚类半径
    'min_samples': 8,# 减少最小样本数
    # 后处理参数
    'smooth_degree': 3,# 样条平滑度
    'interp_step': 0.5, # 插值步长
    
    'closure_kernel_L': (15,15),
    'closure_kernel_R': (25,25),
    'mask_ratio_L': 0.65,
    'mask_ratio_R': 0.4
}

def enhance_contrast(img):
    """对比度增强处理（兼容灰度图）"""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=ENHANCED_CONFIG['clahe_clip'], tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
    else:
        return cv2.cvtColor(cv2.createCLAHE().apply(img), cv2.COLOR_GRAY2BGR)

def adaptive_blur(img):
    """自适应高斯模糊（自动处理彩色/灰度图）"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(img, ENHANCED_CONFIG['blur_kernel'], 0)

def morphology_closure(img, kernel_size):
    """形态学闭运算处理（保持色彩通道）"""
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
    """生成自适应掩膜（保持色彩通道）"""
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
    """抗干扰质心检测"""
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
    """样条插值填补间断"""
    if len(points) < 4: 
        return points
    try:
        tck, _ = splprep(points.T, s=ENHANCED_CONFIG['smooth_degree'])
        u = np.arange(0, 1, ENHANCED_CONFIG['interp_step']/len(points))
        return np.column_stack(splev(u, tck))
    except:
        return points

def detect_connected_lines(img):
    """多线检测主函数"""
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

def save_coordinates(lines, filename):
    """保存坐标数据"""
    with open(filename, 'w') as f:
        for line_idx, line in enumerate(lines):
            sorted_line = line[line[:,1].argsort()]
            valid_points = sorted_line[(sorted_line[:,0] > 0) & (sorted_line[:,1] > 0)]
            f.write(f"# Line {line_idx+1} (Points: {len(valid_points)})\n")
            np.savetxt(f, valid_points, fmt='%d', delimiter=',')
            f.write("\n")

def match_lines(left_lines, right_lines, max_disparity=50, y_tolerance=30):
    """基于视差的左右线匹配"""
    # 对左右线按垂直方向排序
    left_sorted = sorted(left_lines, key=lambda x: np.mean(x[:,1]))
    right_sorted = sorted(right_lines, key=lambda x: np.mean(x[:,1]))
    
    matched_pairs = []
    used_right = set()
    
    for l_idx, l_line in enumerate(left_sorted):
        l_center_y = np.mean(l_line[:,1])
        l_center_x = np.mean(l_line[:,0])
        
        best_r_dist = float('inf')
        best_r_line = None
        
        for r_idx, r_line in enumerate(right_sorted):
            if r_idx in used_right:
                continue
            r_center_y = np.mean(r_line[:,1])
            r_center_x = np.mean(r_line[:,0])
            
            # 垂直方向容忍度和水平视差检查
            if abs(l_center_y - r_center_y) > y_tolerance:
                continue
            disparity = abs(l_center_x - r_center_x)
            if disparity == 0:
                continue
            
            # 优先选择视差较小的匹配
            if disparity < best_r_dist and disparity <= max_disparity:
                best_r_dist = disparity
                best_r_line = r_line
                best_r_idx = r_idx
                
        if best_r_line is not None:
            matched_pairs.append( (l_line, best_r_line) )
            used_right.add(best_r_idx)
    
    return matched_pairs

def visualize_matches(left_img, right_img, matched_pairs, thickness=2):
    """可视化匹配结果"""
    vis_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    vis_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
    
    for idx, (l_line, r_line) in enumerate(matched_pairs[:4]):
        color = colors[idx % len(colors)]
        # 左图绘制
        l_points = l_line.astype(int)
        for i in range(1, len(l_points)):
            cv2.line(vis_left, tuple(l_points[i-1]), tuple(l_points[i]), color, thickness)
        # 右图绘制
        r_points = r_line.astype(int)
        for i in range(1, len(r_points)):
            cv2.line(vis_right, tuple(r_points[i-1]), tuple(r_points[i]), color, thickness)
    
    # 创建对比图
    combined = np.hstack((vis_left, vis_right))
    return combined
def visualize_lines(img, lines, thickness=2):
    """可视化检测结果"""
    vis = img.copy() if len(img.shape)==3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = [(0,255,0), (0,0,255), (255,0,255), (255,255,0)]
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        int_line = line.astype(int)
        for j in range(1, len(int_line)):
            cv2.line(vis, tuple(int_line[j-1]), tuple(int_line[j]), color, thickness)
    return vis

if __name__ == "__main__":
    try:
        left_img = cv2.imread('L.bmp')
        right_img = cv2.imread('R.jpg')
        assert left_img is not None and right_img is not None
    except:
        raise FileNotFoundError("请确保L.jpg和R.jpg存在于当前目录且可读取")

    # 新增预处理流程
    processed_L = enhance_contrast(left_img)
    processed_L = morphology_closure(processed_L, ENHANCED_CONFIG['closure_kernel_L'])
    processed_L = create_adaptive_mask(processed_L, ENHANCED_CONFIG['mask_ratio_L'])
    
    processed_R = enhance_contrast(right_img)
    processed_R = morphology_closure(processed_R, ENHANCED_CONFIG['closure_kernel_R'])
    processed_R = create_adaptive_mask(processed_R, ENHANCED_CONFIG['mask_ratio_R'], is_right=True)

    print("正在处理左图...")
    left_lines = detect_connected_lines(cv2.cvtColor(processed_L, cv2.COLOR_BGR2GRAY))
    print("正在处理右图...")
    right_lines = detect_connected_lines(cv2.cvtColor(processed_R, cv2.COLOR_BGR2GRAY))
    print("正在执行左右匹配...")
    matched_pairs = match_lines(left_lines, right_lines, max_disparity=40, y_tolerance=20)
    with open('matched.txt', 'w') as f:
        f.write(f"Total matched pairs: {len(matched_pairs)}\n")
        for idx, (l_line, r_line) in enumerate(matched_pairs):
            f.write(f"Pair {idx+1}:\n")
            f.write("Left Line:\n")
            np.savetxt(f, l_line, fmt='%d', delimiter=',')
            f.write("Right Line:\n")
            np.savetxt(f, r_line, fmt='%d', delimiter=',')
            f.write("\n------------------\n")

    save_coordinates(left_lines, 'L.txt')
    save_coordinates(right_lines, 'R.txt')
    print(f"坐标文件已保存：L.txt ({len(left_lines)}条线), R.txt ({len(right_lines)}条线)")

    fig = plt.figure(figsize=(18, 9))
    
    # 可视化调整
    fig = plt.figure(figsize=(18, 12))
    
    # 左图流程
    ax1 = fig.add_subplot(241)
    ax1.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Left Original')
    
    ax2 = fig.add_subplot(242)
    ax2.imshow(cv2.cvtColor(processed_L, cv2.COLOR_BGR2RGB))
    ax2.set_title('Left Preprocessed')
    
    ax3 = fig.add_subplot(243)
    ax3.imshow(visualize_lines(left_img, left_lines))
    ax3.set_title('Left Detected')
    
    # 右图流程
    ax4 = fig.add_subplot(245)
    ax4.imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    ax4.set_title('Right Original')
    
    ax5 = fig.add_subplot(246)
    ax5.imshow(cv2.cvtColor(processed_R, cv2.COLOR_BGR2RGB))
    ax5.set_title('Right Preprocessed')
    
    ax6 = fig.add_subplot(247)
    ax6.imshow(visualize_lines(right_img, right_lines))
    ax6.set_title('Right Detected')
    
    # 匹配结果可视化
    ax7 = fig.add_subplot(244)
    ax7.imshow(visualize_matches(left_img, right_img, matched_pairs))
    ax7.set_title('Matched Lines')
    
    plt.tight_layout()
    plt.show()
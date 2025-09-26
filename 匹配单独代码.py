import numpy as np
import cv2

def extract_centers(image):
    """提取每行的中心点列坐标"""
    centers = []
    height, width = image.shape
    for y in range(height):
        cols = np.where(image[y, :] == 255)[0]
        centers.append(cols)  # 存储每行所有中心点列坐标
    return centers

def compute_ncc(patch1, patch2):
    """计算归一化互相关系数"""
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)
    
    numerator = np.sum((patch1 - mean1) * (patch2 - mean2))
    denominator = np.sqrt(np.sum((patch1 - mean1)**2) * np.sum((patch2 - mean2)**2))
    
    return numerator / (denominator + 1e-6)  # 防止除以零

def stereo_match(left_img, right_img, d_min, d_max, window_size=21):
    """
    带视差范围约束的立体匹配
    :param left_img: 左图中心线二值图像
    :param right_img: 右图中心线二值图像
    :param d_min: 最小视差（右图列 = 左图列 - d_max）
    :param d_max: 最大视差（右图列 = 左图列 - d_min）
    :param window_size: 匹配窗口大小（奇数）
    :return: 匹配点对列表[(x_left,y,x_right,y)]
    """
    # 参数校验
    assert window_size % 2 == 1, "窗口大小必须为奇数"
    
    # 初始化参数
    half_win = window_size // 2
    height, width = left_img.shape
    matches = []
    
    # 提取中心点
    left_centers = extract_centers(left_img)
    right_centers = extract_centers(right_img)
    
    # 遍历每一行
    for y in range(half_win, height - half_win):
        # 获取左图当前行所有中心点
        left_cols = left_centers[y]
        if len(left_cols) == 0:
            continue
            
        # 遍历左图当前行每个中心点
        for x_left in left_cols:
            # 检查边界条件
            if x_left < half_win or x_left >= width - half_win:
                continue
                
            # 计算右图搜索范围
            start_col = x_left - d_max
            end_col = x_left - d_min
            start_col = max(half_win, start_col)
            end_col = min(width - half_win - 1, end_col)
            
            if start_col > end_col:
                continue  # 无效的搜索范围
                
            # 准备左图窗口
            left_patch = left_img[y-half_win:y+half_win+1, 
                               x_left-half_win:x_left+half_win+1]
            
            best_ncc = -1
            best_x_right = None
            
            # 获取右图当前行候选点
            right_cols = [x for x in right_centers[y] 
                        if start_col <= x <= end_col]
            
            # 遍历右图候选点
            for x_right in right_cols:
                # 检查边界条件
                if x_right < half_win or x_right >= width - half_win:
                    continue
                
                # 提取右图窗口
                right_patch = right_img[y-half_win:y+half_win+1,
                                    x_right-half_win:x_right+half_win+1]
                
                # 计算相似度
                ncc = compute_ncc(left_patch, right_patch)
                
                # 更新最佳匹配
                if ncc > best_ncc:
                    best_ncc = ncc
                    best_x_right = x_right
            
            # 记录有效匹配
            if best_x_right is not None and best_ncc > 0.7:  # 可调整阈值
                matches.append((x_left, y, best_x_right, y))
    
    return matches

# 使用示例
if __name__ == "__main__":
    # 参数设置（需要根据实际情况调整）
    DISPARITY_MIN = 10   # 最小视差
    DISPARITY_MAX = 100  # 最大视差
    WINDOW_SIZE = 21     # 匹配窗口大小
    
    # 载入中心线图像
    left_center = cv2.imread("left_center.png", cv2.IMREAD_GRAYSCALE)
    right_center = cv2.imread("right_center.png", cv2.IMREAD_GRAYSCALE)
    
    # 执行匹配
    matched_pairs = stereo_match(left_center, right_center,
                                DISPARITY_MIN, DISPARITY_MAX,
                                WINDOW_SIZE)
    
    # 可视化结果（示例）
    disp_img = cv2.cvtColor(left_center, cv2.COLOR_GRAY2BGR)
    for pair in matched_pairs[:100]:  # 显示前100个匹配点
        x1, y1, x2, y2 = pair
        cv2.circle(disp_img, (x1, y1), 2, (0,255,0), -1)
    
    cv2.imshow("Matches", disp_img)
    cv2.waitKey(0)
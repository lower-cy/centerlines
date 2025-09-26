import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_laser_centerline(img):
    """
    基于法线引导的激光中心线提取算法（单图像处理）
    参数:
        img: 输入图像（灰度或彩色）
    返回:
        center_points: 激光中心线坐标列表[(x,y)...]
    """
    # 如果是彩色图转为灰度
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    
    # 1. 图像预处理 - 大津法阈值分割
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # 2. 修补断线 - 形态学闭运算
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 3. 快速定位像素级中心线 - 几何中心法
    edges = cv2.Canny(closed, 50, 150)
    pixel_centers = geometric_center(edges)
    
    # 4. 主成分分析法求取法线方向
    normals = compute_normals(img_gray, pixel_centers)
    
    # 5. 基于法线引导的中心线提取
    subpixel_centers = normal_guided_centers(img_gray, pixel_centers, normals)
    
    return subpixel_centers

def geometric_center(edge_img):
    """几何中心法提取像素级中心线"""
    centers = []
    rows, cols = edge_img.shape
    
    for y in range(rows):
        row_data = edge_img[y, :]
        indices = np.where(row_data > 0)[0]
        
        if len(indices) >= 2:
            left = indices[0]
            right = indices[-1]
            center_x = (left + right) / 2
            centers.append((center_x, y))
    
    return centers

def compute_normals(img, centers, window_size=15):
    """使用PCA计算中心点的法线方向"""
    normals = []
    img_blur = cv2.GaussianBlur(img, (5,5), 1.5)
    
    for x, y in centers:
        # 将 x 和 y 转换为整数类型
        x = int(round(x))
        y = int(round(y))
        
        half = window_size // 2
        roi = img_blur[max(0, y - half):min(img.shape[0], y + half + 1),
                      max(0, x - half):min(img.shape[1], x + half + 1)]
        
        if roi.size == 0:  # 检查 ROI 是否为空
            normals.append(np.array([0, 0]))
            continue
        
        gy, gx = np.gradient(roi.astype(np.float32))
        cov = np.cov(np.vstack((gx.flatten(), gy.flatten())))
        eigvals, eigvecs = np.linalg.eig(cov)
        normal = eigvecs[:, np.argmax(eigvals)]
        normals.append(normal / np.linalg.norm(normal))
    
    return normals

def normal_guided_centers(img, centers, normals, search_range=5):
    """基于法线引导的亚像素中心提取"""
    subpixel_centers = []
    img_float = img.astype(np.float32)
    angle_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    
    for i, ((x, y), normal) in enumerate(zip(centers, normals)):
        angle = np.degrees(np.arctan2(normal[1], normal[0])) % 360
        bin_idx = np.digitize(angle, angle_bins) - 1
        search_dir = [(1,0), (1,1), (0,1), (-1,1), 
                      (-1,0), (-1,-1), (0,-1), (1,-1)][bin_idx % 8]
        
        points = []
        for t in np.linspace(-search_range, search_range, 15):
            px = int(round(x + t * search_dir[0]))
            py = int(round(y + t * search_dir[1]))
            
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                points.append((px, py, img_float[py, px]))
        
        if len(points) >= 3:
            points = np.array(points)
            weights = points[:, 2] - np.min(points[:, 2])
            if np.sum(weights) > 0:
                x_sub = np.sum(points[:, 0] * weights) / np.sum(weights)
                y_sub = np.sum(points[:, 1] * weights) / np.sum(weights)
                subpixel_centers.append((x_sub, y_sub))
                continue
        
        subpixel_centers.append((x, y))
    
    return subpixel_centers

def visualize_centers(img, centers, title="Result"):
    """可视化中心线提取结果"""
    result_img = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pt in centers:
        cv2.circle(result_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
    
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
    return result_img

# 主处理函数
def process_dual_images(img_gray, img_color):
    """
    同时处理灰度图和彩色图
    参数:
        img_gray: 灰度图像
        img_color: 彩色图像
    返回:
        gray_centers: 灰度图中心线坐标
        color_centers: 彩色图中心线坐标
        result_gray: 灰度图结果可视化
        result_color: 彩色图结果可视化
    """
    # 提取灰度图中心线
    gray_centers = extract_laser_centerline(img_gray)
    result_gray = visualize_centers(img_gray, gray_centers, "Gray Image Result")
    
    # 提取彩色图中心线（转换为灰度处理）
    color_centers = extract_laser_centerline(img_color)
    result_color = visualize_centers(img_color, color_centers, "Color Image Result")
    
    return gray_centers, color_centers, result_gray, result_color

# 示例使用
if __name__ == "__main__":
    # 读取图像（假设左图是灰度图，右图是彩色图）
    img_color = cv2.imread('R.jpg')  # 替换为您的彩色图像路径
    img_gray = cv2.imread('L.jpg', cv2.IMREAD_GRAYSCALE)  # 替换为您的灰度图像路径
    
    # 同时处理两张图像
    gray_centers, color_centers, res_gray, res_color = process_dual_images(img_gray, img_color)
    
    # 打印部分结果
    print("灰度图前5个中心点坐标:")
    for pt in gray_centers[:5]:
        print(f"({pt[0]:.2f}, {pt[1]:.2f})")
    
    print("\n彩色图前5个中心点坐标:")
    for pt in color_centers[:5]:
        print(f"({pt[0]:.2f}, {pt[1]:.2f})")
    
    # 保存结果
    cv2.imwrite('result_gray.jpg', res_gray)
    cv2.imwrite('result_color.jpg', res_color)
import cv2
import numpy as np
from scipy import ndimage, signal
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt

def extract_laser_center(img, min_width=3):
    """改进的激光中心线提取函数"""
    # 1. 确保图像是灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 2. 增强对比度（关键修改：降低增强强度）
    gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # 3. 自适应阈值（关键修改：使用更稳健的方法）
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    
    # 4. 形态学处理（关键修改：简化处理流程）
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 5. 骨架化（关键修改：添加错误检查）
    skeleton = np.zeros_like(cleaned)
    try:
        skeleton = cv2.ximgproc.thinning(cleaned, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    except:
        print("警告：骨架化失败")
        return np.zeros((0,2), dtype=np.float32)
    
    # 6. 获取初始点集（关键修改：确保数据类型为整数）
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return np.zeros((0,2), dtype=np.float32)
    
    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    
    return points

def process_images(left_img, right_img):
    """改进的图像处理函数"""
    # 1. 图像读取（添加错误处理）
    try:
        if isinstance(left_img, str):
            left_img = cv2.imread(left_img)
            if left_img is None:
                raise ValueError(f"无法读取左图像: {left_img}")
        
        if isinstance(right_img, str):
            right_img = cv2.imread(right_img)
            if right_img is None:
                raise ValueError(f"无法读取右图像: {right_img}")
    except Exception as e:
        print(f"图像读取错误: {e}")
        return np.zeros((0,2)), np.zeros((0,2))
    
    # 2. 提取中心点
    left_pts = extract_laser_center(left_img)
    right_pts = extract_laser_center(right_img)
    
    print(f"提取完成 - 左图: {len(left_pts)}点, 右图: {len(right_pts)}点")
    
    # 3. 可视化（添加空检查）
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    
    for a, img, pts, side in zip(ax, [left_img, right_img], [left_pts, right_pts], ['左', '右']):
        if len(img.shape) == 2:
            a.imshow(img, cmap='gray')
        else:
            a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if len(pts) > 0:
            a.scatter(pts[:,0], pts[:,1], c='r', s=3)
        a.set_title(f"{side}图像 ({len(pts)}点)")
    
    plt.tight_layout()
    plt.show()
    
    # 4. 保存结果
    np.savetxt('left_points.txt', left_pts, fmt='%.2f')
    np.savetxt('right_points.txt', right_pts, fmt='%.2f')
    
    return left_pts, right_pts

# 主程序（添加完整的异常处理）
if __name__ == "__main__":
    try:
        left_path = 'L.png'  # 替换为您的实际路径
        right_path = 'R.png'  # 替换为您的实际路径
        
        print("正在处理图像...")
        left_points, right_points = process_images(left_path, right_path)
        
        if len(left_points) == 0 or len(right_points) == 0:
            print("警告：未能提取到激光中心线点")
            print("可能原因：")
            print("1. 图像路径不正确")
            print("2. 激光线太暗或太模糊")
            print("3. 参数需要调整（尝试修改min_width）")
        else:
            print("处理完成！")
            
    except Exception as e:
        print(f"程序出错: {str(e)}")
        print("建议检查：")
        print("1. 确保opencv-contrib-python包已安装（包含ximgproc模块）")
        print("2. 确保图像文件存在且格式正确")

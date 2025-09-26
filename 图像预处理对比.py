import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====================== 图像预处理函数 ======================
def adaptive_gamma_correction(img, config):
    """自适应伽马校正（局部抑制高光）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, config['specular_thresh'], 255, cv2.THRESH_BINARY)
    inv_gamma = 1.0 / config['gamma_correct']
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(img, table)
    return cv2.bitwise_and(corrected, corrected, mask=mask) + cv2.bitwise_and(img, img, mask=~mask)

def enhance_laser_channel(img, config):
    """激光通道增强"""
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

def local_contrast_enhancement(gray, config):
    """局部对比度增强"""
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

def multi_scale_preprocess(img, config):
    """
    多尺度预处理流水线
    Args:
        img: 原始输入图像
        config: 配置参数
    Returns:
        预处理后的单通道灰度图
    """
    # 1：伽马校正抑制高光
    corrected = adaptive_gamma_correction(img, config)

    # 2：转换到LAB颜色空间
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 3：自适应直方图均衡化（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip'], tileGridSize=(8,8))
    l = clahe.apply(l)

    # 4：混合模糊
    blur1 = cv2.GaussianBlur(l, config['blur_kernel'], 0)
    blur2 = cv2.medianBlur(l, 5)
    merged = cv2.addWeighted(blur1, 0.6, blur2, 0.4, 0)

    # 5：激光通道增强
    enhanced = enhance_laser_channel(merged, config)

    # 转换为灰度图后进行局部增强
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    enhanced_gray = local_contrast_enhancement(gray, config)
    
    return enhanced_gray

# ====================== 大津阈值法处理 ======================
def otsu_thresholding(gray_img):
    """应用大津阈值法"""
    _, otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 配置参数
    LEFT_CONFIG = {
        'laser_color': 'gray',
        'min_laser_intensity': 75,
        'clahe_clip': 3.5,
        'blur_kernel': (3, 3),
        'gamma_correct': 1.0,
        'specular_thresh': 200,
        'local_enhance_region': (0, 1),
        'clahe_clip_local': 1.5,
        'blend_weights': (0.2, 0.8)
    }

    # 读取图像
    img = cv2.imread('31.1.bmp')  # 使用左图作为示例
    if img is None:
        print("错误：无法读取图像文件！请检查文件路径和名称是否正确。")
        exit()

    # 三种处理方式
    original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 仅用大津阈值法
    otsu_only = otsu_thresholding(original_gray)
    
    # 2. 我们的预处理方法
    our_method = multi_scale_preprocess(img, LEFT_CONFIG)
    
    # 3. 我们的预处理+大津阈值法
    our_method_plus_otsu = otsu_thresholding(our_method)

    # 创建对比图
    plt.figure(figsize=(15, 8))
    
    # 第一行：原始方法和单独阈值法
    plt.subplot(2, 3, 1)
    plt.imshow(original_gray, cmap='gray')
    plt.title('1. Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(otsu_only, cmap='gray')
    plt.title('2. Otsu Only')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(our_method, cmap='gray')
    plt.title('3. Our Method Only')
    plt.axis('off')
    
    # 第二行：我们的方法与组合方法
    plt.subplot(2, 3, 4)
    plt.imshow(original_gray, cmap='gray')
    plt.title('1. Original (Reference)')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(our_method, cmap='gray')
    plt.title('3. Our Method (Reference)')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(our_method_plus_otsu, cmap='gray')
    plt.title('4. Our Method + Otsu')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

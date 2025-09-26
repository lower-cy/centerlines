import cv2
import numpy as np
import cupy as cp  # 替代 NumPy 实现 GPU 加速
from scipy import ndimage, signal
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.feature import hessian_matrix
from sklearn.linear_model import RANSACRegressor
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import time


def adaptive_noise_suppression(img):
    """
    自适应噪声抑制（混合 GPU 和 CPU 版本）
    """
    img_gpu = cp.array(img)  # 将图像数据转移到 GPU

    # 计算局部标准差（CPU 版本）
    img_cpu = cp.asnumpy(img_gpu)  # 转换回 NumPy 数组
    local_var = ndimage.generic_filter(img_cpu, np.std, size=5, mode='reflect') ** 2

    # 将结果转回 GPU 并计算 sigma_map
    local_var_gpu = cp.array(local_var)
    sigma_map = 1.2 + (2.5 - 1.2) * (local_var_gpu / local_var_gpu.max())

    # 高斯模糊（GPU 版本）
    img_smooth_gpu = cv2.GaussianBlur(cp.asnumpy(img_gpu), (0, 0), sigmaX=float(cp.mean(sigma_map)))
    return img_smooth_gpu


def multiscale_light_enhancement(img_smooth):
    """
    多尺度光条增强（混合 GPU 和 CPU 版本）
    """
    # 使用 skimage 计算 Hessian 矩阵（CPU 版本）
    hxx, hxy, hyy = hessian_matrix(img_smooth, sigma=2.0, use_gaussian_derivatives=False)

    # 将 Hessian 矩阵转移到 GPU
    hxx_gpu = cp.array(hxx)
    hxy_gpu = cp.array(hxy)
    hyy_gpu = cp.array(hyy)

    # 构造批量对称矩阵 [[hxx, hxy], [hxy, hyy]]
    batch_size = hxx.shape
    matrix_shape = (batch_size[0], batch_size[1], 2, 2)
    hessian_matrices = cp.zeros(matrix_shape, dtype=cp.float32)
    hessian_matrices[:, :, 0, 0] = hxx_gpu
    hessian_matrices[:, :, 0, 1] = hxy_gpu
    hessian_matrices[:, :, 1, 0] = hxy_gpu
    hessian_matrices[:, :, 1, 1] = hyy_gpu

    # 使用 GPU 计算特征值
    eigvals, _ = cp.linalg.eigh(hessian_matrices)  # eigh 适用于对称矩阵
    lambda1 = cp.min(eigvals, axis=-1)  # 取较小的特征值
    enhanced_img = lambda1 * (lambda1 > 0)

    return cp.asnumpy(enhanced_img)


def morphological_processing(bw, min_width):
    """
    形态学处理（GPU 版本）
    """
    bw_gpu = cp.array(bw)
    line_thickness = max(1, min_width // 2)
    se_length = 5 + 2 * line_thickness

    # 使用垂直结构元素闭运算
    se_vertical = cp.ones((1, se_length), dtype=cp.uint8)
    bw_gpu = cp.clip(cp.convolve(bw_gpu, se_vertical, mode='constant'), 0, 1)

    # 使用水平结构元素开运算
    se_horizontal = cp.ones((3, 1), dtype=cp.uint8)
    bw_gpu = cp.clip(cp.convolve(bw_gpu, se_horizontal, mode='constant'), 0, 1)

    # 去除小区域
    bw_cpu = cp.asnumpy(bw_gpu)
    bw_cpu = remove_small_objects(bw_cpu.astype(np.int32), min_size=min_width * 20)
    return bw_cpu


def local_orientation(img, cx, cy, win_size):
    """
    计算局部方向（CPU 版本）
    """
    xmin = max(0, cx - win_size)
    xmax = min(img.shape[1], cx + win_size)
    ymin = max(0, cy - win_size)
    ymax = min(img.shape[0], cy + win_size)

    patch = img[ymin:ymax, xmin:xmax]
    gy, gx = np.gradient(patch)

    G11 = np.sum(gx**2)
    G12 = np.sum(gx * gy)
    G22 = np.sum(gy**2)
    G = np.array([[G11, G12], [G12, G22]])

    eigvals, eigvecs = np.linalg.eig(G)
    nx, ny = eigvecs[:, np.argmin(eigvals)]
    conf = (G11 - G22)**2 / (G11 + G22)**2 if (G11 + G22) > 0 else 0
    return nx, ny, conf


def search_peak_on_normal(img, cx, cy, nx, ny, min_width):
    """
    在法线方向搜索峰值（CPU 版本）
    """
    search_range = int(min_width * 2)
    steps = np.linspace(-search_range, search_range, 30)
    intensities = []

    for s in steps:
        px = cx + s * nx
        py = cy + s * ny

        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
            intensity = cv2.remap(img,
                                  np.array([[px]], dtype=np.float32),
                                  np.array([[py]], dtype=np.float32),
                                  cv2.INTER_CUBIC)[0, 0]
            intensities.append(intensity)
        else:
            intensities.append(0)

    intensities = np.array(intensities)
    peak_thresh = intensities.max() * 0.7
    peaks, _ = signal.find_peaks(intensities, height=peak_thresh)

    if len(peaks) > 0:
        idx = np.argmax(intensities[peaks])
        s_optimal = steps[peaks[idx]]
        px = cx + s_optimal * nx
        py = cy + s_optimal * ny
        return px, py, True
    return 0, 0, False


def process_point(cx, cy, bw, img_smooth, min_width):
    """
    单点处理函数（CPU 版本）
    """
    dist_transform = ndimage.distance_transform_edt(bw)
    win_size = max(min_width, int(dist_transform[cy, cx] * 1.5))

    nx, ny, conf = local_orientation(img_smooth, cx, cy, win_size)
    if conf < 0.8:
        return None

    px, py, valid = search_peak_on_normal(img_smooth, cx, cy, nx, ny, min_width)
    return [px, py] if valid else None


def search_peaks_parallel(img_smooth, centers, bw, min_width):
    """
    并行化法线引导亚像素定位（CPU 版本）
    """
    def process_point_wrapper(cx, cy):
        return process_point(cx, cy, bw, img_smooth, min_width)

    results = Parallel(n_jobs=-1)(delayed(process_point_wrapper)(cx, cy) for cx, cy in centers)
    return np.array([r for r in results if r is not None])


def ransac_polyfit(points, order=2, thresh=5.0):
    """
    使用RANSAC算法进行多项式拟合
    """
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    model = RANSACRegressor(
        min_samples=max(order + 1, len(points) // 10),
        residual_threshold=thresh,
        max_trials=50
    )

    X_poly = np.column_stack([X**i for i in range(order + 1)])
    model.fit(X_poly, y)
    return model.inlier_mask_


def extract_laser_center_reflective_optimized(img, min_width=3):
    """
    提取激光中心线（优化版）
    """
    # 自适应噪声抑制
    img_smooth = adaptive_noise_suppression(img)

    # 多尺度光条增强
    enhanced_img = multiscale_light_enhancement(img_smooth)

    # 动态阈值分割与形态学处理
    thresh = threshold_otsu(enhanced_img) * 0.7
    bw = enhanced_img > thresh
    bw = morphological_processing(bw, min_width)

    # 法线引导亚像素定位
    y, x = np.where(bw)
    centers = np.column_stack((x, y))
    centers = search_peaks_parallel(img_smooth, centers, bw, min_width)

    # RANSAC 拟合去除离群点
    if len(centers) > 0:
        inliers = ransac_polyfit(centers, order=2, thresh=5.0)
        centers = centers[inliers]

    return centers


def extract_laser_center_pair(left_img_path, right_img_path):
    """
    提取左右图像的激光中心线
    """
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    start = time.time()
    left_centers = extract_laser_center_reflective_optimized(left_img)
    print(f'左图处理完成，耗时{time.time() - start:.2f}s，提取{len(left_centers)}个点')

    start = time.time()
    right_centers = extract_laser_center_reflective_optimized(right_img)
    print(f'右图处理完成，耗时{time.time() - start:.2f}s，提取{len(right_centers)}个点')

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(left_img, cmap='gray')
    plt.plot(left_centers[:, 0], left_centers[:, 1], 'r.', markersize=3)
    plt.title(f'左图像(提取{len(left_centers)}点)')

    plt.subplot(122)
    plt.imshow(right_img, cmap='gray')
    plt.plot(right_centers[:, 0], right_centers[:, 1], 'b.', markersize=3)
    plt.title(f'右图像(提取{len(right_centers)}点)')

    plt.tight_layout()
    plt.show()

    # 保存结果
    np.savetxt('left_centers.txt', left_centers)
    np.savetxt('right_centers.txt', right_centers)
    print("结果已保存为left_centers.txt和right_centers.txt")


if __name__ == "__main__":
    left_img_path = 'L.jpg'
    right_img_path = 'R.jpg'

    # 调用处理函数
    extract_laser_center_pair(left_img_path, right_img_path)
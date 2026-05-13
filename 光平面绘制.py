import numpy as np
import matplotlib
matplotlib.use("Agg")
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ====================== 读取PLY点云 ======================

def read_ply(filename):
    points = []
    colors = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() == "end_header":
                break
        for line in f:
            data = line.strip().split()
            if len(data) >= 6:
                points.append([float(data[0]), float(data[1]), float(data[2])])
                colors.append([int(data[3]), int(data[4]), int(data[5])])
    return np.array(points), np.array(colors)


def write_ply(filename, points, colors):
    n = len(points)
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")


def sample_plane_on_bbox(A, B, C, D, bbox, n1=40, n2=40):
    """
    沿平面法向量的最弱轴求解，在另外两轴的包围盒范围内采样。

    平面方程: Ax + By + Cz + D = 0
    bbox: (x_min, x_max, y_min, y_max, z_min, z_max)

    自动选择求解轴（法向量分量绝对值最大的轴），
    在另外两个轴的平面内做网格采样。
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    abs_n = np.array([abs(A), abs(B), abs(C)])
    dominant = np.argmax(abs_n)  # 0=x, 1=y, 2=z

    if abs_n[dominant] < 1e-12:
        return np.empty((0, 3))

    if dominant == 0:   # |A|最大 → 求解 x = -(By + Cz + D)/A
        margin = (x_max - x_min) * 0.3
        ys = np.linspace(y_min - margin, y_max + margin, n1)
        zs = np.linspace(z_min - margin, z_max + margin, n2)
        yy, zz = np.meshgrid(ys, zs)
        xx = -(B * yy + C * zz + D) / A
        # 裁剪到合理范围
        clip = max(x_max - x_min, 50)
        mask = (xx > x_min - clip) & (xx < x_max + clip)
        pts = np.stack([xx, yy, zz], axis=-1)
        return pts[mask]

    elif dominant == 1:  # |B|最大 → 求解 y = -(Ax + Cz + D)/B
        margin = (y_max - y_min) * 0.3
        xs = np.linspace(x_min - margin, x_max + margin, n1)
        zs = np.linspace(z_min - margin, z_max + margin, n2)
        xx, zz = np.meshgrid(xs, zs)
        yy = -(A * xx + C * zz + D) / B
        clip = max(y_max - y_min, 50)
        mask = (yy > y_min - clip) & (yy < y_max + clip)
        pts = np.stack([xx, yy, zz], axis=-1)
        return pts[mask]

    else:               # |C|最大 → 求解 z = -(Ax + By + D)/C
        margin = (z_max - z_min) * 0.3
        xs = np.linspace(x_min - margin, x_max + margin, n1)
        ys = np.linspace(y_min - margin, y_max + margin, n2)
        xx, yy = np.meshgrid(xs, ys)
        zz = -(A * xx + B * yy + D) / C
        clip = max(z_max - z_min, 50)
        mask = (zz > z_min - clip) & (zz < z_max + clip)
        pts = np.stack([xx, yy, zz], axis=-1)
        return pts[mask]


# ====================== 主程序 ======================

def main():
    # 1. 读取点云
    print("读取 output.ply ...")
    cloud_pts, cloud_colors = read_ply("output.ply")
    print(f"  点云点数: {len(cloud_pts)}")

    x_min, x_max = cloud_pts[:, 0].min(), cloud_pts[:, 0].max()
    y_min, y_max = cloud_pts[:, 1].min(), cloud_pts[:, 1].max()
    z_min, z_max = cloud_pts[:, 2].min(), cloud_pts[:, 2].max()
    bbox = (x_min, x_max, y_min, y_max, z_min, z_max)
    print(f"  点云范围: x=[{x_min:.1f}, {x_max:.1f}]  y=[{y_min:.1f}, {y_max:.1f}]  z=[{z_min:.1f}, {z_max:.1f}]")

    # 2. 读取光平面方程
    print("\n读取 plane_equations.yaml ...")
    with open("plane_equations.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    planes = cfg["plane_equations"]
    print(f"  光平面数量: {len(planes)}")

    # 3. 验证：平面是否经过点云中对应点
    print("\n验证平面方程与点云的一致性:")
    for pl in planes:
        A, B, C, D = pl["A"], pl["B"], pl["C"], pl["D"]
        vals = np.abs(A * cloud_pts[:, 0] + B * cloud_pts[:, 1] +
                      C * cloud_pts[:, 2] + D)
        idx = pl["line_index"]
        print(f"  线{idx:2d}: min_err={vals.min():.3f}  median={np.median(vals):.3f}  "
              f"mean={vals.mean():.3f}  max={vals.max():.3f}")

    # 4. 为每个光平面采样，合并到新点云
    all_pts = [cloud_pts]
    all_colors = [cloud_colors]
    cmap = plt.cm.tab10

    plane_sample_counts = []
    for i, pl in enumerate(planes):
        A, B, C, D = pl["A"], pl["B"], pl["C"], pl["D"]
        idx = pl["line_index"]

        pts = sample_plane_on_bbox(A, B, C, D, bbox, n1=40, n2=40)
        n_pts = len(pts)
        if n_pts == 0:
            print(f"  线{idx}: 采样点为空，跳过")
            continue

        color = (np.array(cmap(i % 10)[:3]) * 255).astype(np.uint8)
        colors = np.tile(color, (n_pts, 1))

        all_pts.append(pts)
        all_colors.append(colors)
        plane_sample_counts.append((idx, n_pts))

    # 5. 合并写入新PLY
    combined_pts = np.vstack(all_pts)
    combined_colors = np.vstack(all_colors).astype(np.uint8)

    out_path = "output_with_planes.ply"
    write_ply(out_path, combined_pts, combined_colors)
    print(f"\n合成点云已保存: {out_path}")
    print(f"  总点数: {len(combined_pts)} (原始{len(cloud_pts)} + 平面{len(combined_pts)-len(cloud_pts)})")
    for idx, n in plane_sample_counts:
        print(f"  线{idx}: {n} 个点")

    # 6. 3D可视化
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(cloud_pts[:, 0], cloud_pts[:, 1], cloud_pts[:, 2],
               c='lightgray', s=1.0, alpha=0.6, label='点云')

    for i, pl in enumerate(planes):
        A, B, C, D = pl["A"], pl["B"], pl["C"], pl["D"]
        idx = pl["line_index"]

        pts = sample_plane_on_bbox(A, B, C, D, bbox, n1=20, n2=20)
        if len(pts) == 0:
            continue
        

        color = cmap(i % 10)
        # 用散点图代替曲面图，避免非矩形网格导致绘图失败
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=[color], s=0.5, alpha=0.3, label=f'线{idx}')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('点云与光平面')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                              markersize=5, label='点云')]
    for i, pl in enumerate(planes):
        idx = pl["line_index"]
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor=cmap(i % 10),
                   markersize=6, label=f'线{idx}')
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig("planes_visualization.png", dpi=200)
    print("可视化已保存: planes_visualization.png")
    plt.show()


if __name__ == "__main__":
    main()

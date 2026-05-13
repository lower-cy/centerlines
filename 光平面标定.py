import cv2
import numpy as np
import yaml
from 基于光平面匹配 import (
    stereo_rectify, detect_laser_lines,
    LEFT_CONFIG, RIGHT_CONFIG,
    CAMERA_MATRIX_LEFT, CAMERA_MATRIX_RIGHT,
    DIST_COEFF_LEFT, DIST_COEFF_RIGHT, R, T
)


class PlaneFitting:
    """SVD平面拟合（保留兼容性）"""

    def fit_plane(self, points):
        points = np.asarray(points)
        centroid = np.mean(points, axis=0)
        _, _, Vt = np.linalg.svd(points - centroid)
        n = Vt[-1, :]
        d = -np.dot(n, centroid)
        n = n / np.linalg.norm(n)
        d = d / np.linalg.norm(n)
        return n, d

    @staticmethod
    def equation_str(a, b, c, d):
        return f"{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0"


class InteractiveCalibrator:
    """交互式光平面标定器 —— 键盘选择左右对应激光线，拟合光平面方程"""

    def __init__(self, left_path, right_path):
        self.left_raw = cv2.imread(left_path)
        self.right_raw = cv2.imread(right_path)
        if self.left_raw is None or self.right_raw is None:
            raise FileNotFoundError("无法读取图像，请检查路径")

        h, w = self.left_raw.shape[:2]

        # 极线矫正
        print("极线矫正...")
        self.left_rect, self.right_rect, self.P1, self.P2, self.Q = \
            stereo_rectify(self.left_raw, self.right_raw)

        # 检测激光线
        print("检测左图激光线...")
        self.left_lines = detect_laser_lines(self.left_rect, LEFT_CONFIG)
        print(f"  -> {len(self.left_lines)} 条")

        print("检测右图激光线...")
        self.right_lines = detect_laser_lines(self.right_rect, RIGHT_CONFIG)
        print(f"  -> {len(self.right_lines)} 条")

        # 相机参数
        self.fx = self.P1[0, 0]
        self.fy = self.P1[1, 1]
        self.cx = self.P1[0, 2]
        self.cy = self.P1[1, 2]
        self.Tx = self.P2[0, 3] / self.P2[0, 0]
        # Q矩阵: W = Q[3,2]*d + Q[3,3], Z = fx/W, X = (u-cx)/W, Y = (v-cy)/W
        # 使用Q矩阵三角测量，与 cv2.reprojectImageTo3D 完全一致
        print(f"fx={self.fx:.1f}  fy={self.fy:.1f}  Tx={self.Tx:.2f} mm")

        # 状态
        self.sel_left = 0
        self.sel_right = 0
        self.calibrated = []       # 已标定的光平面 [(A,B,C,D), ...]
        self.last_plane = None
        self.last_matched_left = None
        self.last_matched_right = None

    # ------------------------------------------------------------------
    #  绘图
    # ------------------------------------------------------------------

    def _draw_lines(self, img, lines, highlight_idx, matched_pts=None):
        if len(img.shape) == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()

        for i, line in enumerate(lines):
            color = (0, 255, 0)
            thickness = 2
            if i == highlight_idx:
                color = (0, 0, 255)
                thickness = 3
            pts = line.astype(int)
            cv2.polylines(vis, [pts], False, color, thickness)
            mid = pts[len(pts) // 2]
            cv2.putText(vis, str(i), (mid[0] - 10, mid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 绘制匹配点
        if matched_pts is not None and len(matched_pts):
            for pt in matched_pts:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)

        return vis

    # ------------------------------------------------------------------
    #  匹配与三角测量
    # ------------------------------------------------------------------

    def _match_and_triangulate(self, left_line, right_line):
        """
        沿极线匹配左右线点，三角测量得到3D坐标。

        Returns:
            pts_left  (M,2)  左图像素
            pts_right (M,2)  右图像素
            pts_3d    (M,3)  左矫正相机坐标系下的3D点
        """
        # 右线按行索引
        right_idx = {}
        for pt in right_line:
            y = int(round(pt[1]))
            right_idx.setdefault(y, []).append(pt[0])

        pts_left, pts_right, pts_3d = [], [], []

        for pt in left_line:
            xL, yL = pt[0], pt[1]
            y_int = int(round(yL))

            best_xR = None
            best_disp = 0
            for dy in range(-2, 3):          # 行方向搜索半径 ±2
                cand = right_idx.get(y_int + dy)
                if cand is None:
                    continue
                for xR in cand:
                    d = xL - xR               # 视差
                    if d > 3 and (best_xR is None or d < best_disp):
                        best_disp = d
                        best_xR = xR
                        best_yR = y_int + dy

            if best_xR is not None:
                # 使用 Q 矩阵三角测量，与 cv2.reprojectImageTo3D 完全一致
                W = self.Q[3, 2] * best_disp + self.Q[3, 3]
                if abs(W) < 1e-12:
                    continue
                Z = self.fx / W
                X = (xL + self.Q[0, 3]) / W   # Q[0,3] = -cx
                Y = (yL + self.Q[1, 3]) / W   # Q[1,3] = -cy

                pts_left.append([xL, yL])
                pts_right.append([best_xR, best_yR])
                pts_3d.append([X, Y, Z])

        return (np.float32(pts_left) if pts_left else np.empty((0, 2)),
                np.float32(pts_right) if pts_right else np.empty((0, 2)),
                np.array(pts_3d) if pts_3d else np.empty((0, 3)))

    @staticmethod
    def _fit_plane_svd(points):
        centroid = np.mean(points, axis=0)
        _, _, Vt = np.linalg.svd(points - centroid)
        n = Vt[-1, :]
        d = -np.dot(n, centroid)
        n = n / np.linalg.norm(n)
        d = d / np.linalg.norm(n)
        return float(n[0]), float(n[1]), float(n[2]), float(d)

    # ------------------------------------------------------------------
    #  交互主循环
    # ------------------------------------------------------------------

    def run(self):
        nL = len(self.left_lines)
        nR = len(self.right_lines)

        print("\n=== 交互式光平面标定 ===")
        print(f"左图 {nL} 条线, 右图 {nR} 条线")
        print("")
        print("  [a/d]     左图线  上一条/下一条")
        print("  [w/s]     右图线  上一条/下一条")
        print("  [Space]   标定当前左右线对 → 拟合光平面")
        print("  [Enter]   保存所有已标定光平面到 YAML")
        print("  [r]       重置已标定结果")
        print("  [q/Esc]   退出")
        print("")

        while True:
            # 绘制
            left_vis = self._draw_lines(self.left_rect, self.left_lines,
                                        self.sel_left if nL else -1,
                                        self.last_matched_left)
            right_vis = self._draw_lines(self.right_rect, self.right_lines,
                                         self.sel_right if nR else -1,
                                         self.last_matched_right)

            h, w = left_vis.shape[:2]
            canvas = np.zeros((h + 90, w * 2, 3), dtype=np.uint8)
            canvas[:h, :w] = left_vis
            canvas[:h, w:] = right_vis

            yb = h + 5
            sel_l_str = f"{self.sel_left}/{nL - 1}" if nL else "-"
            sel_r_str = f"{self.sel_right}/{nR - 1}" if nR else "-"
            cv2.putText(canvas,
                        f"左线 [{sel_l_str}]   右线 [{sel_r_str}]   已标定 {len(self.calibrated)} 个平面",
                        (10, yb + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(canvas,
                        "[a/d:左线] [w/s:右线] [Space:标定] [Enter:保存] [r:重置] [q:退出]",
                        (10, yb + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            if self.last_plane is not None:
                a, b, c, d = self.last_plane
                cv2.putText(canvas,
                            f"最新平面: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0",
                            (10, yb + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("光平面标定", cv2.resize(canvas, (1400, 960)))
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('a'):
                self.sel_left = max(0, self.sel_left - 1) if nL else 0
            elif key == ord('d'):
                self.sel_left = min(nL - 1, self.sel_left + 1) if nL else 0
            elif key == ord('w'):
                self.sel_right = max(0, self.sel_right - 1) if nR else 0
            elif key == ord('s'):
                self.sel_right = min(nR - 1, self.sel_right + 1) if nR else 0
            elif key == ord(' '):
                self._calibrate()
            elif key == 13:          # Enter
                self._save_yaml()
            elif key == ord('r'):
                self.calibrated.clear()
                self.last_plane = None
                self.last_matched_left = None
                self.last_matched_right = None
                print("已重置所有标定结果")

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    #  标定与保存
    # ------------------------------------------------------------------

    def _calibrate(self):
        if not self.left_lines or not self.right_lines:
            print("错误: 左右图中至少有一图未检测到激光线")
            return

        left_line = self.left_lines[self.sel_left]
        right_line = self.right_lines[self.sel_right]

        pts_left, pts_right, pts_3d = self._match_and_triangulate(left_line, right_line)
        if len(pts_3d) < 10:
            print(f"有效3D点不足 ({len(pts_3d)}/10)，无法拟合平面")
            return

        a, b, c, d = self._fit_plane_svd(pts_3d)
        self.last_plane = (a, b, c, d)
        self.calibrated.append((a, b, c, d))
        self.last_matched_left = pts_left
        self.last_matched_right = pts_right

        # 拟合质量
        dist = np.abs(pts_3d @ np.array([a, b, c]) + d)
        print(f"线 {self.sel_left}(左) → {self.sel_right}(右)  |  "
              f"{len(pts_3d)} 点  |  "
              f"平面: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        print(f"  拟合误差: mean={np.mean(dist):.4f}  max={np.max(dist):.4f}  "
              f"std={np.std(dist):.4f} mm")

    def _save_yaml(self):
        if not self.calibrated:
            print("没有已标定的光平面可保存")
            return

        try:
            with open("plane_equations.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        if "plane_equations" not in cfg:
            cfg["plane_equations"] = []

        max_idx = -1
        for entry in cfg["plane_equations"]:
            if "line_index" in entry and entry["line_index"] > max_idx:
                max_idx = entry["line_index"]

        for i, (a, b, c, d) in enumerate(self.calibrated):
            cfg["plane_equations"].append({
                "line_index": max_idx + 1 + i,
                "A": a, "B": b, "C": c, "D": d,
                "equation": f"{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0"
            })

        with open("plane_equations.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        n = len(self.calibrated)
        print(f"已保存 {n} 个光平面到 plane_equations.yaml "
              f"(line_index {max_idx + 1} ~ {max_idx + n})")

        self.calibrated.clear()
        self.last_plane = None
        self.last_matched_left = None
        self.last_matched_right = None


def read_ply_points(filename):
    """从PLY文件提取点云坐标（保留兼容性）"""
    points = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() == "end_header":
                break
        for line in f:
            data = line.strip().split()
            if len(data) >= 3:
                points.append([float(data[0]), float(data[1]), float(data[2])])
    return np.array(points)


def demo():
    """原 PLY 平面拟合示例（保留兼容性）"""
    loaded = read_ply_points("output.ply")
    fitter = PlaneFitting()
    normal, d = fitter.fit_plane(loaded)
    a, b, c = normal
    print("拟合的平面方程:")
    print(PlaneFitting.equation_str(a, b, c, d))
    print(f"参数: A={a}, B={b}, C={c}, D={d}")


if __name__ == "__main__":
    import sys

    left_path = sys.argv[1] if len(sys.argv) > 1 else "31.1.bmp"
    right_path = sys.argv[2] if len(sys.argv) > 2 else "31.0.bmp"

    try:
        cal = InteractiveCalibrator(left_path, right_path)
        cal.run()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

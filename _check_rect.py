"""快速检查 stereoRectify 输出的 Q 矩阵和 P1, P2 参数"""
import numpy as np
import cv2

CAMERA_MATRIX_LEFT = np.array([
    [4703.3840666469305, 0.0, 1133.8966264844476],
    [0.0, 4657.770006641158, 983.7755276735744],
    [0.0, 0.0, 1.0]
])
CAMERA_MATRIX_RIGHT = np.array([
    [4409.199175099535, 0.0, 1531.0013908252736],
    [0.0, 4384.905205883512, 1013.4751888939345],
    [0.0, 0.0, 1.0]
])
DIST_COEFF_LEFT = np.array([-0.19060368249367288, -6.827044122904246, 0.015377030028687984, -0.00750634791176898, 107.39588017569562])
DIST_COEFF_RIGHT = np.array([-0.42270673798875497, 1.378263372731151, 0.009909410979026863, -0.008593483642757997, -1.0961258361436514])
R = np.array([
    [0.9867230542685737, 0.007483211056180142, 0.1622393778562597],
    [-0.005753664364150946, 0.9999215317777955, -0.011127696685821956],
    [-0.16230991812357692, 0.010046483933974946, 0.9866886837494805]
])
T = np.array([-65.930698300496, 0.7317230319931822, -12.020455702540955])

h, w = 1024, 2048  # approximate, from 31.1.bmp

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    CAMERA_MATRIX_LEFT, DIST_COEFF_LEFT,
    CAMERA_MATRIX_RIGHT, DIST_COEFF_RIGHT,
    (w, h), R, T, alpha=-1, flags=0|cv2.CALIB_USE_INTRINSIC_GUESS
)

print("=== P1 ===")
print(P1)
print("\n=== P2 ===")
print(P2)
print("\n=== Q ===")
print(Q)

fx = P1[0, 0]
cx1 = P1[0, 2]
cy1 = P1[1, 2]
cx2 = P2[0, 2]
cy2 = P2[1, 2]
print(f"\nfx = {fx:.2f}")
print(f"cx1 = {cx1:.4f}, cy1 = {cy1:.4f}")
print(f"cx2 = {cx2:.4f}, cy2 = {cy2:.4f}")
print(f"cx1 - cx2 = {cx1 - cx2:.4f}")

# Tx from P2
Tx_from_P2 = P2[0, 3] / P2[0, 0]
print(f"Tx (P2[0,3]/P2[0,0]) = {Tx_from_P2:.6f}")

# Tx from Q
# Q = [[1, 0, 0, -cx1], [0, 1, 0, -cy1], [0, 0, 0, fx], [0, 0, -1/Tx, (cx1-cx2)/Tx]]
# Q[3,2] = -1/Tx  =>  Tx = -1/Q[3,2]
Tx_from_Q = -1.0 / Q[3, 2]
print(f"Tx (from Q[3,2]) = {Tx_from_Q:.6f}")

# Check: if we use the simple formula Z = fx * Tx / d
# vs the Q formula Z = fx / ((-d + cx1 - cx2)/Tx)
print("\n=== 给定一个典型像素和视差，比较两种三角测量结果 ===")
u, v, d = 800, 500, 200  # typical rectified image coords

# Method 1: simple triangulation
Z_simple = fx * Tx_from_P2 / d
X_simple = (u - cx1) * Z_simple / fx
Y_simple = (v - cy1) * Z_simple / fx
print(f"简单公式: X={X_simple:.2f}, Y={Y_simple:.2f}, Z={Z_simple:.2f}")

# Method 2: Q reprojection
W = -d/Q[3,2] + Q[3,3]
Z_Q = fx / W
X_Q = (u - cx1) / W
Y_Q = (v - cy1) / W
print(f"Q公式:    X={X_Q:.2f}, Y={Y_Q:.2f}, Z={Z_Q:.2f}")

# Method 3: correct geometric formula: Z = fx*Tx / (d - (cx1-cx2))
Z_correct = fx * Tx_from_P2 / (d - (cx1 - cx2))
X_correct = (u - cx1) * Z_correct / fx
Y_correct = (v - cy1) * Z_correct / fx
print(f"正确公式: X={X_correct:.2f}, Y={Y_correct:.2f}, Z={Z_correct:.2f}")

# Verify: d = xL - xR = fx*Tx/Z + (cx1-cx2)
d_back = fx * Tx_from_P2 / Z_correct + (cx1 - cx2)
print(f"验证: d输入={d}, d反算={d_back:.2f}")

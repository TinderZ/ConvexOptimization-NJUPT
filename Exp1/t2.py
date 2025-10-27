import numpy as np
from cvxopt import matrix, solvers

def sol_least_squares():
    y = np.array([22, 27.2, 35.1, 47.2, 100.4])
    x1 = np.array([0, 2.3, 5.1, 12.4, 15.3])
    x2 = np.array([4.2, 5.5, 3.6, 1.5, 10.6])
    
    ones = np.ones(len(y))
    A = np.column_stack([ones, x1, x2])  # 5×3 [1, x1, x2]
    b = y  # 5×1

    # 转换为二次规划问题: min (1/2)x'Px + q'x
    P = matrix(2 * A.T @ A)    # 其中 P = 2A'A, q = -2A'b, 推导见小结
    q = matrix(-2 * A.T @ b)
    
    G = matrix(np.array([    # 不等式约束 Gx ≤ h
        [0.0, 1.0, 0.0],   # w₁ ≤ 2  =>  [0, 1, 0]x ≤ 2
        [0.0, 0.0, -1.0]   # w₂ ≥ 10 => [0, 0, -1]x ≤ -10
    ]))
    h = matrix(np.array([2.0, -10.0]))
    
    # 求解二次规划
    sol = solvers.qp(P, q, G, h)
    
    if sol['status'] == 'optimal':
        w = np.array(sol['x']).flatten()
        
        # 计算拟合值和误差
        y_pred = A @ w
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        print(f"\n求解成功!")
        print(f"回归方程: y = {w[0]:.3f} + {w[1]:.3f}x₁ + {w[2]:.3f}x₂")
        print(f"均方误差: {mse:.3f}")
        return w
    else:
        print(f"求解失败: {sol['status']}")
        return None

sol_least_squares()

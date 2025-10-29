import numpy as np
import matplotlib.pyplot as plt

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(filepath):
    data = np.loadtxt(filepath)
    # 第一列是序号，不需要，从第二列开始取
    X = data[:, 1:-1]
    y = data[:, -1]
    y = y.reshape(-1, 1)

    train_size = 350
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 为特征矩阵添加偏置项（一列1）
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    
    return X_train, y_train, X_test, y_test

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def least_squares(X_train, y_train):
    #求解最小二乘法的权重
    # w = (X^T * X)^-1 * X^T * y
    w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    return w

def ridge_regression(X_train, y_train, lambda_val):
    #求解岭回归的权重
    # w = (X^T * X + lambda * I)^-1 * X^T * y
    I = np.identity(X_train.shape[1])
    w = np.linalg.inv(X_train.T @ X_train + lambda_val * I) @ X_train.T @ y_train
    return w

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data('Exp2/data.txt')

    # 最小二乘法
    w_ls = least_squares(X_train, y_train)
    y_pred_ls = X_test @ w_ls
    mse_ls_test = mean_squared_error(y_test, y_pred_ls)
    print(f"测试集上的均方误差 (MSE): {mse_ls_test:.4f}\n")

    # 岭回归
    lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
    train_mses = []
    test_mses = []

    for l in lambdas:
        w_ridge = ridge_regression(X_train, y_train, l)
        
        # 训练集
        y_pred_train_ridge = X_train @ w_ridge
        mse_train = mean_squared_error(y_train, y_pred_train_ridge)
        train_mses.append(mse_train)
        # 测试集
        y_pred_test_ridge = X_test @ w_ridge
        mse_test = mean_squared_error(y_test, y_pred_test_ridge)
        test_mses.append(mse_test)
        
        print(f"λ = {l}")
        print(f"训练集 MSE: {mse_train:.4f}")
        print(f"测试集 MSE: {mse_test:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas[1:], train_mses[1:], 'o-', label='训练集 MSE')
    plt.plot(lambdas[1:], test_mses[1:], 'o-', label='测试集 MSE')
    plt.xscale('log')
    plt.xlabel('拉格朗日乘子(log λ)')
    plt.ylabel('均方误差 (MSE)')
    plt.title('岭回归中λ值与均方误差的关系')
    plt.legend()
    plt.grid(True)
    plt.show()


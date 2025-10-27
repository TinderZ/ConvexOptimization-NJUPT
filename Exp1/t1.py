import numpy as np
from cvxopt import matrix, solvers

def sol_linear_opt():
    foods = ['鸡蛋', '火腿', '香肠', '羊肉', '米饭', '烙饼', '凉粉']
    nutrition_data = np.array([
        [160, 80, 5],    [320, 30, 15],   [508, 40, 10],   
        [217, 60, 4],    [180, 5, 60],    [225, 7, 55],   [37, 10, 30]    
    ])
    n_foods = len(foods)
    
    # 目标函数系数 (最小化卡路里)
    c = matrix(nutrition_data[:, 0].astype(float))

    G_list = []  # 不等式约束 Gx <= h
    G_list.append(-nutrition_data[:, 1])  # 维生素约束
    G_list.append(-nutrition_data[:, 2])  # 碳水化合物约束
    
    # 添加非负约束 -x <= 0
    for i in range(n_foods):
        constraint_row = np.zeros(n_foods)
        constraint_row[i] = -1.0
        G_list.append(constraint_row)
    
    G = matrix(np.vstack(G_list).astype(float))
    h = matrix(np.array([-200.0, -300.0] + [0.0] * n_foods))
    
    print("-" * 40)
    sol = solvers.lp(c, G, h)
    
    if sol['status'] == 'optimal':
        x = np.array(sol['x']).flatten()
        
        total_calories = np.dot(x, nutrition_data[:, 0])
        total_vitamins = np.dot(x, nutrition_data[:, 1])
        total_carbs = np.dot(x, nutrition_data[:, 2])
        print(f"求解成功! 最小卡路里: {total_calories:.2f}")    
        print("\n最优食谱:")
        print(f"{'食材':<6} {'份数':<8} {'卡路里':<8} {'维生素':<8} {'碳水':<8}")
        print("-" * 40)
        
        for i, food in enumerate(foods):
            if x[i] > 1e-6: 
                weight = x[i]
                calories = weight * nutrition_data[i, 0]
                vitamins = weight * nutrition_data[i, 1]
                carbs = weight * nutrition_data[i, 2]
                print(f"{food:<6} {weight:<8.2f} {calories:<8.1f} {vitamins:<8.1f} {carbs:<8.1f}")
        print("-" * 50)
        print(f"总计: 卡路里={total_calories:.1f}, 维生素={total_vitamins:.1f}mg, 碳水={total_carbs:.1f}g")
        return x, sol
    else:
        print(f"求解失败，状态: {sol['status']}")
        return None, sol

sol_linear_opt()
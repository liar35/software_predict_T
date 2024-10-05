# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:20:30 2024

@author: ad
"""


import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from datetime import datetime

# 删除之前的调优结果目录（如果存在）
shutil.rmtree('tuner_dir', ignore_errors=True)

# 记录开始时间
start_time = datetime.now()

# 1. 从Excel读取数据
io = r"C:\Users\86152\Desktop\TEST_ML.xlsx"
data = pd.read_excel(io, sheet_name=3)

# 2. 去除空值
dataset = data.dropna()

# 3. 定义输入特征和输出目标
X = dataset.iloc[:, 1:9].values  # 假设特征在前8列
Y = dataset.iloc[:, 9:13].values  # 4个气体的突破时间
Y = np.log10(Y)  # 取log10变换

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=None)

# 5. LightGBM 参数设置
lgbm_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.07,
    'max_depth': 10,
    'min_child_weight': 6.0,
    'n_estimators': 737,
    'reg_alpha': 1.0,
    'reg_lambda': 0.0,
    'subsample': 0.8,
    'objective': 'regression',
    'n_jobs': -1
}

# 6. 交叉验证的设置
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5折交叉验证

# 7. 为每个气体分别训练 LightGBM 模型，并进行交叉验证
lgbm_models = []
cv_r2_scores = []  # 存储交叉验证R²分数
cv_rmse_scores = []  # 存储交叉验证RMSE

for i in range(4):
    lgbm_model = LGBMRegressor(**lgbm_params)
    
    # 使用交叉验证来评估模型表现 (R²和RMSE)
    r2_scores_cv = cross_val_score(lgbm_model, X_train, y_train[:, i], cv=kf, scoring='r2')
    neg_mse_scores_cv = cross_val_score(lgbm_model, X_train, y_train[:, i], cv=kf, scoring='neg_mean_squared_error')
    
    # 将交叉验证的平均结果记录
    cv_r2_scores.append(r2_scores_cv.mean())
    cv_rmse_scores.append(np.sqrt(-neg_mse_scores_cv.mean()))
    
    # 在整个训练集上拟合模型
    lgbm_model.fit(X_train, y_train[:, i])
    lgbm_models.append(lgbm_model)

# 8. 使用 LightGBM 进行预测
lgbm_preds = np.zeros_like(y_test)
for i in range(4):
    lgbm_preds[:, i] = lgbm_models[i].predict(X_test)

# 9. 模型评估 (R², RMSE, MAE)
r2_scores = []
rmse_scores = []
mae_scores = []

for i in range(4):
    R2_test = r2_score(y_test[:, i], lgbm_preds[:, i])
    RMSE_test = np.sqrt(mean_squared_error(y_test[:, i], lgbm_preds[:, i]))
    MAE_test = np.mean(np.abs(10**y_test[:, i] - 10**lgbm_preds[:, i]))  # 反log10还原后计算MAE
    r2_scores.append(R2_test)
    rmse_scores.append(RMSE_test)
    mae_scores.append(MAE_test)

# 10. 可视化
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.scatter(10 ** y_test[:, i], 10 ** lgbm_preds[:, i], label='Predicted Values', color='red', alpha=0.6)
    plt.plot([min(10 ** y_test[:, i]), max(10 ** y_test[:, i])], 
             [min(10 ** y_test[:, i]), max(10 ** y_test[:, i])], 
             color='blue', lw=2, label='True Values')
    plt.title(f'Gas {i+1} - Predicted vs True')
    plt.xlabel('True Breakthrough Time')
    plt.ylabel('Predicted Breakthrough Time')
    plt.legend()

# 展示所有子图
plt.tight_layout()
plt.show()

# 11. 在所有评估和可视化完成后，输出 R²、RMSE、MAE 和交叉验证结果
print("\nEvaluation Results:")
for i in range(4):
    print(f"Gas {i+1} - R²: {r2_scores[i]:.4f}, RMSE: {rmse_scores[i]:.4f}, MAE: {mae_scores[i]:.4f}")
    print(f"Gas {i+1} - Cross-Validated R²: {cv_r2_scores[i]:.4f}, Cross-Validated RMSE: {cv_rmse_scores[i]:.4f}")

# 记录结束时间
end_time = datetime.now()
duration = end_time - start_time
print('\nDuration:', duration)

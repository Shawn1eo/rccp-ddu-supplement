#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2025/3/31 10:23
# @Author: WANG SHENGRAN
import pandas as pd
from pycaret.regression import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

data = pd.read_csv("result.csv")
# 设置 PyCaret 回归环境
# 指定目标变量为 'final_unit_price'
reg_setup = setup(data, target='quantity', session_id=123,
                  numeric_features=['final_unit_price', 'original_unit_price','attribute1', 'attribute2', 'weekday'],
                  categorical_features=['sku_ID'])

# 获取所有可用的回归模型名称
all_models = models().index.to_list()

# 比较模型
best_model = compare_models(include=all_models, fold=5, sort='RMSE')

# 选择最优模型
print("最优模型：", best_model)

# 训练和评估最优模型
# 创建最优模型的详细报告
best_model_report = pull()
print("最优模型的性能指标：")
print(best_model_report)

# 保存模型
save_model(best_model, 'best_regression_model')  # 保存模型到文件

model = load_model('best_regression_model')

# 准备新数据
data2 = data

# 进行预测
predictions = predict_model(model, data=data2)

# 保存或使用预测结果
predictions.to_csv('predictions.csv', index=False)  # 将预测结果保存到文件
print(predictions.head())  # 打印预测结果的前几行

# 按 sku_ID 分组
grouped = predictions.groupby('sku_ID')

# 计算每个组的 RMSE和 R^2
rmse_results = []
r2_results = []
for sku, group in grouped:
    y_true = group['quantity']  # 实际值
    y_pred = group['prediction_label']  # 预测值
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    rmse_results.append({'sku_ID': sku, 'RMSE': rmse})
    r2 = r2_score(y_true,y_pred)
    r2_results.append({'sku_ID': sku, 'R2': r2})

# 输出结果
rmse_df = pd.DataFrame(rmse_results)
print(rmse_df)
r2_df = pd.DataFrame(r2_results)
print(r2_df)

# 如果需要将结果保存到文件
rmse_df.to_csv('rmse_results.csv', index=False)
r2_df.to_csv('r2_results.csv', index=False)
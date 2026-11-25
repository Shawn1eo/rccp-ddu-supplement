from Scheme import scheme4_knn, scheme4_lsa, scheme4_cart
import copy
import pandas as pd
from pycaret.regression import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("result.csv")
order_date_d = data[['order_date']].copy()
data = data.drop(columns=['order_date'])
# # 设置 PyCaret 回归环境
# reg_setup = setup(data, target='quantity', session_id=123,
#                   numeric_features=['final_unit_price', 'original_unit_price','attribute1', 'attribute2', 'weekday'],
#                   categorical_features=['sku_ID'])
#
# # 获取所有可用的回归模型名称
# all_models = models().index.to_list()
#
# # 比较模型
# best_model = compare_models(include=all_models, fold=5, sort='RMSE')
#
# # 选择最优模型
# print("最优模型：", best_model)
#
# # 训练和评估最优模型
# # 创建最优模型的详细报告
# best_model_report = pull()
# print("最优模型的性能指标：")
# print(best_model_report)
#
# # 保存模型
# save_model(best_model, 'best_regression_model')  # 保存模型到文件

model = load_model('best_regression_model')

# 准备新数据
data2 = data

# 进行预测
predictions = predict_model(model, data=data2)

# 保存或使用预测结果
# print(predictions.head())  # 打印预测结果的前几行
# predictions_date = pd.concat([order_date_d,predictions], axis=1)
# print(predictions_date.head())

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
# print(rmse_df)
r2_df = pd.DataFrame(r2_results)
# print(r2_df)

r2_88 = r2_df.query("R2 >= 0.88")
sku_88 = r2_88["sku_ID"]
df_88 = predictions[predictions["sku_ID"].isin(sku_88)]
residuals = df_88["quantity"] - df_88["prediction_label"]
df_88.drop(columns=["prediction_label"], inplace=True)
df_88 = df_88.rename(columns={"quantity": "demand"})
# plt.hist(residuals, bins=50)
# plt.show()


def pi_true(p, q, c, s, x, v):
    x1 = copy.deepcopy(x)
    x1["final_unit_price"] = [p]
    meanQuantity = model.predict(x1)[0]

    avgProfit = 0
    feasibleRate = 0
    for res in list(residuals):
        profit = (p - c) * q - (p - s) * np.maximum(0, q - (meanQuantity + res))
        if profit >= v:
            avgProfit += profit
            feasibleRate += 1
    avgProfit /= len(residuals)
    feasibleRate /= len(residuals)
    return avgProfit, feasibleRate


np.random.seed(42)
testset = df_88.sample(n=80)

ParameterSet = [(0, 0.1), (0, 0.2), (0, 0.5), (0, 0.9), (5, 0.1), (5, 0.2), (5, 0.5), (5, 0.9), (10, 0.1), (10, 0.2), (10, 0.5), (10, 0.9)]
Result = []
for v, alpha in ParameterSet:
    for i in range(80):
        x = pd.DataFrame(testset.iloc[i]).T
        c = float(x["final_unit_price"]/(1 + 0.021))
        s = c*(1 - 0.003)

        M = 3e5
        D = 35
        P = [int(x["final_unit_price"]) - 2 + 0.1 * t for t in range(D)]  # All possible discrete values of p
        P = [num for num in P if num > c]
        Data_standard = copy.deepcopy(df_88)
        Data_standard = pd.get_dummies(Data_standard, "sku_ID")
        originalColumns = Data_standard.columns
        for col in originalColumns:
            if sum(Data_standard[col]) < 1:
                Data_standard = Data_standard.drop(columns=[col])
        Data_standard["attribute1"] = (Data_standard["attribute1"] - 1)/3
        Data_standard["attribute2"] = (Data_standard["attribute2"] - 30)/70
        Data_standard["original_unit_price"] = (Data_standard["original_unit_price"] - 50)/150
        Data_standard["final_unit_price"] = (Data_standard["final_unit_price"] - min(P))/(max(P) - min(P))
        Data_standard = Data_standard.rename(columns={"final_unit_price": "price"})

        x_standard = Data_standard.loc[x.index]
        q0 = float(x_standard["demand"])
        x_standard = np.array(x_standard.drop(columns=["price", "demand"]))
        x_standard = x_standard.reshape(-1)

        Data_standard_reindex = copy.deepcopy(Data_standard).reset_index(drop=True)

        try:
            pstar_knn, qstar_knn, _ = scheme4_knn(Data_standard_reindex, x_standard, P, c, s, alpha, v, 0.9*q0, 1.1*q0)
            pstar_lsa, qstar_lsa, _ = scheme4_lsa(Data_standard_reindex, x_standard, P, c, s, alpha, v, 0.9*q0, 1.1*q0)
            pstar_cart, qstar_cart, _ = scheme4_cart(Data_standard_reindex, x_standard, P, c, s, alpha, v, 0.9*q0, 1.1*q0)
            
        except Exception as e:
            continue

        x = x.drop(columns=["demand"])
        print("kNN给出的解：")
        print(pstar_knn, qstar_knn)
        print("kNN给出的解的目标函数：")
        kNNobj, kNNrate = pi_true(pstar_knn, qstar_knn, c, s, x, v)
        print(kNNobj, kNNrate)

        print("lsa给出的解：")
        print(pstar_lsa, qstar_lsa)
        print("lsa给出的解的目标函数：")
        lsaobj, lsarate = pi_true(pstar_lsa, qstar_lsa, c, s, x, v)
        print(lsaobj, lsarate)

        print("cart给出的解：")
        print(pstar_cart, qstar_cart)
        print("cart给出的解的目标函数：")
        cartobj, cartrate = pi_true(pstar_cart, qstar_cart, c, s, x, v)
        print(cartobj, cartrate)
        print("原解：")
        print(float(x["final_unit_price"]), q0)
        print("原解的目标函数：")
        originObj, originRate = pi_true(float(x["final_unit_price"]), q0, c, s, x, v)
        print(originObj, originRate)
        if pstar_knn == 0 or pstar_lsa == 0 or pstar_cart == 0 or originObj < 10:
            continue
        Result.append([v, alpha, originObj, kNNobj, lsaobj, cartobj, originRate, kNNrate, lsarate, cartrate])


Result = pd.DataFrame(Result, columns=["v", "alpha", "originalObj", "kNNobj", "lsaobj", "cartobj", "originRate", "kNNrate", "lsarate", "cartrate"])
Result.to_excel("CaseStudy.xlsx")
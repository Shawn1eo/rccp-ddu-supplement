# The simplified Benders decomposition method for the discretization form of the original problem

import numpy as np
from gurobipy import *
import pandas as pd
from copy import deepcopy
from dualsubproblem_accelerate import lagrangian
import random

# M = 3e3
# alpha = 0.7
# p_min = 10
# q_max = 100
# c = 5
# s = 2
# D = 200
# k = 50
# P = [p_min + 0.1 * t for t in range(D)]  # All possible discrete values of p
# # m = 10  # 单iteration添加多少cut
#
# np.random.seed(1)
# Data = pd.read_excel("Data1.xlsx")
# cov = [[0.5 ** abs(i - j) for i in range(10)] for j in range(10)]
# mean = [0 for i in range(10)]
# X = np.random.multivariate_normal(mean, cov, 1)[0]
# N = Data.shape[0]
# DataFeature = Data.drop(["demand", "price"], axis=1)


def insert_into_sorted_list(sorted_list_to_insert, number_to_insert):
    sorted_list = deepcopy(sorted_list_to_insert)
    # 如果列表为空或者要插入的数字大于列表中的所有数字，直接插入到末尾
    if not sorted_list or number_to_insert > sorted_list[-1]:
        sorted_list.append(number_to_insert)
    else:
        # 遍历列表，找到第一个大于等于number_to_insert的索引
        for index, num in enumerate(sorted_list):
            if num >= number_to_insert:
                # 在这个索引位置插入数字
                sorted_list.insert(index, number_to_insert)
                break
    return sorted_list


def subProblem(p, Data, k, c, s, knn, dmin, dmax):
    n = Data.shape[0]
    Data_withknn = deepcopy(Data)
    Data_withknn["knn"] = knn
    knnDemand = sorted(list(Data_withknn.query("knn >= 0.9")["demand"]))
    N_star = math.ceil(k * (p - c) / (p - s))
    q = min(max(dmin, knnDemand[N_star - 1]),dmax)
    obj = k * (c - p) * q + sum([knn[i] * (p - s) * max(q - Data["demand"][i], 0) for i in range(n)])
    return q, obj


def subProblem_basic(p, Data, P, k, c, s, z, dmin, dmax, qmax, M=3e3):
    N = Data.shape[0]
    Np = len(P)
    demand = Data["demand"]
    index = P.index(p)
    g = [1 if t == index else 0 for t in range(Np)]
    model = Model("LinearizedReformulation")
    model.setParam("OutputFlag", 0)
    model.setParam('DualReductions', 0)

    # Define variables
    q = model.addVar(lb=0, ub=qmax, vtype=GRB.CONTINUOUS, name="q")
    r = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="r")
    y = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="y")
    h = model.addVars(Np, lb=0, vtype=GRB.CONTINUOUS, name="h")
    u = model.addVars(N, Np, lb=0, vtype=GRB.CONTINUOUS, name="u")
    pq = model.addVar(vtype=GRB.CONTINUOUS, name="pq")
    py = model.addVars(N, vtype=GRB.CONTINUOUS, name="py")

    model.setObjective(k * c * q - k * pq + quicksum(r[i] for i in range(N)), GRB.MINIMIZE)

    model.addConstr(pq == quicksum(P[t] * h[t] for t in range(Np)), name="11c_Part1")
    model.addConstr(q >= dmin, name="q_range1")
    model.addConstr(q <= dmax, name="q_range2")
    for i in range(N):
        model.addConstr(y[i] >= q - demand[i], name="10b-%d" % i)
        model.addConstr(py[i] == quicksum(P[t] * u[i, t] for t in range(Np)), name="11c_Part2-%d" % i)
        model.addConstr(py[i] - s * y[i] - (1 - z[i]) * M <= r[i], name="11f_Left-%d" % i)
        model.addConstr(r[i] <= M * z[i], name="11f_Right_Part1-%d" % i)
        model.addConstr(r[i] <= py[i] - s * y[i], name="11f_Right_Part2-%d" % i)
        for t in range(Np):
            model.addConstr(y[i] + qmax * g[t] - qmax <= u[i, t], name="11e_Left-%d-%d" % (i, t))
            model.addConstr(u[i, t] <= y[i], name="11e_Right_Part1-%d-%d" % (i, t))
            model.addConstr(u[i, t] <= qmax * g[t], name="11e_Right_Part2-%d-%d" % (i, t))
    for t in range(Np):
        model.addConstr(q + qmax * g[t] - qmax <= h[t], name="11d_Left-%d" % t)
        model.addConstr(h[t] <= q, name="11d_Right_Part1-%d" % t)
        model.addConstr(h[t] <= qmax * g[t], name="11d_Right_Part2-%d" % t)

    model.optimize()
    return q.X, model.ObjVal


def initialized(Data, X, P, c, s, alpha, v, k, M=3e3):
    '''
    此函数用于进行初始化，主要是计算boundary矩阵
    :param p_min: 最小的价格
    :param Data: 历史数据集
    :param k: kNN数量
    :return:一系列的kNN
    '''
    p_min, p_max = min(P), max(P)
    N = Data.shape[0]  # 数据量
    Price = np.array(Data["price"])
    DataFeature = Data.drop(["demand", "price"], axis=1)
    Diff = []  # 记录历史数据点的context到当前的context的距离
    for i in range(N):
        Feature = np.array(DataFeature.iloc[i])
        diff = Feature - X
        diff = diff.T.dot(diff)
        Diff.append(diff)
    Diff = np.array(Diff)

    B = []  # Boundary矩阵
    for i in range(N):
        for j in range(i + 1, N):
            if Price[i] - Price[j] != 0:
                B.append((0.5 * ((Diff[i] - Diff[j]) / (Price[i] - Price[j]) + Price[i] + Price[j]), i, j))
                # 3个元素，界限，两个有关index
            else:
                B.append((-100, i, j))

    P_standard = [(num - p_min)/(p_max - p_min) for num in P]

    pmin = min(P_standard)
    D = len(P)
    for i in range(N):
        Diff[i] += (Price[i] - pmin)**2
    kNN_index = np.argsort(Diff)[:k]
    z = [1 if i in kNN_index else 0 for i in range(N)]  # 最小距离的k个点成为knn

    kNN_list = [z]
    dmin_list = [v / (P[t] - c) for t in range(D)]
    currentKNN = list(kNN_index)
    currentKNNDemand = sorted([Data["demand"][i] for i in currentKNN])
    dmax_list = [((p_min-s)*currentKNNDemand[-math.ceil(k*alpha)]-v)/(c-s)]
    sorted_B = sorted(B, key=lambda z:z[0])
    currentGateIndex = -1  # 此处的index指sorted_B的index
    for i in range(len(B)):
        if sorted_B[i][0] >= pmin:
            currentGateIndex = i  # CurrentGateIndex是当前的p往上走时碰到的第一道门
            break
    for t in range(1, D):
        p_destination = P_standard[t]  # 下一个可行p
        while sorted_B[currentGateIndex][0] < p_destination and currentGateIndex < len(B):  # 如果P或B被遍历完了就停
            # 当前面的门的两个index不全在当前kNN里时
            if not (sorted_B[currentGateIndex][1] in currentKNN and sorted_B[currentGateIndex][2] in currentKNN):
                if sorted_B[currentGateIndex][1] in currentKNN:  # 1在2不在
                    currentKNN = [sorted_B[currentGateIndex][2] if x == sorted_B[currentGateIndex][1] else x for x in
                                  currentKNN]
                    # currentKNNDemand = sorted([Data["demand"][i] for i in currentKNN])
                    currentKNNDemand.remove(Data["demand"][sorted_B[currentGateIndex][1]])
                    currentKNNDemand = insert_into_sorted_list(currentKNNDemand,
                                                               Data["demand"][sorted_B[currentGateIndex][2]])
                elif sorted_B[currentGateIndex][2] in currentKNN:  # 2在1不在
                    currentKNN = [sorted_B[currentGateIndex][1] if x == sorted_B[currentGateIndex][2] else x for x in
                                  currentKNN]
                    # currentKNNDemand = sorted([Data["demand"][i] for i in currentKNN])
                    currentKNNDemand.remove(Data["demand"][sorted_B[currentGateIndex][2]])
                    currentKNNDemand = insert_into_sorted_list(currentKNNDemand,
                                                               Data["demand"][sorted_B[currentGateIndex][1]])
                # 剩一个全不在的直接跳过
            currentGateIndex += 1
        kNN_list.append([1 if i in currentKNN else 0 for i in range(N)])
        dmax_list.append(((P[t]-s)*currentKNNDemand[-math.ceil(k*alpha)]-v)/(c-s))
    return kNN_list, dmin_list, dmax_list


# print("checkpoint1")
# knn_list, dmin_list, dmax_list = initialized(p_min, Data, k, 70)
# # print(pd.DataFrame({"dmin":dmin_list, "dmax":dmax_list}))
# print("checkpoint2")
#
# # 使用列表推导式进行筛选
# kNN_filtered, P_filtered, dmin_filtered, dmax_filtered = zip(*[
#     (knn, p, dmin, dmax) for knn, p, dmin, dmax in zip(knn_list, P, dmin_list, dmax_list) if dmax >= dmin
# ])
#
# # 将结果转换回列表
# knn_list = pd.DataFrame(list(kNN_filtered))
# P = list(P_filtered)
# dmin_list = list(dmin_filtered)
# dmax_list = list(dmax_filtered)
#
# D = len(P)
#
# unused_P = deepcopy(P)
# p0 = P[0]
# index = P.index(p0)
# unused_P.remove(p0)
# print("checkpoint3")
#
#
# #  主问题初始化
# model = Model("master")
# model.setParam("OutputFlag", 0)
# p = model.addVar(lb=p_min, vtype=GRB.CONTINUOUS, name="p")
# dmin = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="dmin")
# dmax = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="dmax")
# g = model.addVars(D, vtype=GRB.BINARY, name=["g_%d" % t for t in range(D)])
# z = model.addVars(N, vtype=GRB.BINARY, name=["x_%d" % i for i in range(N)])
# e = model.addVar(lb=-1e6, vtype=GRB.CONTINUOUS, name="e")
#
# model.setObjective(e, GRB.MINIMIZE)
#
# model.addConstr(p == quicksum(g[t] * P[t] for t in range(D)), name="p-discretization1")
# model.addConstr(quicksum(g[t] for t in range(D)) == 1, name="p-discretization2")
# model.addConstr(dmin == quicksum(g[t]*dmin_list[t] for t in range(D)), name="dmin")
# model.addConstr(dmax == quicksum(g[t]*dmax_list[t] for t in range(D)), name="dmax")
# model.addConstrs(quicksum(g[t]*knn_list.iloc[t, i] for t in range(D)) == z[i] for i in range(N))
# model.addConstr(e >= - k * (p-c) * max(Data["demand"]), "lowerBoundPerP")  # The best situation is to sell everything
# #  求解子问题、对偶子问题、添加最优割
# q0, UB = subProblem(p0, Data, k, c, s, knn=list(knn_list.iloc[index]), dmin=dmin_list[index], dmax=dmax_list[index])
# print(q0, UB)
# l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, time = lagrangian(k, c, s,
#                                         [1 if t == index else 0 for t in range(D)], p0, list(knn_list.iloc[index]),
#                                         q0, dmin_list[index], dmax_list[index], q_max, P, D, Data["demand"])
# model.addConstr(e >= - quicksum(Data["demand"][i] * l1[i] for i in range(N))
#                        + quicksum(q_max * ((g[t] - 1) * l5[t] - g[t] * l4[t]) for t in range(D))
#                        + quicksum(q_max * ((g[t] - 1) * l9[i][t] - g[t] * l8[i][t]) for t in range(D) for i in range(N))
#                        - M * quicksum(z[i] * l10[i] + (1 - z[i]) * l12[i] for i in range(N)) + dmin * l13 - dmax * l14, name="initial-optimality")
# model.Params.PoolSolutions = 10
# model.Params.PoolSearchMode = 2
# model.optimize()
# LB = e.X
# print(LB, UB, time)
#
# #  初始最优割已生成，进入循环
# iteration = 0
# while True:
#     if iteration < 9:
#         L = 10
#     else:
#         L = 1
#         model.Params.PoolSolutions = 1
#     for solutionCount in range(L):
#         model.Params.SolutionNumber = solutionCount
#         # 防止重复使用p进入子问题
#         if p.Xn not in unused_P:
#             pt = random.choice(unused_P)
#         else:
#             pt = p.Xn
#
#         index = P.index(pt)
#         unused_P.remove(pt)
#
#         # 求子问题，对偶子问题，加最优割
#         q, sub_obj = subProblem(pt, Data, k, c, s, knn=list(knn_list.iloc[index]), dmin=dmin_list[index], dmax=dmax_list[index])
#         UB = min(UB, sub_obj)
#         l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, time = lagrangian(k, c, s,
#                                         [1 if t == index else 0 for t in range(D)], pt, list(knn_list.iloc[index]),
#                                         q, dmin_list[index], dmax_list[index], q_max, P, D, Data["demand"])
#         model.addConstr(e >= - quicksum(Data["demand"][i] * l1[i] for i in range(N))
#                            + quicksum(q_max * ((g[t] - 1) * l5[t] - g[t] * l4[t]) for t in range(D))
#                            + quicksum(q_max * ((g[t] - 1) * l9[i][t] - g[t] * l8[i][t]) for t in range(D) for i in range(N))
#                            - M * quicksum(z[i] * l10[i] + (1 - z[i]) * l12[i] for i in range(N)) + dmin * l13 - dmax * l14, name="optimality_%d" % iteration)
#
#     model.optimize()
#     LB = max(LB, e.X)
#     print(iteration, LB, UB, p.X)
#     iteration += 1
#     if UB <= LB + 10:
#         break
#
# q, UB = subProblem(p.X, Data, k, c, s, [z[i].X for i in range(N)], dmin.X, dmax.X)
# print()
# print("最优解的p=" + str(p.X) + ", q=" + str(q))
# print("最优解为" + str(model.ObjVal))

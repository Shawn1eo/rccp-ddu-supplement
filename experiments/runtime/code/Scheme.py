#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/11/28 20:26
# @Author: WANG SHENGRAN, LIU XIANGTING
from copy import deepcopy
from SimplifiedBD import subProblem, subProblem_basic
from DataGeneration import l
from gurobipy import *
from time import time
import pandas as pd
import numpy as np
import math
from SimplifiedBD import initialized
from dualsubproblem_accelerate import lagrangian
import random
from collections import Counter
from dualsubproblem_basic import lagrangian_basic


class HonestDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = {}
        self.node_count = 0  # 用于生成唯一的节点编号

    def _find_best_split(self, X):
        """基于特征分布（无目标变量）找最佳分割点"""
        best_feature, best_value = None, None
        min_variance = float('inf')

        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]

            # 连续特征：选择近似中位数
            if len(np.unique(values)) > 10:  # 粗略判断连续特征
                sorted_vals = np.sort(values)
                mid_idx = len(sorted_vals) // 2
                candidate = sorted_vals[mid_idx]

                # 计算左右子节点方差
                left_mask = values <= candidate
                right_mask = values > candidate
                var_left = np.var(X[left_mask], axis=0).sum()
                var_right = np.var(X[right_mask], axis=0).sum()
                total_variance = var_left + var_right

                if total_variance < min_variance:
                    min_variance = total_variance
                    best_feature = feature_idx
                    best_value = candidate

        return best_feature, best_value

    def _build_tree(self, X, depth=0):
        """递归构建决策树"""
        n_samples = X.shape[0]

        # 停止条件：样本不足或达到最大深度
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            self.node_count += 1  # 为叶节点分配编号
            return {"node_id": self.node_count, "samples": n_samples}  # 叶节点存储样本量和编号

        # 寻找最佳分割点
        feature, value = self._find_best_split(X)
        if feature is None:
            self.node_count += 1  # 为叶节点分配编号
            return {"node_id": self.node_count, "samples": n_samples}

        # 根据分割点划分数据
        left_mask = X[:, feature] <= value
        right_mask = X[:, feature] > value

        # 递归构建子树
        left_subtree = self._build_tree(X[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], depth + 1)

        return {
            "feature": feature,
            "value": value,
            "left": left_subtree,
            "right": right_subtree,
            "samples": n_samples
        }

    def fit(self, X):
        self.tree = self._build_tree(np.array(X))

    def print_tree(self, node=None, indent=""):
        """打印树结构（可视化）"""
        if node is None:
            node = self.tree

        if "feature" in node:
            print(f"{indent}Feature {node['feature']} <= {node['value']:.2f} [samples={node['samples']}]")
            self.print_tree(node["left"], indent + "  ")
            self.print_tree(node["right"], indent + "  ")
        else:
            print(f"{indent}Leaf [samples={node['samples']}]")

    def predict_leaf(self, x):
        """预测单个点落入的叶节点编号"""

        def _predict_leaf(node, x):
            if "node_id" in node:  # 检查是否为叶节点
                return node["node_id"]
            if x[node["feature"]] <= node["value"]:
                return _predict_leaf(node["left"], x)
            else:
                return _predict_leaf(node["right"], x)

        return _predict_leaf(self.tree, x)


def CART_initial(Data, x, Z):
    """
    定义一个CART_initial函数，输入参数为Data, x, Z。
    Data表示历史数据集，其中包括N个历史数据点，每个历史数据点的组成包括x（contextual information）、z（价格）和demand。
    Z是决策变量z的取值范围。

    :param Data: 历史数据集，一个N×(M+2)的numpy数组，其中N是数据点数量，M是contextual information的维度。
    :param x: 当前上下文信息，一个M维的numpy数组。
    :param Z: 决策变量z的取值范围，一个Np维的numpy数组。
    :return: 一个Np×N的numpy数组，每一行代表一个z取值下对应的相同叶节点历史数据点标记。
    """
    N = Data.shape[0]  # 历史数据点的数量
    M = Data.shape[1] - 2  # contextual information的维度
    Np = len(Z)  # Z的规模大小

    # 提取特征和目标变量
    X = Data[:, :-1]  # 特征：x和z
    # y = Data[:, -1]   # 目标：demand

    # 训练决策树模型
    tree = HonestDecisionTree(min_samples_split=math.ceil(N ** 0.5))
    tree.fit(X)
    tree.print_tree()

    # 初始化输出数组
    result = np.zeros((Np, N), dtype=int)

    # 遍历Z中的每一个取值
    for i, z in enumerate(Z):
        # 构造查询点
        query_point = np.concatenate((x, [z]))

        # 预测查询点落入的叶节点
        leaf_node = tree.predict_leaf(query_point)

        # 找出所有落入相同叶节点的历史数据点
        for j in range(N):
            if tree.predict_leaf(X[j]) == leaf_node:
                result[i, j] = 1

    return result


def scheme1(Data, X, P, c, s, alpha, v, qmax, M=3e5):
    """
    :param M: a big number, default to be 3e5
    :param Data: generated data, including price and demand
    :param X: generated contextual information
    :param P: range of price
    :param c: unit cost
    :param s: unit salvage price
    :param alpha: ragarding chance constraint
    :param v: regarding chance constraint
    :param qmax: maximum quantity
    :return: operation time
    """

    demand = Data["demand"]  # demand: a list of demand recorded in the historical data
    p_history = Data["price"] # p_history: unit price in historical data
    x_history = Data.drop(["demand", "price"], axis=1)  # x_history: contextual information in historical data
    N = len(demand)
    Np = len(P)
    k = math.ceil(2 * math.sqrt(N))
    k_alpha = math.ceil(k * alpha)

    model = Model("LinearizedReformulation")
    model.setParam("OutputFlag", 0)
    model.setParam('DualReductions', 0)

    # Define variables
    p = model.addVar(lb=max(s, P[0]), ub=P[Np - 1], vtype=GRB.CONTINUOUS, name="p")
    q = model.addVar(lb=0, ub=qmax, vtype=GRB.CONTINUOUS, name="q")
    r = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="r")
    y = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="y")
    h = model.addVars(Np, lb=0, vtype=GRB.CONTINUOUS, name="h")
    u = model.addVars(N, Np, lb=0, vtype=GRB.CONTINUOUS, name="u")
    z = model.addVars(N, vtype=GRB.BINARY, name="z")
    g = model.addVars(Np, vtype=GRB.BINARY, name="g")
    pq = model.addVar(vtype=GRB.CONTINUOUS, name="pq")
    py = model.addVars(N, vtype=GRB.CONTINUOUS, name="py")
    gamma = model.addVars(N, vtype=GRB.BINARY, name="gamma")
    w = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="w")

    # Set objective
    model.setObjective(k * c * q - k * pq + quicksum(r[i] for i in range(N)), GRB.MINIMIZE)

    # Define constraints
    model.addConstr(quicksum(z[i] for i in range(N)) == k, name="10c")
    model.addConstr(quicksum(g[t] for t in range(Np)) == 1, name="11b_Part1")
    model.addConstr(p == quicksum(P[t] * g[t] for t in range(Np)), name="11b_Part2")
    model.addConstr(pq == quicksum(P[t] * h[t] for t in range(Np)), name="11c_Part1")
    model.addConstr(quicksum(w[i] for i in range(N)) <= k_alpha, name="11h_Part2")
    for i in range(N):
        model.addConstr(y[i] >= q - demand[i], name="10b-%d" % i)
        model.addConstr(py[i] == quicksum(P[t] * u[i, t] for t in range(Np)), name="11c_Part2-%d" % i)
        model.addConstr(py[i] - s * y[i] - (1 - z[i]) * M <= r[i], name="11f_Left-%d" % i)
        model.addConstr(r[i] <= M * z[i], name="11f_Right_Part1-%d" % i)
        model.addConstr(r[i] <= py[i] - s * y[i], name="11f_Right_Part2-%d" % i)
        model.addConstr(pq - c * q - py[i] + s * y[i] + M * gamma[i] >= v, name="11g-%d" % i)
        model.addConstr(z[i] + gamma[i] - 1 <= w[i], name="11h_Part1_Left-%d" % i)
        model.addConstr(w[i] <= z[i], name="11h_Part1_Right1-%d" % i)
        model.addConstr(w[i] <= gamma[i], name="11h_Part1_Right2-%d" % i)
        for j in range(N):
            if j == i:
                continue
            else:
                model.addConstr((p - p_history[i]) ** 2 - (p - p_history[j]) ** 2
                                + quicksum((X[m] - x_history.iloc[i, m]) ** 2 for m in range(x_history.shape[1]))
                                - quicksum((X[m] - x_history.iloc[j, m]) ** 2 for m in range(x_history.shape[1]))
                                <= M * (z[j] - z[i] + 1), name="10d-%d-%d" % (i, j))
        for t in range(Np):
            model.addConstr(y[i] + qmax * g[t] - qmax <= u[i, t], name="11e_Left-%d-%d" % (i, t))
            model.addConstr(u[i, t] <= y[i], name="11e_Right_Part1-%d-%d" % (i, t))
            model.addConstr(u[i, t] <= qmax * g[t], name="11e_Right_Part2-%d-%d" % (i, t))
    for t in range(Np):
        model.addConstr(q + qmax * g[t] - qmax <= h[t], name="11d_Left-%d" % t)
        model.addConstr(h[t] <= q, name="11d_Right_Part1-%d" % t)
        model.addConstr(h[t] <= qmax * g[t], name="11d_Right_Part2-%d" % t)

    # Record time
    time0 = time()
    model.optimize()
    # print(model.status)
    # obj = model.ObjVal
    # print(obj)
    time1 = time() - time0
    # print(time1)

    return time1


def scheme2(Data, X, P, c, s, alpha, v, qmax, M=3e5):
    """
    :param M: a big number, default to be 3e5
    :param Data: generated data, including price and demand
    :param X: generated contextual information
    :param P: range of price
    :param c: unit cost
    :param s: unit salvage price
    :param alpha: ragarding chance constraint
    :param v: regarding chance constraint
    :param qmax: maximum quantity
    :return: operation time
    """

    demand = Data["demand"]  # demand: a list of demand recorded in the historical data
    p_history = Data["price"] # p_history: unit price in historical data
    x_history = Data.drop(["demand", "price"], axis=1)  # x_history: contextual information in historical data
    N = len(demand)
    Np = len(P)
    k = math.ceil(2 * math.sqrt(N))
    knn_list, dmin_list, dmax_list = initialized(Data, X, P, c, s, alpha, v, k)

    model = Model("LinearizedReformulation")
    model.setParam("OutputFlag", 0)
    model.setParam('DualReductions', 0)

    # Define variables
    p = model.addVar(lb=max(s, P[0]), ub=P[Np - 1], vtype=GRB.CONTINUOUS, name="p")
    q = model.addVar(lb=0, ub=qmax, vtype=GRB.CONTINUOUS, name="q")
    r = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="r")
    y = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="y")
    h = model.addVars(Np, lb=0, vtype=GRB.CONTINUOUS, name="h")
    u = model.addVars(N, Np, lb=0, vtype=GRB.CONTINUOUS, name="u")
    z = model.addVars(N, vtype=GRB.BINARY, name="z")
    g = model.addVars(Np, vtype=GRB.BINARY, name="g")
    pq = model.addVar(vtype=GRB.CONTINUOUS, name="pq")
    py = model.addVars(N, vtype=GRB.CONTINUOUS, name="py")
    dmin = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="dmin")
    dmax = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="dmax")

    # Set objective
    model.setObjective(k * c * q - k * pq + quicksum(r[i] for i in range(N)), GRB.MINIMIZE)

    # Define constraints
    model.addConstr(quicksum(z[i] for i in range(N)) == k, name="10c")
    model.addConstr(quicksum(g[t] for t in range(Np)) == 1, name="11b_Part1")
    model.addConstr(p == quicksum(P[t] * g[t] for t in range(Np)), name="11b_Part2")
    model.addConstr(pq == quicksum(P[t] * h[t] for t in range(Np)), name="11c_Part1")
    model.addConstr(q >= dmin, name="q_range1")
    model.addConstr(q <= dmax, name="q_range2")
    for i in range(N):
        model.addConstr(y[i] >= q - demand[i], name="10b-%d" % i)
        model.addConstr(py[i] == quicksum(P[t] * u[i, t] for t in range(Np)), name="11c_Part2-%d" % i)
        model.addConstr(py[i] - s * y[i] - (1 - z[i]) * M <= r[i], name="11f_Left-%d" % i)
        model.addConstr(r[i] <= M * z[i], name="11f_Right_Part1-%d" % i)
        model.addConstr(r[i] <= py[i] - s * y[i], name="11f_Right_Part2-%d" % i)
        model.addConstr(dmin == quicksum(g[t] * dmin_list[t] for t in range(Np)), name="dmin")
        model.addConstr(dmax == quicksum(g[t] * dmax_list[t] for t in range(Np)), name="dmax")
        for j in range(N):
            if j == i:
                continue
            else:
                model.addConstr((p - p_history[i]) ** 2 - (p - p_history[j]) ** 2
                                + quicksum((X[m] - x_history.iloc[i, m]) ** 2 for m in range(x_history.shape[1]))
                                - quicksum((X[m] - x_history.iloc[j, m]) ** 2 for m in range(x_history.shape[1]))
                                <= M * (z[j] - z[i] + 1), name="10d-%d-%d" % (i, j))
        for t in range(Np):
            model.addConstr(y[i] + qmax * g[t] - qmax <= u[i, t], name="11e_Left-%d-%d" % (i, t))
            model.addConstr(u[i, t] <= y[i], name="11e_Right_Part1-%d-%d" % (i, t))
            model.addConstr(u[i, t] <= qmax * g[t], name="11e_Right_Part2-%d-%d" % (i, t))
    for t in range(Np):
        model.addConstr(q + qmax * g[t] - qmax <= h[t], name="11d_Left-%d" % t)
        model.addConstr(h[t] <= q, name="11d_Right_Part1-%d" % t)
        model.addConstr(h[t] <= qmax * g[t], name="11d_Right_Part2-%d" % t)

    # Record time
    time0 = time()
    model.optimize()
    # print(model.status)
    # obj = model.ObjVal
    # print(obj)
    time1 = time() - time0
    # print(time1)

    return time1


def scheme3(Data, X, P, c, s, alpha, v, qmax, M=3e5):
    demand = Data["demand"]  # demand: a list of demand recorded in the historical data
    N = len(demand)
    k = math.ceil(2 * math.sqrt(N))
    # np.random.seed(1)

    TotalTime = 0
    timestamp = time()
    knn_list, dmin_list, dmax_list = initialized(Data, X, P, c, s, alpha, v, k)
    TotalTime += time() - timestamp
    # print(pd.DataFrame({"dmin":dmin_list, "dmax":dmax_list}))
    # print("checkpoint2")

    feasibility = 0
    for i in range(len(dmax_list)):
        if dmax_list[i] >= dmin_list[i]:
            feasibility = 1
            break

    if feasibility == 0:
        return TotalTime

    # 使用列表推导式进行筛选
    kNN_filtered, P_filtered, dmin_filtered, dmax_filtered = zip(*[
        (knn, p, dmin, dmax) for knn, p, dmin, dmax in zip(knn_list, P, dmin_list, dmax_list) if dmax >= dmin
    ])

    # 将结果转换回列表
    knn_list = pd.DataFrame(list(kNN_filtered))
    P = list(P_filtered)
    dmin_list = list(dmin_filtered)
    dmax_list = list(dmax_filtered)

    D = len(P)
    p_min = min(P)

    unused_P = deepcopy(P)
    p0 = P[0]
    index = P.index(p0)
    unused_P.remove(p0)
    # print("checkpoint3")

    #  主问题初始化
    model = Model("master")
    model.setParam("OutputFlag", 0)
    p = model.addVar(lb=p_min, vtype=GRB.CONTINUOUS, name="p")
    dmin = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="dmin")
    dmax = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="dmax")
    g = model.addVars(D, vtype=GRB.BINARY, name=["g_%d" % t for t in range(D)])
    z = model.addVars(N, vtype=GRB.BINARY, name=["x_%d" % i for i in range(N)])
    e = model.addVar(lb=-1e6, vtype=GRB.CONTINUOUS, name="e")

    model.setObjective(e, GRB.MINIMIZE)

    model.addConstr(p == quicksum(g[t] * P[t] for t in range(D)), name="p-discretization1")
    model.addConstr(quicksum(g[t] for t in range(D)) == 1, name="p-discretization2")
    model.addConstr(dmin == quicksum(g[t] * dmin_list[t] for t in range(D)), name="dmin")
    model.addConstr(dmax == quicksum(g[t] * dmax_list[t] for t in range(D)), name="dmax")
    model.addConstrs(quicksum(g[t] * knn_list.iloc[t, i] for t in range(D)) == z[i] for i in range(N))
    model.addConstr(e >= - k * (p - c) * max(Data["demand"]), "lowerBoundPerP")
    #  求解子问题、对偶子问题、添加最优割
    timestamp = time()
    q0, UB = subProblem_basic(p0, Data, P, k, c, s, list(knn_list.iloc[index]), dmin_list[index], dmax_list[index], qmax)
    TotalTime += time() - timestamp
    # print(q0, UB)
    l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, ltime = lagrangian_basic(k, c, s,
                                                                                    [1 if t == index else 0 for t in
                                                                                     range(D)], list(knn_list.iloc[index]),
                                                                                    dmin_list[index], dmax_list[index],
                                                                                    qmax, P, D, Data["demand"])
    TotalTime += ltime
    model.addConstr(e >= - quicksum(Data["demand"][i] * l1[i] for i in range(N))
                    + quicksum(qmax * ((g[t] - 1) * l5[t] - g[t] * l4[t]) for t in range(D))
                    + quicksum(qmax * ((g[t] - 1) * l9[i][t] - g[t] * l8[i][t]) for t in range(D) for i in range(N))
                    - M * quicksum(z[i] * l10[i] + (1 - z[i]) * l12[i] for i in range(N)) + dmin * l13 - dmax * l14,
                    name="initial-optimality")
    timestamp = time()
    model.optimize()
    TotalTime += time() - timestamp
    LB = e.X

    #  初始最优割已生成，进入循环
    iteration = 0
    while True:
        L = 1
        for solutionCount in range(L):
            # 防止重复使用p进入子问题
            if p.X not in unused_P:
                pt = random.choice(unused_P)
            else:
                pt = p.X

            index = P.index(pt)
            unused_P.remove(pt)

            # 求子问题，对偶子问题，加最优割
            timestamp = time()
            q, sub_obj = subProblem_basic(pt, Data, P, k, c, s, list(knn_list.iloc[index]), dmin_list[index],
                                    dmax_list[index], qmax)
            TotalTime += time() - timestamp
            UB = min(UB, sub_obj)
            l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, ltime = lagrangian_basic(k, c, s,
                                                                                            [1 if t == index else 0 for
                                                                                             t in range(D)],
                                                                                            list(knn_list.iloc[index]),
                                                                                            dmin_list[index],
                                                                                            dmax_list[index], qmax, P,
                                                                                            D, Data["demand"])
            TotalTime += ltime
            model.addConstr(e >= - quicksum(Data["demand"][i] * l1[i] for i in range(N))
                            + quicksum(qmax * ((g[t] - 1) * l5[t] - g[t] * l4[t]) for t in range(D))
                            + quicksum(
                qmax * ((g[t] - 1) * l9[i][t] - g[t] * l8[i][t]) for t in range(D) for i in range(N))
                            - M * quicksum(
                z[i] * l10[i] + (1 - z[i]) * l12[i] for i in range(N)) + dmin * l13 - dmax * l14,
                            name="optimality_%d" % iteration)

        timestamp = time()
        model.optimize()
        TotalTime += time() - timestamp
        LB = max(LB, e.X)
        # print(TotalTime, iteration, LB, UB, p.X)
        iteration += 1
        if UB <= LB + 10 or len(unused_P) == 0:
            break
    return TotalTime


def scheme4(Data, X, P, c, s, alpha, v, qmax, M=3e5):
    demand = Data["demand"]  # demand: a list of demand recorded in the historical data
    N = len(demand)
    k = math.ceil(0.5 * math.sqrt(N))

    TotalTime = 0
    timestamp = time()
    knn_list, dmin_list, dmax_list = initialized(Data, X, P, c, s, alpha, v, k)
    TotalTime += time() - timestamp
    # print(pd.DataFrame({"dmin":dmin_list, "dmax":dmax_list}))
    # print("checkpoint2")

    feasibility = 0
    for i in range(len(dmax_list)):
        if dmax_list[i] >= dmin_list[i]:
            feasibility = 1
            break

    if feasibility == 0:
        return TotalTime

    # 使用列表推导式进行筛选
    kNN_filtered, P_filtered, dmin_filtered, dmax_filtered = zip(*[
        (knn, p, dmin, dmax) for knn, p, dmin, dmax in zip(knn_list, P, dmin_list, dmax_list) if dmax >= dmin
    ])

    # 将结果转换回列表
    knn_list = pd.DataFrame(list(kNN_filtered))
    P = list(P_filtered)
    dmin_list = list(dmin_filtered)
    dmax_list = list(dmax_filtered)

    D = len(P)
    p_min = min(P)

    unused_P = deepcopy(P)
    p0 = P[0]
    index = P.index(p0)
    unused_P.remove(p0)
    # print("checkpoint3")

    #  主问题初始化
    model = Model("master")
    model.setParam("OutputFlag", 0)
    p = model.addVar(lb=p_min, vtype=GRB.CONTINUOUS, name="p")
    dmin = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="dmin")
    dmax = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="dmax")
    g = model.addVars(D, vtype=GRB.BINARY, name=["g_%d" % t for t in range(D)])
    z = model.addVars(N, vtype=GRB.BINARY, name=["x_%d" % i for i in range(N)])
    e = model.addVar(lb=-1e6, vtype=GRB.CONTINUOUS, name="e")

    model.setObjective(e, GRB.MINIMIZE)

    model.addConstr(p == quicksum(g[t] * P[t] for t in range(D)), name="p-discretization1")
    model.addConstr(quicksum(g[t] for t in range(D)) == 1, name="p-discretization2")
    model.addConstr(dmin == quicksum(g[t] * dmin_list[t] for t in range(D)), name="dmin")
    model.addConstr(dmax == quicksum(g[t] * dmax_list[t] for t in range(D)), name="dmax")
    model.addConstrs(quicksum(g[t] * knn_list.iloc[t, i] for t in range(D)) == z[i] for i in range(N))
    model.addConstr(e >= - k * (p - c) * max(Data["demand"]),
                    "lowerBoundPerP")  # The best situation is to sell everything
    #  求解子问题、对偶子问题、添加最优割
    timestamp = time()
    q0, UB = subProblem(p0, Data, k, c, s, knn=list(knn_list.iloc[index]), dmin=dmin_list[index], dmax=dmax_list[index])
    TotalTime += time() - timestamp
    # print(q0, UB)
    l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, ltime = lagrangian(k, c, s,
                                                                                    [1 if t == index else 0 for t in
                                                                                     range(D)], p0,
                                                                                    list(knn_list.iloc[index]),
                                                                                    q0, dmin_list[index],
                                                                                    dmax_list[index], qmax, P, D,
                                                                                    Data["demand"])
    TotalTime += ltime
    model.addConstr(e >= - quicksum(Data["demand"][i] * l1[i] for i in range(N))
                    + quicksum(qmax * ((g[t] - 1) * l5[t] - g[t] * l4[t]) for t in range(D))
                    + quicksum(qmax * ((g[t] - 1) * l9[i][t] - g[t] * l8[i][t]) for t in range(D) for i in range(N))
                    - M * quicksum(z[i] * l10[i] + (1 - z[i]) * l12[i] for i in range(N)) + dmin * l13 - dmax * l14,
                    name="initial-optimality")
    model.Params.PoolSolutions = 10
    model.Params.PoolSearchMode = 2
    timestamp = time()
    model.optimize()
    TotalTime += time() - timestamp
    LB = e.X

    #  初始最优割已生成，进入循环
    iteration = 0
    while True:
        if iteration < D / 10 - 2:
            L = 10
        else:
            L = 1
            model.Params.PoolSolutions = 1
        for solutionCount in range(L):
            model.Params.SolutionNumber = solutionCount
            # 防止重复使用p进入子问题
            if p.Xn not in unused_P:
                pt = random.choice(unused_P)
            else:
                pt = p.Xn

            index = P.index(pt)
            unused_P.remove(pt)

            # 求子问题，对偶子问题，加最优割
            timestamp = time()
            q, sub_obj = subProblem(pt, Data, k, c, s, knn=list(knn_list.iloc[index]), dmin=dmin_list[index],
                                    dmax=dmax_list[index])
            TotalTime += time() - timestamp
            UB = min(UB, sub_obj)
            l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, ltime = lagrangian(k, c, s,
                                                                                            [1 if t == index else 0 for
                                                                                             t in range(D)], pt,
                                                                                            list(knn_list.iloc[index]),
                                                                                            q, dmin_list[index],
                                                                                            dmax_list[index], qmax, P,
                                                                                            D, Data["demand"])
            TotalTime += ltime
            model.addConstr(e >= - quicksum(Data["demand"][i] * l1[i] for i in range(N))
                            + quicksum(qmax * ((g[t] - 1) * l5[t] - g[t] * l4[t]) for t in range(D))
                            + quicksum(
                qmax * ((g[t] - 1) * l9[i][t] - g[t] * l8[i][t]) for t in range(D) for i in range(N))
                            - M * quicksum(
                z[i] * l10[i] + (1 - z[i]) * l12[i] for i in range(N)) + dmin * l13 - dmax * l14,
                            name="optimality_%d" % iteration)

        timestamp = time()
        model.optimize()
        TotalTime += time() - timestamp
        LB = max(LB, e.X)
        print(TotalTime, iteration, LB, UB, p.X)
        iteration += 1
        if UB <= LB + 10 or len(unused_P) == 0:
            break

    # index = P.index(p.X)
    q, obj = subProblem(p.X, Data, k, c, s, knn=[z[i].X for i in range(N)], dmin=dmin.X, dmax=dmax.X)
    return TotalTime, p.X, q, obj/k


# 以下三个走的枚举，目的是加速咱们的整体运算效率（难绷）

def scheme4_lsa(Data, X, P, c, s, alpha, v, q_min, q_max, M=3e5):
    demand = Data["demand"]  # demand: a list of demand recorded in the historical data
    DataFeature = Data.drop(['demand', 'price'], axis=1)
    Diff0 = DataFeature - X
    Dist0 = (Diff0 ** 2).sum(axis=1)  # 不考虑p的时候先算一版距离
    N = len(demand)
    h = 50 * N**(-0.2)

    Obj = 1e5
    pStar = 0
    qStar = 0
    for p in P:
        Dist1 = (Data["price"] - p)**2
        Dist = Dist0 + Dist1

        lsa = Dist[Dist <= h ** 2].index
        # print(N, len(lsa))
        lsa_D = Data["demand"][lsa]
        lsa_p = Data["price"][lsa]
        qmin = max(v / (p - c), q_min)
        qmax = min(((p - s) * sorted(lsa_D)[-math.ceil(len(lsa) * alpha)] - v) / (c - s), q_max)

        if qmin > qmax:
            continue

        else:
            N_star = math.ceil(len(lsa) * (p - c) / (p - s))
            qstar = min(qmax, max(qmin, sorted(lsa_D)[N_star - 1]))
            obj = 1/len(lsa) * sum(-(p - c) * qstar + (p - s) * max(qstar - di, 0) for di in lsa_D)
            if obj < Obj:
                pStar = p
                qStar = qstar
                Obj = obj

    return pStar, qStar, Obj



def scheme4_knn(Data, X, P, c, s, alpha, v, q_min, q_max, M=3e5):
    demand = Data["demand"]  # demand: a list of demand recorded in the historical data
    DataFeature = Data.drop(['demand', 'price'], axis=1)
    Diff0 = DataFeature - X
    Dist0 = (Diff0 ** 2).sum(axis=1)  # 不考虑p的时候先算一版距离
    N = len(demand)
    k = math.ceil(N ** 0.5)  # Case Study

    Obj = 1e5
    pStar = 0
    qStar = 0
    for p in P:
        Dist1 = (Data["price"] - p)**2
        Dist = Dist0 + Dist1

        knn = Dist.nsmallest(k).index
        knn_D = Data["demand"][knn]
        knn_p = Data["price"][knn]
        qmin = max(v / (p - c), q_min)
        qmax = min(((p-s)*sorted(knn_D)[-math.ceil(k*alpha)]-v)/(c-s), q_max)

        if qmin > qmax:
            print("YYYYYYYYYYYYYYYYYYYYYYYYY")
            continue

        else:
            N_star = math.ceil(k * (p - c) / (p - s))
            qstar = min(qmax, max(qmin, sorted(knn_D)[N_star - 1]))
            obj = 1/k * sum(-(p - c) * qstar + (p - s) * max(qstar - di, 0) for di in knn_D)
            if obj < Obj:
                pStar = p
                qStar = qstar
                Obj = obj

    return pStar, qStar, Obj


def scheme4_cart(Data, X, P, c, s, alpha, v, q_min, q_max, M=3e5):
    demand = Data["demand"]  # demand: a list of demand recorded in the historical data
    data = Data.to_numpy()
    Cart = CART_initial(data, X, P)

    Obj = 1e5
    pStar = 0
    qStar = 0
    for t in range(len(P)):
        cart = np.nonzero(Cart[t] == 1)[0]
        cart_D = list(Data["demand"][cart])
        cart_p = list(Data["price"][cart])
        qmin = max(v / (P[t] - c), q_min)
        qmax = min(((P[t] - s) * sorted(cart_D)[-math.ceil(len(cart) * alpha)] - v) / (c - s), q_max)

        if qmin > qmax:
            continue

        else:
            N_star = math.ceil(len(cart) * (P[t] - c) / (P[t] - s))
            qstar = min(qmax, max(qmin, sorted(cart_D)[N_star - 1]))
            obj = 1/len(cart) * sum(-(P[t] - c) * qstar + (P[t] - s) * max(qstar - di, 0) for di in cart_D)
            if obj < Obj:
                pStar = P[t]
                qStar = qstar
                Obj = obj

    return pStar, qStar, Obj


#
#
# M = 3e3
# alpha = 0.7
# p_min = 10
# q_max = 100
# c = 5
# s = 2
# D = 200
# v = 100
# P = [p_min + 0.1 * t for t in range(D)]  # All possible discrete values of p
# cov = [[0.5 ** abs(i - j) for i in range(10)] for j in range(10)]
# mean = [0 for i in range(10)]
# np.random.seed(42)
# X = np.random.multivariate_normal(mean, cov, 1)[0]
# Data = pd.read_excel("Data.xlsx")
#
# print(scheme4_cart(Data, X, P, c, s, alpha, v, q_max))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/11/26 15:48
# @Author: WANG SHENGRAN
from gurobipy import *
from time import time
import pandas as pd
import numpy as np


M = 3e5
# D = 200
# P = [10 + 0.1*t for t in range(D)]  # All possible discrete values of p


def lagrangian(k, c, s, g, p, z, q, d_min, d_max, q_max, P, Np, demand):
    """
    The function to calculate all the lagrangian for the sub-problem
    :param k: key parameter to kNN
    :param c: unit cost
    :param p: unit price
    :param s: unit salvage price
    :param z: member of knn, list of N
    :param d_min: the left boundary of d
    :param d_max: the right boundary of d
    :param q: order quantity
    :param q_max: maximum quantity
    :param P: number of possible values for p
    :param Np: number of possible values for p
    :param demand: a list of demand recorded in the historical data
    :return: a list of lagrangian from l1 to l12
    """
    N = len(demand)
    y = [max(q - demand[i], 0) for i in range(N)]
    r = [z[i] * y[i] * (p - s) for i in range(N)]
    h = [0 if g[t] == 0 else q for t in range(Np)]
    u = [[0 if g[t] == 0 else y[i] for t in range(Np)] for i in range(N)]

    model = Model("Laplacian")
    model.setParam("OutputFlag", 0)
    model.setParam("DualReductions", 0)

    # Define all lagrangian
    l1 = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="l1")
    l2 = model.addVar(lb=-1e6, vtype=GRB.CONTINUOUS, name="l2")
    l3 = model.addVars(Np, lb=0, vtype=GRB.CONTINUOUS, name="l3")
    l4 = model.addVars(Np, lb=0, vtype=GRB.CONTINUOUS, name="l4")
    l5 = model.addVars(Np, lb=0, vtype=GRB.CONTINUOUS, name="l5")
    l6 = model.addVars(N, lb=-1e6, vtype=GRB.CONTINUOUS, name="l6")
    l7 = model.addVars(N, Np, lb=0, vtype=GRB.CONTINUOUS, name="l7")
    l8 = model.addVars(N, Np, lb=0, vtype=GRB.CONTINUOUS, name="l8")
    l9 = model.addVars(N, Np, lb=0, vtype=GRB.CONTINUOUS, name="l9")
    l10 = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="l10")
    l11 = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="l11")
    l12 = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="l12")
    l13 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="l13")
    l14 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="l14")

    model.setObjective(- quicksum(demand[i] * l1[i] for i in range(N))
                       + quicksum(q_max * ((g[t] - 1) * l5[t] - g[t] * l4[t]) for t in range(Np))
                       + quicksum(q_max * ((g[t] - 1) * l9[i, t] - g[t] * l8[i, t]) for t in range(Np) for i in range(N))
                       - M * quicksum(z[i] * l10[i] + (1 - z[i]) * l12[i] for i in range(N)) + d_min * l13 - d_max * l14
                       , GRB.MAXIMIZE)

    # define original constraints (combines complementary slackness)
    if q == 0:
        model.addConstr(
            quicksum(l3[t] - l5[t] for t in range(Np)) - quicksum(l1[i] for i in range(N)) + l13 - l14 <= k * c,
            name="dual-q")  # a
        model.addConstr(l2 <= -k, name="dual-pq")  # b
    else:
        model.addConstr(
            quicksum(l3[t] - l5[t] for t in range(Np)) - quicksum(l1[i] for i in range(N)) + l13 - l14 == k * c,
            name="dual-q")
        model.addConstr(l2 == -k, name="dual-pq")

    for i in range(N):
        if r[i] == 0:
            model.addConstr(-l10[i] - l11[i] + l12[i] <= 1, name="dual-r%d" % i)  # c
        else:
            model.addConstr(-l10[i] - l11[i] + l12[i] == 1, name="dual-r%d" % i)

        if y[i] == 0:
            model.addConstr(s * (l12[i] - l11[i]) + quicksum(l7[i, t] - l9[i, t] for t in range(Np)) + l1[i] <= 0,
                            name="dual-y%d" % i)  # d
            model.addConstr(l6[i] - l12[i] + l11[i] <= 0, name="dual-py%d" % i)  # e
        else:
            model.addConstr(s * (l12[i] - l11[i]) + quicksum(l7[i, t] - l9[i, t] for t in range(Np)) + l1[i] == 0,
                            name="dual-y%d" % i)
            model.addConstr(l6[i] - l12[i] + l11[i] == 0, name="dual-py%d" % i)

        for t in range(Np):
            if u[i][t] == 0:
                model.addConstr(-P[t] * l6[i] - l7[i, t] - l8[i, t] + l9[i, t] <= 0,
                                name="dual-u_%d_%d" % (i, t))  # g
            else:
                model.addConstr(-P[t] * l6[i] - l7[i, t] - l8[i, t] + l9[i, t] == 0,
                                name="dual-u_%d_%d" % (i, t))

    for t in range(Np):
        if h[t] == 0:
            model.addConstr(-P[t] * l2 - l3[t] - l4[t] + l5[t] <= 0,
                            name="dual-h_%d" % t)  # f
        else:
            model.addConstr(-P[t] * l2 - l3[t] - l4[t] + l5[t] == 0,
                            name="dual-h_%d" % t)

    # fix some laplacian from complementary slackness
    for i in range(N):
        if z[i] == 1:
            model.addConstr(l10[i] == 0, name="fix-l10_%d" % i)
        else:
            model.addConstr(l12[i] == 0, name="fix-l12_%d" % i)
        model.addConstr(l11[i] == 0, name="fix-l11_%d" % i)

        if y[i] - q + demand[i] > 1e-3:
            model.addConstr(l1[i] == 0, name="fix-l1_%d" % i)

    for t in range(Np):
        if g[t] == 0:
            model.addConstr(l5[t] == 0, name="fix-l5_%d" % t)
            if q > 0:
                model.addConstr(l3[t] == 0, name="fix-l3_%d" % t)
        else:
            model.addConstr(l4[t] == 0, name="fix-l4_%d" % t)

        for i in range(N):
            if g[t] == 0:
                model.addConstr(l9[i, t] == 0, name="fix-l9_%d_%d" % (i, t))
                if y[i] > 0:
                    model.addConstr(l7[i, t] == 0, name="fix-l7_%d_%d" % (i, t))
            else:
                model.addConstr(l8[i, t] == 0, name="fix-l8_%d_%d" % (i, t))

    if q - d_min > 1e3:
        model.addConstr(l13 == 0, name="fix-l13")
    if d_max - q > 1e3:
        model.addConstr(l14 == 0, name="fix-l14")

    # model.addConstr(
    #     quicksum(l3[t] - l5[t] for t in range(Np)) - quicksum(l1[i] for i in range(N)) + l13 - l14 <= k * c,
    #     name="dual-q")
    # model.addConstr(l2 <= -k, name="dual-pq")
    #
    # for i in range(N):
    #     model.addConstr(-l10[i] - l11[i] + l12[i] <= 1, name="dual-r%d" % i)
    #     model.addConstr(s * (l12[i] - l11[i]) + quicksum(l7[i, t] - l9[i, t] for t in range(Np)) + l1[i] <= 0,
    #                     name="dual-y%d" % i)
    #     model.addConstr(l6[i] - l12[i] + l11[i] <= 0, name="dual-py%d" % i)
    #
    #     for t in range(Np):
    #         model.addConstr(-P[t] * l6[i] - l7[i, t] - l8[i, t] + l9[i, t] <= 0,
    #                         name="dual-u_%d_%d" % (i, t))
    #
    # for t in range(Np):
    #     model.addConstr(-P[t] * l2 - l3[t] - l4[t] + l5[t] <= 0,
    #                     name="dual-h_%d" % t)

    time0 = time()
    model.optimize()
    # print(model.ObjVal)
    # print(model.status)
    # if model.status != 2:
    #     for t in range(Np):
    #         if g[t] == 0:
    #             model.remove(model.getConstrByName("fix-l5_%d" % t))
    #             if q > 0:
    #                 model.remove(model.getConstrByName("fix-l3_%d" % t))
    #         else:
    #             model.remove(model.getConstrByName("fix-l4_%d" % t))
    #     model.optimize()
    time1 = time() - time0

    return  [l1[i].X for i in range(N)], l2.X, [l3[t].X for t in range(Np)], [l4[t].X for t in range(Np)], \
            [l5[t].X for t in range(Np)], [l6[i].X for i in range(N)], \
            [[l7[i, t].X for t in range(Np)] for i in range(N)], [[l8[i, t].X for t in range(Np)] for i in range(N)], \
            [[l9[i, t].X for t in range(Np)] for i in range(N)], [l10[i].X for i in range(N)], \
            [l11[i].X for i in range(N)], [l12[i].X for i in range(N)], l13.X, l14.X, time1
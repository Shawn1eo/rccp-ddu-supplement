#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/11/26 15:26
# @Author: WANG SHENGRAN
from gurobipy import *
from time import time
import pandas as pd
import numpy as np


M = 3e5
# D = 200
# P = [10 + 0.1*t for t in range(D)]  # All possible discrete values of p


def lagrangian_basic(k, c, s, g, z, d_min, d_max, q_max, P, Np, demand):
    """
    The function to calculate all the lagrangian for the sub-problem
    :param k: key parameter to kNN
    :param c: unit cost
    :param p: unit price
    :param s: unit salvage price
    :param z: member of knn, list of N
    :param d_min: the left boundary of d
    :param d_max: the right boundary ofd
    :param q: order quantity
    :param q_max: maximum quantity
    :param P: number of possible values for p
    :param Np: number of possible values for p
    :param demand: a list of demand recorded in the historical data
    :return: a list of lagrangian from l1 to l12
    """
    N = len(demand)

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

    # define constraints
    model.addConstr(
         quicksum(l3[t] - l5[t] for t in range(Np)) - quicksum(l1[i] for i in range(N)) + l13 - l14 <= k * c,
         name="dual-q")
    model.addConstr(l2 <= -k, name="dual-pq")

    for i in range(N):
         model.addConstr(-l10[i] - l11[i] + l12[i] <= 1, name="dual-r%d" % i)
         model.addConstr(s * (l12[i] - l11[i]) + quicksum(l7[i, t] - l9[i, t] for t in range(Np)) + l1[i] <= 0,
                        name="dual-y%d" % i)
         model.addConstr(l6[i] - l12[i] + l11[i] <= 0, name="dual-py%d" % i)

         for t in range(Np):
             model.addConstr(-P[t] * l6[i] - l7[i, t] - l8[i, t] + l9[i, t] <= 0,
                             name="dual-u_%d_%d" % (i, t))

    for t in range(Np):
         model.addConstr(-P[t] * l2 - l3[t] - l4[t] + l5[t] <= 0,
                        name="dual-h_%d" % t)

    time0 = time()
    model.optimize()
    # print(model.ObjVal)
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

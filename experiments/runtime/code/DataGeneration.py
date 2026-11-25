import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm, lognorm, t
from scipy.optimize import root_scalar


def DataGeneration(n, P: list, relationshipMode: int = 1, uncertaintyMode: int = 1, covariateNum: int = 10):
    '''
    Generate DataSet
    :param n: scale of the Dataset
    :param P: Possible value of price
    :param relationshipMode: 1 for linear 2 for quadratic demand-price location,
    :param uncertaintyMode: 1 for norm(0,1), 2 for lognorm(0,1), 3 for t(3)
    :param covariateNum: Number of covariates
    :return: a Dataset of pd.Dataframe
    '''
    # np.random.seed(1)

    b = np.arange(-math.ceil(covariateNum / 6), math.ceil(covariateNum / 6), 1)  # 留1/3的有效信息
    gamma = np.repeat(1, len(b))
    b = np.pad(b, (0, covariateNum - len(b)), 'constant', constant_values=(0,))  # 补剩下的为0
    gamma = np.pad(gamma, (0, covariateNum - len(gamma)), 'constant', constant_values=(0,))  # 补剩下的为0
    b = b / np.linalg.norm(b)  # 归一化
    gamma = gamma / np.linalg.norm(gamma)

    cov = [[0.5 ** abs(i - j) for i in range(covariateNum)] for j in range(covariateNum)]
    mean = [0 for i in range(covariateNum)]
    X = np.random.multivariate_normal(mean, cov, n)

    Data = pd.DataFrame(X)
    price = []
    demand = []
    for i in range(n):
        if uncertaintyMode == 1:
            e = np.random.normal()
        elif uncertaintyMode == 2:
            e = np.random.lognormal()
        else:
            e = np.random.standard_t(3)

        p = np.random.choice(P)
        if relationshipMode == 1:
            d = 200 - 10*p + X[i].T.dot(b) + (20 - 2.1*p + 0.2*p**2 + X[i].T.dot(gamma))*e
        else:
            d = 200 - 10*p - 0.2*p**2 + X[i].T.dot(b) + (20 - 2.1*p)*min(0, e) + 0.2*p**2*max(e, 0)
        d = max(0, d)
        price.append(p)
        demand.append(d)

    Data["price"] = price
    Data["demand"] = demand

    X = np.random.multivariate_normal(mean, cov, 1)[0]
    return Data, X


def showProfit(data):
    profit = list(data["price"]*data["demand"])
    plt.hist(profit, bins=100)
    plt.show()


# 函数 l(p, q, d)
def l(p, q, d, c, s):
    """
    定义损失函数 l(p, q, d)
    """
    return -(p - c) * q + (p - s) * max(q - d, 0)


# d 的生成
def d_function(e, p, X_mean, b, gamma, relationshipMode):
    """
    根据关系模式和 e 生成 d 的公式
    """
    X_term = X_mean @ b
    gamma_term = X_mean @ gamma

    if relationshipMode == 1:
        return max(0, 200 - 3 * p + X_term + (20 - 2.1 * p + 0.2 * p ** 2 + gamma_term) * e)
    else:
        return max(0, 200 - 3 * p - 0.2 * p ** 2 + X_term
                   + (20 - 2.1 * p) * min(0, e) + 0.2 * p ** 2 * max(0, e))


# e 的反函数 (将 d 映射回 e)
def e_inverse(d_target, p, X_mean, b, gamma, relationshipMode):
    """
    根据给定的 d, 求出对应的 e
    """
    X_term = X_mean @ b
    gamma_term = X_mean @ gamma

    if relationshipMode == 1:
        coeff = 20 - 2.1 * p + 0.2 * p ** 2 + gamma_term
        if coeff == 0:
            return None
        e = (d_target - (200 - 3 * p + X_term)) / coeff
        return e
    else:
        # RelationshipMode == 2 的情况更复杂，需要通过段定义求解
        coeff_positive = 0.2 * p ** 2
        coeff_negative = 20 - 2.1 * p

        if d_target <= 200 - 3 * p - 0.2 * p ** 2 + X_term:  # e <= 0 的情况
            return (d_target - (200 - 3 * p - 0.2 * p ** 2 + X_term)) / coeff_negative
        else:  # e > 0 的情况
            return (d_target - (200 - 3 * p + X_term)) / coeff_positive


# 计算 d 的分位数
def calculate_d_quantile(p, X_mean, b, gamma, alpha, uncertaintyMode, relationshipMode):
    """
    根据给定的 p, X_mean, alpha, 和 mode (uncertaintyMode & relationshipMode)
    计算 d 的 (1-alpha) 分位数。
    """
    # 确定对应的随机变量 e 的分布
    if uncertaintyMode == 1:  # e ~ N(0, 1)
        e_cdf = norm.cdf
        e_bounds = (-np.inf, np.inf)
    elif uncertaintyMode == 2:  # e ~ LogNormal(0, 1)
        lognorm_dist = lognorm(s=1, scale=np.exp(0))  # 对数正态分布，均值为 0，标准差为 1
        e_cdf = lognorm_dist.cdf
        e_bounds = (0, np.inf)
    else:  # e ~ t(3)
        t_dist = t(df=3)  # 自由度为 3 的 t 分布
        e_cdf = t_dist.cdf
        e_bounds = (-np.inf, np.inf)

    # 目标分位数 CDF 概率
    target_cdf = 1 - alpha

    # 定义目标函数，用于求解 d_{1-alpha}
    def target_function(d):
        e = e_inverse(d, p, X_mean, b, gamma, relationshipMode)
        if e is None:
            return -np.inf  # 系数为零时无法计算
        return e_cdf(e) - target_cdf

    # 用数值方法求解目标分位数
    result = root_scalar(target_function, bracket=(0, 1e3), method='brentq')  # 搜索范围可根据 d 的预期范围调整
    if result.converged:
        return result.root
    else:
        raise ValueError("Failed to find the (1-alpha) quantile of d.")


# 计算机会约束是否成立
def true_chance(p, q, c, s, X_mean, v, covariateNum, uncertaintyMode, relationshipMode):
    """
    检查机会约束 Pr(l(p, q, d) > -v) <= alpha 是否成立
    """
    # 根据 uncertaintyMode 确定 e 的概率密度函数 (PDF)
    b = np.arange(-math.ceil(covariateNum / 6), math.ceil(covariateNum / 6), 1)  # 留1/3的有效信息
    gamma = np.repeat(1, len(b))
    b = np.pad(b, (0, covariateNum - len(b)), 'constant', constant_values=(0,))  # 补剩下的为0
    gamma = np.pad(gamma, (0, covariateNum - len(gamma)), 'constant', constant_values=(0,))  # 补剩下的为0
    b = b / np.linalg.norm(b)  # 归一化
    gamma = gamma / np.linalg.norm(gamma)

    if uncertaintyMode == 1:  # e ~ N(0, 1)
        e_pdf = norm.pdf
        e_cdf = norm.cdf
        e_bounds = (-np.inf, np.inf)
    elif uncertaintyMode == 2:  # e ~ LogNormal(0, 1)
        lognorm_dist = lognorm(s=1, scale=np.exp(0))  # 对数正态分布，均值为 0，标准差为 1
        e_pdf = lognorm_dist.pdf
        e_cdf = lognorm_dist.cdf
        e_bounds = (0, np.inf)  # 对数正态的支持范围是正数
    else:  # e ~ t(3)
        t_dist = t(df=3)  # 自由度为 3 的 t 分布
        e_pdf = t_dist.pdf
        e_cdf = t_dist.cdf
        e_bounds = (-np.inf, np.inf)

    # 定义目标积分 Pr(l(p, q, d) > -v)
    def chance_integrand(e):
        d = d_function(e, p, X_mean, b, gamma, relationshipMode)
        if l(p, q, d, c, s) > -v:
            return e_pdf(e)
        else:
            return 0

    # 计算机会约束概率
    p_chance, _ = quad(chance_integrand, *e_bounds)
    return 1 - p_chance  # 检查是否满足约束


def true_chance_simu(p, q, c, s, X, v, covariateNum, uncertaintyMode, relationshipMode):
    n = 10000  # 模拟用数据

    b = np.arange(-math.ceil(covariateNum / 6), math.ceil(covariateNum / 6), 1)  # 留1/3的有效信息
    gamma = np.repeat(1, len(b))
    b = np.pad(b, (0, covariateNum - len(b)), 'constant', constant_values=(0,))  # 补剩下的为0
    gamma = np.pad(gamma, (0, covariateNum - len(gamma)), 'constant', constant_values=(0,))  # 补剩下的为0
    b = b / np.linalg.norm(b)  # 归一化
    gamma = gamma / np.linalg.norm(gamma)

    Amount = 0

    for i in range(n):
        if uncertaintyMode == 1:
            e = np.random.normal()
        elif uncertaintyMode == 2:
            e = np.random.lognormal()
        else:
            e = np.random.standard_t(3)

        if relationshipMode == 1:
            d = 200 - 10 * p + X.T.dot(b) + (20 - 2.1 * p + 0.2 * p ** 2 + X.T.dot(gamma)) * e
        else:
            d = 200 - 10 * p - 0.2 * p ** 2 + X.T.dot(b) + (20 - 2.1 * p) * min(0, e) + 0.2 * p ** 2 * max(e, 0)
        d = max(0, d)

        if l(p, q, d, c, s) <= -v:
            Amount += 1

        # if q >= d:
        #     Amount += 1

        # print(d, l(p, q, d, c, s))

    return Amount / n




# 期望计算
def expected_l(p, q, c, s, v, X_mean, b, gamma, uncertaintyMode, relationshipMode):
    """
    通过数值积分计算 l(p, q, d) 关于 d 的期望
    """
    # 根据 uncertaintyMode 确定 e 的概率密度函数 (PDF)
    if uncertaintyMode == 1:  # e ~ N(0, 1)
        e_pdf = norm.pdf
        e_bounds = (-np.inf, np.inf)
    elif uncertaintyMode == 2:  # e ~ LogNormal(0, 1)
        lognorm_dist = lognorm(s=1, scale=np.exp(0))  # 对数正态分布，均值为 0，标准差为 1
        e_pdf = lognorm_dist.pdf
        e_bounds = (0, np.inf)  # 对数正态的支持范围是正数
    else:  # e ~ t(3)
        t_dist = t(df=3)  # 自由度为 3 的 t 分布
        e_pdf = t_dist.pdf
        e_bounds = (-np.inf, np.inf)

    # 定义被积函数：l(p, q, d(e)) * f(e)
    def integrand(e):
        d = d_function(e, p, X_mean, b, gamma, relationshipMode)
        return l(p, q, d, c, s) * e_pdf(e) if l(p, q, d, c, s) < -v else 0

    # 使用数值积分计算期望
    result, _ = quad(integrand, *e_bounds)
    return result


def trueObjective(p, q, c, s, X, v, alpha, covariateNum, relationshipMode, uncertaintyMode):
    b = np.arange(-math.ceil(covariateNum / 6), math.ceil(covariateNum / 6), 1)  # 留1/3的有效信息
    gamma = np.repeat(1, len(b))
    b = np.pad(b, (0, covariateNum - len(b)), 'constant', constant_values=(0,))  # 补剩下的为0
    gamma = np.pad(gamma, (0, covariateNum - len(gamma)), 'constant', constant_values=(0,))  # 补剩下的为0
    b = b / np.linalg.norm(b)  # 归一化
    gamma = gamma / np.linalg.norm(gamma)

    # if not check_chance_constraint(p, q, c, s, X, b, gamma, v, alpha, uncertaintyMode, relationshipMode):
    #     return 1e5
    # else:
    return expected_l(p, q, c, s, v, X, b, gamma, relationshipMode, uncertaintyMode)


def trueOptima(P, X, c, s, alpha, v, covariateNum, relationshipMode, uncertaintyMode):
    b = np.arange(-math.ceil(covariateNum / 6), math.ceil(covariateNum / 6), 1)  # 留1/3的有效信息
    gamma = np.repeat(1, len(b))
    b = np.pad(b, (0, covariateNum - len(b)), 'constant', constant_values=(0,))  # 补剩下的为0
    gamma = np.pad(gamma, (0, covariateNum - len(gamma)), 'constant', constant_values=(0,))  # 补剩下的为0
    b = b / np.linalg.norm(b)  # 归一化
    gamma = gamma / np.linalg.norm(gamma)
    obj = []
    qopt = []
    for p in P:
        dmin = v / (p - c)
        dmax = calculate_d_quantile(p, X, b, gamma, alpha, uncertaintyMode, relationshipMode) if alpha > 0.1 else 5e3
        dmax = ((p-s)*dmax-v)/(c-s)
        if dmin > dmax:
            obj.append(1e5)
            qopt.append(-1)
            continue
        else:
            dstar = calculate_d_quantile(p, X, b, gamma, (c-s)/(p-s), uncertaintyMode, relationshipMode)
            qopt.append(min(max(dmin, dstar),dmax))
            obj.append(trueObjective(p,qopt[-1],c,s,X,v,alpha,covariateNum,relationshipMode,uncertaintyMode))
    pstar = obj.index(min(obj))
    return P[pstar], qopt[pstar], obj[pstar]


# def trueOptima_2(P, X, c, s, alpha, v, covariateNum, relationshipMode, uncertaintyMode):
#     def objective_to_be_found(x):
#         p, q = x
#         b = np.arange(-math.ceil(covariateNum / 6), math.ceil(covariateNum / 6), 1)  # 留1/3的有效信息
#         gamma = np.repeat(1, len(b))
#         b = np.pad(b, (0, covariateNum - len(b)), 'constant', constant_values=(0,))  # 补剩下的为0
#         gamma = np.pad(gamma, (0, covariateNum - len(gamma)), 'constant', constant_values=(0,))  # 补剩下的为0
#         b = b / np.linalg.norm(b)  # 归一化
#         gamma = gamma / np.linalg.norm(gamma)
#
#         if not check_chance_constraint(p, q, c, s, X, b, gamma, v, alpha, uncertaintyMode, relationshipMode):
#             return 1e5
#         else:
#             return expected_l(p, q, c, s, X, b, gamma, relationshipMode, uncertaintyMode)
#
#     x0 = [15, 150]
#
#     result = basinhopping(objective_to_be_found, x0)
#     return result.x, result.fun



# P = [10 + 0.1 * t for t in range(200)]
# data, X = DataGeneration(5000, P, 1, 1)
# c = 5
# s = 2
# covariateNum = 10
#
# # print(trueObjective(15, 200, 5, 2, X, 1500, 0.7, 10, 2, 1))
# print(trueOptima(P,X,c,s,0.7,1500,10,1,1))
# time, p,q,obj = scheme4(data, X, P, c, s, 0.7, 1500, 5000)
# print(p, q, obj)
# print(trueObjective(p, q, 5, 2, X, 1500, 0.7, 10, 1, 1))
# # showProfit(data)

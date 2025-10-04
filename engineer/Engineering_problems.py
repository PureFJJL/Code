# -*- coding: utf-8 -*-
"""
工程优化设计问题
 注意各变量的物理含义,必要时自己可查文献,自行修改
参考文献：
 https://doi.org/10.1155/2021/8548639
 https://doi.org/10.1016/j.engappai.2022.104805
 https://doi.org/10.1016/j.future.2019.02.028
 https://doi.org/10.1016/j.swevo.2020.100693

关注微信公众号：优化算法侠   Swarm-Opti
https://mbd.pub/o/author-a2mVmGpsYw==

"""

import numpy as np
import math
import cmath
from scipy.optimize import fminbound

'''
lb -> 变量下限
ub -> 变量上限
dim -> 变量维数
fobj -> 待优化目标函数

'''


def Problem_models(problem_number):
    ''' 压力容器设计(Pressure vessel design) '''
    if problem_number == 1:
        lb = [0, 0, 10, 10]
        ub = [99, 99, 200, 200]
        dim = len(lb)
        fobj = pressure_vessel
    ''' 滚动轴承设计(Rolling element bearing design) '''
    if problem_number == 2:
        D = 160
        d = 90
        lb = [0.5 * (D + d), 0.15 * (D - d), 4, 0.515, 0.515, 0.4, 0.6, 0.3, 0.02, 0.6]
        ub = [0.6 * (D + d), 0.45 * (D - d), 50, 0.6, 0.6, 0.5, 0.7, 0.4, 0.1, 0.85]
        dim = len(lb)
        fobj = rolling_element_bearing
    ''' 拉伸/压缩弹簧设计(Tension/compression spring design) '''
    if problem_number == 3:
        lb = [0.05, 0.25, 2]
        ub = [2.0, 1.3, 15]
        dim = len(lb)
        fobj = Tension_compression_spring
    ''' 悬臂梁设计(Cantilever beam design) '''
    if problem_number == 4:
        lb = [0.01, 0.01, 0.01, 0.01, 0.01]
        ub = [100, 100, 100, 100, 100]
        dim = len(lb)
        fobj = Cantilever_beam
    ''' 轮系设计(Gear train design) '''
    if problem_number == 5:
        lb = [12, 12, 12, 12]
        ub = [60, 60, 60, 60]
        dim = len(lb)
        fobj = Gear_train
    ''' 三杆桁架设计(Three bar truss design) '''
    if problem_number == 6:
        lb = [1e-20, 1e-20]  # 避免为0
        ub = [1, 1]
        dim = len(lb)
        fobj = Three_bar_truss
    ''' 焊接梁设计(Welded beam design) '''
    if problem_number == 7:
        lb = [0.1, 0.1, 0.1, 0.1]
        ub = [2, 10, 10, 2]
        dim = len(lb)
        fobj = Welded_beam
    ''' 多盘离合器制动器设计问题(Multiple disk clbtch brake design problem) '''
    if problem_number == 8:
        lb = [60, 90, 1, 0, 2]
        ub = [80, 110, 3, 1000, 9]
        dim = len(lb)
        fobj = Multiple_disk_clbtch
    ''' 步进圆锥滑轮问题(Step-cone pulley problem) '''
    if problem_number == 9:
        lb = [0, 0, 0, 0, 0]
        ub = [60, 60, 90, 90, 90]
        dim = len(lb)
        fobj = Step_cone_pulley
    ''' 减速机设计问题(Speed reducer design problem) '''
    if problem_number == 10:
        lb = [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5]
        ub = [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]
        dim = len(lb)
        fobj = Speed_reducer
    ''' 行星轮系设计优化问题(Planetary gear train design optimization problem) '''
    if problem_number == 11:
        lb = [16.51, 13.51, 13.51, 16.51, 13.51, 47.51, 0.51, 0.51, 0.51]
        ub = [96.49, 54.49, 51.49, 46.49, 51.49, 124.49, 3.49, 6.49, 6.49]
        dim = len(lb)
        fobj = Planetary_gear_train
    ''' 机器人夹持器问题(Robot gripper problem) '''
    if problem_number == 12:
        lb = [10, 10, 100, 0, 10, 100, 1]
        ub = [150, 150, 200, 50, 150, 300, 3.14]
        dim = len(lb)
        fobj = Robot_gripper
    ''' 工字钢垂直挠度问题(I-beam vertical deflection) '''
    if problem_number == 13:
        lb = [10, 10, 0.9, 0.9]
        ub = [50, 80, 5, 5]
        dim = len(lb)
        fobj = I_beam
    ''' 管状柱设计问题(Tubular column design) '''
    if problem_number == 14:
        lb = [2, 0.2]
        ub = [14, 0.8]
        dim = len(lb)
        fobj = Tubular_column
    ''' 波纹舱壁设计问题(Corrugated bulkhead design ) '''
    if problem_number == 15:
        lb = [0, 0, 0, 0]
        ub = [100, 100, 100, 5]
        dim = len(lb)
        fobj = Corrugated_bulkhead
    ''' 活塞杆设计问题(Piston lever design  ) '''
    if problem_number == 16:
        lb = [0.05, 0.05, 0.05, 0.05]
        ub = [500, 500, 120, 500]
        dim = len(lb)
        fobj = Piston_lever
    ''' 钢筋混凝土梁设计问题(Reinforced concrete beam design ) '''
    if problem_number == 17:
        lb = [6, 28, 5]
        ub = [8, 40, 10]
        dim = len(lb)
        fobj = Reinforced_concrete_beam
    ''' 汽车侧面碰撞设计问题(Car side impact design) '''
    if problem_number == 18:
        lb = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0, 0, -30, -30]
        ub = [1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1, 1, 30, 30]
        dim = len(lb)
        fobj = Car_side_impact
    ''' 锯木厂运行问题(Sawmill operation problem) '''
    if problem_number == 19:
        lb = [0, 0, 0, 0]
        ub = [200, 200, 200, 200]
        dim = len(lb)
        fobj = Sawmill_operation
    ''' 静压推力轴承设计问题(hydro-static thrust bearing design ) '''
    if problem_number == 20:
        lb = [1e-6, 1, 1, 1]
        ub = [1.6e-5, 16, 16, 16]
        dim = len(lb)
        fobj = hydro_static_thrust_bearing
    ''' 热交换网络设计问题(Heat exchanger network design problem) '''
    if problem_number == 21:
        lb = [10 ** 4, 10 ** 4, 10 ** 4, 0, 0, 0, 100, 100, 100, 100, 100]
        ub = [10 ** 5, 1.131 * 10 ** 6, 2.05 * 10 ** 6, 5.074e-2, 5.074e-2, 5.074e-2, 200, 300, 300, 300, 400]
        dim = len(lb)
        fobj = heat_exchanger_network
    ''' 热交换网络设计问题2(Heat exchanger network design problem 2) '''
    if problem_number == 22:
        lb = [0, 0, 0, 0, 1000, 0, 100, 100, 100]
        ub = [10, 200, 100, 200, 2e6, 600, 600, 600, 900]
        dim = len(lb)
        fobj = heat_exchanger_network_2
    ''' Haverly's Pooling Problem '''
    if problem_number == 23:
        lb = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ub = [100, 200, 100, 100, 100, 100, 200, 100, 200]
        dim = len(lb)
        fobj = Haverly_pooling
    ''' Blending-Pooling-Separation problem '''
    if problem_number == 24:
        lb = [0] * 38
        ub = [90, 150, 90, 150, 90, 90, 150, 90, 90, 90, 150, 150, 90, 90, 150, 90, 150, 90, 150, 90, 1, 1.2, 1, 1, 1, 0.5, 1, 1, 0.5, 0.5,
              0.5, 1.2, 0.5, 1.2, 1.2, 0.5, 1.2, 1.2]
        dim = len(lb)
        fobj = Blending_Pooling_Separation
    ''' 反应堆网络设计 Reactor network design '''
    if problem_number == 25:
        lb = [0, 0, 0, 0, 0.00001, 0.00001]
        ub = [1, 1, 1, 1, 16, 16]
        dim = len(lb)
        fobj = Reactor_Network
    ''' 烷基化装置的优化操作 Optimal operation of alkylation unit '''
    if problem_number == 26:
        lb = [1000, 0, 2000, 0, 0, 0, 0]
        ub = [2000, 100, 4000, 100, 100, 20, 200]
        dim = len(lb)
        fobj = Operation_Alkylation_Unit

    ''' 输气压缩机设计 Gas transmission compressor design '''
    if problem_number == 27:
        lb = [20, 1, 20, 0.1]
        ub = [50, 10, 50, 60]
        dim = len(lb)
        fobj = Gas_transmission_compressor
    ''' 两个反应器问题 Two-reactor problem '''
    if problem_number == 28:
        lb = [0, 0, 0, 0, 0, 0, 0, 0]
        ub = [100, 100, 100, 100, 100, 100, 1, 1]
        dim = len(lb)
        fobj = Two_reactor
    ''' 工业制冷系统 industrial refrigeration system '''
    if problem_number == 29:
        dim = 14
        lb = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
        ub = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        fobj = Design29
    ''' Himmelblau's function '''
    if problem_number == 30:
        dim = 5
        lb = [78, 33, 27, 27, 27]
        ub = [102, 45, 45, 45, 45]
        fobj = Design30
    ''' Process synthesis problem '''
    if problem_number == 31:
        dim = 2
        lb = [0, 0]
        ub = [1.6, 1]
        fobj = Design31
    ''' Process synthesis and design problem '''
    if problem_number == 32:
        dim = 2
        lb = [0.5, 0]
        ub = [1.4, 1]
        fobj = Design32
    ''' Process flow sheeting problem  '''
    if problem_number == 33:
        dim = 3
        lb = [0.2, -2.22554, 0]
        ub = [1, -1, 1]
        fobj = Design33

    return lb, ub, dim, fobj


'''
Fit -> 目标函数
g -> 约束条件
g_p -> 使用罚函数处理
factor -> 惩罚系数（可自行修改）

'''


def pressure_vessel(x):
    factor = 10 ** 20

    g = [[] for _ in range(4)]
    g[0] = -x[0] + 0.0193 * x[2]
    g[1] = -x[1] + 0.00954 * x[2]
    g[2] = -math.pi * (x[2] ** 2) * x[3] - (4 / 3) * math.pi * (x[2] ** 3) + 1296000
    g[3] = x[3] - 240
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = 0.6224 * x[0] * x[2] * x[3] + 1.7781 * x[1] * (x[2] ** 2) + 3.1661 * (x[0] ** 2) * x[3] + 19.84 * (x[0] ** 2) * x[2]
    Fit = f + g_p
    return Fit


def rolling_element_bearing(x):
    factor = 10 ** 20

    D = 160
    d = 90
    Bw = 30
    T = D - d - 2 * x[1]
    phio = 2 * math.pi - 2 * math.acos(((((D - d) / 2) - 3 * (T / 4)) ** 2 + (D / 2 - T / 4 - x[1]) ** 2 - (d / 2 + T / 4) ** 2)
                                       / (2 * ((D - d) / 2 - 3 * (T / 4)) * (D / 2 - T / 4 - x[1])))

    g = [[] for _ in range(9)]
    g[0] = -phio / (2 * math.asin(x[1] / x[0])) + x[2] - 1
    g[1] = -2 * x[1] + x[5] * (D - d)
    g[2] = -x[6] * (D - d) + 2 * x[1]
    g[3] = -(0.5 + x[8]) * (D + d) + x[0]
    g[4] = -x[0] + 0.5 * (D + d)
    g[5] = -0.5 * (D - x[0] - x[1]) + x[7] * x[1]
    g[6] = x[9] * Bw - x[1]
    g[7] = 0.515 - x[3]
    g[8] = 0.515 - x[4]
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    gama = x[1] / x[0]
    fc = 37.91 * ((1 + (1.04 * ((1 - gama / 1 + gama) ** 1.72) * ((x[3] * (2 * x[4] - 1) / x[4] *
                                                                   (2 * x[3] - 1)) ** 0.41)) ** (10 / 3)) ** -0.3) * (
                 (gama ** 0.3 * (1 - gama) ** 1.39) /
                 (1 + gama) ** (1 / 3)) * (2 * x[3] / (2 * x[3] - 1)) ** 0.41
    if x[1] <= 25.4:
        f = -fc * x[2] ** (2 / 3) * x[1] ** 1.8
    else:
        f = -3.647 * fc * x[2] ** (2 / 3) * x[1] ** 1.4

    Fit = f + g_p
    return Fit


def Tension_compression_spring(x):
    x[2] = round(x[2])
    factor = 10 ** 20

    g = [[] for _ in range(4)]
    g[0] = 1 - x[2] * x[1] ** 3 / (71785 * x[0] ** 4)
    g[1] = (4 * x[1] ** 2 - x[0] * x[1]) / (12566 * (x[1] * x[0] ** 3 - x[0] ** 4)) + 1 / (5108 * x[0] ** 2) - 1
    g[2] = 1 - 140.45 * x[0] / (x[1] ** 2 * x[2])
    g[3] = (x[0] + x[1]) / 1.5 - 1
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = (2 + x[2]) * x[0] ** 2 * x[1]
    Fit = f + g_p
    return Fit


def Cantilever_beam(x):
    factor = 10 ** 20
    g1 = 61 / x[0] ** 3 + 37 / x[1] ** 3 + 19 / x[2] ** 3 + 7 / x[3] ** 3 + 1 / x[4] ** 3 - 1
    g1_p = factor * (max(0, g1)) ** 2
    f = 0.0624 * (x[0] + x[1] + x[2] + x[3] + x[4])

    Fit = f + g1_p
    return Fit


def Gear_train(x):
    x = [round(k) for k in x]
    Fit = (1 / 6.931 - (x[2] * x[1] / (x[0] * x[3]))) ** 2

    return Fit


def Three_bar_truss(x):
    factor = 10 ** 20
    l = 100
    P = 2
    o = 2

    g = [[] for _ in range(3)]
    g[0] = ((math.sqrt(2) * x[0] + x[1]) / (math.sqrt(2) * x[0] ** 2 + 2 * x[0] * x[1])) * P - o
    g[1] = (x[1] / (math.sqrt(2) * x[0] ** 2 + 2 * x[0] * x[1])) * P - o
    g[2] = (1 / (math.sqrt(2) * x[1] + x[0])) * P - o
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = (2 * math.sqrt(2) * x[0] + x[1]) * l
    Fit = f + g_p
    return Fit


def Welded_beam(x):
    factor = 10 ** 20

    f = 1.10471 * x[0] ** 2 * x[1] + 0.04811 * x[2] * x[3] * (14 + x[1])
    P = 6000
    L = 14
    delta_max = 0.25
    E = 30 * 1e6
    G = 12 * 1e6
    T_max = 13600
    sigma_max = 30000
    Pc = 4.013 * E * math.sqrt(x[2] ** 2 * x[3] ** 6 / 30) / L ** 2 * (1 - x[2] / (2 * L) * math.sqrt(E / (4 * G)))
    sigma = 6 * P * L / (x[3] * x[2] ** 2)
    delta = 6 * P * L ** 3 / (E * x[2] ** 2 * x[3])
    J = 2 * (math.sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 4 + (x[0] + x[2]) ** 2 / 4))
    R = math.sqrt(x[1] ** 2 / 4 + (x[0] + x[2]) ** 2 / 4)
    M = P * (L + x[1] / 2)
    ttt = M * R / J
    tt = P / (math.sqrt(2) * x[0] * x[1])
    t = math.sqrt(tt ** 2 + 2 * tt * ttt * x[1] / (2 * R) + ttt ** 2)
    # constraints
    g = [[] for _ in range(5)]
    g[0] = t - T_max
    g[1] = sigma - sigma_max
    g[2] = x[0] - x[3]
    g[3] = delta - delta_max
    g[4] = P - Pc
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    Fit = f + g_p
    return Fit


def Multiple_disk_clbtch(x):
    factor = 10 ** 20
    x = [round(k) for k in x]

    # parameters
    Mf = 3
    Ms = 40
    Iz = 55
    n = 250
    Tmax = 15
    s = 1.5
    delta = 0.5
    Vsrmax = 10
    rho = 0.0000078
    pmax = 1
    mu = 0.6
    Lmax = 30
    delR = 20
    Rsr = 2 / 3 * (x[1] ** 3 - x[0] ** 3) / (x[1] ** 2 * x[0] ** 2)
    Vsr = math.pi * Rsr * n / 30
    A = math.pi * (x[1] ** 2 - x[0] ** 2)
    Prz = x[3] / A
    w = math.pi * n / 30
    Mh = 2 / 3 * mu * x[3] * x[4] * (x[1] ** 3 - x[0] ** 3) / (x[1] ** 2 - x[0] ** 2)
    T = Iz * w / (Mh + Mf)
    #
    f = math.pi * (x[1] ** 2 - x[0] ** 2) * x[2] * (x[4] + 1) * rho

    g = [[] for _ in range(8)]
    g[0] = -x[1] + x[0] + delR
    g[1] = (x[4] + 1) * (x[2] + delta) - Lmax
    g[2] = Prz - pmax
    g[3] = Prz * Vsr - pmax * Vsrmax
    g[4] = Vsr - Vsrmax
    g[5] = T - Tmax
    g[6] = s * Ms - Mh
    g[7] = -T
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    Fit = f + g_p
    return Fit


def Step_cone_pulley(x):
    factor = 10 ** 20

    # parameter Initialization
    d1 = x[0] * 1e-3
    d2 = x[1] * 1e-3
    d3 = x[2] * 1e-3
    d4 = x[3] * 1e-3
    w = x[4] * 1e-3
    N = 350
    N1 = 750
    N2 = 450
    N3 = 250
    N4 = 150
    rho = 7200
    a = 3
    mu = 0.35
    s = 1.75 * 1e6
    t = 8 * 1e-3
    # objective function
    f = rho * w * math.pi / 4 * (d1 ** 2 * (1 + (N1 / N) ** 2) + d2 ** 2 * (1 + (N2 / N) ** 2) + d3 ** 2 * (1 + (N3 / N) ** 2) + d4 ** 2 * (
            1 + (N4 / N) ** 2))
    # constraint
    C1 = math.pi * d1 / 2 * (1 + N1 / N) + (N1 / N - 1) ** 2 * d1 ** 2 / (4 * a) + 2 * a
    C2 = math.pi * d2 / 2 * (1 + N2 / N) + (N2 / N - 1) ** 2 * d2 ** 2 / (4 * a) + 2 * a
    C3 = math.pi * d3 / 2 * (1 + N3 / N) + (N3 / N - 1) ** 2 * d3 ** 2 / (4 * a) + 2 * a
    C4 = math.pi * d4 / 2 * (1 + N4 / N) + (N4 / N - 1) ** 2 * d4 ** 2 / (4 * a) + 2 * a
    R1 = math.exp(mu * (math.pi - 2 * math.asin((N1 / N - 1) * d1 / (2 * a))))
    R2 = math.exp(mu * (math.pi - 2 * math.asin((N2 / N - 1) * d2 / (2 * a))))
    R3 = math.exp(mu * (math.pi - 2 * math.asin((N3 / N - 1) * d3 / (2 * a))))
    R4 = math.exp(mu * (math.pi - 2 * math.asin((N4 / N - 1) * d4 / (2 * a))))
    P1 = s * t * w * (1 - math.exp(-mu * (math.pi - 2. * math.asin((N1 / N - 1) * d1 / (2. * a))))) * math.pi * d1 * N1 / 60
    P2 = s * t * w * (1 - math.exp(-mu * (math.pi - 2. * math.asin((N2 / N - 1) * d2 / (2. * a))))) * math.pi * d2 * N2 / 60
    P3 = s * t * w * (1 - math.exp(-mu * (math.pi - 2. * math.asin((N3 / N - 1) * d3 / (2. * a))))) * math.pi * d3 * N3 / 60
    P4 = s * t * w * (1 - math.exp(-mu * (math.pi - 2. * math.asin((N4 / N - 1) * d4 / (2. * a))))) * math.pi * d4 * N4 / 60

    g = [[] for _ in range(8)]
    g[0] = -R1 + 2
    g[1] = -R2 + 2
    g[2] = -R3 + 2
    g[3] = -R4 + 2
    g[4] = -P1 + (0.75 * 745.6998)
    g[5] = -P2 + (0.75 * 745.6998)
    g[6] = -P3 + (0.75 * 745.6998)
    g[7] = -P4 + (0.75 * 745.6998)
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    h = [[] for _ in range(3)]
    h[0] = C1 - C2
    h[1] = C1 - C3
    h[2] = C1 - C4
    h_p = factor * sum([var ** 2 for var in h if var != 0])

    Fit = f + g_p + h_p
    return Fit


def Speed_reducer(x):
    factor = 10 ** 20

    g = [[] for _ in range(11)]
    g[0] = -x[0] * (x[1] ** 2) * x[2] + 27
    g[1] = -x[0] * (x[1] ** 2) * x[2] ** 2 + 397.5
    g[2] = -x[1] * (x[5] ** 4) * x[2] * (x[3] ** (-3)) + 1.93
    g[3] = -x[1] * (x[6] ** 4) * x[2] * (x[4] ** (-3)) + 1.93
    g[4] = 10 * (x[5] ** (-3)) * math.sqrt(16.91 * (10 ** 6) + (745 * x[3] * ((x[1] * x[2]) ** (-1))) ** 2) - 1100
    g[5] = 10 * (x[6] ** (-3)) * math.sqrt(157.5 * (10 ** 6) + (745 * x[4] * ((x[1] * x[2]) ** (-1))) ** 2) - 850
    g[6] = x[1] * x[2] - 40
    g[7] = -x[0] * x[1] ** (-1) + 5
    g[8] = x[0] * x[1] ** (-1) - 12
    g[9] = 1.5 * x[5] - x[3] + 1.9
    g[10] = 1.1 * x[6] - x[4] + 1.9
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = 0.7854 * x[0] * (x[1] ** 2) * (3.3333 * (x[2] ** 2) + 14.9334 * x[2] - 43.0934) - 1.508 * x[0] * (x[5] ** 2 + x[6] ** 2)
    +7.477 * (x[5] ** 3 + x[6] ** 3) + 0.7854 * (x[3] * x[5] ** 2 + x[4] * x[6] ** 2)

    Fit = f + g_p
    return Fit


def Planetary_gear_train(x):
    factor = 10 ** 20

    # parameter Initialization
    x = [round(k) for k in x]
    pind = [3, 4, 5]
    mind = [1.75, 2, 2.25, 2.5, 2.75, 3.0]
    N1 = x[0]
    N2 = x[1]
    N3 = x[2]
    N4 = x[3]
    N5 = x[4]
    N6 = x[5]
    p = pind[x[6] - 1]
    m1 = mind[x[7] - 1]
    m2 = mind[x[8] - 1]
    # objective function
    i1 = N6 / N4
    i01 = 3.11
    i2 = N6 * (N1 * N3 + N2 * N4) / (N1 * N3 * (N6 - N4))
    i02 = 1.84
    iR = -(N2 * N6 / (N1 * N3))
    i0R = -3.11
    f = max([i1 - i01, i2 - i02, iR - i0R])
    # constraints
    Dmax = 220
    dlt22 = 0.5
    dlt33 = 0.5
    dlt55 = 0.5
    dlt35 = 0.5
    dlt34 = 0.5
    dlt56 = 0.5
    beta = cmath.acos(((N6 - N3) ** 2 + (N4 + N5) ** 2 - (N3 + N5) ** 2) / (2 * (N6 - N3 + 1e-8) * (N4 + N5)))
    g = [[] for _ in range(11)]
    g[0] = m2 * (N6 + 2.5) - Dmax
    g[1] = m1 * (N1 + N2) + m1 * (N2 + 2) - Dmax
    g[2] = m2 * (N4 + N5) + m2 * (N5 + 2) - Dmax
    g[3] = abs(m1 * (N1 + N2) - m2 * (N6 - N3)) - m1 - m2
    g[4] = -((N1 + N2) * math.sin(math.pi / p) - N2 - 2 - dlt22)
    g[5] = -((N6 - N3) * math.sin(math.pi / p) - N3 - 2 - dlt33)
    g[6] = -((N4 + N5) * math.sin(math.pi / p) - N5 - 2 - dlt55)
    g[7] = ((N3 + N5 + 2 + dlt35) ** 2 - (
            (N6 - N3) ** 2 + (N4 + N5) ** 2 - 2 * (N6 - N3) * (N4 + N5) * cmath.cos(2 * math.pi / p - beta))).real
    g[8] = -(N6 - 2. * N3 - N4 - 4 - 2. * dlt34)
    g[9] = -(N6 - N4 - 2. * N5 - 4 - 2. * dlt56)
    g[10] = (N6 - N4 % p) - 0.0001
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    Fit = f + g_p
    return Fit


def Robot_gripper(x):
    factor = 10 ** 20
    a, b, c, e, ff, l, delta = x
    Ymin = 50
    Ymax = 100
    YG = 150
    Zmax = 99.999
    P = 100

    def OBJ1(z):
        g = math.sqrt(e ** 2 + (z - l) ** 2)
        zeta = math.atan(e / (l - z))
        beta = cmath.acos((b ** 2 + g ** 2 - a ** 2) / 2 * b * g) - zeta
        y = 2 * (ff + e + c * cmath.sin(beta + delta))
        return y.real

    def OBJ2(z):
        g = math.sqrt(e ** 2 + (z - l) ** 2)
        zeta = math.atan(e / (l - z))
        alpha = cmath.acos((a ** 2 + g ** 2 - b ** 2) / 2 * a * g) + zeta
        beta = cmath.acos((b ** 2 + g ** 2 - a ** 2) / 2 * b * g) - zeta
        Fk = P * b * cmath.sin(alpha + beta) / (2 * c * cmath.cos(alpha))
        return Fk.real

    def OBJ3(z):
        g = math.sqrt(e ** 2 + (z - l) ** 2)
        zeta = math.atan(e / (l - z))
        alpha = cmath.acos((a ** 2 + g ** 2 - b ** 2) / 2 * a * g) + zeta
        beta = cmath.acos((b ** 2 + g ** 2 - a ** 2) / 2 * b * g) - zeta
        Fk = -P * b * cmath.sin(alpha + beta) / (2 * c * cmath.cos(alpha))
        return Fk.real

    g = [[] for _ in range(7)]
    g[0] = -Ymin + OBJ1(Zmax)
    g[1] = -OBJ1(Zmax)
    g[2] = Ymax - OBJ1(0)
    g[3] = OBJ1(0) - YG
    g[4] = l ** 2 + e ** 2 - (a + b) ** 2
    g[5] = b ** 2 - (a - e) ** 2 - (l - Zmax) ** 2
    g[6] = Zmax - l
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = fminbound(OBJ2, 0, Zmax)
    f2 = fminbound(OBJ3, 0, Zmax)
    f = -f2 - f

    Fit = f + g_p
    return Fit


def I_beam(x):
    factor = 10 ** 20

    g = [[] for _ in range(2)]
    g[0] = 2 * x[0] * x[2] + x[2] * (x[1] - 2 * x[3]) - 300
    term1 = x[2] * (x[1] - 2 * x[3]) ** 3
    term2 = 2 * x[0] * x[2] * (4 * x[3] ** 2 + 3 * x[1] * (x[1] - 2 * x[3]))
    term3 = (x[1] - 2 * x[3]) * x[2] ** 3
    term4 = 2 * x[2] * x[0] ** 3
    g[1] = ((18 * x[1] * 10 ** 4) / (term1 + term2)) + ((15 * x[0] * 10 ** 3) / (term3 + term4)) - 56
    term1 = x[2] * (x[1] - 2 * x[3]) ** 3 / 12
    term2 = x[0] * x[3] ** 3 / 6
    term3 = 2 * x[0] * x[3] * ((x[1] - x[3]) / 2) ** 2
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    term1 = x[2] * (x[1] - 2 * x[3]) ** 3 / 12
    term2 = x[0] * x[3] ** 3 / 6
    term3 = 2 * x[0] * x[3] * ((x[1] - x[3]) / 2) ** 2
    Fit = -5000 / (term1 + term2 + term3) + g_p
    return Fit


def Tubular_column(x):
    factor = 10 ** 20  # 惩罚因子
    P = 2300  # compressive load (kg_f)
    o_y = 500  # yield stress (kg_f/cm**2)
    E = 0.85e6  # elasticity (kg_f/cm**2)
    L = 300  # length of the column (cm)

    g = [[] for _ in range(6)]
    g[0] = P / (math.pi * x[0] * x[1] * o_y) - 1
    g[1] = 8 * P * L ** 2 / (math.pi ** 3 * E * x[0] * x[1] * (x[0] ** 2 + x[1] ** 2) - 1)
    g[2] = 2 / x[0] - 1
    g[3] = x[0] / 14 - 1
    g[4] = 0.2 / x[1] - 1
    g[5] = x[1] / 8 - 1
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = 9.8 * x[0] * x[1] + 2 * x[0]
    Fit = f + g_p
    return Fit


def Corrugated_bulkhead(x):
    factor = 10 ** 20  # 惩罚因子

    g = [[] for _ in range(6)]
    g[0] = -x[3] * x[1] * (0.4 * x[0] + x[2] / 6) + 8.94 * (x[0] + (abs(x[2] ** 2 - x[1] ** 2)) ** 0.5)
    g[1] = -x[3] * x[1] ** 2 * (0.2 * x[0] + x[2] / 12) + 2.2 * (8.94 * (x[0] + (abs(x[2] ** 2 - x[1] ** 2)) ** 0.5)) ** (4 / 3)
    g[2] = -x[3] + 0.0156 * x[0] + 0.15
    g[3] = -x[3] + 0.0156 * x[2] + 0.15
    g[4] = -x[3] + 1.05
    g[5] = -x[2] + x[1]
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = (5.885 * x[3] * (x[0] + x[2])) / (x[0] + (abs(x[2] ** 2 - x[1] ** 2)) ** 0.5)
    Fit = f + g_p
    return Fit


def Piston_lever(x):
    factor = 10 ** 20  # 惩罚因子
    teta = 0.25 * math.pi
    P = 1500
    Q = 10000
    L = 240
    Mmax = 1.8e+6
    R = abs(-x[3] * (x[3] * math.sin(teta) + x[0]) + x[0] * (x[1] - x[3] * math.cos(teta))) / math.sqrt((x[3] - x[1]) ** 2 + x[0] ** 2)
    F = 0.25 * math.pi * P * x[2] ** 2
    L2 = ((x[3] * math.sin(teta) + x[0]) ** 2 + (x[1] - x[3] * math.cos(teta)) ** 2) ** 0.5
    L1 = ((x[3] - x[1]) ** 2 + x[0] ** 2) ** 0.5

    g = [[] for _ in range(4)]
    g[0] = Q * L * math.cos(teta) - R * F
    g[1] = Q * (L - x[3]) - Mmax
    g[2] = 1.2 * (L2 - L1) - L1
    g[3] = x[2] / 2 - x[1]
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = 0.25 * math.pi * (x[2] ** 2) * (L2 - L1)
    Fit = f + g_p
    return Fit


def Reinforced_concrete_beam(x):
    factor = 10 ** 20  # 惩罚因子
    x[1] = round(x[1])

    g = [[] for _ in range(2)]
    g[0] = x[1] / x[2] - 4
    g[1] = 180 + (7.375 * x[0] ** 2) / x[2] - x[0] * x[1]
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = 2.9 * x[0] + 0.6 * x[1] * x[2]
    Fit = f + g_p
    return Fit


def Car_side_impact(x):
    factor = 10 ** 20  # 惩罚因子
    rng = [0.192, 0.345]
    x[7] = rng[min(math.floor(x[7] * len(rng) + 1), len(rng)) - 1]
    x[8] = rng[min(math.floor(x[8] * len(rng) + 1), len(rng)) - 1]

    # Subjective
    g = [[] for _ in range(10)]
    g[0] = 1.16 - 0.3717 * x[1] * x[3] - 0.00931 * x[1] * x[9] - 0.484 * x[2] * x[8] + 0.01343 * x[5] * x[9] - 1
    g[1] = 46.36 - 9.9 * x[1] - 12.9 * x[0] * x[7] + 0.1107 * x[2] * x[9] - 32
    g[2] = 33.86 + 2.95 * x[2] + 0.1792 * x[9] - 5.057 * x[0] * x[1] - 11 * x[1] * x[7] - 0.0215 * x[4] * x[9] - 9.98 * x[6] * x[7] + 22 * \
           x[7] * x[8] - 32
    g[3] = 28.98 + 3.818 * x[2] - 4.2 * x[0] * x[1] + 0.0207 * x[4] * x[9] + 6.63 * x[5] * x[8] - 7.7 * x[6] * x[7] + 0.32 * x[8] * x[
        9] - 32
    g[4] = 0.261 - 0.0159 * x[0] * x[1] - 0.188 * x[0] * x[7] - 0.019 * x[1] * x[6] + 0.0144 * x[2] * x[4] + 0.0008757 * x[4] * x[
        9] + 0.08045 * x[5] * x[8] + 0.00139 * x[7] * x[10] + 0.00001575 * x[9] * x[10] - 0.32
    g[5] = 0.214 + 0.00817 * x[4] - 0.131 * x[0] * x[7] - 0.0704 * x[0] * x[8] + 0.03099 * x[1] * x[5] - 0.018 * x[1] * x[6] + 0.0208 * x[
        2] * x[7] + 0.121 * x[2] * x[8] - 0.00364 * x[4] * x[5] + 0.0007715 * x[4] * x[9]
    -0.0005354 * x[5] * x[9] + 0.00121 * x[7] * x[10] + 0.00184 * x[8] * x[9] - 0.02 * x[1] ** 2 - 0.32
    g[6] = 0.74 - 0.61 * x[1] - 0.163 * x[2] * x[7] + 0.001232 * x[2] * x[9] - 0.166 * x[6] * x[8] + 0.227 * x[1] ** 2 - 0.32
    g[7] = 4.72 - 0.5 * x[3] - 0.19 * x[1] * x[2] - 0.0122 * x[3] * x[9] + 0.009325 * x[5] * x[9] + 0.000191 * x[10] ** 2 - 4
    g[8] = 10.58 - 0.674 * x[0] * x[1] - 1.95 * x[1] * x[7] + 0.02054 * x[2] * x[9]
    -0.0198 * x[3] * x[9] + 0.028 * x[5] * x[9] - 9.9
    g[9] = 16.45 - 0.489 * x[2] * x[6] - 0.843 * x[4] * x[5] + 0.0432 * x[8] * x[9] - 0.0556 * x[8] * x[10] - 0.000786 * x[10] ** 2
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = 1.98 + 4.9 * x[0] + 6.67 * x[1] + 6.98 * x[2] + 4.01 * x[3] + 1.78 * x[4] + 2.73 * x[6]
    Fit = f + g_p
    return Fit


def Sawmill_operation(x):
    factor = 10 ** 20  # 惩罚因子

    g = [[] for _ in range(5)]
    g[0] = x[0] + x[1] - 240
    g[1] = x[2] + x[3] - 300
    g[2] = x[0] + x[2] - 200
    g[3] = x[1] + x[3] - 200
    g[4] = 300 - (x[0] + x[1] + x[2] + x[3])
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = 10 * (24 * x[0] + 20.5 * x[1] + 17.2 * x[2] + 10 * x[3])
    Fit = f + g_p
    return Fit


def hydro_static_thrust_bearing(x):
    factor = 10 ** 20  # 惩罚因子
    miu, Q, R, R0 = x
    P = (math.log10(math.log10(8.122 * 10 ** 6 * miu + 0.8)) + 3.55) / 10.04
    Ef = 9336 * Q * 0.0307 * (10 ** P - 559.7)
    h = ((math.pi * 750 / 60) ** 2) * (2 * math.pi * miu / Ef) * (R ** 4 / 4 - R0 ** 4 / 4)
    P0 = math.log(R / R0) * 6 * miu * Q / (math.pi * h ** 3)
    W = (math.pi * P0 / 2) * (R - P0 ** 2) / math.log(R / R0)

    g = [[] for _ in range(7)]
    g[0] = 1000 - P0
    g[1] = W - 101000
    g[2] = 5000 - W / (math.pi * (R ** 2 - R0 ** 2))
    g[3] = 50 - P0
    g[4] = 0.001 - 0.0307 * Q / (386.4 * P0 * math.pi * R * h)
    g[5] = R - R0
    g[6] = h - 0.001
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = Q * P0 / 0.7 + Ef
    Fit = f + g_p
    return Fit


def heat_exchanger_network(x):
    factor = 10 ** 20  # 惩罚因子

    h = [[] for _ in range(9)]
    h[0] = x[0] - 1e4 * (x[6] - 100)
    h[1] = x[1] - 1e4 * (x[7] - x[6])
    h[2] = x[2] - 1e4 * (500 - x[7])
    h[3] = x[0] - 1e4 * (300 - x[8])
    h[4] = x[1] - 1e4 * (400 - x[9])
    h[5] = x[2] - 1e4 * (600 - x[10])
    h[6] = x[3] * math.log(x[8] - 100 + 1e-8) - x[3] * math.log(300 - x[6] + 1e-8) - x[8] - x[6] + 400
    h[7] = x[4] * math.log(abs(x[9] - x[6]) + 1e-8) - x[4] * math.log(400 - x[7] + 1e-8) - x[9] + x[6] - x[7] + 400
    h[8] = x[5] * math.log(abs(x[10] - x[7]) + 1e-8) - x[5] * math.log(100) - x[10] + x[7] + 100
    h_p = factor * sum([var ** 2 for var in h if var != 0])

    f = (x[0] / (120 * x[3])) ** 0.6 + (x[1] / (80 * x[4])) ** 0.6 + (x[2] / (40 * x[5])) ** 0.6
    Fit = f + h_p
    return Fit


def heat_exchanger_network_2(x):
    factor = 10 ** 20  # 惩罚因子

    h = [[] for _ in range(8)]
    h[0] = 200 * x[0] * x[3] - x[2]
    h[1] = 200 * x[1] * x[5] - x[4]
    h[2] = x[2] - 10000 * (x[6] - 100)
    h[3] = x[4] - 10000 * (300 - x[6])
    h[4] = x[2] - 10000 * (600 - x[7])
    h[5] = x[4] - 10000 * (900 - x[8])
    h[6] = x[3] * math.log(x[7] - 100 + 1e-8) - x[3] * math.log((600 - x[6]) + 1e-8) - x[7] + x[6] + 500
    h[7] = x[5] * math.log(abs(x[8] - x[6]) + 1e-8) - x[5] * math.log(600) - x[8] + x[6] + 600
    h_p = factor * sum([var ** 2 for var in h if var != 0])

    f = 35 * x[0] ** 0.6 + 35 * x[1] ** 0.6
    Fit = f + h_p
    return Fit


def Haverly_pooling(x):
    factor = 10 ** 20  # 惩罚因子

    h = [[] for _ in range(4)]
    h[0] = x[6] + x[7] - x[2] - x[3]
    h[1] = x[0] - x[6] - x[4]
    h[2] = x[1] - x[7] - x[5]
    h[3] = x[8] * x[6] + x[8] * x[7] - 3 * x[2] - x[3]
    h_p = factor * sum([var ** 2 for var in h if var != 0])

    g = [[] for _ in range(2)]
    g[0] = x[8] * x[6] + 2 * x[4] - 2.5 * x[0]
    g[1] = x[8] * x[7] + 2 * x[5] - 1.5 * x[1]
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = -(9 * x[0] + 15 * x[1] - 6 * x[2] - 16 * x[3] - 10 * (x[4] + x[5]))
    Fit = f + g_p + h_p
    return Fit


def Blending_Pooling_Separation(x):
    factor = 10 ** 20  # 惩罚因子

    h = [[] for _ in range(32)]
    h[0] = x[0] + x[1] + x[2] + x[3] - 300
    h[1] = x[5] - x[6] - x[7]
    h[2] = x[8] - x[9] - x[10] - x[11]
    h[3] = x[13] - x[14] - x[15] - x[16]
    h[4] = x[17] - x[18] - x[19]
    h[5] = x[4] * x[20] - x[5] * x[21] - x[8] * x[22]
    h[6] = x[4] * x[23] - x[5] * x[24] - x[8] * x[25]
    h[7] = x[4] * x[26] - x[5] * x[27] - x[8] * x[28]
    h[8] = x[12] * x[29] - x[13] * x[30] - x[17] * x[31]
    h[9] = x[12] * x[32] - x[13] * x[33] - x[17] * x[34]
    h[10] = x[12] * x[35] - x[13] * x[36] - x[17] * x[36]
    h[11] = 1 / 3 * x[0] + x[14] * x[30] - x[4] * x[20]
    h[12] = 1 / 3 * x[0] + x[14] * x[33] - x[4] * x[23]
    h[13] = 1 / 3 * x[0] + x[14] * x[36] - x[4] * x[26]
    h[14] = 1 / 3 * x[1] + x[9] * x[22] - x[12] * x[29]
    h[15] = 1 / 3 * x[1] + x[9] * x[25] - x[12] * x[32]
    h[16] = 1 / 3 * x[1] + x[9] * x[28] - x[12] * x[35]
    h[17] = 1 / 3 * x[2] + x[6] * x[21] + x[10] * x[22] + x[15] * x[30] + x[18] * x[31] - 30
    h[18] = 1 / 3 * x[2] + x[6] * x[24] + x[10] * x[25] + x[15] * x[33] + x[18] * x[34] - 50
    h[19] = 1 / 3 * x[2] + x[6] * x[27] + x[10] * x[28] + x[15] * x[36] + x[18] * x[37] - 30
    h[20] = x[20] + x[23] + x[26] - 1
    h[21] = x[21] + x[24] + x[27] - 1
    h[22] = x[22] + x[25] + x[28] - 1
    h[23] = x[29] + x[32] + x[35] - 1
    h[24] = x[30] + x[33] + x[36] - 1
    h[25] = x[31] + x[34] + x[37] - 1
    h[26] = x[24]
    h[27] = x[27]
    h[28] = x[22]
    h[29] = x[36]
    h[30] = x[31]
    h[31] = x[34]
    h_p = factor * sum([var ** 2 for var in h if var != 0])

    f = 0.9979 + 0.00432 * x[4] + 0.01517 * x[12]
    Fit = f + h_p
    return Fit


def Reactor_Network(x):
    factor = 10 ** 20  # 惩罚因子
    k1 = 0.09755988
    k2 = 0.99 * k1
    k3 = 0.0391908
    k4 = 0.9 * k3

    h = [[] for _ in range(4)]
    h[0] = x[0] + k1 * x[1] * x[4] - 1
    h[1] = x[1] - x[0] + k2 * x[1] * x[5]
    h[2] = x[2] + x[0] + k3 * x[2] * x[4] - 1
    h[3] = x[3] - x[2] + x[1] - x[0] + k4 * x[3] * x[5]
    h_p = factor * sum([var ** 2 for var in h if var != 0])

    g = [[] for _ in range(1)]
    g[0] = x[4] ** 0.5 + x[5] ** 0.5 - 4
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = -x[3]
    Fit = f + g_p + h_p
    return Fit


def Operation_Alkylation_Unit(x):
    factor = 10 ** 20  # 惩罚因子

    g = [[] for _ in range(14)]
    g[0] = 0.0059553571 * x[5] ** 2 * x[0] + 0.88392857 * x[2] - 0.1175625 * x[5] * x[0] - x[0]
    g[1] = 1.1088 * x[0] + 0.1303533 * x[0] * x[5] - 0.0066033 * x[0] * x[5] ** 2 - x[2]
    g[2] = 6.66173269 * x[5] ** 2 + 172.39878 * x[4] - 56.596669 * x[3] - 191.20592 * x[5] - 10000
    g[3] = 1.08702 * x[5] + 0.32175 * x[3] - 0.03762 * x[5] ** 2 - x[4] + 56.85075
    g[4] = 0.006198 * x[6] * x[3] * x[2] + 2462.3121 * x[1] - 25.125634 * x[1] * x[3] - x[2] * x[3]
    g[5] = 161.18996 * x[2] * x[3] + 5000.0 * x[1] * x[3] - 489510.0 * x[1] - x[2] * x[3] * x[6]
    g[6] = 0.33 * x[6] - x[4] + 44.333333
    g[7] = 0.022556 * x[4] - 0.007595 * x[6] - 1.0
    g[8] = 0.00061 * x[2] - 0.0005 * x[0] - 1.0
    g[9] = 0.819672 * x[0] - x[2] + 0.819672
    g[10] = 24500.0 * x[1] - 250.0 * x[1] * x[3] - x[2] * x[3]
    g[11] = 1020.4082 * x[3] * x[1] + 1.2244898 * x[2] * x[3] - 100000 * x[1]
    g[12] = 6.25 * x[0] * x[5] + 6.25 * x[0] - 7.625 * x[2] - 100000
    g[13] = 1.22 * x[2] - x[5] * x[0] - x[0] + 1.0
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = -1.715 * x[0] - 0.035 * x[0] * x[5] - 4.0565 * x[2] - 10.0 * x[1] + 0.063 * x[2] * x[4]
    Fit = f + g_p
    return Fit


def Gas_transmission_compressor(x):
    factor = 10 ** 20  # 惩罚因子
    g = [[] for _ in range(1)]
    g[0] = x[3] * x[1] ** (-2) + x[1] ** (-2) - 1
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = 8.61e5 * x[0] ** 0.5 * x[1] * x[2] ** (-2 / 3) * x[3] ** -0.5 + 3.69e4 * x[2] + 7.72e8 * x[0] ** -1 * x[1] ** 0.219 - 765.43e6 / x[
        0]
    Fit = f + g_p
    return Fit


def Two_reactor(x):
    factor = 10 ** 20  # 惩罚因子
    x[6] = round(x[6])
    x[7] = round(x[7])
    g = [[] for _ in range(4)]
    g[0] = x[4] - 10 * x[6]
    g[1] = x[5] - 10 * x[7]
    g[2] = x[0] - 20 * x[6]
    g[3] = x[1] - 20 * x[7]
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    h = [[] for _ in range(5)]
    h[0] = x[6] + x[7] - 1
    h[1] = x[2] - 0.9 * (1 - math.exp(0.5 * x[4])) * x[0]
    h[2] = x[3] - 0.8 * (1 - math.exp(0.4 * x[5])) * x[1]
    h[3] = x[2] + x[3] - 10
    h[4] = x[2] * x[6] + x[3] * x[7] - 10
    h_p = factor * sum([var ** 2 for var in h if var != 0])

    f = 7.5 * x[6] + 5.5 * x[7] + 7 * x[4] + 6 * x[5] + 5 * (x[0] + x[1])
    Fit = f + g_p + h_p
    return Fit


def Design29(x):
    factor = 10 ** 20  # 惩罚因子
    g = [[] for _ in range(15)]
    g[0] = 1.524 / x[6] - 1
    g[1] = 1.524 / x[7] - 1
    g[2] = 0.07789 * x[0] - 2 * x[8] / x[6] - 1
    g[3] = 7.05305 * x[0] ** 2 * x[9] / (x[8] * x[7] * x[1] * x[13]) - 1
    g[4] = 0.0833 * x[13] / x[12] - 1
    g[5] = 47.136 * x[1] ** 0.333 * x[11] / x[9] - 1.333 * x[7] * x[12] ** 2.1195 + 62.08 * x[12] ** 2.1195 * x[7] ** 0.2 / (
            x[11] * x[9]) - 1
    g[6] = 0.04771 * x[9] * x[7] ** 1.8812 * x[11] ** 0.3424 - 1
    g[7] = 0.0488 * x[8] * x[6] ** 1.893 * x[10] ** 0.316 - 1
    g[8] = 0.0099 * x[0] / x[2] - 1
    g[9] = 0.0193 * x[1] / x[3] - 1
    g[10] = 0.0298 * x[0] / x[4] - 1
    g[11] = 0.056 * x[1] / x[5] - 1
    g[12] = 2 / x[8] - 1
    g[13] = 2 / x[9] - 1
    g[14] = x[11] / x[10] - 1
    g_p = factor * sum([max(0, var) ** 2 for var in g])

    f = 63098.88 * x[1] * x[3] * x[11] + 5441.5 * x[1] ** 2 * x[11] + 115055.5 * x[1] ** 1.664 * x[5]
    + 6172.27 * x[1] ** 2 * x[5] + 63098.88 * x[0] * x[2] * x[10] + 5441.5 * x[0] ** 2 * x[10]
    + 115055.5 * x[0] ** 1.664 * x[4] + 6172.27 * x[0] ** 2 * x[4] + 140.53 * x[0] * x[10]
    + 281.29 * x[2] * x[10] + 70.26 * x[0] ** 2 + 281.29 * x[0] * x[2] + 281.29 * x[2] ** 2
    + 14437 * x[7] ** 1.8812 * x[11] ** 0.3424 * x[9] * x[0] ** 2 * x[6] / (x[13] * x[8])
    + 20470.2 * x[6] ** 2.893 * x[10] ** 0.316 * x[11]

    Fit = f + g_p

    return Fit


def Design30(x):
    factor = 10 ** 20  # 惩罚因子

    G1 = 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4]
    G2 = 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2
    G3 = 9.300961 + 0.0047026 * x[2] * x[4] + 0.00125447 * x[0] * x[2] + 0.0019085 * x[2] * x[3]

    g = [[] for _ in range(6)]
    g[0] = -G1
    g[1] = G1 - 92
    g[2] = 90 - G2
    g[3] = G2 - 110
    g[4] = 20 - G3
    g[5] = G3 - 25

    # 罚函数
    g_p = factor * sum([max(0, var) ** 2 for var in g])
    f = 5.3578547 * x[2] ** 2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141

    Fit = f + g_p

    return Fit


def Design31(x):
    factor = 10 ** 20  # 惩罚因子
    x[1] = round(x[1])
    g = [[] for _ in range(2)]
    g[0] = -(x[0] ** 2) - x[1] + 1.25
    g[1] = x[0] + x[1] - 1.6

    # 罚函数
    g_p = factor * sum([max(0, var) ** 2 for var in g])
    f = x[1] + 2 * x[0]

    Fit = f + g_p

    return Fit


def Design32(x):
    factor = 10 ** 20  # 惩罚因子
    x[1] = round(x[1])

    newx = 2 * np.exp(-x[0])

    g = [[] for _ in range(3)]
    g[0] = x[0] - newx + x[1]
    g[1] = 0.5 - newx
    g[2] = newx - 1.4

    # 罚函数
    g_p = factor * sum([max(0, var) ** 2 for var in g])
    f = -x[1] + x[0] + 2 * newx

    Fit = f + g_p

    return Fit


def Design33(x):
    factor = 10 ** 20  # 惩罚因子
    x[2] = round(x[2])
    g = [[] for _ in range(3)]
    g[0] = -np.exp(x[0] - 0.2) - x[1]
    g[1] = x[1] + 1.1 * x[2] + 1
    g[2] = x[0] - x[2] - 0.2

    # 罚函数
    g_p = factor * sum([max(0, var) ** 2 for var in g])
    f = -0.7 * x[2] + 0.8 + 5 * (x[0] - 0.5) ** 2

    Fit = f + g_p

    return Fit

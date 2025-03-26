#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
import random
from matplotlib.animation import FuncAnimation
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatus, value

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

# 解析输入数据
def parse_input(input_data):
    lines = input_data.split('\n')
    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(Facility(i - 1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3]))))

    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(Customer(i - 1 - facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    return facility_count, customer_count, facilities, customers


# 构建设施选址问题模型
def build_facility_location_model(facility_count, customer_count, facilities, customers):
    model = LpProblem("Facility_Location", LpMinimize)
    
    # 创建决策变量
    x = [[LpVariable(f"x_{c}_{f}", lowBound=0, upBound=1) for f in range(facility_count)] for c in range(customer_count)]
    y = [LpVariable(f"y_{f}", lowBound=0, upBound=1) for f in range(facility_count)]
    
    # 计算距离矩阵
    distance_matrix = np.zeros((facility_count, customer_count))
    for i in range(facility_count):
        for j in range(customer_count):
            distance_matrix[i][j] = length(facilities[i].location, customers[j].location)
    
    # 添加目标函数：最小化设施设置成本和运输成本
    model += lpSum(y[f] * facilities[f].setup_cost for f in range(facility_count)) + \
             lpSum(x[c][f] * distance_matrix[f][c] for c in range(customer_count) for f in range(facility_count))
    
    # 添加客户分配约束
    for c in range(customer_count):
        model += lpSum(x[c][f] for f in range(facility_count)) == 1
    
    # 添加设施容量约束
    for f in range(facility_count):
        model += lpSum(x[c][f] * customers[c].demand for c in range(customer_count)) <= facilities[f].capacity * y[f]
    
    # 添加设施启用约束
    for c in range(customer_count):
        for f in range(facility_count):
            model += x[c][f] <= y[f]
    
    return model, x, y

# 添加Gomory切割平面
def add_gomory_cut(model, fractional_vars):
    for var in fractional_vars:
        fractional_part = var.varValue - math.floor(var.varValue)
        if fractional_part > 0:
            # 生成Gomory切割平面，目标是去掉该分数解
            cut = (lpSum(fractional_part * v for v in model.variables() if v.name == var.name) <= fractional_part)
            model += cut
            print(f"Added Gomory cut for variable {var.name} with fractional part {fractional_part}")


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data):
    # 解析输入数据
    facility_count, customer_count, facilities, customers = parse_input(input_data)
    
    # 构建初始模型
    model, x, y = build_facility_location_model(facility_count, customer_count, facilities, customers)
    
    # main process
    iteration = 0
    max_iterations = 100000  # 限制最大迭代次数
    while True:
        # 求解模型
        model.solve()
        
        # 检查解的整数性
        fractional_vars = [v for v in model.variables() if v.varValue is not None and v.varValue != int(v.varValue)]
        
        # 输出当前解
        print(f"Iteration {iteration}, Objective Value: {value(model.objective)}")
        for var in model.variables():
            print(f"{var.name} = {var.varValue}")

        # 如果所有变量都是整数，结束循环
        if not fractional_vars:
            print("Found an integer solution.")
            break
        
        # 添加Gomory切割平面
        add_gomory_cut(model, fractional_vars)
        
        # 更新迭代次数
        iteration += 1
        if iteration >= max_iterations:
            print("Reached maximum iteration limit.")
            break
    
    # 提取解
    solution = [int(x[c].varValue) for c in range(customer_count)]
    print('v.varValue',model.variables())
    used = [int(v.varValue) for v in y]
    # calculate the cost of the solution
    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')


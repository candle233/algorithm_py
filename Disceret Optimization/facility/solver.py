#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog
import pandas as pd
import time
import networkx as nx
import random
from matplotlib.animation import FuncAnimation

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Branch and Bound function
def branch_and_bound(c, A_eq, b_eq, A_ub, b_ub, bounds, best_solution=None, best_cost=math.inf):
    # print('best_solution:',best_solution,'best_cost:',best_cost)
    # 求解松弛问题
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    # print('res:',res.x)
    # print('res:',res)
    # show the result
    print(f"当前解：{res.x}，目标值：{res.fun}")

    # 检查解的可行性和成功状态
    if not res.success:
        return best_solution, best_cost
    solution = res.x

    # 统计非整数解的数量
    non_integer_count = sum(1 for x in solution if abs(x - round(x)) >= 1e-4)
    print(f"当前解中有 {non_integer_count} 个非整数变量")

    # 如果解是近似整数解（接近 0 或 1 的解），检查其是否优于当前最优解
    if all(abs(x - round(x)) < 1e-4 for x in solution):  # 允许接近 0 或 1 的值
        current_cost = res.fun
        if current_cost < best_cost:
            best_solution = solution
            best_cost = current_cost
        return best_solution, best_cost
    
    # 寻找第一个非整数（非 0 或 1）变量来生成分支
    for i, x in enumerate(solution):
        if abs(x - round(x)) >= 1e-4:  # 判断 x 是否接近整数 0 或 1
            print(i,x)
            # 左分支：将 x[i] 的上界设为接近 0
            bounds_left = bounds.copy()
            bounds_left[i] = [0,0]
            print('bounds_left:',bounds_left)
            left_solution,left_cost = branch_and_bound(c, A_eq, b_eq, A_ub, b_ub, bounds_left, best_solution)
            
            # 右分支：将 x[i] 的下界设为接近 1
            bounds_right = bounds.copy()
            bounds_right[i] = [1, 1]
            print('bounds_right:',bounds_right)
            right_solution ,right_cost = branch_and_bound(c, A_eq, b_eq, A_ub, b_ub, bounds_right, best_solution)
            
            # 如果两种分支均无解，则跳到下一个非整数变量进行分支
            if left_solution is None and right_solution is None:
                continue
            
            # 否则，更新最优解
            if left_solution and (best_solution is None or left_solution['cost'] < best_solution['cost']):
                best_solution,best_cost = left_solution,left_cost
            if right_solution and (best_solution is None or right_solution['cost'] < best_solution['cost']):
                best_solution,best_cost = right_solution,right_cost
            # 一旦我们分支并搜索了这两个分支，就可以停止当前循环
            break

    return best_solution, best_cost
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    # show the data
    # print('facilities:',facilities)
    # print('customers:',customers)

    # main process
    
    # 0. initialize the parameters and variables
    min_factorys,max_factorys = 0,0 # the number of factorys which will be opened
    factory_cost = [i.setup_cost for i in facilities] # the cost of the factory
    factory_consumer = np.zeros((facility_count,customer_count)) # the matrix of the relationship between factory and consumer
    distance_matrix = np.zeros((facility_count,customer_count)) # the matrix of the distance between factory and consumer
    for i in range(facility_count):
        for j in range(customer_count):
            distance_matrix[i][j] = length(facilities[i].location,customers[j].location)
    # print('distance_matrix:',distance_matrix)
    # 1. calculate the bounds of the number of factorys which will be opened
    max_capacity,min_capacity = 0,facilities[0].capacity
    for i in facilities:
        if i.capacity > max_capacity:
            max_capacity = i.capacity
        if i.capacity !=0 and i.capacity<min_capacity:
            min_capacity = i.capacity
    # print('max_capacity:',max_capacity,'min_capacity',min_capacity)
    sum_demand = 0
    for i in customers:
        sum_demand += i.demand
    # print('sum_demand:',sum_demand)
    min_factorys = math.ceil(sum_demand/max_capacity)
    max_factorys = math.ceil(sum_demand/min_capacity)
    costume_demand = np.array([i.demand for i in customers])
    factory_capacity = np.array([i.capacity for i in facilities])
    # print('max_factorys:',max_factorys,'min_factorys:',min_factorys)

    # 2. add the constrains
    # 2.1 demand constrains
    A_demand = np.zeros((customer_count,facility_count*customer_count+facility_count))
    b_demand = np.ones(customer_count)
    for i in range(customer_count):
        A_demand[i][i*facility_count:(i+1)*facility_count] = [1]*facility_count
    # print maxtrix
    # print('A_demand:',A_demand)
    # print('b_demand:',b_demand)

    # 2.2 capacity constrains at least and at most
    A_capacity_1 = np.zeros((2,facility_count*customer_count+facility_count))
    b_capacity_1 = np.array([max_factorys,-1*min_factorys])
    A_capacity_1[0][(facility_count*customer_count):] = [1]*facility_count
    A_capacity_1[1][(facility_count*customer_count):] = [-1]*facility_count
    # print
    # print('A_capacity_1:',A_capacity_1)
    # print('b_capacity_1:',b_capacity_1)

    # 2.3 capacity constrains
    A_capacity_2 = np.zeros((facility_count,customer_count*facility_count+facility_count))
    b_capacity_2 = np.zeros(facility_count)
    for i in range(facility_count):
        for j in range(customer_count):
            A_capacity_2[i][j*facility_count+i] = costume_demand[j]
        A_capacity_2[i][facility_count*customer_count+i] = -1*factory_capacity[i]
    # print('temp:',temp)
    # print('A_capacity_2:',np.shape(A_capacity_2))
    # print
    # print('A_capacity_2:',A_capacity_2)
    # print('b_capacity_2:',b_capacity_2)

    # 2.4 设施启用约束：只有在设施 f 启用时（即 y_f = 1)，才能分配客户给该设施。通过以下不等式可以实现：
    A_facility = np.zeros((facility_count*customer_count,facility_count*customer_count+facility_count))
    b_facility = np.zeros(facility_count*customer_count)
    for i in range(facility_count):
        for j in range(customer_count):
            A_facility[i*customer_count+j][i*customer_count+j] = 1
            A_facility[i*customer_count+j][facility_count*customer_count+i] = -1
    # print('A_facility:',np.shape(A_facility))
    # np.savetxt('A_facility.csv',A_facility,delimiter=',')
    # print
    # print('A_facility:',A_facility)
    # print('b_facility:',b_facility)

    # 3. obejective function
    setup_cost = np.array([i.setup_cost for i in facilities])
    c = np.zeros(facility_count*customer_count)
    for i in range(facility_count):
        c[i*customer_count:(i+1)*customer_count] = distance_matrix[i]
    
    c1 = np.append(c,setup_cost)
    # print('c:',c)
    # print('c:',np.shape(c))
    # 4. solve the problem
    A_eq=A_demand
    b_eq=b_demand
    A_ub = np.vstack([A_capacity_1,A_capacity_2,A_facility])
    b_ub = np.concatenate((b_capacity_1,b_capacity_2,b_facility),axis=0)
    bounds = np.array([(0,1)]*(facility_count*customer_count+facility_count))
    # print all matrix
    # np.savetxt('distance_matrix.csv',distance_matrix,delimiter=',')
    # np.savetxt('A_eq.csv',A_eq,delimiter=',')
    # np.savetxt('b_eq.csv',b_eq,delimiter=',')
    # np.savetxt('A_ub.csv',A_ub,delimiter=',')
    # np.savetxt('b_ub.csv',b_ub,delimiter=',')
    # np.savetxt('c.csv',c1,delimiter=',')

    # res = linprog(c1,A_eq=A_eq,b_eq=b_eq,A_ub=A_ub,b_ub=b_ub,method='highs',bounds=bounds)
    # print(res)

    # 使用分支定界法求解
    #print all matrix
    print('A_eq:',A_eq)
    print('b_eq:',b_eq)
    print('A_ub:',A_ub)
    print('b_ub:',b_ub)
    print('bounds:',bounds)

    best_solution, best_cost = branch_and_bound(c1, A_eq, b_eq, A_ub, b_ub, bounds)
    print('best_solution:',best_solution)
    print('best_cost:',best_cost)
    # solution: customer_i to which facility
    solution = best_solution[:facility_count*customer_count].reshape((customer_count,facility_count))
    solution = np.argmax(solution,axis=1)
    print('solution:',solution)
    # used: whether the facility is used
    used = best_solution[facility_count*customer_count:]
    # 5. gomoery algorithm
    # res = linprog(c1, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, method='highs', bounds=bounds)
    

    # cutting plane failed
    # max_iterations = 100
    # iteration = 0
    # while iteration < max_iterations:
    #     # test
    #     print(res.x)
    #     print('A_ub',A_ub,'b_ub',b_ub)
    #     if res.success:
    #         # 检查是否所有变量为整数
    #         if all(math.isclose(x, round(x)) for x in res.x):
    #             print("整数解已找到")
    #             break
    #         else:
    #             # 添加 Gomory 切割平面
    #             print(f"添加 Gomory 切割平面，第 {iteration + 1} 次迭代")
    #             A_ub, b_ub = add_gomory_cut(A_ub, b_ub, res.x)
    #             # 重新求解
    #             res = linprog(c1, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, method='highs', bounds=bounds)
                
    #             iteration += 1
    #     else:
    #         print("优化失败")
    #         break
    
    # 计算结果
    # if res.success:
    #     solution = res.x[:facility_count * customer_count].reshape((facility_count, customer_count))
    #     used = res.x[facility_count * customer_count:]
    
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


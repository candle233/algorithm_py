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
fig, ax = plt.subplots()

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def draw_solution(solution,customers,facilities,time = 0.1):
    ax.clear()
    GG = nx.DiGraph()
    color_map = []
    GG.add_nodes_from(range(len(solution)))
    for i in range(len(solution)):
        color_map.append('green')
    # print([f'A{i}' for i in range(len(facilities))])
    GG.add_nodes_from([f'A{i}' for i in range(len(facilities))])
    for i in range(len(facilities)):
        color_map.append('red')
    for i in range(len(solution)):
        GG.add_edge(i, f'A{solution[i]}')
    # print('GG:',GG.edges)
    # show the graph
    pos = {i: (customers[i].location.x, customers[i].location.y) for i in range(len(customers))}
    pos.update({f'A{i}': (facilities[i].location.x, facilities[i].location.y) for i in range(len(facilities))})
    # print('pos:',pos)
    nx.draw(GG, pos, with_labels=True, node_color=color_map)
    plt.draw()
    # plt.show()
    plt.pause(time)  # 设置更新频率

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# greedy initial solution
def greedy_initial_solution(facility_count,customer_count,facilities_capacity,customers_demand,distance_matrix,unused = []):
    used = [0]*facility_count # whether the facility is used
    solution = [-1]*customer_count # the facility that the customer is assigned to

    facilities_capacity_left = facilities_capacity.copy() # the left capacity of the facilities
    for i in range(customer_count):
        min_cost = 1e9
        facilities_index = [j for j in range(facility_count) if (j not in unused) and (facilities_capacity_left[j] >= customers_demand[i])]
        # print('facilities_index:',facilities_index)
        for j in facilities_index:
            # find the minimum cost which should satisfy the capacity
            if distance_matrix[j][i] < min_cost:
                min_cost = distance_matrix[j][i]
                solution[i] = j
        facilities_capacity_left[solution[i]] -= customers_demand[i] 
        used[solution[i]] = 1
    return solution,used

# calculate the cost of the solution
def calculate_cost(solution,facilities,customers):
    obj = sum([f.setup_cost for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)
    return obj

# main function
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
    # 0. init the variables
    facilities_capacity = [f.capacity for f in facilities] # the capacity of the facilities
    customers_demand = [c.demand for c in customers] # the demand of the customers
    
    # distance matrix
    distance_matrix = np.zeros((facility_count,customer_count)) # the matrix of the distance between factory and consumer
    for i in range(facility_count):
        for j in range(customer_count):
            distance_matrix[i][j] = length(facilities[i].location,customers[j].location)
    # main process greedy algorithm
    # greedy arrange
    solution,used = greedy_initial_solution(facility_count,customer_count,facilities_capacity,customers_demand,distance_matrix)
    
    # calculate the cost of the solution
    min_obj = calculate_cost(solution,facilities,customers)
    
    # close the factories whihc is not used
    unused = [i for i in range(facility_count) if used[i] == 0]
    # print('unused:',unused)
    # Adjust the solution
    used_index = []
    for i in range(facility_count):
        if i in unused:
            continue
        used_index.append(i)
    # print('used_index:',used_index)
    setup_cost_sorted = sorted(used_index, key=lambda x:facilities[x].setup_cost,reverse=True) # from large to small
    # for i in setup_cost_sorted:
    #     print(f'{i}_cost',facilities[i].setup_cost)
    # print('setup_cost_sorted:',setup_cost_sorted)
    for i in setup_cost_sorted:
        solution,used = greedy_initial_solution(facility_count,customer_count,facilities_capacity,customers_demand,distance_matrix,unused+[i])
        obj_now = calculate_cost(solution,facilities,customers)
        if obj_now < min_obj:
            unused.append(i)
            print('cutting')
        # draw_solution(solution,customers,facilities)
    solution,used = greedy_initial_solution(facility_count,customer_count,facilities_capacity,customers_demand,distance_matrix,unused)
    # print('solution:',solution)
    # show the graph
    # print('unused:',unused)
    # 
    
    # calculate the cost of the solution
    obj = calculate_cost(solution,facilities,customers)
    # obj = sum([f.setup_cost*used[f.index] for f in facilities])
    # for customer in customers:
    #     obj += length(customer.location, facilities[solution[customer.index]].location)

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


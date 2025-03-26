#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from matplotlib import pyplot as plt
import numpy as np
import random
import time
import networkx as nx
import math
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
Point = namedtuple("Point", ['x', 'y'])

# show the graph
def draw_solution(solution, points,time = 0.1):
    ax.clear()
    GG = nx.DiGraph()
    GG.add_nodes_from(range(len(solution)))
    for i in range(len(solution)):
        GG.add_edge(solution[i], solution[(i+1)%len(solution)])
    GG.add_edge(solution[-1], solution[0])
    
    # show the graph
    pos = {i: (points[i].x, points[i].y) for i in range(len(points))}
    nx.draw(GG, pos, with_labels=True, node_color='r')
    # plt.draw()
    plt.savefig('high_res_plot.png', dpi=300, bbox_inches='tight')
    plt.pause(time)  # 设置更新频率

# calculate the length of the two points
def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# calculate the total distance of the tour with distance matrix
def total_distance(solution, distance_matrix):
    total_dis = 0
    for i in range(len(solution)-1):
        total_dis += distance_matrix[solution[i]][solution[i+1]]
    total_dis += distance_matrix[solution[-1]][solution[0]]
    return total_dis

# greedy initial solution
def greedy_initial_solution(distance_matrix):
    num_cities = len(distance_matrix)
    start_city = random.randint(0, num_cities - 1)
    unvisited = set(range(num_cities))
    unvisited.remove(start_city)
    path = [start_city]
    
    current_city = start_city
    while unvisited:
        next_city = min(unvisited, key=lambda city: distance_matrix[current_city][city])
        path.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
    
    return path

# 2-opt algorithm randomly, optimize the exchange
def two_opt(solution):
    new_solution = solution[:]
    i,j = random.sample(range(len(solution)), 2)
    if i>j:
        i,j = j,i
    new_solution[i:j] = new_solution[i:j][::-1]
    return new_solution

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # print('points:', points)
    # visit the nodes in the order they appear in the file
    # solution = [i for i in range(0, nodeCount)] # this is the trivial solution
    # main code simulated annealing
    K = 100000 # intial temperature
    alpha = 0.999 # cooling rate
    min_K = 0.001 # minimum temperature
    no_change_count = 0 # count the number of no change
    max_no_change = 1000 # maximum number of no change
    distance_matrix = np.zeros((nodeCount, nodeCount))
    solution = greedy_initial_solution(distance_matrix) # greedy initial solution
    # calculate the distance matrix
    for i in range(nodeCount):
        for j in range(nodeCount):
            distance_matrix[i][j] = length(points[i], points[j])
    # print('distance_matrix:', distance_matrix)

    # calculate the total distance
    total_dis = total_distance(solution, distance_matrix)
    best_total_dis = total_dis
    best_solution = solution
    # start simulated annealing
    while K>min_K:
        # preform 2-opt algorithm
        # print('solution:', solution)
        new_solution = two_opt(solution)
        # print('new_solution:', new_solution)
        total_dis_new = total_distance(new_solution, distance_matrix)
        delta = total_dis_new - total_dis

        # decide whether to accept the new solution
        if delta < 0:
            solution = new_solution
            total_dis = total_dis_new

            if total_dis < best_total_dis:
                best_total_dis = total_dis
                best_solution = solution
                draw_solution(best_solution, points)
                # print(best_solution)
        elif random.random() < math.exp(-delta/K):
            solution = new_solution
            total_dis = total_dis_new
        else:
            new_solution = solution
        # update the temperature
        K=K*alpha

    # visualize the solution
    draw_solution(best_solution, points,-1)

    # prepare the solution in the specified output format
    output_data = '%.2f' % best_total_dis + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, best_solution))

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

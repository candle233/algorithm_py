#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import gurobipy as gp
from gurobipy import GRB, quicksum
from collections import namedtuple

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0]) # n

    points = [] # information of each node
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # create the model
    model = gp.Model('tsp')

    # set the variables
    x = model.addVars(nodeCount, nodeCount, vtype=GRB.BINARY, name='x')
    u = model.addVars(nodeCount, vtype=GRB.CONTINUOUS, lb=0, name='u')

    # Set objective function
    model.setObjective(quicksum(x[i, j] * length(points[i], points[j]) for i in range(nodeCount) for j in range(nodeCount) if i != j), GRB.MINIMIZE)

    # add constraints
    # each node is visited exactly once
    for j in range(1,nodeCount):
        model.addConstr(gp.quicksum(x[i, j] for i in range(nodeCount) if i != j) == 1)
    
    # each node is left exactly once
    for i in range(0,nodeCount-1):
        model.addConstr(gp.quicksum(x[i, j] for i in range(nodeCount) if i != j) == 1)
    
    # subtour elimination
    for i in range(1,nodeCount):
        for j in range(0,nodeCount-1):
            if i != j:
                model.addConstr(u[i] - u[j] + nodeCount * x[i, j] <= nodeCount - 1)

    # optimize the model
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        obj = model.objVal
        solution = sorted([(i,u[i].x) for i in range(nodeCount)], key=lambda x: x[1])
        solution = [i[0] for i in solution]
    # build a trivial solution
    # visit the nodes in the order they appear in the file
    # solution = range(0, nodeCount)

    # # calculate the length of the tour
    # obj = length(points[solution[-1]], points[solution[0]])
    # for index in range(0, nodeCount-1):
    #     obj += length(points[solution[index]], points[solution[index+1]])

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')


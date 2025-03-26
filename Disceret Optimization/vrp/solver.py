#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import copy
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import random
Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def calculate_route_cost(route,depot):
    if route == []:
        return 0
    cost = length(depot, route[0]) + length(route[-1], depot)
    for i in range(len(route) - 1):
        cost += length(route[i], route[i + 1])
    
    return cost

# draw the graph
def draw_graph(customers, routes, depot,customer_count):
    plt.clf()  # 清除之前的图像
    # init the graph
    GG = nx.DiGraph()

    # calculate cuurent vehicle count
    routes = [route for route in routes if route != []]
    current_vehicle_count = len(routes)

    # routes in list
    routes_list = []
    node_routes_list = []
    for i in range(len(routes)):
        route = []
        node_list=[]
        route.append((0, routes[i][0].index))
        for j in range(len(routes[i])-1):
            route.append((routes[i][j].index, routes[i][j+1].index))
            node_list.append(routes[i][j].index)
        node_list.append(routes[i][-1].index)
        route.append((routes[i][-1].index, 0))
        print('route:',len(route))
        print('node_list:',len(node_list))
        routes_list.append(route)
        node_routes_list.append(node_list)
    # print('routes_list:',routes_list)
    # 定义深红和浅红的 RGB 值
    deep_red = np.array([120, 0, 0]) / 255  # 深红 (darkred) in RGB
    light_red = np.array([255, 128, 128]) / 255  # 浅红 (mistyrose) in RGB
    # generate gradient colors
    gradient_colors = [deep_red + (light_red - deep_red) * i /current_vehicle_count for i in range(current_vehicle_count)]
    # print('gradient_colors:',gradient_colors)

    GG.add_nodes_from(range(customer_count)) # add nodes
    for i in range(current_vehicle_count):
        GG.add_edge(depot.index, routes[i][0].index)
        for j in range(len(routes[i])-1):
            GG.add_edge(routes[i][j].index, routes[i][j+1].index)
        GG.add_edge(routes[i][-1].index, depot.index)
    # show the graph
    pos = {i: (customers[i].x, customers[i].y) for i in range(customer_count)}
    select_color = []
    node_color = [] 
    for i in GG.edges:
        for j in range(len(routes_list)):
            if i in routes_list[j]:
                select_color.append(gradient_colors[j])
    # print('nodes',GG.nodes)
    # print(GG.edges)
    print(len(GG.nodes)==customer_count) # True
    for i in GG.nodes:
        # print('i:',i)
        if i == 0:
            node_color.append('green')
        else:
            for j in range(current_vehicle_count):
                if i in node_routes_list[j]:
                    node_color.append(gradient_colors[j])
                    # print('i:',i)
                    break
            # print('j',j,'i',i)
    # print('node_color:',node_color)
    # print('node_color_len:',len(node_color))
    # print('nodes',GG.nodes)
    # print('select_color:',len(select_color))
    # print('routes_list:',routes_list)
    # print('edges:',len(GG.edges))
    
    nx.draw(GG, pos, with_labels=True, node_color=node_color,edge_color=select_color)
    plt.draw()
    plt.pause(1)

# calculate the cost of the solution
def calculated_cost(vehicle_tours,depot):
    vehicle_tours = [route for route in vehicle_tours if route != []]
    vehicle_count = len(vehicle_tours)
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            obj += length(depot,vehicle_tour[0])
            for i in range(0, len(vehicle_tour)-1):
                obj += length(vehicle_tour[i],vehicle_tour[i+1])
            obj += length(vehicle_tour[-1],depot)
    return obj

# main function
def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    # random.seed(958349089)
    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    #the depot is always the first customer in the input
    depot = customers[0] 


    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = []
    
    remaining_customers = set(customers)
    remaining_customers.remove(depot)

    # Clarke-Wright Savings Heuristic
    # Initialize each customer as a separate route
    routes = [[customer] for customer in customers if customer != depot]
    current_vehicle_count = len(routes)

    # Calculate savings
    savings = []
    for i in range(1, customer_count):
        for j in range(i + 1, customer_count):
            saving = length(depot, customers[i]) + length(depot, customers[j]) - length(customers[i], customers[j])
            savings.append((i, j, saving))
    savings.sort(key=lambda x: x[2], reverse=True)

    # Merge routes based on savings
    for i, j, saving in savings:
        route_i = None
        route_j = None

        # Find routes containing customers i and j
        for route in routes:
            if customers[i] in route:
                route_i = route
            if customers[j] in route:
                route_j = route

        # If i and j are in different routes and merging does not exceed capacity
        if route_i is not None and route_j is not None and route_i != route_j:
            total_demand = sum(customer.demand for customer in route_i + route_j)
            if total_demand <= vehicle_capacity:
                # Merge routes
                route_i.extend(route_j)
                routes.remove(route_j)
                current_vehicle_count -= 1
                # print('current_vehicle_count:',current_vehicle_count)
                # Stop if the number of vehicles is within the limit
                if current_vehicle_count <= vehicle_count:
                    break
    
    # If the number of vehicles is greater than the limit, try to adjust the routes
    while current_vehicle_count > vehicle_count:
        # Find the route with the highest demand and try to split or redistribute customers
        routes.sort(key=lambda r: sum(customer.demand for customer in r), reverse=True)
        route_to_adjust = routes[0]
        
        # Try to redistribute customers to other routes if possible
        for customer in route_to_adjust:
            # print('customer',customer)
            for route in routes[1:]:
                if sum(c.demand for c in route) + customer.demand <= vehicle_capacity:
                    route.append(customer)
                    break

        # If the route is still too large, split it
        if sum(customer.demand for customer in route_to_adjust) > vehicle_capacity:
            split_point = len(route_to_adjust) // 2
            new_route_1 = route_to_adjust[:split_point]
            new_route_2 = route_to_adjust[split_point:]
            routes = routes[1:] + [new_route_1, new_route_2]
            # print('test')
            current_vehicle_count += 1
        else:
            # print('route_to_adjust:',route_to_adjust)
            routes.remove(route_to_adjust)
            current_vehicle_count -= 1
    print('routes:',routes)

    
    # draw the graph with the initial solution
    draw_graph(customers, routes, depot,customer_count)
    
    # plt.show()

    # Construct the final solution
    vehicle_tours = []
    for route in routes:
        vehicle_tours.append([customer for customer in route])
    # print('vehicle_tours:',vehicle_tours)
    # print('vehicle_tours:',vehicle_tours)
    if len(vehicle_tours) < vehicle_count:
        vehicle_tours.extend([[] for _ in range(vehicle_count - len(vehicle_tours))])

    # for v in range(0, vehicle_count):
    #     # print "Start Vehicle: ",v
    #     vehicle_tours.append([])
    #     capacity_remaining = vehicle_capacity
    #     while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
    #         used = set()
    #         order = sorted(remaining_customers, key=lambda customer: -customer.demand*customer_count + customer.index)
    #         for customer in order:
    #             if capacity_remaining >= customer.demand:
    #                 capacity_remaining -= customer.demand
    #                 vehicle_tours[v].append(customer)
    #                 # print '   add', ci, capacity_remaining
    #                 used.add(customer)
    #         remaining_customers -= used

    # 2. do the global optimization
    current_solution = copy.deepcopy(vehicle_tours)
    current_solution_cost = calculated_cost(current_solution,depot)
    iter = 0
    max_interations = 1000 # set the max inter
    unchanged_num = 0

    while iter < max_interations:
        # destory the current solution
        k = max(1, int(len(customers) * 0.1))  # remove some customers in a route
        removed_customers = random.sample(customers[1:], k)  # remove customers randomly
        removed_customers.sort(key=lambda c: -c.demand)  # Sort by descending demand
        print('removed_customers:',removed_customers)
        # calculate where the removed customers are in which route
        remaining_routes = []
        for route in current_solution:
            remain_route=[]
            for c in route:
                if c not in removed_customers:
                    remain_route.append(c)
            remaining_routes.append(remain_route)
        # print('remaining_routes:',remaining_routes)
        # repair the solution
        for customer in removed_customers:
            # print('customer:',customer)
            best_position,best_route = None,None
            best_cost_increase = float('inf')
            # find the best position
            for i in range(len(remaining_routes)):
                remain_route = remaining_routes[i]
                if sum(c.demand for c in remain_route)+customer.demand <= vehicle_capacity:
                    for j in range(len(remain_route)+1):
                        new_route = remain_route[:j] + [customer] + remain_route[j:]
                        new_cost = calculate_route_cost(new_route,depot)
                        original_cost = calculate_route_cost(remain_route, depot)
                        cost_increase = new_cost - original_cost

                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            best_position = j
                            best_route = i
                

            # insert to the best position
            if best_route is not None and best_position is not None:
                remaining_routes[best_route] = remaining_routes[best_route][:best_position] + [customer] + remaining_routes[best_route][best_position:]
                # delete removed customer from the current_solution
                # add the customer to the current_solution

                print('best_route',best_route,'best_position:',best_position)
            else: # if not change
                # add the customer to the current_solution
                pass
                '''
                如果一个要更新位置的customer，没有找到合适的位置，
                这个时候应该如何处理呢？
                1. 将这个customer加入到一个新的route中 ！！！可能会出现这个route中所有的customer都不满足添加到其他地方，容量的上限
                2. 将这个customer加入到一个最近的route中
                
                '''



        # print('remaining_routes:',remaining_routes)
        # print('cuurent_solution:',current_solution)
        iter += 1
        print('iter:',iter)
        # calculate the cost of the current solution and the best solution
        current_remaining_routes_cost = calculated_cost(remaining_routes,depot)

        if unchanged_num > int(max_interations*0.1):
            unchanged_num = 0
            break
        
        if current_remaining_routes_cost < current_solution_cost*0.9:
            current_solution_cost = current_remaining_routes_cost
            current_solution = remaining_routes
            # print('remaining_routes:',remaining_routes)
            draw_graph(customers, current_solution, depot, customer_count)
        else:
            unchanged_num += 1
        
        

    #
    vehicle_tours = remaining_routes
    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = calculated_cost(vehicle_tours,depot)
    # obj = 0
    # for v in range(0, vehicle_count):
    #     vehicle_tour = vehicle_tours[v]
    #     if len(vehicle_tour) > 0:
    #         obj += length(depot,vehicle_tour[0])
    #         for i in range(0, len(vehicle_tour)-1):
    #             obj += length(vehicle_tour[i],vehicle_tour[i+1])
    #         obj += length(vehicle_tour[-1],depot)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += str(depot.index) + ' ' + ' '.join([str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'

    return outputData


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')


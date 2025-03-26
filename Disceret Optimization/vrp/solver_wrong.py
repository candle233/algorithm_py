#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import random
from collections import namedtuple

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def branch_and_bound_reinsertion(route, removed_customers, current_cost, depot):
    """
    Branch-and-bound reinsertion of removed customers into the given route.
    """
    best_cost = current_cost
    best_route = route[:]

    def dfs(partial_route, remaining_customers, current_cost):
        nonlocal best_cost, best_route
        if not remaining_customers:
            if current_cost < best_cost:
                best_cost = current_cost
                best_route = partial_route[:]
            return

        for customer in remaining_customers:
            for i in range(len(partial_route) + 1):
                new_route = partial_route[:i] + [customer] + partial_route[i:]
                new_cost = (
                    length(depot,new_route[0])
                    + sum(
                        length(new_route[k],new_route[k + 1])
                        for k in range(len(new_route) - 1)
                    )
                    + length(new_route[-1],depot)
                )
                if new_cost >= best_cost:
                    continue  # Prune
                dfs(new_route, [c for c in remaining_customers if c != customer], new_cost)

    dfs(route, removed_customers, current_cost)
    return best_route, best_cost

# 
def calculate_route_cost(route, depot):
    """
    Calculate the cost of a route.
    """
    if not route:
        return 0
    cost = length(depot,route[0]) + length(depot,route[-1])
    for i in range(len(route) - 1):
        cost += length(route[i],route[i + 1])
    return cost

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

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
    
    print('customer_count:',customer_count)
    print('vehicle_count:',vehicle_count)
    print('vehicle_capacity:',vehicle_capacity)
    print('remaining_customers:',remaining_customers)
    print('depot:',depot)
    # main method to solve the problem
    # 1. Clarke/Wright savings algorithm
    savings = []
    for i in range(1, len(customers)):
        for j in range(i+1, len(customers)):
            savings.append((length(depot, customers[i]) + length(depot, customers[j]) - length(customers[i], customers[j]), i, j))
    
    savings.sort(reverse=True)
    # print savings
    print('savings:',savings)

    # 2. merge the customers
    for s in savings:
        # print('s:',s)
        remain_customer_index = [c.index for c in remaining_customers]
        if s[1] in remain_customer_index and s[2] in remain_customer_index:
            # print('s[1]:',s[1])
            # print('s[2]:',s[2])
            vehicle_tours.append([depot, customers[s[1]], customers[s[2]], depot])
            remaining_customers.remove(customers[s[1]])
            remaining_customers.remove(customers[s[2]])

    
    # print vehicle_tours
    print('vehicle_tours:',vehicle_tours)

    # 3. Compute total cost
    total_cost  = 0
    for v in range(vehicle_count-len(vehicle_tours)):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            total_cost  += length(depot, vehicle_tour[1])
            for i in range(1, len(vehicle_tour)-1):
                total_cost  += length(vehicle_tour[i], vehicle_tour[i+1])
            total_cost  += length(vehicle_tour[-1], depot)


    best_route, best_cost = vehicle_tours[:], total_cost
    
    # 4. branch and bound
    max_iterations = 1000
    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        # Select a route and remove a subset of customers
        if not vehicle_tours:
            break

        selected_route = random.choice(vehicle_tours)
        if len(selected_route) == 0:
            continue
        num_removed = max(1, len(selected_route) // 3)
        removed_customers = random.sample(selected_route, num_removed)

        # Update the selected route
        for customer in removed_customers:
            selected_route.remove(customer)

        # Reinsertion: Reinsert removed customers into any route using greedy insertion
        for customer in removed_customers:
            best_increase = float('inf')
            best_route_index = -1
            best_position = -1

            for r_idx, route in enumerate(vehicle_tours):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [customer] + route[pos:]
                    # print('new_route:',new_route)
                    if sum(c.demand for c in new_route) <= vehicle_capacity:
                        increase = calculate_route_cost(new_route, depot) - calculate_route_cost(route, depot)
                        if increase < best_increase:
                            best_increase = increase
                            best_route_index = r_idx
                            best_position = pos

            if best_route_index == -1:
                # Create a new route if reinsertion fails into existing routes
                vehicle_tours.append([customer])
            else:
                # Insert customer into the best route
                vehicle_tours[best_route_index].insert(best_position, customer)

        # Remove empty routes
        vehicle_tours = [route for route in vehicle_tours if route]

        # Merge routes if exceeding max_vehicles
        while len(vehicle_tours) > vehicle_count:
            # Find two smallest routes to merge
            vehicle_tours.sort(key=lambda r: calculate_route_cost(r, distance_matrix, depot))
            route1, route2 = vehicle_tours[:2]
            if sum(demand[c] for c in route1 + route2) <= vehicle_capacity:
                new_route = route1 + route2
                vehicle_tours = vehicle_tours[2:]  # Remove the two smallest routes
                vehicle_tours.append(new_route)

    print('best_route:',best_route)
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

    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            obj += length(depot,vehicle_tour[0])
            for i in range(0, len(vehicle_tour)-1):
                obj += length(vehicle_tour[i],vehicle_tour[i+1])
            obj += length(vehicle_tour[-1],depot)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += str(depot.index) + ' ' + ' '.join([str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'

    return outputData


import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')


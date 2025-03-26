#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import random

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def calculate_route_cost(route, depot):
    """Calculate the cost of a single vehicle's route."""
    cost = length(depot, route[0])
    for i in range(len(route) - 1):
        cost += length(route[i], route[i + 1])
    cost += length(route[-1], depot)
    return cost

def clarke_wright_savings(customers, depot, vehicle_capacity, vehicle_count):
    """Initial solution using Clarke/Wright Savings heuristic."""
    routes = {customer.index: [customer] for customer in customers if customer.index != depot.index}
    capacities = {customer.index: customer.demand for customer in customers if customer.index != depot.index}

    savings = []
    for i in customers:
        if i.index == depot.index:
            continue
        for j in customers:
            if j.index <= i.index or j.index == depot.index:
                continue
            saving = length(depot, i) + length(depot, j) - length(i, j)
            savings.append((saving, i.index, j.index))
    
    savings.sort(reverse=True, key=lambda x: x[0])

    for _, i, j in savings:
        if i in routes and j in routes and routes[i] != routes[j]:
            if capacities[i] + capacities[j] <= vehicle_capacity:
                routes[i].extend(routes[j])
                capacities[i] += capacities[j]
                del routes[j]
            # Stop merging if we reach the vehicle count limit
            if len(routes) <= vehicle_count:
                break
    
    if len(routes) > vehicle_count:
        all_customers = []
        for route in routes.values():
            all_customers.extend(route)
    
    customers = sorted(customers, key=lambda c: -c.demand)  # Sort by descending demand
    routes = []
    for _ in range(vehicle_count):
        route = []
        capacity_remaining = vehicle_capacity
        for customer in list(customers[1:]):  # Iterate over a copy of the list
            if customer.demand <= capacity_remaining:
                route.append(customer)
                capacity_remaining -= customer.demand
                customers.remove(customer)
        routes.append(route)
    # print('routes:',routes)
    return routes

def lns_with_branch_and_bound(routes, depot, vehicle_capacity, customers, max_iterations=1000):
    """Refine the solution using Local Neighborhood Search with branch-and-bound."""
    best_routes = routes
    best_cost = sum(calculate_route_cost(route, depot) for route in best_routes)
    print('init_best_routes:', best_routes)
    print('init_best_cost:', best_cost)
    for iteration in range(max_iterations):
        removed_customers = []
        removal_size = random.randint(1, len(customers) // 10 + 1)
        
        # Remove customers
        for _ in range(removal_size):
            route = random.choice(best_routes) # select one route randomly
            if route: # not empty
                removed_customers.append(route.pop(random.randint(0, len(route) - 1)))
        
        # Reinsert customers via branch-and-bound
        for customer in removed_customers:
            best_insertion = None
            best_cost_increase = float('inf')

            for route in best_routes:
                for i in range(len(route) + 1):
                    new_route = route[:i] + [customer] + route[i:]
                    if sum(c.demand for c in new_route) <= vehicle_capacity:
                        # print('new_route:',new_route)
                        # print('route:',route)
                        cost_increase = calculate_route_cost(new_route, depot) - calculate_route_cost(route, depot)
                        if cost_increase < best_cost_increase:
                            best_insertion = (route, i)
                            best_cost_increase = cost_increase
                            print('best_cost_increase:',best_cost_increase)
            
            if best_insertion :
                route, index = best_insertion
                route.insert(index, customer)
            else:
                # keep the customer removed if no cost-reducing insertion found
                print(f"No cost-reducing insertion found for customer {customer.index}")
        # Update best solution if improved
        current_cost = sum(calculate_route_cost(route, depot) for route in best_routes)
        if current_cost < best_cost:
            best_cost = current_cost
        else:
            break  # Stop if no improvement
    print('best_routes',best_routes)
    return best_routes

def solve_it(input_data):
    lines = input_data.split('\n')
    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = []
    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i - 1, int(parts[0]), float(parts[1]), float(parts[2])))

    depot = customers[0]

    # Step 1: Initial solution with Clarke/Wright Savings heuristic
    initial_routes = clarke_wright_savings(customers, depot, vehicle_capacity,vehicle_count)

    # Step 2: Optimize using LNS with branch-and-bound
    optimized_routes = lns_with_branch_and_bound(initial_routes, depot, vehicle_capacity, customers)

    # Calculate final objective value
    obj = sum(calculate_route_cost(route, depot) for route in optimized_routes)

    # Prepare the solution output
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for route in optimized_routes:
        outputData += str(depot.index) + ' ' + ' '.join([str(customer.index) for customer in route]) + ' ' + str(depot.index) + '\n'

    return outputData

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file. Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

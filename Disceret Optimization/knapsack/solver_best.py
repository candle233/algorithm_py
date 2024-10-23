#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    print(items) # show the data types
    print(capacity) # show how to get the property in the data type
    # a trivial algorithm for filling the knapsack
    # dynamic programming
    dynamic_table = [[0 for _ in range(item_count+1)] for __ in range(capacity+1)]
    
    for i in range(item_count+1):
        dynamic_table[i][0]=0

    for j in range(1,item_count+1):
        for i in range(capacity+1):
            if i >= items[j-1].weight:
                dynamic_table[i][j] = max(dynamic_table[i-items[j-1].weight][j-1]+items[j-1].value, dynamic_table[i][j-1])
            else:
                dynamic_table[i][j] = dynamic_table[i][j-1]
    
    value = dynamic_table[-1][-1]
    

    # traceback algorithm
    i,j = capacity,item_count
    selected_items = [0 for i in range(item_count)]
    while i>0 and j>0:
        if dynamic_table[i][j] == dynamic_table[i][j-1]:
            j -= 1
        else:
            j -= 1
            i -= items[j].weight
            selected_items[j] = 1
    
    print(selected_items)
    taken = selected_items

    # it takes items in-order until the knapsack is full
    '''
    value = 0
    weight = 0
    taken = [0]*len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight
    '''
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')


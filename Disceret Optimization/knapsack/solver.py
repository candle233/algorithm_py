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
    # dynamic programming with a 1D table
    dynamic_table = [0 for _ in range(capacity+1)]

    for j in range(1, item_count+1):
        for i in range(capacity, items[j-1].weight - 1, -1):
            dynamic_table[i] = max(dynamic_table[i], dynamic_table[i - items[j-1].weight] + items[j-1].value)
    
    value = dynamic_table[-1]
    
    # traceback algorithm to determine selected items
    i = capacity
    selected_items = [0 for _ in range(item_count)]
    for j in range(item_count, 0, -1):
        if i >= items[j-1].weight and dynamic_table[i] == dynamic_table[i - items[j-1].weight] + items[j-1].value:
            selected_items[j-1] = 1
            i -= items[j-1].weight
    
    print(selected_items)
    taken = selected_items
    sum_test=0
    for i in range(item_count):
        print(f"Item {i}: value = {items[i].value}, selected = {selected_items[i]}")
        sum_test += items[i].value*selected_items[i]
        print('sum',sum_test)
    
    print(sum_test)

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

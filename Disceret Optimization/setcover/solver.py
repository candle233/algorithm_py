#!/usr/bin/python
# -*- coding: utf-8 -*-

# The MIT License (MIT)
#
# Copyright (c) 2014 Carleton Coffrin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from collections import namedtuple

Set = namedtuple("Set", ['index', 'cost', 'items'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    item_count = int(parts[0])
    set_count = int(parts[1])
    
    sets = []
    for i in range(1, set_count+1):
        parts = lines[i].split()
        # print(type(parts)) # type is list
        sets.append(Set(i-1, float(parts[0]), set(map(int, parts[1:]))))

    # build a trivial solution
    # print(sets) # test the sets
    # pick add sets one-by-one until all the items are covered
    solution = [0]*set_count
    covered = set()
    # attention: iteration can be used onely once
    # sorted_sets = sorted(sets, key=lambda s: -s.cost * len(set(s.items) - covered))
    # print(sorted_sets)  # 这应该打印排序后的列表
    # print(list(sets[0].items))
    
    while len(covered) < item_count:
        sorted_sets = sorted(sets, key=lambda s: -s.cost*len(set(s.items)-covered) if len(set(s.items)-covered) > 0 else s.cost)
        # print(sorted_sets)
        # print [-s.cost*len(set(s.items)-covered) for s in sorted_sets]
        # print(sorted_sets)
        for s in sorted_sets:
            # print('s',s)
            # print(s.index,set(s.items))

            if solution[s.index] < 1: 
                solution[s.index] = 1
                covered |= set(s.items)
                break

    # for s in sets:
    #     solution[s.index] = 1
    #     coverted |= set(s.items)
    #     if len(coverted) >= item_count:
    #         break
    
    print('solution:',solution)
    # calculate the cost of the solution
    obj = sum([s.cost*solution[s.index] for s in sets])

    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/sc_6_1)')


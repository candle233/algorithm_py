#!/usr/bin/python
# -*- coding: utf-8 -*-

edges_each_node=[]
nodes_color=[]
over_edge=[]

# if the item not in list, we can also remove it
def safe_remove(lst, item):
    # Remove the item only if it exists in the list
    if item in lst:
        lst.remove(item)
    return lst

def Reverse(tuples): # reverse the tuple
    new_tup = ()
    for k in reversed(tuples):
        new_tup = new_tup + (k,)
    return new_tup

def iter_nodes(node,edge=(None,None)):
    # node: now the program in which nod
    # edge: updated value along which edege
    # over_edge: which edges are completed
    if edge == (None,None): # to start
        # print(nodes_color,node)
        nodes_color[node].remove(-1)
        for i in edges_each_node[node]:
            if (i not in over_edge) and (Reverse(i) not in over_edge):
                iter_nodes(i[0 if node==i[1] else 1],i)
    else:
        over_edge.append(edge) # add the edeg which is finished
        current_node = node

        prev_node = edge[0] if edge[0] != node else edge[1]
        print('previous:',prev_node,'node',node)
        nodes_color[node] = safe_remove(nodes_color[node],nodes_color[prev_node][0])
        #nodes_color[node].remove(nodes_color[prev_node][0]) # remove choice when updating
        if nodes_color[node][0] == -1: # if we never vist this node, then add flag 
            nodes_color[node].remove(-1)
        elif nodes_color[node][0] == nodes_color[prev_node][0]: # if we already visit this node, then remove the the color choice of previous node
            nodes_color[edge[prev_node]].remove(nodes_color[node][0])
        
        for i in edges_each_node[node]:
            if (i not in over_edge) and (Reverse(i) not in over_edge):
                iter_nodes(i[0 if node==i[1] else 1],i)



def solve_it(input_data):
    global edges_each_node
    global nodes_color
    global over_edge
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # build a trivial solution
    print(node_count) # for test
    print(edge_count)
    print(edges)

     #init color of each node
    nodes_color = [[-1,0,1,2,3] for _ in range(node_count)]
    print('nodes_color:',nodes_color)

    # find all edges for each node
    edges_each_node = [[] for _ in range(node_count)]
    for i in range(node_count):
        for j in edges:
            if (i in j) or (i in Reverse(j)):
                edges_each_node[i].append(j)
    
    print('edge_each_node',edges_each_node)
    
    # find the number of edges for a node
    node_edges = []
    for i in range(node_count):
        count_temp = 0
        for j in edges:
            if i in j:
                count_temp += 1
        node_edges.append(count_temp)
    
    print(node_edges)
    print('select the index of lowest element:', node_edges.index(min(node_edges)))

    # main method
    iter_nodes(node_edges.index(min(node_edges)),)
    
    print('color',nodes_color)
    # every node has its own color
    # solution = range(0, node_count)

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    # import sys
    # if len(sys.argv) > 1:
    #     file_location = sys.argv[1].strip()
    #     with open(file_location, 'r') as input_data_file:
    #         input_data = input_data_file.read()
    #     print(solve_it(input_data))
    # else:
    #     print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
     # test
        with open(r"C:\Users\FUN\Desktop\codepackage\algorithm_py\Disceret Optimization\constraint optimization\data\gc_20_1",'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))

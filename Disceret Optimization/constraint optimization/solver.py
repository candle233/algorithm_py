edges_each_node=[]
nodes_color=[]
over_edge=[]

def iter_nodes(node,edge=(None,None)):
    # node: now the program in which nod
    # edge: updated value along which edege
    # over_edge: which edges are completed
    
    if edge == (None,None): # to start
        # print(nodes_color,node)
        current_node = node
        nodes_color[current_node] = 0
        for i in edges_each_node[current_node]:
            
            if ((i,current_node) not in over_edge) and ((current_node,i) not in over_edge):
                iter_nodes(i,(current_node,i))
    else:
        # find all colors around this point
        adjacent_colors = []
        current_node = node
        for neighbor in edges_each_node[current_node]:
            if nodes_color[neighbor] != -1:
                    adjacent_colors.append(nodes_color[neighbor])

        over_edge.append(edge) # add the edge which is finished

        # Assign the smallest color that is not used by adjacent nodes
        color = 0
        while color in adjacent_colors:
            color += 1
        nodes_color[current_node] = color  # Limit to 4 colors
        
        for i in edges_each_node[current_node]:
            if ((i,current_node) not in over_edge) and ((current_node,i) not in over_edge):
                # print(nodes_color)
                iter_nodes(i,(current_node,i))



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
    # print(node_count) # for test
    # print(edge_count)
    # print(edges)

    # init node's color
    nodes_color = [-1 for _ in range(node_count)]
    # Find all edges for each node
    edges_each_node=[[] for _ in range(node_count)]
    for i in edges:
        edges_each_node[i[0]].append(i[1])
        edges_each_node[i[1]].append(i[0])
    # print(edges_each_node)

    # find the number of edges for a node
    node_edges = []
    for i in range(node_count):
        count_temp = 0
        for j in edges:
            if i in j:
                count_temp += 1
        node_edges.append(count_temp)

    # Traverse all nodes using BFS to color them
    node = node_edges.index(min(node_edges))

    # main method
    iter_nodes(node,)
    while -1 in nodes_color:
        iter_nodes(nodes_color.index(-1),)
    
    
    # print('color',nodes_color)
    # every node has its own color
    print(nodes_color)
    solution = nodes_color

    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

if __name__ == '__main__':
        with open(r".\data\gc_100_1",'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
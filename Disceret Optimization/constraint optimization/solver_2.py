edges_each_node=[]
nodes_color=[]
over_edge=[]

def iter_nodes_iterative(start_node):
    # 初始化一个栈，用于手动管理遍历过程
    stack = [(start_node, (None, None))]
    
    # 当栈不为空时，不断进行迭代
    while stack:
        # 弹出当前节点和对应的边
        current_node, edge = stack.pop()
        
        if edge == (None, None):  # 如果是起始节点
            nodes_color[current_node] = 0
            # 将相邻节点加入栈中
            for neighbor in edges_each_node[current_node]:
                if ((neighbor, current_node) not in over_edge) and ((current_node, neighbor) not in over_edge):
                    stack.append((neighbor, (current_node, neighbor)))
        else:
            # 查找当前节点的所有相邻颜色
            adjacent_colors = set()
            for neighbor in edges_each_node[current_node]:
                if nodes_color[neighbor] != -1:
                    adjacent_colors.add(nodes_color[neighbor])
            
            over_edge.append(edge)  # 将已完成的边添加到over_edge中

            # 分配当前节点的颜色
            color = 0
            while color in adjacent_colors:
                color += 1
            nodes_color[current_node] = color

            # 将相邻节点加入栈中
            for neighbor in edges_each_node[current_node]:
                if ((neighbor, current_node) not in over_edge) and ((current_node, neighbor) not in over_edge):
                    # if nodes_color[neighbor] == -1:
                    stack.append((neighbor, (current_node, neighbor)))

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
    
    iter_nodes_iterative(node)
    min_colors = max(nodes_color)+1
    colors = 0
    while min_color > colors:
        min_color = colors
        iter_nodes_iterative(nodes_color.index(-1))
        colors = max(nodes_color)+1
    
    
    # print('color',nodes_color)
    # every node has its own color
    print(nodes_color)
    solution = nodes_color

    # prepare the solution in the specified output format
    output_data = str(max(solution)+1) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
    # with open(r"D:\candles\Downloads\gc_1000_5.txt",'r') as input_data_file:
    #     input_data = input_data_file.read()
    # print(solve_it(input_data))
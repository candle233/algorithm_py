edges_each_node = []
nodes_color = []

def bfs_coloring(start_node):
    pass


def solve_it(input_data):
    global edges_each_node
    global nodes_color

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

    # Initialize color of each node as -1 (no color assigned)
    nodes_color = [-1] * node_count

    # Find all edges for each node
    edges_each_node=[[] for _ in range(node_count)]
    for i in edges:
        edges_each_node[i[0]].append(i[1])
        edges_each_node[i[1]].append(i[0])
    print(edges_each_node)

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
    while -1 in nodes_color:
        bfs_coloring()

    # Prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(max(nodes_color) + 1) + '\n'
    output_data += ' '.join(map(str, nodes_color))

    return output_data


if __name__ == '__main__':
    with open(r"C:\Users\FUN\Desktop\codepackage\algorithm_py\Disceret Optimization\constraint optimization\data\gc_100_1", 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))

edges_each_node = []
nodes_color = []

def bfs_coloring(start_node):
    queue = [start_node]
    nodes_color[start_node] = 0  # Start coloring with color 0
    
    while queue:
        current_node = queue.pop(0)
        # Find colors of adjacent nodes
        adjacent_colors = set()
        for neighbor in edges_each_node[current_node]:
            if nodes_color[neighbor] != -1:
                adjacent_colors.add(nodes_color[neighbor])
            elif neighbor not in queue:
                queue.append(neighbor)

        # Assign the smallest color that is not used by adjacent nodes
        color = 0
        while color in adjacent_colors and color < 4:
            color += 1
        nodes_color[current_node] = color if color < 4 else 3  # Limit to 4 colors


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
    edges_each_node = [[] for _ in range(node_count)]
    for edge in edges:
        edges_each_node[edge[0]].append(edge[1])
        edges_each_node[edge[1]].append(edge[0])

    # Traverse all nodes using BFS to color them
    for node in range(node_count):
        if nodes_color[node] == -1:
            bfs_coloring(node)

    # Prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(max(nodes_color) + 1) + '\n'
    output_data += ' '.join(map(str, nodes_color))

    return output_data


if __name__ == '__main__':
    with open(r"C:\Users\FUN\Desktop\codepackage\algorithm_py\Disceret Optimization\constraint optimization\data\gc_20_1", 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))

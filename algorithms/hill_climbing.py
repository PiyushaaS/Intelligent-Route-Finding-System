from time import perf_counter

from algorithms._common import make_result


def hill_climbing_search(graph, start, goal, heuristic):
    """Follow strictly improving heuristic neighbors until goal or a local optimum."""
    start_time = perf_counter()

    if start not in graph or goal not in graph:
        return make_result(graph, [], 0, start_time)

    if start == goal:
        return make_result(graph, [start], 0, start_time)

    path = [start]
    visited = {start}
    current = start
    current_heuristic = heuristic(current, goal)
    nodes_explored = 0

    while current != goal:
        nodes_explored += 1
        neighbors = [
            neighbor
            for neighbor in graph.neighbors(current)
            if neighbor not in visited
        ]

        if goal in neighbors:
            path.append(goal)
            return make_result(graph, path, nodes_explored, start_time)

        if not neighbors:
            return make_result(graph, [], nodes_explored, start_time)

        next_node = min(neighbors, key=lambda node: heuristic(node, goal))
        next_heuristic = heuristic(next_node, goal)
        if next_heuristic >= current_heuristic:
            return make_result(graph, [], nodes_explored, start_time)

        path.append(next_node)
        visited.add(next_node)
        current = next_node
        current_heuristic = next_heuristic

    return make_result(graph, path, nodes_explored, start_time)
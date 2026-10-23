from time import perf_counter

from algorithms._common import make_result, reconstruct_path


def dfs_search(graph, start, goal):
    """Find a route with depth-first search; the route is not necessarily optimal."""
    start_time = perf_counter()

    if start not in graph or goal not in graph:
        return make_result(graph, [], 0, start_time)

    if start == goal:
        return make_result(graph, [start], 0, start_time)

    stack = [start]
    parent = {start: None}
    nodes_explored = 0

    while stack:
        current = stack.pop()

        if current == goal:
            break

        nodes_explored += 1
        for neighbor in graph.neighbors(current):
            if neighbor not in parent:
                parent[neighbor] = current
                stack.append(neighbor)

    if goal not in parent:
        return make_result(graph, [], nodes_explored, start_time)

    return make_result(
        graph,
        reconstruct_path(parent, goal),
        nodes_explored,
        start_time,
    )

from collections import deque
from time import perf_counter

from algorithms._common import make_result, reconstruct_path


def bfs_search(graph, start, goal):
    """Find a minimum-edge route with breadth-first search."""
    start_time = perf_counter()

    if start not in graph or goal not in graph:
        return make_result(graph, [], 0, start_time)

    if start == goal:
        return make_result(graph, [start], 0, start_time)

    queue = deque([start])
    parent = {start: None}
    nodes_explored = 0

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        nodes_explored += 1
        for neighbor in graph.neighbors(current):
            if neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)

    if goal not in parent:
        return make_result(graph, [], nodes_explored, start_time)

    return make_result(
        graph,
        reconstruct_path(parent, goal),
        nodes_explored,
        start_time,
    )

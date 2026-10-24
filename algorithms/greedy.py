from heapq import heappop, heappush
from itertools import count
from time import perf_counter

from algorithms._common import graph_neighbors, make_result, reconstruct_path


def greedy_search(graph, start, goal, heuristic):
    """Find a route by prioritizing the lowest heuristic value."""
    start_time = perf_counter()

    if start not in graph or goal not in graph:
        return make_result(graph, [], 0, start_time)

    if start == goal:
        return make_result(graph, [start], 0, start_time)

    sequence = count()
    frontier = [(heuristic(start, goal), next(sequence), start)]
    parents = {start: None}
    visited = set()
    nodes_explored = 0

    while frontier:
        _, _, current = heappop(frontier)
        if current in visited:
            continue

        if current == goal:
            return make_result(
                graph,
                reconstruct_path(parents, goal),
                nodes_explored,
                start_time,
            )

        visited.add(current)
        nodes_explored += 1

        for neighbor in graph_neighbors(graph, current):
            if neighbor not in parents:
                parents[neighbor] = current
                heappush(
                    frontier,
                    (heuristic(neighbor, goal), next(sequence), neighbor),
                )

    return make_result(graph, [], nodes_explored, start_time)
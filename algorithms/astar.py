from heapq import heappop, heappush
from itertools import count
from math import inf
from time import perf_counter

from algorithms._common import edge_weight, graph_neighbors, make_result, reconstruct_path


def astar_search(graph, start, goal, heuristic):
    """Find a least-cost route using f(n) = g(n) + h(n).

    With non-negative edge weights and an admissible heuristic that is zero
    at the goal, the returned route is optimal.
    """
    start_time = perf_counter()

    if start not in graph or goal not in graph:
        return make_result(graph, [], 0, start_time)

    if start == goal:
        return make_result(graph, [start], 0, start_time)

    sequence = count()
    frontier = [(heuristic(start, goal), next(sequence), 0, start)]
    parents = {start: None}
    cost_so_far = {start: 0}
    nodes_explored = 0

    while frontier:
        _, _, current_cost, current = heappop(frontier)
        if current_cost != cost_so_far[current]:
            continue

        if current == goal:
            return make_result(
                graph,
                reconstruct_path(parents, goal),
                nodes_explored,
                start_time,
            )

        nodes_explored += 1
        for neighbor in graph_neighbors(graph, current):
            edge_cost = edge_weight(graph, current, neighbor)
            tentative_cost = current_cost + edge_cost

            if tentative_cost < cost_so_far.get(neighbor, inf):
                cost_so_far[neighbor] = tentative_cost
                parents[neighbor] = current
                priority = tentative_cost + heuristic(neighbor, goal)
                heappush(
                    frontier,
                    (priority, next(sequence), tentative_cost, neighbor),
                )

    return make_result(graph, [], nodes_explored, start_time)
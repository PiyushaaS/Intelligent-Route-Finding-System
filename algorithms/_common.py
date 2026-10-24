from collections import deque
from collections.abc import Mapping
from time import perf_counter
from typing import Any, Hashable


def graph_neighbors(graph: Any, node: Hashable):
    if isinstance(graph, Mapping):
        return graph.get(node, {}).keys()
    return graph.neighbors(node)


def edge_weight(graph: Any, start: Hashable, end: Hashable):
    if isinstance(graph, Mapping):
        weight = graph[start][end]
        return 1 if weight is None else weight
    return graph[start][end].get("weight", 1)


def make_result(
    graph: Any,
    path: list[Hashable],
    nodes_explored: int,
    start_time: float,
) -> dict[str, Any]:
    cost = None
    if path:
        cost = sum(edge_weight(graph, start, end) for start, end in zip(path, path[1:]))

    return {
        "path": path,
        "cost": cost,
        "nodes_explored": nodes_explored,
        "execution_time": perf_counter() - start_time,
    }


def make_admissible_heuristic(graph: Any):
    """Build a lower bound from hop distance and the graph's minimum edge cost."""
    if isinstance(graph, Mapping):
        nodes = list(graph)
    else:
        nodes = list(graph.nodes)

    minimum_edge_cost = None
    reverse_neighbors = {node: [] for node in nodes}
    for node in nodes:
        for neighbor in graph_neighbors(graph, node):
            weight = edge_weight(graph, node, neighbor)
            if minimum_edge_cost is None or weight < minimum_edge_cost:
                minimum_edge_cost = weight

            if isinstance(graph, Mapping):
                reverse_neighbors.setdefault(neighbor, []).append(node)

    if not isinstance(graph, Mapping):
        if graph.is_directed():
            reverse_neighbors = {
                node: list(graph.predecessors(node))
                for node in nodes
            }
        else:
            reverse_neighbors = {
                node: list(graph_neighbors(graph, node))
                for node in nodes
            }

    minimum_edge_cost = minimum_edge_cost or 0
    distances_by_goal = {}

    def heuristic(node: Hashable, goal: Hashable):
        if goal not in graph or node not in graph:
            return 0

        if goal not in distances_by_goal:
            distances = {goal: 0}
            queue = deque([goal])
            while queue:
                current = queue.popleft()
                for predecessor in reverse_neighbors.get(current, ()):
                    if predecessor not in distances:
                        distances[predecessor] = distances[current] + 1
                        queue.append(predecessor)
            distances_by_goal[goal] = distances

        hop_distance = distances_by_goal[goal].get(node)
        if hop_distance is None:
            return 0
        return hop_distance * minimum_edge_cost

    return heuristic


def reconstruct_path(
    parents: dict[Hashable, Hashable | None],
    goal: Hashable,
) -> list[Hashable]:
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path

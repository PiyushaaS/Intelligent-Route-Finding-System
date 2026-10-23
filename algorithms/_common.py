from time import perf_counter
from typing import Any, Hashable


def make_result(
    graph: Any,
    path: list[Hashable],
    nodes_explored: int,
    start_time: float,
) -> dict[str, Any]:
    cost = None
    if path:
        cost = sum(
            graph[start][end].get("weight", 1)
            for start, end in zip(path, path[1:])
        )

    return {
        "path": path,
        "cost": cost,
        "nodes_explored": nodes_explored,
        "execution_time": perf_counter() - start_time,
    }


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

from algorithms._common import make_admissible_heuristic
from algorithms.astar import astar_search
from algorithms.bfs import bfs_search
from algorithms.dfs import dfs_search
from algorithms.greedy import greedy_search
from algorithms.hill_climbing import hill_climbing_search
from graph_data import graph, locations


def main():
    start = "A"
    goal = "F"
    heuristic = make_admissible_heuristic(graph)
    algorithms = (
        ("BFS", bfs_search, False),
        ("DFS", dfs_search, False),
        ("Greedy Best-First", greedy_search, True),
        ("A*", astar_search, True),
        ("Hill Climbing", hill_climbing_search, True),
    )

    rows = []
    for name, search, uses_heuristic in algorithms:
        if uses_heuristic:
            result = search(graph, start, goal, heuristic)
        else:
            result = search(graph, start, goal)

        route = (
            " -> ".join(
                "{} ({})".format(node, locations[node]["name"])
                for node in result["path"]
            )
            if result["path"]
            else "No route"
        )
        cost = str(result["cost"]) if result["cost"] is not None else "-"
        rows.append(
            (
                name,
                route,
                cost,
                str(result["nodes_explored"]),
                f'{result["execution_time"]:.8f}',
            )
        )

    headers = ("Algorithm", "Route", "Cost", "Nodes explored", "Time (s)")
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    print(
        "Route search comparison: {} ({}) -> {} ({})".format(
            start,
            locations[start]["name"],
            goal,
            locations[goal]["name"],
        )
    )
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


if __name__ == "__main__":
    main()

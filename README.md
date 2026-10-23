# Intelligent Route-Finding System

A modular Python project that compares five AI search strategies for finding
routes in a weighted NetworkX graph. The included `utils/graph_data.py` graph is
a small sample; it can be replaced by the final location dataset without
coupling the algorithms to a user interface.

## Algorithms

- **Breadth-First Search (BFS):** finds a route with the fewest edges. It does
  not minimize weighted cost when edge weights differ.
- **Depth-First Search (DFS):** finds a reachable route but does not guarantee
  minimum edge count or minimum cost.
- **Greedy Best-First Search:** prioritizes the lowest heuristic estimate. It
  terminates on finite graphs, but is not cost-optimal and is not generally
  complete for infinite state spaces.
- **A\* Search:** prioritizes `f(n) = g(n) + h(n)` and updates/reopens nodes
  when a cheaper route is found. It is optimal when edge costs are
  non-negative and the heuristic is admissible (never overestimates remaining
  cost) and zero at the goal. The sample's Euclidean coordinate heuristic is
  compatible with its sample edge-cost scale.
- **Hill Climbing:** repeatedly chooses an unvisited neighbor with a strictly
  lower heuristic value. It detects revisits and stops at a local optimum or
  dead end; it is neither complete nor optimal and does not use route cost to
  choose neighbors.

All algorithms treat a missing edge `weight` attribute as a cost of `1`.
`nodes_explored` counts nodes whose neighbors were examined; a goal node is
not counted as expanded.

## Install (Windows PowerShell)

```powershell
python -m pip install -r requirements.txt
```

## Run the comparison

```powershell
python main.py
```

The runner compares all five algorithms on the same sample graph, from
`College` to `Airport`. It is independent of Streamlit.

## Run tests

```powershell
python -m pytest -v
```

## Algorithm API

```python
bfs_search(graph, start, goal)
dfs_search(graph, start, goal)
greedy_search(graph, start, goal, heuristic)
astar_search(graph, start, goal, heuristic)
hill_climbing_search(graph, start, goal, heuristic)
```

`graph` is a NetworkX graph. A heuristic is a callable of the form
`heuristic(node, goal) -> number`. Each function returns the same dictionary:

```python
{
    "path": ["start", "...", "goal"],  # [] if no route is found
    "cost": 3,                          # None if no route is found
    "nodes_explored": 2,
    "execution_time": 0.00001,          # seconds
}
```

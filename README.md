# Intelligent Route-Finding System

A modular Python project that compares five AI search strategies for finding
routes in a weighted graph. The project-root `graph_data.py` contains Member 1's
adjacency-dictionary dataset (locations `A`–`F`); the algorithms also retain
support for weighted and unweighted NetworkX graphs.

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
  cost) and zero at the goal. For Member 1's graph, the runner uses a
  graph-derived lower bound: the minimum edge cost multiplied by the minimum
  number of remaining hops. This is admissible for its non-negative edge
  weights. The supplied coordinates are display coordinates, not kilometer
  distances, so they are not used as a distance heuristic.
- **Hill Climbing:** repeatedly chooses an unvisited neighbor with a strictly
  lower heuristic value. It detects revisits and stops at a local optimum or
  dead end; it is neither complete nor optimal and does not use route cost to
  choose neighbors.

For NetworkX graphs, a missing edge `weight` attribute is treated as a cost of
`1`. Adjacency dictionaries use their numeric edge values directly.
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

The runner compares all five algorithms on Member 1's graph, from `A`
(`College Gate`) to `F` (`Hostel`). It is independent of Streamlit.

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

`graph` can be a NetworkX graph or an adjacency dictionary in which each
neighbor maps to its numeric edge cost. A heuristic is a callable of the form
`heuristic(node, goal) -> number`. Each function returns the same dictionary:

```python
{
    "path": ["start", "...", "goal"],  # [] if no route is found
    "cost": 3,                          # None if no route is found
    "nodes_explored": 2,
    "execution_time": 0.00001,          # seconds
}
```

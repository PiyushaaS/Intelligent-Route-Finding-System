import networkx as nx
import pytest

from algorithms.astar import astar_search
from algorithms.bfs import bfs_search
from algorithms.dfs import dfs_search
from algorithms.greedy import greedy_search
from algorithms.hill_climbing import hill_climbing_search
from utils.graph_data import G, heuristic


SEARCHES = (
    (bfs_search, False),
    (dfs_search, False),
    (greedy_search, True),
    (astar_search, True),
    (hill_climbing_search, True),
)


def run_search(search, graph, start, goal, uses_heuristic, estimate=heuristic):
    if uses_heuristic:
        return search(graph, start, goal, estimate)
    return search(graph, start, goal)


@pytest.mark.parametrize(("search", "uses_heuristic"), SEARCHES)
def test_returns_consistent_result_structure(search, uses_heuristic):
    result = run_search(search, G, "College", "Airport", uses_heuristic)

    assert set(result) == {"path", "cost", "nodes_explored", "execution_time"}
    assert result["path"][0] == "College"
    assert result["path"][-1] == "Airport"
    assert result["cost"] is not None
    assert isinstance(result["nodes_explored"], int)
    assert result["nodes_explored"] >= 0
    assert isinstance(result["execution_time"], float)
    assert result["execution_time"] >= 0


def test_known_sample_route_and_cost():
    result = astar_search(G, "College", "Airport", heuristic)

    assert result["path"][0] == "College"
    assert result["path"][-1] == "Airport"
    assert result["cost"] == 6


def test_sample_coordinate_heuristic_is_admissible_for_edge_costs():
    for start in G:
        distances = nx.single_source_dijkstra_path_length(G, start, weight="weight")
        for goal, actual_cost in distances.items():
            assert heuristic(start, goal) <= actual_cost


@pytest.mark.parametrize(("search", "uses_heuristic"), SEARCHES)
def test_start_equal_goal(search, uses_heuristic):
    result = run_search(search, G, "College", "College", uses_heuristic)

    assert result["path"] == ["College"]
    assert result["cost"] == 0
    assert result["nodes_explored"] == 0


@pytest.mark.parametrize(("search", "uses_heuristic"), SEARCHES)
@pytest.mark.parametrize(("start", "goal"), [("Missing", "Airport"), ("College", "Missing")])
def test_invalid_start_or_goal(search, uses_heuristic, start, goal):
    result = run_search(search, G, start, goal, uses_heuristic)

    assert result["path"] == []
    assert result["cost"] is None
    assert result["nodes_explored"] == 0


@pytest.mark.parametrize(("search", "uses_heuristic"), SEARCHES)
def test_disconnected_goal(search, uses_heuristic):
    graph = nx.Graph()
    graph.add_edge("start", "nearby")
    graph.add_node("goal")

    result = run_search(
        search,
        graph,
        "start",
        "goal",
        uses_heuristic,
        estimate=lambda node, goal: {"start": 2, "nearby": 1, "goal": 0}[node],
    )

    assert result["path"] == []
    assert result["cost"] is None


@pytest.mark.parametrize(("search", "uses_heuristic"), SEARCHES)
def test_missing_edge_weights_default_to_one(search, uses_heuristic):
    graph = nx.Graph()
    graph.add_edge("start", "middle")
    graph.add_edge("middle", "goal", weight=2)

    result = run_search(
        search,
        graph,
        "start",
        "goal",
        uses_heuristic,
        estimate=lambda node, goal: {"start": 2, "middle": 1, "goal": 0}[node],
    )

    assert result["path"] == ["start", "middle", "goal"]
    assert result["cost"] == 3


@pytest.mark.parametrize(("search", "uses_heuristic"), SEARCHES)
def test_cyclic_graph_terminates_with_valid_route(search, uses_heuristic):
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            ("start", "first", 1),
            ("first", "second", 1),
            ("second", "start", 1),
            ("second", "goal", 1),
        ]
    )
    estimates = {"start": 3, "first": 2, "second": 1, "goal": 0}
    result = run_search(
        search,
        graph,
        "start",
        "goal",
        uses_heuristic,
        estimate=lambda node, goal: estimates[node],
    )

    assert result["path"][0] == "start"
    assert result["path"][-1] == "goal"
    assert len(result["path"]) <= len(graph)


def test_bfs_minimizes_edges_while_astar_minimizes_weighted_cost():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            ("start", "short", 1),
            ("short", "goal", 100),
            ("start", "detour", 2),
            ("detour", "middle", 2),
            ("middle", "goal", 2),
        ]
    )
    zero_heuristic = lambda node, goal: 0

    bfs_result = bfs_search(graph, "start", "goal")
    astar_result = astar_search(graph, "start", "goal", zero_heuristic)

    assert len(bfs_result["path"]) == 3
    assert bfs_result["cost"] == 101
    assert astar_result["cost"] == 6


def test_astar_reopens_a_node_when_a_better_path_is_discovered():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            ("start", "a", 2),
            ("start", "b", 1),
            ("b", "a", 0.5),
            ("a", "goal", 2),
            ("b", "goal", 5),
        ]
    )
    estimates = {"start": 2, "a": 2, "b": 2, "goal": 0}

    result = astar_search(
        graph,
        "start",
        "goal",
        lambda node, goal: estimates[node],
    )

    assert result["path"] == ["start", "b", "a", "goal"]
    assert result["cost"] == pytest.approx(3.5)


def test_hill_climbing_stops_at_a_local_optimum():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            ("start", "local", 1),
            ("start", "detour", 2),
            ("local", "dead_end", 1),
            ("detour", "goal", 1),
        ]
    )
    estimates = {"start": 3, "local": 1, "dead_end": 2, "detour": 2, "goal": 0}

    result = hill_climbing_search(
        graph,
        "start",
        "goal",
        lambda node, goal: estimates[node],
    )

    assert result["path"] == []
    assert result["cost"] is None
    assert result["nodes_explored"] == 2
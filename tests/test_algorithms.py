import networkx as nx
import pytest

from algorithms._common import make_admissible_heuristic
from algorithms.astar import astar_search
from algorithms.bfs import bfs_search
from algorithms.dfs import dfs_search
from algorithms.greedy import greedy_search
from algorithms.hill_climbing import hill_climbing_search
from graph_data import graph as member_graph
from graph_data import locations
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


def run_member_search(search, uses_heuristic, start, goal, graph=member_graph):
    if uses_heuristic:
        return search(graph, start, goal, make_admissible_heuristic(graph))
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


@pytest.mark.parametrize(("search", "uses_heuristic"), SEARCHES)
def test_member_graph_returns_valid_route_and_actual_weighted_cost(search, uses_heuristic):
    result = run_member_search(search, uses_heuristic, "A", "F")
    path = result["path"]

    assert path[0] == "A"
    assert path[-1] == "F"
    assert all(end in member_graph[start] for start, end in zip(path, path[1:]))
    assert result["cost"] == sum(
        member_graph[start][end] for start, end in zip(path, path[1:])
    )
    assert result["cost"] is not None
    assert set(result) == {"path", "cost", "nodes_explored", "execution_time"}


def test_member_graph_a_star_finds_minimum_weighted_cost():
    result = astar_search(
        member_graph,
        "A",
        "F",
        make_admissible_heuristic(member_graph),
    )

    assert result["path"] == ["A", "B", "D", "F"]
    assert result["cost"] == 12


def test_member_graph_heuristic_is_admissible():
    weighted_graph = nx.Graph()
    for start, neighbours in member_graph.items():
        for end, weight in neighbours.items():
            weighted_graph.add_edge(start, end, weight=weight)

    estimate = make_admissible_heuristic(member_graph)
    for start in member_graph:
        distances = nx.single_source_dijkstra_path_length(
            weighted_graph,
            start,
            weight="weight",
        )
        for goal, actual_cost in distances.items():
            assert estimate(start, goal) <= actual_cost


@pytest.mark.parametrize(("search", "uses_heuristic"), SEARCHES)
def test_member_graph_start_equal_goal(search, uses_heuristic):
    result = run_member_search(search, uses_heuristic, "C", "C")

    assert result["path"] == ["C"]
    assert result["cost"] == 0
    assert result["nodes_explored"] == 0


@pytest.mark.parametrize(("search", "uses_heuristic"), SEARCHES)
@pytest.mark.parametrize(("start", "goal"), [("Z", "F"), ("A", "Z")])
def test_member_graph_missing_locations(search, uses_heuristic, start, goal):
    result = run_member_search(search, uses_heuristic, start, goal)

    assert result["path"] == []
    assert result["cost"] is None
    assert result["nodes_explored"] == 0


@pytest.mark.parametrize(("search", "uses_heuristic"), SEARCHES)
def test_member_graph_unreachable_goal(search, uses_heuristic):
    disconnected_graph = {
        "start": {"nearby": 2},
        "nearby": {"start": 2},
        "goal": {},
    }
    result = run_member_search(
        search,
        uses_heuristic,
        "start",
        "goal",
        disconnected_graph,
    )

    assert result["path"] == []
    assert result["cost"] is None


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


def test_member_coordinates_are_display_only_for_route_heuristic():
    coordinate_distance = (
        (locations["A"]["x"] - locations["F"]["x"]) ** 2
        + (locations["A"]["y"] - locations["F"]["y"]) ** 2
    ) ** 0.5
    route_cost = astar_search(
        member_graph,
        "A",
        "F",
        make_admissible_heuristic(member_graph),
    )["cost"]

    assert coordinate_distance > route_cost


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
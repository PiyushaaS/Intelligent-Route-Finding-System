import networkx as nx
import math

G = nx.Graph()

G.add_weighted_edges_from([
    ("College", "Library", 2),
    ("College", "Hospital", 3),
    ("Library", "Hospital", 1),
    ("Library", "Railway Station", 2),
    ("Hospital", "Airport", 4),
    ("Railway Station", "Airport", 2),
])

POSITIONS = {
    "College": (0, 0),
    "Library": (2, 0),
    "Hospital": (2, 1),
    "Railway Station": (4, 0),
    "Airport": (5, 1),
}

def heuristic(node, goal):
    x1, y1 = POSITIONS[node]
    x2, y2 = POSITIONS[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
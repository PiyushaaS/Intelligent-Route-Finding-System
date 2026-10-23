# Graph and Route Data Management
# Member 1

# Locations connected by roads and their distances in km

graph = {
    "A": {"B": 4, "C": 2},
    "B": {"A": 4, "D": 5, "E": 7},
    "C": {"A": 2, "D": 8},
    "D": {"B": 5, "C": 8, "F": 3},
    "E": {"B": 7, "F": 2},
    "F": {"D": 3, "E": 2}
}

# Location details for display on the map

locations = {
    "A": {"name": "College Gate", "x": 50, "y": 300},
    "B": {"name": "Library", "x": 180, "y": 220},
    "C": {"name": "Main Road", "x": 180, "y": 380},
    "D": {"name": "Canteen", "x": 340, "y": 270},
    "E": {"name": "Bus Stop", "x": 400, "y": 150},
    "F": {"name": "Hostel", "x": 500, "y": 300}
}

def get_locations():
    return {
        node: details["name"]
        for node, details in locations.items()
    }

def is_valid_location(node):
    return node in graph

def get_neighbours(node):
    return graph.get(node, {})

def get_distance(start, end):
    return graph.get(start, {}).get(end)


if __name__ == "__main__":
    print("Available Locations:", get_locations())
    print("Neighbours of A:", get_neighbours("A"))
    print("Distance from A to B:", get_distance("A", "B"))
    print("Is Z valid?", is_valid_location("Z"))

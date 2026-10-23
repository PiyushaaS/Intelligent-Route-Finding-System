
from graph_data import graph, locations

def validate_graph():
    errors = []

    # Check if all graph nodes have location details
    if set(graph.keys()) != set(locations.keys()):
        errors.append("Graph and location IDs do not match.")

    # Check roads and distances
    for node, neighbours in graph.items():
        for neighbour, distance in neighbours.items():

            if neighbour not in graph:
                errors.append(
                    f"Unknown location: {neighbour}"
                )
                continue

            if not isinstance(distance, (int, float)) or distance <= 0:
                errors.append(
                    f"Invalid distance: {node} to {neighbour}"
                )
                continue

            reverse_distance = graph[neighbour].get(node)

            if reverse_distance != distance:
                errors.append(
                    f"Distance mismatch: {node} to {neighbour}"
                )

    if errors:
        print("Graph validation failed:")
        for error in errors:
            print("-", error)
    else:
        print("Graph validation successful!")

if __name__ == "__main__":
    validate_graph()

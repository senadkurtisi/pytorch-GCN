import numpy as np
import igraph as ig


def visualize_graph(edges, node_labels):
    """ Most of the code within this function was taken and "fine-tuned"
        from the Aleksa Gordić's repo:
        https://github.com/gordicaleksa/pytorch-GAT
    """
    num_of_nodes = len(node_labels)
    edge_index_tuples = list(zip(edges[:, 0], edges[:, 1]))

    ig_graph = ig.Graph()
    ig_graph.add_vertices(num_of_nodes)
    ig_graph.add_edges(edge_index_tuples)

    # Prepare the visualization settings dictionary
    visual_style = {"bbox": (1000, 1000), "margin": 50}

    # Normalization of the edge weights
    edge_weights_raw = np.clip(np.log(np.asarray(ig_graph.edge_betweenness()) + 1e-16), a_min=0, a_max=None)
    edge_weights_raw_normalized = edge_weights_raw / np.max(edge_weights_raw)
    edge_weights = [w/3 for w in edge_weights_raw_normalized]
    visual_style["edge_width"] = edge_weights

    # A simple heuristic for vertex size. Multiplying with 0.75 gave decent visualization
    visual_style["vertex_size"] = [0.75*deg for deg in ig_graph.degree()]

    # Color map for each class
    cora_label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}
    visual_style["vertex_color"] = [cora_label_to_color_map[label] for label in node_labels]

    # Display the cora graph
    visual_style["layout"] = ig_graph.layout_kamada_kawai()
    out = ig.plot(ig_graph, **visual_style)
    out.save("cora_visualized.png")

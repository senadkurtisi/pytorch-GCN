import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Color map for each class
cora_label_to_color_map = {0: "red", 1: "blue", 2: "green",
                           3: "orange", 4: "yellow", 5: "pink", 6: "gray"}


def visualize_embedding_tSNE(labels, out_features, num_classes):
    """ https://github.com/gordicaleksa/pytorch-GAT """
    node_labels = labels.cpu().numpy()
    out_features = out_features.cpu().numpy()
    t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(out_features)

    plt.figure()
    for class_id in range(num_classes):
        plt.scatter(t_sne_embeddings[node_labels == class_id, 0],
                    t_sne_embeddings[node_labels == class_id, 1], s=20,
                    color=cora_label_to_color_map[class_id],
                    edgecolors='black', linewidths=0.15)

    plt.axis("off")
    plt.title("t-SNE projection of the learned features")
    plt.show()


def visualize_graph(edges, node_labels, save=False):
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

    visual_style["vertex_color"] = [cora_label_to_color_map[label] for label in node_labels]

    # Display the cora graph
    visual_style["layout"] = ig_graph.layout_kamada_kawai()
    out = ig.plot(ig_graph, **visual_style)

    if save:
        out.save("cora_visualized.png")


def visualize_validation_performance(val_acc, val_loss):
    f, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    axs[0].plot(val_loss, linewidth=2, color="red")
    axs[0].set_title("Validation loss")
    axs[0].set_ylabel("Cross Entropy Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].grid()

    axs[1].plot(val_acc, linewidth=2, color="red")
    axs[1].set_title("Validation accuracy")
    axs[1].set_ylabel("Acc")
    axs[1].set_xlabel("Epoch")
    axs[1].grid()

    plt.show()

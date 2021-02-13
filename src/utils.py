import torch

import numpy as np
import scipy.sparse as sparse


def accuracy(output, labels):
    y_pred = output.max(1)[1].type_as(labels)
    correct = y_pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def prepare_dataset(labels, num_classes, config):
    """ Splits the loaded dataset into train/validation/test sets. """
    if not config.follow_paper:
        # Follow train/val/test indices as in the official implementation
        # on GitHub: https://github.com/tkipf/pygcn
        train_set = range(140)
        validation_set = range(200, 500)
        test_set = range(500, 1500)
    else:
        # https://arxiv.org/pdf/1609.02907.pdf
        # The original paper proposes that the training set is composed
        # out of 20 samples per class -> 140 samples, but the indices
        # above (range(140)) do not contain 20 samples per class
        # The remaining val/test indices were selected empirically
        classes = [ind for ind in range(num_classes)]
        train_set = []

        # Construct train set (indices) out of 20 samples per each class
        for class_label in classes:
            target_indices = torch.nonzero(labels == class_label, as_tuple=False).tolist()
            train_set += [ind[0] for ind in target_indices[:config.train_size_per_class]]

        # Extract the remaining samples
        validation_test_set = [ind for ind in range(len(labels)) if ind not in train_set]
        # Split the remaining samples into validation/test set
        validation_set = validation_test_set[:config.validation_size]
        test_set = validation_test_set[config.validation_size:config.validation_size+config.test_size]

    return train_set, validation_set, test_set


def enumerate_labels(labels):
    """ Converts the labels from the original
        string form to the integer [0:MaxLabels-1]
    """
    unique = list(set(labels))
    labels = np.array([unique.index(label) for label in labels])
    return labels


def normalize_adjacency(adj):
    """ Normalizes the adjacency matrix according to the
        paper by Kipf et al.
        https://arxiv.org/pdf/1609.02907.pdf
    """
    adj = adj + sparse.eye(adj.shape[0])

    node_degrees = np.array(adj.sum(1))
    node_degrees = np.power(node_degrees, -0.5).flatten()
    node_degrees[np.isinf(node_degrees)] = 0.0
    node_degrees[np.isnan(node_degrees)] = 0.0
    degree_matrix = sparse.diags(node_degrees, dtype=np.float32)

    adj = degree_matrix @ adj @ degree_matrix
    return adj


def convert_scipy_to_torch_sparse(matrix):
    matrix_helper_coo = matrix.tocoo().astype('float32')
    data = torch.FloatTensor(matrix_helper_coo.data)
    rows = torch.LongTensor(matrix_helper_coo.row)
    cols = torch.LongTensor(matrix_helper_coo.col)
    indices = torch.vstack([rows, cols])

    shape = torch.Size(matrix_helper_coo.shape)
    matrix = torch.sparse.FloatTensor(indices, data, shape)
    return matrix


def load_data(config):
    """ Loads the graph data and stores them using
        efficient sparse matrices approach.
    """
    print("Loading Cora dataset...")
    ###############################
    # Loading Graph Nodes Data
    ###############################
    raw_nodes_data = np.genfromtxt(config.nodes_path, dtype="str")
    raw_node_ids = raw_nodes_data[:, 0].astype('int32')  # unique identifier of each node
    raw_node_labels = raw_nodes_data[:, -1]
    labels_enumerated = enumerate_labels(raw_node_labels)  # target labels as integers
    node_features = sparse.csr_matrix(raw_nodes_data[:, 1:-1], dtype="float32")

    ################################
    # Loading Graph Structure Data
    ################################
    ids_ordered = {raw_id: order for order, raw_id in enumerate(raw_node_ids)}
    raw_edges_data = np.genfromtxt(config.edges_path, dtype="int32")
    edges_ordered = np.array(list(map(ids_ordered.get, raw_edges_data.flatten())),
                             dtype='int32').reshape(raw_edges_data.shape)
    ####################
    # ADJACENCY MATRIX
    ####################
    adj = sparse.coo_matrix((np.ones(edges_ordered.shape[0]), (edges_ordered[:, 0], edges_ordered[:, 1])),
                            shape=(labels_enumerated.shape[0], labels_enumerated.shape[0]),
                            dtype=np.float32)
    # Make the adjacency matrix symmetric
    adj = adj + adj.T.multiply(adj.T > adj)
    adj = normalize_adjacency(adj)

    ####################################
    # Adapt the data to PyTorch format
    ####################################
    features = torch.FloatTensor(node_features.toarray())
    labels = torch.LongTensor(labels_enumerated)
    adj = convert_scipy_to_torch_sparse(adj)

    print("Dataset loaded.")

    return features, labels, adj, edges_ordered

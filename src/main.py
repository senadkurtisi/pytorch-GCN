from globals import config
from utils import *


if __name__ == "__main__":
    features, labels, adj = load_data(config)
    NUM_CLASSES = int(labels.max().item() + 1)

    dataset = prepare_dataset(features, labels, NUM_CLASSES, config)

    model = GCN(features.shape[1], config.hidden_dim,
                NUM_CLASSES, config.dropout, config.use_bias)

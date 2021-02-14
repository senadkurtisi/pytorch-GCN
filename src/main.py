from model import GCN

from utils import *
from globals import config
from training_evaluation import *
from visualization import *

if __name__ == "__main__":
    features, labels, adj, edges = load_data(config)
    visualize_graph(edges, labels.cpu().tolist(), save=False)
    NUM_CLASSES = int(labels.max().item() + 1)

    train_set_ind, val_set_ind, test_set_ind = prepare_dataset(labels, NUM_CLASSES, config)

    model = GCN(features.shape[1], config.hidden_dim,
                NUM_CLASSES, config.dropout, config.use_bias)

    if not config.multiple_runs:
        print("Started training with 1 run.",
              f"Early stopping: {'Yes' if config.use_early_stopping else 'No'}")
        val_acc, val_loss = training_loop(model, features, labels, adj, train_set_ind, val_set_ind, config)
        out_features = evaluate_on_test(model, features, labels, adj, test_set_ind, config)

        visualize_validation_performance(val_acc, val_loss)
        visualize_embedding_tSNE(labels, out_features, NUM_CLASSES)
    else:
        print(f"Started training with {config.num_of_runs} runs.",
              f"Early stopping: {'Yes' if config.use_early_stopping else 'No'}")
        multiple_runs(model, features, labels, adj,
                      [train_set_ind, val_set_ind, test_set_ind],
                      config, training_loop, evaluate_on_test)

import torch
from argparse import ArgumentParser

use_cuda = torch.cuda.is_available()

parser = ArgumentParser()
parser.add_argument("--cuda", type=bool, default=use_cuda)
parser.add_argument("--nodes_path", type=str, default="../data/cora.content")
parser.add_argument("--edges_path", type=str, default="../data/cora.cites")
parser.add_argument("--hidden_dim", type=int, default=16)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--use_bias", type=bool, default=True)
parser.add_argument("--train_size_per_class", type=int, default=20)
parser.add_argument("--validation_size", type=int, default=300)

config = parser.parse_args()

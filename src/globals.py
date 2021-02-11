import torch
from argparse import ArgumentParser

use_cuda = torch.cuda.is_available()

parser = ArgumentParser()
parser.add_argument("--cuda", type=bool, default=use_cuda)
parser.add_argument("--nodes_path", type=str, default="../data/cora.content")
parser.add_argument("--edges_path", type=str, default="../data/cora.cites")

config = parser.parse_args()

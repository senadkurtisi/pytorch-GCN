# PyTorch Graph Convolutional Network
PyTorch implementation of the [Graph Convolutional Network](https://arxiv.org/abs/1609.02907) paper by Kipf et al.

### Table of Contents
* [Graph Neural Networks](#graph-neural-networks)
* [Dataset](#dataset)
* [GCN Architecture](#gcn-architecture)
* [Results](#results)
* [Instructions](#setup-&-instructions)
* [Acknowledgements](#acknowledgements)

## Graph Neural Networks 
Graph Neural networks are a family of neural networks that can deal with the data which isn't organized in some strict manner. </br>
For example images are organized in a MxN grid, where M is the number of vertical and N is the number of horizontal pixels. </br>Time-series data is organized in some sequential manner. </br>
</br>
On the other hand there is a specific class of problems which can be represented as nodes/vertices which can (but don't have to) be connected via some edges. An example of this is friendship representation of some social media platform. Since there aren't any hard constraints on how the graph which represents, for example that social media platform, should look like we must use a specific family of neural networks called **Graph Neural Networks** or **GNNs** for short.

## Dataset
The dataset used in this implementation is **Cora**. Cora consists out of **2708 nodes** and **5429 edges**. Each node represents a particular science paper, and each edge represents the citation between the two connected papers. These edges are directioned in the original form, since paper A cites paper B, so the direction of edge has certain meaning, but the authors in the Kipf et al. transformed all of the edges in the undirectioned form, and so did I.
</br>

### Visualization
Below you can see the Cora dataset visualized. The size of each node is directly proportional to the degree of that node. The degree of a node corresponds to the number of outgoing and ingoing edges of a node. Since this "transformed" edges are undirected, degree of a node is just a number of edges connected to that node.

<img src="imgs/cora_visualized.png" width="750" height="750">


## GCN Architecture

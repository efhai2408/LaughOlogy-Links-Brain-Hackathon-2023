## Network parameter testing

## mean clustering 
The networkx.clustering function calculates the clustering coefficient for each node in the graph. 

Each node's clustering coefficient has a range of 0 to 1 (0 indicates the node's neighbors didn't connect with each other, while 1 indicates that this node's neighbors connected with all possible connections).

We can find the mean value of every clustering coefficient of every node in the graph with the code below.
```python
clustering_coeffsA = nx.clustering(A)
mean_clusteringA = statistics.mean(clustering_coeffsA.values())
```
References
- https://www.bluebirz.net/th/note-of-data-science-training-ep-9-th/#clustering
- https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html#networkx.algorithms.cluster.clustering
### Graph Density

Networkx.density returns the density of a graph.

The formula to calculate graph density between directed graphs and undirected graphs is different; in this study, we use the formula for directed graphs.

The density value has a range of 0 to 1. The value is 0 when the graph has no edge and 1 when the graph is fully connected.
References
- https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.density.html
### Assortativity
Assortativity measures the tendency for nodes to connect to other nodes with similar properties, such as the node degree.

We use degree_assortativity_coefficient in this study; it returns the assortativity of the graph by degree in the range between -1 and 1. 
References 
- https://math.libretexts.org/Bookshelves/Scientific_Computing_Simulations_and_Modeling/Book%3A_Introduction_to_the_Modeling_and_Analysis_of_Complex_Systems_(Sayama)/17%3A_Dynamical_Networks_II__Analysis_of_Network_Topologies/17.06%3A_Assortativity
- https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.assortativity.degree_assortativity_coefficient.html#networkx.algorithms.assortativity.degree_assortativity_coefficient
## Network comparison testing

### Graph Edit Distance
Graph Edit Distance (GED) is one strategy for network comparison between two graphs; it is the minimum cost of an edit path (sequence of node and edge) between two graphs. 
References
- https://www.sciencedirect.com/science/article/abs/pii/S095070511830488X
- https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.similarity.graph_edit_distance.html

### Isomorphic 
A graph can exist in different forms with the same number of edges, nodes, and graph connectivity. We can call two graphs isomorphic if they are the same.

The networkx.is_isomorphic function returns True if two graphs are isomorphic, and false otherwise.
References
- https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.isomorphism.is_isomorphic.html
- https://www.tutorialspoint.com/graph_theory/graph_theory_isomorphism.htm
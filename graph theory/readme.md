Graph theory is the mathematical study of graphs.

A graph structure consists of **vertices (or nodes)** and **edges** (it represents relationships between objects).

graph G = (V, E), where:
  - V is set of vertices
  - E is a set of edges.

Algorithm - Shortest Path

Dijkstra's algorithm finds the shortest path between nodes in weighted graph. It is greedy algorithm that always exploresthe closet unvisited node first.

Overview of STRING Database

A STRING database is a web resource focused on **protein-protein interactions** and fucntional associations.

![Network nodes](https://github.com/user-attachments/assets/f34bce42-69b9-4b52-b730-cb17f15e5583)
randomly generated output from the STRING database

This is a protein interaction network represented as a graph. The network consists of nodes and edges, where nodes represent proteins and edges represent protein-protein associations. The network was generated starting from an uncharacterized protein (the red node ending with 'N10') and shows its interaction partners. The network contains 11 nodes and 31 edges. Inteaction score is medium confidence (0.4) which is mean only protein interactions with ≥40% confidence are shown. 

NetworkX overview

In the web resource of NetworkX there is a definition. It is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

To deepen my understanding of how NetworkX implements graph traversal, I chose to examine Beam Search from its extensive gallery of algorithms. The specific implementation showcased is a Progressive Widening Beam Search.

This variant involves performing a series of beam searches, starting with a very small beam width and then progressively increasing (widening) the beam on each iteration. The process returns the first node that matches a user-defined termination condition.

On the web NetworkX's page, a directed graph is generated, and the progressive widening search fucntion is demonstrated. 

![beam search ](https://github.com/user-attachments/assets/9b094e49-b1fb-4d04-96bb-d81d0ce8f784)

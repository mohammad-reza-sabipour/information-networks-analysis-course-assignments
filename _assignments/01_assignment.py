import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import math

edges = []
# url : https://networks.skewed.de/net/karate

with open('edges.csv', 'r') as f:
    reader = csv.DictReader(f)
    for line in reader:
        edges.append((int(line['# source']), int(line[' target'])))
        
g = ig.Graph(35)
g.add_edges(edges)
betweenness = [0 if math.isnan(v) else v for v in g.betweenness()]
g.vs['betweenness'] = betweenness
#g.write_graphml('graph.graphml')
index = [i for i in range(0,35)]
sns.barplot(x=index, y=betweenness)
plt.show()
communities = g.community_fastgreedy()
print(communities.as_clustering().modularity)

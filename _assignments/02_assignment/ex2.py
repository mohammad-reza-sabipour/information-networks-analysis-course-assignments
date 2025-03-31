import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler



g = ig.Graph.Famous('Zachary')
e = g.get_edgelist()
v_count = g.vcount()

jaccard = np.zeros([v_count, v_count])
preferential_attachment = np.zeros([v_count, v_count])
adamic_adar = np.zeros([v_count, v_count])

for i in range(v_count):
    v = set(g.neighbors(i, mode='all'))
    for j in range(i+1, v_count):
        u = set(g.neighbors(j, mode='all'))
        jaccard[i,j] = len(u.intersection(v))/len(u.union(v))
        preferential_attachment[i,j] = len(u) * len(v)
        common_neighbors = v.intersection(u)
        score = sum(1/np.log(g.degree(w)) for w in common_neighbors if g.degree(w) > 1 )
        adamic_adar[i,j] = score

X = np.column_stack([
    jaccard[np.triu_indices(v_count, k=1)],
    preferential_attachment[np.triu_indices(v_count, k=1)],
    adamic_adar[np.triu_indices(v_count, k=1)]
])

Y = np.array([(1 if (i, j) in e or (j, i) in e else 0) for i in range(v_count) for j in range(i+1, v_count)])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=4, test_size=0.2)

"""scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)"""

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
for i in range(len(Y_pred)):
    if Y_pred[i] != Y_test[i]:
        print(X_test[i])

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

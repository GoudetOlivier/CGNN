import cgnn
import sys
import pandas as pd

# Params
cgnn.SETTINGS.GPU = True
cgnn.SETTINGS.NB_GPU = 1
cgnn.SETTINGS.NB_JOBS = 1

#Setting for CGNN-Fourier
cgnn.SETTINGS.use_Fast_MMD = True
cgnn.SETTINGS.NB_RUNS = 64 

#Setting for CGNN-MMD
# cgnn.SETTINGS.use_Fast_MMD = False
#cgnn.SETTINGS.NB_RUNS = 32


datafile = "Example_graph_numdata.csv"
skeletonfile = "Example_graph_skeleton.csv"

print("Processing " + datafile + "...")
undirected_links = pd.read_csv(skeletonfile)

umg = cgnn.UndirectedGraph(undirected_links)
data = pd.read_csv(datafile)

GNN = cgnn.GNN(backend="TensorFlow")
p_directed_graph = GNN.orient_graph(data, umg, printout=datafile + '_printout.csv')
gnn_res = pd.DataFrame(p_directed_graph.get_list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
gnn_res.to_csv(datafile + "_pairwise_predictions.csv")

CGNN = cgnn.CGNN(backend="TensorFlow")
directed_graph = CGNN.orient_directed_graph(data, p_directed_graph)
cgnn_res = pd.DataFrame(directed_graph.get_list_edges(descending=True), columns=['Cause', 'Effect', 'Score'])
cgnn_res.to_csv(datafile + "_predictions.csv")

print('Processed ' + datafile)

import cgnn
from cgnn.utils import CCEPC_PairsFileReader as CC
import pandas as pd

# Params
cgnn.SETTINGS.GPU = True
cgnn.SETTINGS.NB_GPU = 2
cgnn.SETTINGS.NB_JOBS = 8
cgnn.SETTINGS.h_layer_dim = 30

#Setting for CGNN-Fourier
cgnn.SETTINGS.use_Fast_MMD = True
cgnn.SETTINGS.NB_RUNS = 64 

#Setting for CGNN-MMD
# cgnn.SETTINGS.use_Fast_MMD = False
#cgnn.SETTINGS.NB_RUNS = 32

datafile = "Example_pairwise_pairs.csv"

print("Processing " + datafile + "...")
data = CC(datafile, scale=True)
model = cgnn.GNN(backend="TensorFlow")
predictions = model.predict_dataset(data, printout=datafile + '_printout.csv')
predictions = pd.DataFrame(predictions, columns=["Predictions"])

print('Processed ' + datafile)
predictions.to_csv(datafile + "_predictions_GNN.csv")

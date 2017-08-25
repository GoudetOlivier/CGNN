Code provided to reproduce the results from the article "Learning Functional Causal Models with Generative Neural Networks"

Requirements:
numpy
scipy
scikit-learn
tensorflow
joblib
pandas

In order to run the CGNN and launch the experiments:
1) First install the CGNN package. Enter in the CGNN repertory. Run the command line "python setup.py install develop --user"

2) Launch the example python script for pairwise inference: "python run_GNN_pairwise_inference.py"

3) Launch the example python script for graph reconstruction from a skeleton: "python run_CGNN_graph.py"

4) Launch the example python script for graph reconstruction in presence if hidden variables: "python run_CGNN_graph_hidden_variables.py"

5) The complete datasets used in the article may be found at the following url:
- pairwise datasets : http://dx.doi.org/10.7910/DVN/3757KX
- graph datasets : http://dx.doi.org/10.7910/DVN/UZMB69

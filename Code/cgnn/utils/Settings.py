"""
Settings file for CGNN algorithm
Defining all global variables
Authors : Anonymous Author
Date : 8/05/2017
"""


class DefaultSettings(object):
    __slots__ = ("h_layer_dim",
                 "train_epochs",
                 "test_epochs",
                 "NB_RUNS",
                 "NB_JOBS",
                 "GPU",
                 "NB_GPU",
                 "GPU_OFFSET",
                 "learning_rate",
                 "init_weights",
                 "use_Fast_MMD",
                 "nb_vectors_approx_MMD",
                 "complexity_graph_param",
		          "max_nb_points")

    def __init__(self):  # Define here the default values of the parameters
        self.NB_RUNS = 32
        self.NB_JOBS = 1
        self.GPU = True
        self.NB_GPU = 1
        self.GPU_OFFSET = 0
        self.learning_rate = 0.01
        self.init_weights = 0.05
        self.max_nb_points = 1500

        # CGNN
        self.h_layer_dim = 20
        self.train_epochs = 1000
        self.test_epochs = 500
        self.use_Fast_MMD = False
        self.nb_vectors_approx_MMD = 100
        self.complexity_graph_param = 0.00005



SETTINGS = DefaultSettings()

"""
CGNN_graph_model
Author : Olivier Goudet & Diviyan Kalainathan
Ref :
Date : 09/5/17
"""

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import Parallel, delayed
from pandas import DataFrame
from sklearn.preprocessing import scale

from .GNN import GNN
from .utils.Loss import MMD_loss_tf, Fourier_MMD_Loss_tf
from .utils.Settings import SETTINGS
from .GraphModel import GraphModel


def init(size, **kwargs):
    """ Initialize a random tensor, normal(0,kwargs(SETTINGS.init_weights)).

    :param size: Size of the tensor
    :param kwargs: init_std=(SETTINGS.init_weights) Std of the initialized normal variable
    :return: Tensor
    """
    init_std = kwargs.get('init_std', SETTINGS.init_weights)
    return tf.random_normal(shape=size, stddev=init_std)


class CGNN_decomposable_tf(object):
    def __init__(self, N, n_parents, n_all_other_variables, run, idx, **kwargs):
        """ Build the tensorflow graph of the CGNN structure

        :param N: Number of points
        :param graph: Graph to be run
        :param run: number of the run (only for print)
        :param idx: number of the idx (only for print)
        :param kwargs: learning_rate=(SETTINGS.learning_rate) learning rate of the optimizer
        :param kwargs: h_layer_dim=(SETTINGS.h_layer_dim) Number of units in the hidden layer
        :param kwargs: use_Fast_MMD=(SETTINGS.use_Fast_MMD) use fast MMD option
        :param kwargs: nb_vectors_approx_MMD=(SETTINGS.nb_vectors_approx_MMD) nb vectors
        """
        learning_rate = kwargs.get('learning_rate', SETTINGS.learning_rate)
        h_layer_dim = kwargs.get('h_layer_dim', SETTINGS.h_layer_dim)
        use_Fast_MMD = kwargs.get('use_Fast_MMD', SETTINGS.use_Fast_MMD)
        nb_vectors_approx_MMD = kwargs.get('nb_vectors_approx_MMD', SETTINGS.nb_vectors_approx_MMD)

        self.run = run
        self.idx = idx

        self.all_other_variables = tf.placeholder(tf.float32, shape=[None, n_all_other_variables])
        self.target_variables = tf.placeholder(tf.float32, shape=[None, 1])
        self.parents_variables = tf.placeholder(tf.float32, shape=[None, n_parents])

        W_in = tf.Variable(init([n_parents + 1, h_layer_dim], **kwargs))
        b_in = tf.Variable(init([h_layer_dim], **kwargs))
        W_out = tf.Variable(init([h_layer_dim, 1], **kwargs))
        b_out = tf.Variable(init([1], **kwargs))

        theta_G = [W_in, b_in, W_out, b_out]

        input_v = tf.concat([self.parents_variables, tf.random_normal([N, 1], mean=0, stddev=1)], 1)

        out_v = tf.nn.relu(tf.matmul(input_v, W_in) + b_in)
        out_v = tf.matmul(out_v, W_out) + b_out

        self.all_real_variables = tf.concat([self.all_other_variables, self.target_variables], 1)
        self.all_generated_variables = tf.concat([self.all_other_variables, out_v] , 1)

        if(use_Fast_MMD):
            self.G_dist_loss_xcausesy = Fourier_MMD_Loss_tf(self.all_real_variables, self.all_generated_variables,nb_vectors_approx_MMD)
        else:
            self.G_dist_loss_xcausesy = MMD_loss_tf(self.all_real_variables, self.all_generated_variables)

        self.G_solver_xcausesy = (tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.G_dist_loss_xcausesy,
                                                  var_list=theta_G))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, data_target, data_all_other_variables, data_parents, verbose=True, **kwargs):
        """ Train the initialized model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :param kwargs: train_epochs=(SETTINGS.train_epochs) number of train epochs
        :return: None
        """
        train_epochs = kwargs.get('train_epochs', SETTINGS.train_epochs)
        for it in range(train_epochs):

            _, G_dist_loss_xcausesy_curr = self.sess.run(
                [self.G_solver_xcausesy, self.G_dist_loss_xcausesy],
                feed_dict={self.all_other_variables: data_all_other_variables, self.target_variables: data_target, self.parents_variables: data_parents }
            )

            if verbose:
                if it % 100 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                          format(self.idx, self.run,
                                 it, G_dist_loss_xcausesy_curr))

    def evaluate(self, data_target, data_all_other_variables, data_parents, verbose=True, **kwargs):
        """ Test the model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :param kwargs: test_epochs=(SETTINGS.test_epochs) number of test epochs
        :return: mean MMD loss value of the CGNN structure on the data
        """
        test_epochs = kwargs.get('test_epochs', SETTINGS.test_epochs)
        sumMMD_tr = 0

        for it in range(test_epochs):

            MMD_tr = self.sess.run([self.G_dist_loss_xcausesy], feed_dict={self.all_other_variables: data_all_other_variables, self.target_variables: data_target, self.parents_variables: data_parents })

            sumMMD_tr += MMD_tr[0]

            if verbose and it % 100 == 0:
                print('Pair:{}, Run:{}, Iter:{}, score:{}'
                          .format(self.idx, self.run, it, MMD_tr[0]))

        tf.reset_default_graph()

        return sumMMD_tr / test_epochs

    def generate(self, data, **kwargs):

        generated_variables = self.sess.run([self.all_generated_variables], feed_dict={self.all_real_variables: data})

        tf.reset_default_graph()
        return np.array(generated_variables)[0, :, :]


def run_CGNN_decomposable_tf(df_data, graph, target, idx=0, run=0, **kwargs):
    """ Execute the CGNN, by init, train and eval either on CPU or GPU

    :param df_data: data corresponding to the graph
    :param graph: Graph to be run
    :param run: number of the run (only for print)
    :param idx: number of the idx (only for print)
    :param kwargs: gpu=(SETTINGS.GPU) True if GPU is used
    :param kwargs: nb_gpu=(SETTINGS.nb_gpu) Number of available GPUs
    :param kwargs: gpu_offset=(SETTINGS.gpu_offset) number of gpu offsets
    :return: MMD loss value of the given structure after training
    """
    gpu = kwargs.get('gpu', SETTINGS.GPU)
    nb_gpu = kwargs.get('nb_gpu', SETTINGS.NB_GPU)
    gpu_offset = kwargs.get('gpu_offset', SETTINGS.GPU_OFFSET)

    data_target = df_data[target].as_matrix()
    data_target = data_target.reshape(data_target.shape[0], 1)


    list_all_other_variables = graph.get_list_nodes()
    list_all_other_variables.remove(target)

    data_all_other_variables = df_data[list_all_other_variables].as_matrix()
    data_all_other_variables = data_all_other_variables.reshape(data_all_other_variables.shape[0], data_all_other_variables.shape[1])


    list_parents = graph.get_parents(target)
    data_parents = df_data[list_parents].as_matrix()
    data_parents = data_parents.reshape(data_parents.shape[0], data_parents.shape[1])

    n_parents =  data_parents.shape[1]
    n_all_other_variables = data_all_other_variables.shape[1]

    print("target " + str(target))
    print(list_parents)

    if gpu:
        with tf.device('/gpu:' + str(gpu_offset + run % nb_gpu)):
            model = CGNN_decomposable_tf(df_data.shape[0], n_parents, n_all_other_variables, run, idx, **kwargs)
            model.train(data_target, data_all_other_variables, data_parents, **kwargs)
            score_node = model.evaluate(data_target, data_all_other_variables, data_parents, **kwargs)
    else:
        model = CGNN_decomposable_tf(df_data.shape[0], n_parents, n_all_other_variables, run, idx, **kwargs)
        model.train(data_target, data_all_other_variables, data_parents, **kwargs)
        score_node = model.evaluate(data_target, data_all_other_variables, data_parents, **kwargs)

    print("score node : " + str(score_node))

    return score_node


def hill_climbing(graph, data, run_cgnn_function, **kwargs):
    """ Optimize graph using CGNN with a hill-climbing algorithm

    :param graph: graph to optimize
    :param data: data
    :param run_cgnn_function: name of the CGNN function (depending on the backend)
    :param kwargs: nb_jobs=(SETTINGS.NB_JOBS) number of jobs
    :param kwargs: nb_runs=(SETTINGS.NB_RUNS) number of runs, of different evaluations
    :return: improved graph
    """
    nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
    nb_runs = kwargs.get("nb_runs", SETTINGS.NB_RUNS)
    loop = 0
    tested_configurations = [graph.get_dict_nw()]
    improvement = True
    result = []
    list_eval_nodes = graph.get_list_nodes()

    for target in list_eval_nodes:
        result_node = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
            data, graph, target, 0, run, **kwargs) for run in range(nb_runs))

        score_node = np.mean([i for i in result_node if np.isfinite(i)])
        graph.score_node[target] = score_node

    globalscore = graph.get_total_score()

    print("Graph score : " + str(globalscore))

    while improvement:
        loop += 1
        improvement = False
        list_edges = graph.get_list_edges()
        for idx_pair in range(len(list_edges)):
            edge = list_edges[idx_pair]
            test_graph = deepcopy(graph)
            test_graph.reverse_edge(edge[0], edge[1])

            if (test_graph.is_cyclic()
                or test_graph.get_dict_nw() in tested_configurations):
                print('No Evaluation for {}'.format([edge]))
            else:
                print('Edge {} in evaluation :'.format(edge))
                tested_configurations.append(test_graph.get_dict_nw())

                for target in [edge[0],edge[1]]:
                    result_node = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(data, test_graph, target, idx_pair, run, **kwargs) for run in range(nb_runs))
                    score_node = np.mean([i for i in result_node if np.isfinite(i)])
                    test_graph.score_node[target] = score_node

                score_network = test_graph.get_total_score()

                print("Current score : " + str(score_network))
                print("Best score : " + str(globalscore))

                if score_network < globalscore:
                    graph = deepcopy(test_graph)
                    improvement = True
                    print('Edge {} got reversed !'.format(edge))
                    globalscore = score_network

                df_edge_result = pd.DataFrame(graph.get_list_edges(),
                                              columns=['Cause', 'Effect',
                                                       'Weight'])
                df_edge_result.to_csv('results/DAG-loop{}.csv'.format(loop), index=False)

    return graph


def exploratory_hill_climbing(graph, data, run_cgnn_function, **kwargs):
    """ Optimize graph using CGNN with a hill-climbing algorithm

    :param graph: graph to optimize
    :param data: data
    :param run_cgnn_function: name of the CGNN function (depending on the backend)
    :param kwargs: nb_jobs=(SETTINGS.NB_JOBS) number of jobs
    :param kwargs: nb_runs=(SETTINGS.NB_RUNS) number of runs, of different evaluations
    :return: improved graph
    """
    nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
    nb_runs = kwargs.get("nb_runs", SETTINGS.NB_RUNS)

    nb_loops = 150
    exploration_factor = 10  # Average of number of edges to reverse at the beginning.
    assert exploration_factor < len(graph.get_list_edges())

    loop = 0
    tested_configurations = [graph.get_dict_nw()]
    result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
        data, graph, 0, run, **kwargs) for run in range(nb_runs))

    score_network = np.mean([i for i in result_pairs if np.isfinite(i)])
    globalscore = score_network

    print("Graph score : " + str(globalscore))

    while loop < nb_loops:
        loop += 1
        list_edges = graph.get_list_edges()

        possible_solution=False
        while not possible_solution:
            test_graph = deepcopy(graph)
            selected_edges = np.random.choice(len(list_edges),
                                              max(int(exploration_factor * ((nb_loops-loop)/nb_loops)**2), 1))
            for edge in list_edges[selected_edges]:
                test_graph.reverse_edge()
            if not (test_graph.is_cyclic()
                    or test_graph.get_dict_nw() in tested_configurations):
                possible_solution = True

            print('Reversed Edges {} in evaluation :'.format(list_edges[selected_edges]))
            tested_configurations.append(test_graph.get_dict_nw())
            result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                data, test_graph, loop, run, **kwargs) for run in range(nb_runs))

            score_network = np.mean([i for i in result_pairs if np.isfinite(i)])

            print("Current score : " + str(score_network))
            print("Best score : " + str(globalscore))

            if score_network < globalscore:
                graph.reverse_edge(edge[0], edge[1])
                print('Edge {} got reversed !'.format(list_edges[selected_edges]))
                globalscore = score_network

    return graph


def tabu_search(graph, data, run_cgnn_function, **kwargs):
    """ Optimize graph using CGNN with a hill-climbing algorithm

    :param graph: graph to optimize
    :param data: data
    :param run_cgnn_function: name of the CGNN function (depending on the backend)
    :param kwargs: nb_jobs=(SETTINGS.NB_JOBS) number of jobs
    :param kwargs: nb_runs=(SETTINGS.NB_RUNS) number of runs, of different evaluations
    :return: improved graph
    """
    nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
    nb_runs = kwargs.get("nb_runs", SETTINGS.NB_RUNS)
    raise ValueError('Not Yet Implemented')


class CGNN_decomposable(GraphModel):
    """
    CGNN Model ; Using generative models, generate the whole causal graph and improve causal
    direction predictions in the graph.
    """

    def __init__(self, backend='PyTorch'):
        """ Initialize the CGNN Model.

        :param backend: Choose the backend to use, either 'PyTorch' or 'TensorFlow'
        """
        super(CGNN_decomposable, self).__init__()
        self.backend = backend

        if self.backend == 'TensorFlow':
            self.infer_graph = run_CGNN_decomposable_tf
        else:
            print('No backend known as {}'.format(self.backend))
            raise ValueError

    def create_graph_from_data(self, data):
        print("The CGNN model is not able (yet?) to model the graph directly from raw data")
        raise ValueError

    def orient_directed_graph(self, data, dag, alg='HC', **kwargs):
        """ Improve a directed acyclic graph using CGNN

        :param data: data
        :param dag: directed acyclic graph to optimize
        :param alg: type of algorithm
        :param log: Save logs of the execution
        :return: improved directed acyclic graph
        """
        data = DataFrame(scale(data.as_matrix()), columns=data.columns)
        alg_dic = {'HC': hill_climbing, 'tabu': tabu_search, 'EHC': exploratory_hill_climbing}
        return alg_dic[alg](dag, data, self.infer_graph, **kwargs)

    def orient_undirected_graph(self, data, umg, **kwargs):
        """ Orient the undirected graph using GNN and apply CGNN to improve the graph

        :param data: data
        :param umg: undirected acyclic graph
        :return: directed acyclic graph
        """

        warnings.warn("The pairwise GNN model is computed on each edge of the UMG "
                      "to initialize the model and start CGNN with a DAG")
        gnn = GNN(backend=self.backend, **kwargs)
        dag = gnn.orient_graph(data, umg, **kwargs)  # Pairwise method
        return self.orient_directed_graph(data, dag, **kwargs)

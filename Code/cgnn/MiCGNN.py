"""
MiCGNN : Mixed Variables Causal Generative Neural Networks for causal inference
Authors : Olivier Goudet & Diviyan Kalainathan
Ref:
Date : 7/09/2017
"""
import tensorflow as tf
import numpy as np
from .utils.Loss import MMD_loss_tf as MMD_tf
from .utils.Loss import Fourier_MMD_Loss_tf as Fourier_MMD_tf
from .utils.Settings import SETTINGS
from .utils.Graph import DirectedGraph
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
from .PairwiseModel import Pairwise_Model
from .GraphModel import GraphModel
import pandas as pd
import warnings
from copy import deepcopy


def init(size, **kwargs):
    """ Initialize a random tensor, normal(0,kwargs(SETTINGS.init_weights)).

    :param size: Size of the tensor
    :param kwargs: init_std=(SETTINGS.init_weights) Std of the initialized normal variable
    :return: Tensor
    """
    init_std = kwargs.get('init_std', SETTINGS.init_weights)
    return tf.random_normal(shape=size, stddev=init_std)


class MiGNN_tf(object):
    def __init__(self, N, typeX, typeY, run=0, pair=0, **kwargs):
        """ Build the tensorflow graph, the first column is set as the cause and the second as the effect

        :param N: Number of examples to generate
        :param run: for log purposes (optional)
        :param pair: for log purposes (optional)
        :param kwargs: h_layer_dim=(SETTINGS.h_layer_dim) Number of units in the hidden layer
        :param kwargs: learning_rate=(SETTINGS.learning_rate) learning rate of the optimizer
        :param kwargs: use_Fast_MMD=(SETTINGS.use_Fast_MMD) use fast MMD option
        :param kwargs: nb_vectors_approx_MMD=(SETTINGS.nb_vectors_approx_MMD) nb vectors
        """

        h_layer_dim = kwargs.get('h_layer_dim', SETTINGS.h_layer_dim)
        learning_rate = kwargs.get('learning_rate', SETTINGS.learning_rate)
        use_Fast_MMD = kwargs.get('use_Fast_MMD', SETTINGS.use_Fast_MMD)
        nb_vectors_approx_MMD = kwargs.get('nb_vectors_approx_MMD', SETTINGS.nb_vectors_approx_MMD)

        self.run = run
        self.pair = pair
        self.X = tf.placeholder(tf.float32, shape=[None, typeX])
        self.Y = tf.placeholder(tf.float32, shape=[None, typeY])

        W_in = tf.Variable(init([typeX + 1, h_layer_dim], **kwargs))
        b_in = tf.Variable(init([h_layer_dim], **kwargs))
        W_out = tf.Variable(init([h_layer_dim, typeY], **kwargs))
        b_out = tf.Variable(init([typeY], **kwargs))

        theta_G = [W_in, b_in,
                   W_out, b_out]

        e = tf.random_normal([N, 1], mean=0, stddev=1)

        hid = tf.nn.relu(tf.matmul(tf.concat([self.X, e], 1), W_in) + b_in)
        if typeY > 1:
            out_y = tf.nn.softmax(tf.matmul(hid, W_out) + b_out)
        else:
            out_y = tf.matmul(hid, W_out) + b_out
        if (use_Fast_MMD):
            self.G_dist_loss_xcausesy = Fourier_MMD_tf(tf.concat([self.X, self.Y], 1), tf.concat([self.X, out_y], 1),
                                                       nb_vectors_approx_MMD)
        else:
            self.G_dist_loss_xcausesy = MMD_tf(tf.concat([self.X, self.Y], 1), tf.concat([self.X, out_y], 1))

        self.G_solver_xcausesy = (tf.train.AdamOptimizer(learning_rate=learning_rate)
                                  .minimize(self.G_dist_loss_xcausesy, var_list=theta_G))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, data, verbose=True, **kwargs):
        """ Train the GNN model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :param kwargs: train_epochs=(SETTINGS.nb_epoch_train) number of train epochs
        :return: None
        """
        train_epochs = kwargs.get('train_epochs', SETTINGS.train_epochs)

        for it in range(train_epochs):
            _, G_dist_loss_xcausesy_curr = self.sess.run(
                [self.G_solver_xcausesy, self.G_dist_loss_xcausesy],
                feed_dict={self.X: data[:, [0]], self.Y: data[:, [1]]}
            )

            if verbose:
                if it % 100 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                          format(self.pair, self.run,
                                 it, G_dist_loss_xcausesy_curr))

    def evaluate(self, data, verbose=True, **kwargs):
        """ Test the model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :param kwargs: test_epochs=(SETTINGS.nb_epoch_test) number of test epochs
        :return: mean MMD loss value of the CGNN structure on the data
        """
        test_epochs = kwargs.get('test_epochs', SETTINGS.test_epochs)
        avg_score = 0

        for it in range(test_epochs):
            score = self.sess.run([self.G_dist_loss_xcausesy], feed_dict={self.X: data[:, [0]], self.Y: data[:, [1]]})

            avg_score += score[0]

            if verbose:
                if it % 100 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.format(self.pair, self.run, it, score[0]))

        tf.reset_default_graph()

        return avg_score / test_epochs


def tf_evalcausalscore_pairwise(df, typeX, typeY, idx, run, **kwargs):
    MiGNN = MiGNN_tf(df.shape[0], typeX, typeY, run, idx, **kwargs)
    MiGNN.train(df, **kwargs)
    return MiGNN.evaluate(df, **kwargs)


def tf_run_instance(m, typeX, typeY, idx, run, **kwargs):
    """ Execute the CGNN, by init, train and eval either on CPU or GPU

    :param m: data corresponding to the config : (N, 2) data, [:, 0] cause and [:, 1] effect
    :param run: number of the run (only for print)
    :param idx: number of the idx (only for print)
    :param kwargs: gpu=(SETTINGS.GPU) True if GPU is used
    :param kwargs: nb_gpu=(SETTINGS.NB_GPU) Number of available GPUs
    :param kwargs: gpu_offset=(SETTINGS.GPU_OFFSET) number of gpu offsets
    :return: MMD loss value of the given structure after training
    """
    gpu = kwargs.get('gpu', SETTINGS.GPU)
    nb_gpu = kwargs.get('nb_gpu', SETTINGS.NB_GPU)
    gpu_offset = kwargs.get('gpu_offset', SETTINGS.GPU_OFFSET)

    if m.shape[0] > SETTINGS.max_nb_points:
        p = np.random.permutation(m.shape[0])
        m = m[p[:int(SETTINGS.max_nb_points)], :]

    run_i = run
    if gpu:
        with tf.device('/gpu:' + str(gpu_offset + run_i % nb_gpu)):
            XY = tf_evalcausalscore_pairwise(m, typeX, typeY, idx, run, **kwargs)
        with tf.device('/gpu:' + str(gpu_offset + run_i % nb_gpu)):
            YX = tf_evalcausalscore_pairwise(m[:, [1, 0]], typeY, typeX, idx, run, **kwargs)
            return [XY, YX]
    else:
        return [tf_evalcausalscore_pairwise(m, typeX, typeY, idx, run, **kwargs),
                tf_evalcausalscore_pairwise(m[:, [1, 0]], typeY, typeX, idx, run, **kwargs)]


class MiGNN(Pairwise_Model):
    """
    Shallow Generative Neural networks, models the causal directions x->y and y->x with a 1-hidden layer neural network
    and a MMD loss. The causal direction is considered as the "best-fit" between the two directions
    """

    # ToDo : One Hot encoded data management

    def __init__(self, backend="TensorFlow"):
        super(MiGNN, self).__init__()
        self.backend = backend

    def predict_proba(self, a, b, idx=0, **kwargs):
        typea = kwargs.get("typeX", 1)
        typeb = kwargs.get("typeY", 1)
        backend_alg_dic = {"TensorFlow": tf_run_instance}
        if len(np.array(a).shape) == 1:
            a = np.array(a).reshape((-1, 1))
            b = np.array(b).reshape((-1, 1))

        nb_jobs = kwargs.get("nb_jobs", SETTINGS.NB_JOBS)
        nb_runs = kwargs.get("nb_runs", SETTINGS.NB_RUNS)
        m = np.hstack((a, b))
        m = m.astype('float32')

        result_pair = Parallel(n_jobs=nb_jobs)(delayed(backend_alg_dic[self.backend])(
            m, typea, typeb, idx, run, **kwargs) for run in range(nb_runs))

        score_AB = np.mean([runpair[0] for runpair in result_pair])
        score_BA = np.mean([runpair[1] for runpair in result_pair])

        for runpair in result_pair:
            print(runpair[0])
        print(score_AB)

        for runpair in result_pair:
            print(runpair[1])
        print(score_BA)

        return (score_BA - score_AB) / (score_BA + score_AB)


class MiCGNN_tf(object):
    def __init__(self, N, graph, types, run, idx, **kwargs):
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
        list_nodes = graph.get_list_nodes()
        n_input = sum([types[i] for i in types])
        n_var = len(list_nodes)
        self.all_real_variables = tf.placeholder(tf.float32, shape=[None, n_input])

        generated_variables = {}
        theta_G = []

        while len(generated_variables) < n_var:
            # Need to generate all variables in the graph using its parents : possible because of the DAG structure
            for var in list_nodes:
                # Check if all parents are generated
                par = graph.get_parents(var)
                if (var not in generated_variables and
                        set(par).issubset(generated_variables)):
                    # Generate the variable

                    W_in = tf.Variable(init([sum([types[i] for i in par]) + 1, h_layer_dim], **kwargs))
                    b_in = tf.Variable(init([h_layer_dim], **kwargs))
                    W_out = tf.Variable(init([h_layer_dim, types[var]], **kwargs))
                    b_out = tf.Variable(init([types[var]], **kwargs))

                    input_v = [generated_variables[i] for i in par]
                    input_v.append(tf.random_normal([N, 1], mean=0, stddev=1))
                    input_v = tf.concat(input_v, 1)

                    out_v = tf.nn.relu(tf.matmul(input_v, W_in) + b_in)
                    if types[var] > 1:
                        out_v = tf.nn.softmax(tf.matmul(out_v, W_out) + b_out)
                    else:
                        out_v = tf.matmul(out_v, W_out) + b_out

                    generated_variables[var] = out_v
                    theta_G.extend([W_in, b_in, W_out, b_out])

        listvariablegraph = []
        for var in list_nodes:
            listvariablegraph.append(generated_variables[var])

        self.all_generated_variables = tf.concat(listvariablegraph, 1)

        if (use_Fast_MMD):
            self.G_dist_loss_xcausesy = Fourier_MMD_tf(self.all_real_variables, self.all_generated_variables,
                                                       nb_vectors_approx_MMD)
        else:
            self.G_dist_loss_xcausesy = MMD_tf(self.all_real_variables, self.all_generated_variables)

        self.G_solver_xcausesy = (tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.G_dist_loss_xcausesy,
                                                  var_list=theta_G))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, data, verbose=True, **kwargs):
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
                feed_dict={self.all_real_variables: data}
            )

            if verbose:
                if it % 100 == 0:
                    print('Pair:{}, Run:{}, Iter:{}, score:{}'.
                          format(self.idx, self.run,
                                 it, G_dist_loss_xcausesy_curr))

    def evaluate(self, data, verbose=True, **kwargs):
        """ Test the model

        :param data: data corresponding to the graph
        :param verbose: verbose
        :param kwargs: test_epochs=(SETTINGS.test_epochs) number of test epochs
        :return: mean MMD loss value of the CGNN structure on the data
        """
        test_epochs = kwargs.get('test_epochs', SETTINGS.test_epochs)
        sumMMD_tr = 0

        for it in range(test_epochs):

            MMD_tr = self.sess.run([self.G_dist_loss_xcausesy], feed_dict={
                self.all_real_variables: data})

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


def run_MiCGNN_tf(df_data, graph, types, idx=0, run=0, **kwargs):
    """ Execute the CGNN, by init, train and eval either on CPU or GPU

    :param df_data: data corresponding to the graph
    :param graph: Graph to be run
    :param types:
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

    list_nodes = graph.get_list_nodes()
    # df_data = df_data[list_nodes].as_matrix()
    # ToDo: Conversion Into OneHotEncoded Vars.
    data = pd.DataFrame()
    for idx, node in enumerate(list_nodes):
        if types[node] > 1:
            data = pd.DataFrame(np.concatenate((data.as_matrix(),
                                                pd.get_dummies(df_data[node], prefix=node).as_matrix()), axis=1))
        else:
            data = pd.DataFrame(np.concatenate((data.as_matrix(), df_data[node].as_matrix()), axis=1))

    data = data.as_matrix().astype('float32')

    if data.shape[0] > SETTINGS.max_nb_points:
        p = np.random.permutation(data.shape[0])
        data = data[p[:int(SETTINGS.max_nb_points)], :]

    if gpu:
        with tf.device('/gpu:' + str(gpu_offset + run % nb_gpu)):
            model = MiCGNN_tf(data.shape[0], graph, types, run, idx, **kwargs)
            model.train(data, **kwargs)
            return model.evaluate(data, **kwargs)
    else:
        model = MiCGNN_tf(data.shape[0], graph, types, run, idx, **kwargs)
        model.train(data, **kwargs)
        return model.evaluate(data, **kwargs)


def hill_climbing(graph, data, types, run_cgnn_function, **kwargs):
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
    result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
        data, graph, types, 0, run, **kwargs) for run in range(nb_runs))

    score_network = np.mean([i for i in result_pairs if np.isfinite(i)])
    globalscore = score_network

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
                result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                    data, test_graph, types, idx_pair, run, **kwargs) for run in range(nb_runs))

                score_network = np.mean([i for i in result_pairs if np.isfinite(i)])

                print("Current score : " + str(score_network))
                print("Best score : " + str(globalscore))

                if score_network < globalscore:
                    graph.reverse_edge(edge[0], edge[1])
                    improvement = True
                    print('Edge {} got reversed !'.format(edge))
                    globalscore = score_network

    return graph


def exploratory_hill_climbing(graph, data, types, run_cgnn_function, **kwargs):
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
        data, graph, types, 0, run, **kwargs) for run in range(nb_runs))

    score_network = np.mean([i for i in result_pairs if np.isfinite(i)])
    globalscore = score_network

    print("Graph score : " + str(globalscore))

    while loop < nb_loops:
        loop += 1
        list_edges = graph.get_list_edges()

        possible_solution = False
        while not possible_solution:
            test_graph = deepcopy(graph)
            selected_edges = np.random.choice(len(list_edges),
                                              max(int(exploration_factor * ((nb_loops - loop) / nb_loops) ** 2), 1))
            for edge in list_edges[selected_edges]:
                test_graph.reverse_edge()
            if not (test_graph.is_cyclic()
                    or test_graph.get_dict_nw() in tested_configurations):
                possible_solution = True

            print('Reversed Edges {} in evaluation :'.format(list_edges[selected_edges]))
            tested_configurations.append(test_graph.get_dict_nw())
            result_pairs = Parallel(n_jobs=nb_jobs)(delayed(run_cgnn_function)(
                data, types, test_graph, loop, run, **kwargs) for run in range(nb_runs))

            score_network = np.mean([i for i in result_pairs if np.isfinite(i)])

            print("Current score : " + str(score_network))
            print("Best score : " + str(globalscore))

            if score_network < globalscore:
                graph.reverse_edge(edge[0], edge[1])
                print('Edge {} got reversed !'.format(list_edges[selected_edges]))
                globalscore = score_network

    return graph


def tabu_search(graph, data, types, run_cgnn_function, **kwargs):
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


class MiCGNN(GraphModel):
    """
    CGNN Model ; Using generative models, generate the whole causal graph and improve causal
    direction predictions in the graph.
    """

    def __init__(self, backend='TensorFlow'):
        """ Initialize the CGNN Model.

        :param backend: Choose the backend to use, either 'PyTorch' or 'TensorFlow'
        """
        super(MiCGNN, self).__init__()
        self.backend = backend

        if self.backend == 'TensorFlow':
            self.infer_graph = run_MiCGNN_tf
        else:
            print('No backend known as {}'.format(self.backend))
            raise ValueError

    def create_graph_from_data(self, data, **kwargs):
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
        types = kwargs.get("types", "err")
        if types == "err":
            raise ValueError("Need type")
        data = pd.DataFrame(scale(data.as_matrix()), columns=data.columns)
        alg_dic = {'HC': hill_climbing, 'tabu': tabu_search, 'EHC': exploratory_hill_climbing}
        return alg_dic[alg](dag, data, types, self.infer_graph, **kwargs)

    def orient_undirected_graph(self, data, umg, **kwargs):
        """ Orient the undirected graph using GNN and apply CGNN to improve the graph

        :param data: data
        :param umg: undirected acyclic graph
        :return: directed acyclic graph
        """

        warnings.warn("The pairwise GNN model is computed on each edge of the UMG "
                      "to initialize the model and start CGNN with a DAG")
        # gnn = GNN(backend=self.backend, **kwargs)
        # dag = gnn.orient_graph(data, umg, **kwargs)  # Pairwise method
        # ToDo : Random Orientation?
        dag = DirectedGraph()
        list_edges = umg.get_list_edges_without_duplicate()
        for edge in list_edges:
            if np.random.randint(0, 2, dtype=bool):
                edge = edge[::-1]
            test_graph = deepcopy(dag)
            test_graph.add(edge[0], edge[1])
            if not test_graph.is_cyclic():
                dag = test_graph
            else:
                dag.add(edge[1], edge[0])
                if dag.is_cyclic():
                    raise ValueError
        return self.orient_directed_graph(data, dag, **kwargs)

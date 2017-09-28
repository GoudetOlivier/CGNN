import tensorflow as tf
from .utils.Graph import DirectedGraph, UndirectedGraph
from .CGNN import CGNN
from .CGNN_decomposable import CGNN_decomposable
from .CGNN_confounders import CGNN_confounders
from .GNN import GNN
from .generators import __init__
from .utils import Loss
from .utils.Settings import SETTINGS
import os
# avoid the thousands of stderr output lines
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


__all__ = ['DirectedGraph', 'UndirectedGraph',
           'CGNN', 'CGNN_confounders', 'GNN', 'CGNN_decomposable']

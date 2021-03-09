from utils import *
import numpy as np


class Tensor:
    def __init__(self, shape):
        self.handle = None
        self.shape = shape

    def assign_value(self, value):
        if isinstance(value, np.ndarray):
            assert value.shape == self.shape
        self.handle = value


class Executor:
    def __init__(self, graph):
        self.graph = graph
        self.output_of_nodes = dict()
        self.finished_loop_id = set()
        self.init_data_edges()

    @bfs
    def init_data_edges(self, current_node):
        current_node.generate_data_edges()

    @bfs
    def infer_data(self, current_node):
        current_node.infer_data()

    @bfs
    def run(self, current_node):
        current_node(self)

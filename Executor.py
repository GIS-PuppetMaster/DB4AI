from utils import *
import numpy as np


class Tensor:
    def __init__(self, shape):
        self.handle = None
        self.shape = shape

    def assign_value(self, value):
        if isinstance(value, np.ndarray):
            assert value.shape == self.shape
        else:
            # TODO shape of table
            pass
        self.handle = value

    def to_cpu(self):
        data = np.empty(shape=self.shape)
        table = None # TODO query the table from DB
        # TODO convert
        self.handle = data

    def to_relation(self):
        # TODO
        pass

class Executor:
    def __init__(self, graph):
        self.graph = graph
        self.var_dict = dict()
        self.finished_loop_id = set()
        self.init_data_edges()
        self.infer_data()

    @bfs
    def init_data_edges(self, current_node):
        current_node.generate_data_edges()

    @bfs
    def infer_data(self, current_node):
        current_node.infer_data()

    @bfs
    def execute(self, current_node):
        current_node(self)

    def run(self):
        self.execute()

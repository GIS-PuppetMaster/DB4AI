from utils import *
import numpy as np


class Tensor(np.ndarray):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handle = None
        self.shape = kwargs['shape']

    def assign_value(self, value):
        if isinstance(value, np.ndarray):
            assert value.shape == self.shape
        else:
            # TODO shape of table
            pass
        self.handle = value

    def to_cpu(self):
        data = np.empty(shape=self.shape)
        table = None  # TODO query the table from DB
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
        self.infer_nodes()

    @bfs
    def infer_nodes(self, current_node):
        current_node.generate_data_edges()
        current_node.infer_nodes()
        current_node.executor = self

    @bfs
    def execute(self, current_node):
        current_node.start()

    def run(self):
        self.execute()

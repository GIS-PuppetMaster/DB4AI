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

    def run(self):
        current_node = self.graph.nodes[0]
        while current_node is not None:
            current_node.run(self)

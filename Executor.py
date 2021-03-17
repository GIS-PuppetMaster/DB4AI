from utils import *
import numpy as np


class Executor:
    def __init__(self, graph):
        self.graph = graph
        self.var_dict = dict()
        self.finished_loop_id = set()
        self.infer_nodes()

    @bfs
    def infer_nodes(self, current_node):
        current_node.generate_data_edges()
        current_node.infer_shape()
        current_node.infer_data()
        self.var_dict[current_node.vars[0]] = np.empty(current_node.shape)
        current_node.executor = self

    @bfs
    def execute(self, current_node):
        current_node.start()

    def run(self):
        self.execute()

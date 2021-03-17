from functools import wraps
from queue import Queue
import numpy as np


def bfs(fun):
    @wraps(fun)
    def decorated(executor):
        queue = Queue()
        visited = set()
        root = executor.graph.nodes[0]
        queue.put(root)
        visited.add(root)
        while not queue.empty():
            current_node = queue.get()
            visited.add(current_node)
            fun(executor, current_node)
            next_nodes = current_node.next_nodes(executor)
            for node in next_nodes:
                if node not in visited:
                    queue.put(node)

    return decorated


def get_slice(batch: np.ndarray):
    # TODO
    start_index = int((np.byte_bounds(batch)[0] - np.byte_bounds(batch.base)[0]) / batch.itemsize)


def batch_stream(fun):
    @wraps(fun)
    def decorated(node):
        while True:
            input_buffer = []
            for edge in node.input_data_edges:
                if not edge.data.empty():
                    input_buffer.append(edge.get_data())
            res = fun(input_buffer)
            for edge in node.out_edges:
                edge.put_data(res)

    return decorated

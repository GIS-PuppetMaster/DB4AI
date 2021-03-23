from functools import wraps
from queue import Queue
import numpy as np
from copy import copy, deepcopy


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
            fun(executor, current_node)
            next_nodes = current_node.next_nodes()
            for node in next_nodes:
                if node not in visited:
                    queue.put(node)
                    visited.add(node)

    return decorated


def get_slice(batch: np.ndarray):
    # TODO
    start_index = int((np.byte_bounds(batch)[0] - np.byte_bounds(batch.base)[0]) / batch.itemsize)


def check_buffer(buffer, batch_size, bigger_than_buffer=True):
    # if all data in the buffer is bigger than batch_size, then return true
    for data in buffer:
        if bigger_than_buffer:
            if data.shape[0] < batch_size:
                return True
        else:
            if data.shape[0] > batch_size:
                return True
    return False



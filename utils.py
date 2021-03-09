from functools import wraps
from queue import Queue
from Digraph import Graph
from Executor import Executor

def bfs(fun):
    @wraps(fun)
    def decorated(executor: Executor):
        queue = Queue()
        visited = set()
        root = executor.graph.nodes[0]
        queue.put(root)
        visited.add(root)
        while not queue.empty():
            current_node = queue.get()
            visited.add(current_node)
            fun(current_node)
            next_nodes = current_node.next_nodes(executor)
            for node in next_nodes:
                if node not in visited:
                    queue.put(node)

    return decorated

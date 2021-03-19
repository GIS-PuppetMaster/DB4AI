from functools import wraps
from queue import Queue
import numpy as np

from Executor import BatchedTensor


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


def batch_stream(fun):
    # 对所有接收数据流并输出数据流的节点使用此装饰器
    @wraps(fun)
    def decorated(executor, node):
        while True:
            if node.batch_counter * node.batch_size >= node.out_edges[0].shape[0]:
                node.batch_counter = 0
            input_buffer = [None for _ in range(len(node.input_data_edges))]
            # 累积输入的batch直到全部达到当前节点的batch size设置
            # TODO: 对于不使用batch的节点的适配，例如val常量? -- 疑似split batch可以自动适配
            while check_buffer(input_buffer, node.batch_size):
                for idx, var_name in enumerate(node.vars[1:]):
                    pipeline = executor.pipeline[var_name]
                    batch = pipeline.queue[0]
                    # 当这个operator接收的变量对应的queue不为空，且队首的bath的目的节点为此node，则说明轮到这个operator执行操作
                    # 如果二者的branch吻合，则说明batch属于这个分支
                    if not pipeline.empty() and node in batch.next_nodes and (node.branch in batch.branches or (node.branch is None and len(batch.branches) == 0)):
                        pipeline.get()
                        if input_buffer[idx] is None:
                            input_buffer[idx] = pipeline.get()
                        else:
                            input_buffer[idx].merge_batch(pipeline.get())
            # 如果当前节点的batch size比输入的若干个batch中的至少一个要小
            while check_buffer(input_buffer, node.batch_size, False):
                # 将输入的batch切分
                for idx, var_name in enumerate(node.vars[1:]):
                    pipeline = executor.pipeline[var_name]
                    batch = pipeline.queue[0]
                    if not pipeline.empty() and node in batch.next_nodes and (node.branch in batch.branches or (node.branch is None and len(batch.branches) == 0)):
                        pipeline.get()
                        buffer = batch.split_batch(node.batch_size)
                        if input_buffer[idx] is None:
                            input_buffer[idx] = buffer
                        else:
                            input_buffer[idx].extend(buffer)
            if not isinstance(input_buffer[0], list):
                fun(node, input_buffer)
            else:
                for buffer in input_buffer:
                    fun(node, buffer)


            node.batch_counter += 1

    return decorated


def operator_wrapper(fun):
    @wraps(fun)
    def decorated(node, input_buffer):
        for i in range(len(input_buffer)):
            assert input_buffer[i]().shape[0] == node.batch_size, f'i={i}'
        node.executor.var_dict[node.vars[0]][...] = fun(node, input_buffer)
        node.executor.pipeline[node.vars[0]].put(BatchedTensor(node.executor.var_dict[node.vars[0]], node.next_nodes(), batch_size=node.batch_size, start_index=node.batch_counter * node.batch_size))

    return decorated

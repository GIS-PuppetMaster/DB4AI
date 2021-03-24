from collections import defaultdict

from Nodes import *
from utils import *
import numpy as np
import yaml

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
        deepest_branch = None
        deepest_branch_depth = -1
        for i in range(len(input_buffer)):
            assert input_buffer[i]().shape[0] == node.batch_size, f'i={i}'
            if len(input_buffer[i].branches) > deepest_branch_depth:
                deepest_branch = input_buffer[i].branches
                deepest_branch_depth = len(deepest_branch)
        node.executor.var_dict[node.vars[0]][...] = fun(node, input_buffer)
        current_branches = copy(deepest_branch)
        for next_node in node.sons:
            if isinstance(next_node, If):
                current_branches.append(next_node.id)
                break
            elif isinstance(next_node, IfEnd):
                current_branches.pop()
        batch = BatchedTensor(node.executor.var_dict[node.vars[0]], node.next_nodes(), current_branches, batch_size=node.batch_size, start_index=node.batch_counter * node.batch_size)
        node.executor.pipeline[node.vars[0]].put(batch)

    return decorated

class BatchedTensor:
    def __init__(self, source_tensor, next_nodes, branches, batch_size: int, start_index: int, batch_axis: int = 0, step: int = 1):
        # 包装tensor
        self._source_tensor = source_tensor
        self._batch_axis = batch_axis
        self._batch_size = batch_size
        self._start_index = start_index
        self._step = step
        self._slice = slice(self.start_index, self.start_index + self.batch_size, self._step)
        self.branches = branches
        self.next_nodes = next_nodes

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        raise Exception('do not allow to set step manually')

    @property
    def slice(self):
        return self._slice

    @slice.setter
    def slice(self, value):
        raise Exception('do not allow to set slice manually')

    @property
    def source_tensor(self):
        return self._source_tensor

    @source_tensor.setter
    def source_tensor(self, tensor):
        raise Exception('do not allow to change source tensor manually')

    @property
    def batch_axis(self):
        return self._batch_axis

    @batch_axis.setter
    def batch_axis(self, axis):
        raise Exception('do not allow to change batch axis')

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size):
        raise Exception('do not allow to change batch axis manually')

    @property
    def start_index(self):
        return self._start_index

    @start_index.setter
    def start_index(self, index):
        raise Exception('do not allow to change start_index manually')

    def merge_batch(self, target_batch):
        # 合并两个连续的 batch
        assert self.source_tensor == target_batch.source_tensor
        assert self.batch_axis == target_batch.batch_axis
        assert self.slice.step == target_batch.slice.step
        if self.slice.stop == target_batch.slice.start:
            self._batch_size = self.batch_size + target_batch.batch_size
            self._slice = slice(self.slice.start, target_batch.slice.stop, self.slice.step)
        elif self.slice.start == target_batch.slice.stop:
            self._start_index = target_batch.start_index
            self._batch_size = self.batch_size + target_batch.batch_size
            self._slice = slice(target_batch.slice.start, self.slice.stop, self.slice.step)
        else:
            raise Exception(f'this bath of tensor:{self} is not adjacency with the target batch of tensor:{target_batch}')

    def split_batch(self, new_batch_size):
        res = []
        num = int(np.trunc(self.batch_size / new_batch_size))
        for i in range(num):
            if i != num - 1:
                bs = new_batch_size
            else:
                bs = self.batch_size % new_batch_size
            res.append(BatchedTensor(self.source_tensor, self.next_nodes, bs, self.start_index + i * new_batch_size, self.batch_axis, self.step))
        return res

    def __call__(self):
        return self.source_tensor.__getitem__(self.slice)


class Executor:
    def __init__(self, graph):
        with open('./config.yaml', encoding='utf-8') as f:
            config = yaml.load_all(f)
        self.config = config
        # self.default_batch_size = config['default_batch_size']
        self.graph = graph
        # 存放每个完整Tensor的dict, key=变量名, value=ndarray
        self.var_dict = dict()
        # # 存放流水线队列的dict, key=变量名, value=Queue(BatchedTensor)，表示node输出该变量的节点
        # self.pipeline = dict()
        self.finished_loop_id = set()
        self.init_nodes()
        self.finished_nodes=set()
        # self.init_branches(self.graph.nodes[0], None)

    @bfs(True)
    def init_nodes(self, current_node):
        current_node.fathers = [edge.start for edge in current_node.in_edges]
        current_node.sons = [edge.end for edge in current_node.out_edges]
        # current_node.generate_data_edges()
        current_node.infer_data()
        # if len(current_node.vars)>0:
        #     self.var_dict[current_node.vars[0]] = np.empty(current_node.data_shape)
        # self.pipeline[current_node.vars[0]] = Queue()
        current_node.executor = self

        # current_node.default_batch_size = self.default_batch_size

    def init_branches(self, node, current_branch):
        if isinstance(node, IfBranch):
            current_branch = node.id
        elif not (isinstance(node, If) or isinstance(node, IfEnd) or isinstance(node, Loop) or isinstance(node, LoopEnd)):
            node.branch = current_branch
        next_nodes = node.next_nodes()
        if len(next_nodes) == 0:
            return
        else:
            for next_node in next_nodes:
                self.init_branches(next_node, current_branch)

    @bfs(False)
    def execute(self, current_node):
        # 确保父节点都执行完了再执行他
        for father in current_node.fathers:
            if father not in self.finished_nodes:
                return False
        current_node.run()
        self.finished_nodes.add(current_node)
        return True

    def run(self):
        self.execute()

from collections import defaultdict

from Nodes import *
from utils import *
import numpy as np
import yaml


class BatchedTensor:
    def __init__(self, source_tensor, next_nodes, batch_size: int, start_index: int, batch_axis: int = 0, step: int = 1):
        # 包装tensor
        self._source_tensor = source_tensor
        self._batch_axis = batch_axis
        self._batch_size = batch_size
        self._start_index = start_index
        self._step = step
        self._slice = slice(self.start_index, self.start_index + self.batch_size, self._step)
        self.branches = []
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
        self.default_batch_size = config['default_batch_size']
        self.graph = graph
        # 存放每个完整Tensor的dict, key=变量名, value=ndarray
        self.var_dict = dict()
        # 存放流水线队列的dict, key=变量名, value=Queue(BatchedTensor)，表示node输出该变量的节点
        self.pipeline = dict()
        self.finished_loop_id = set()
        self.init_nodes()

    @bfs
    def init_nodes(self, current_node):
        current_node.generate_data_edges()
        current_node.infer_data()
        self.var_dict[current_node.vars[0]] = np.empty(current_node.shape)
        self.pipeline[current_node.vars[0]] = Queue()
        current_node.executor = self
        current_node.default_batch_size = self.default_batch_size

    def init_branches(self):
        # TODO: dfs, set branch id for Nodes
        pass

    @bfs
    def execute(self, current_node):
        current_node.start()

    def run(self):
        self.execute()

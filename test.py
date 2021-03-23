from Executor import Executor
from Digraph import *
from Nodes import *

graph = Graph()
root = Root(id=0, branches=[0])
graph.InsertNode(root)
ct_x = CreateTensor((4, 3), id=1, branches=[0], var=[])
ct_x.vars = ['x']
graph.InsertNode(ct_x)
ct_w = CreateTensor((3, 5), id=2, var=[], branches=[0])
ct_w.vars = ['w']
graph.InsertNode(ct_w)
ct_y = CreateTensor((4, 5), id=3, var=[], branches=[0])
ct_y.vars = ['y']
graph.InsertNode(ct_y)
random_x = Random((0, 1), (4, 3), id=4, branches=[0])
random_x.vars = ['random_x']
graph.InsertNode(random_x)
random_w = Random((0, 1), (3, 5), id=5, branches=[0])
random_w.vars = ['random_w']
graph.InsertNode(random_w)
ass_x = Assignment(id=6, branches=[0],var_li=['x', 'random_x'])
graph.InsertNode(ass_x)
ass_w = Assignment(id=7, branches=[0], var_li=['w', 'random_w'])
graph.InsertNode(ass_w)
var_x = Var(id=8, branches=[0])
graph.InsertNode(var_x)
var_w = Var(id=9, branches=[0])
graph.InsertNode(var_w)
matmul = MATMUL(id=10, branches=[0])
matmul.vars = ['temp1', 'x', 'w']
graph.InsertNode(matmul)
ass_y = Assignment(id=11, branches=[0], var_li=['y', 'temp1'])
graph.InsertNode(ass_y)
graph.InsertEdge(root, ct_x)
graph.InsertEdge(root, ct_w)
graph.InsertEdge(root, ct_y)
graph.InsertEdge(root, random_x)
graph.InsertEdge(root, random_w)
graph.InsertEdge(ct_x, ass_x)
graph.InsertEdge(random_x, ass_x)
graph.InsertEdge(ct_w, ass_w)
graph.InsertEdge(random_w, ass_w)
graph.InsertEdge(ass_x, var_x)
graph.InsertEdge(ass_w, var_w)
graph.InsertEdge(var_x, matmul)
graph.InsertEdge(var_w, matmul)
graph.InsertEdge(matmul, ass_y)
graph.InsertEdge(ct_y, ass_y)
graph.Show()
executor = Executor(graph)
executor.run()
print(executor.var_dict)

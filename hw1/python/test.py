import needle as ndl
import numpy as np
import torch


def test_matmul_batch():
    '''
        find why the previous test 
        failed on (6, 6, 4, 3) @ (3, 4)
    '''
    a = np.ones((1, 2)) 
    b = np.ones((2, 2, 2, 1)) 
    print(a)
    print(b)
    ta = torch.tensor(a, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)
    tc = ta @ tb
    tc.backward(torch.ones_like(tc))

    na = ndl.Tensor(a)
    nb = ndl.Tensor(b)
    nc = na @ nb
    nc = nc.sum()
    nc.backward()
    # mask = np.ones_like(nc)
    ga, gb = na.grad, nb.grad
    # ga, gb = ndl.ops.MatMul().gradient(ndl.Tensor(mask), nc)
    return ga, gb, ta, tb

def test_broadcast_to():
    a = np.random.randn()
    shape = (3, 3, 3)
    
    # print(a)
    ta = torch.tensor(a, requires_grad=True)
    tc = ta + torch.ones(shape)
    tc = torch.sum(tc)
    tc.backward()
    

    na = ndl.Tensor(a)
    nc = ndl.ops.broadcast_to(na, shape)
    mask = np.ones_like(nc.numpy())

    ga = ndl.ops.BroadcastTo(shape).gradient(ndl.Tensor(mask), nc)
    return ga, ta

def test_summation():
    a = np.ones((5, 4))
    axes = None

    ta = torch.tensor(a, requires_grad=True)
    tc = torch.sum(ta)
    tc = torch.sum(tc)
    tc.backward()
    
    na = ndl.Tensor(a)
    nc = ndl.ops.summation(na, axes).sum()
    nc.backward()
    # mask = np.ones_like(nc.numpy())

    # ga = ndl.ops.Summation(axes).gradient(ndl.Tensor(mask), nc)
    ga = na.grad
    return ga, ta

def valid_two():
    ga, gb, ta, tb = test_matmul_batch()

    print(ga)
    print(gb)

    print(ta.grad)
    print(tb.grad)

    print(ga.shape == ta.shape)
    print(gb.shape == tb.shape)
    print(np.allclose(ga.numpy(), ta.grad.data.numpy()))
    print(np.allclose(gb.numpy(), tb.grad.data.numpy()))

def valid_one():
    # ga, ta = test_broadcast_to()
    ga, ta = test_summation()
    print(ga)
    print(ta.grad)

    print(ga.shape == ta.grad.shape)
    print(np.allclose(ga.numpy(), ta.grad.data.numpy()))


def test_comutational_graph():
    # (A@B+C)*(A@B)
    A = np.ones((4, 3))
    B = np.ones((3, 4))
    C = np.ones((4, 4))

    ta = torch.tensor(A, requires_grad=True)
    tb = torch.tensor(B, requires_grad=True)
    tc = torch.tensor(C, requires_grad=True)

    tout = (ta @ tb + tc) * (ta @ tb)
    tout = tout.sum()
    tout.backward()

    na = ndl.Tensor(A)
    nb = ndl.Tensor(B)
    nc = ndl.Tensor(C)

    nout = (na @ nb + nc) * (na @ nb)
    nout = nout.sum()
    nout.backward()

    return na.grad, nb.grad, nc.grad, ta, tb, tc

def valid_three():
    ga, gb, gc, ta, tb, tc = test_comutational_graph()
    print(ga.shape == ta.grad.shape)
    print(gb.shape == tb.grad.shape)
    print(gc.shape == tc.grad.shape)
    print(np.allclose(ga.numpy(), ta.grad.data.numpy()))
    print(np.allclose(gb.numpy(), tb.grad.data.numpy()))
    print(np.allclose(gc.numpy(), tc.grad.data.numpy()))

    print(ga)
    print(ta.grad)
    print(gb)
    print(tb.grad)
    print(gc)
    print(tc.grad)

from typing import List

def find_topo_sort(node_list: List[ndl.Value]) -> List[ndl.Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    #raise NotImplementedError()
    visited = set()
    topo_order = []
    for node in node_list:
        if node not in visited: topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node: ndl.Value, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    if node in visited: return

    for next in node.inputs:
        topo_sort_dfs(next, visited, topo_order)
    
    visited.add(node)
    topo_order.append(node)
    ### END YOUR SOLUTION

if __name__ == '__main__':
    
    valid_three()

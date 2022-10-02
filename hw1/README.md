## Operators implementations

[ops.py](./python/needle/ops.py)

### gradient function

Take ``AddScalar.gradient()`` as an example:

```python
# out_grad is the backward incoming gradient. `nodes.inputs` contain the Tensor used in `compute()` function.
def gradient(self, out_grad: Tensor, node: Tensor):
	return out_grad
```

- We need the returned `out_grad` to be a `ndl.Tensor` instead of `numpy.ndarray`, which is different from the `compute()` function. So we can use the completed function below such as `add() multiply()...` to return a `ndl.Tensor`.

### Matmul

Note that `Matmul` may be used in batch matmul (i.e. `matmul(a, b), a has shape(6, 6, 3, 4), b has shape (4, 3)`). Be careful in `Matmul.gradient()` function.

### Broadcast_to & Summation

Actually these two functions are complementary in the gradient function. In `Broadcast_to.gradient()`  we may use `summation()`, at the same time, we may use `broadcast_to()` while coding `Summation.gradient()`.

Another small point we need to consider is about the `shape` in `Broadcast_to` and the `axes` in `Summation`. In fact, I don't know whether my solutions dealing with it in the `gradient()` function are good.

## Topo_sort & Gradient Computation

[autograd.py](./python/needle/autograd.py)

### Topo_sort

We need to implement postorder dfs in `find_topo_sort` function. There is another topo_sort algorithm using the degree of the nodes.

### Gradient Computation

~~My codes in this part seems a little bit redundant. There must exist a better solution.~~

*Update:* `op.gradient_as_tuple()` has helped me remove some redundant codes.

## Loss function & Training 

[simple_ml.py](./apps/simple_ml.py)

Almost the same as [hw0](../hw0/simple_ml.py)

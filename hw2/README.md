Until now, I have passed all mugrade tests except for the first case in `mnist_train`.

## Question 0

## Question 1

Implement some simple init functions, just take advatange of functions below.

## Question 2

### Linear

I had made a mistake here. The original wrong code is:

```python
# Wrong!
self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True)).reshape((1, out_features))
```

It first uses `Parameter()` and then `reshape()`, which will turn `Parameter` into `Tensor` object and then `self.bias` will not be considered as a parameter.

So we should first `reshape` and then `Parameter()`, just like:

```python
# Right!
self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True).reshape((1, out_features)))
```

### LogSumExp

1. It is wrong to use `np.squeeze()` to adjust shape before return because it may remove dimensions with size 1 in the original input `Z`.
2. The first case in submit need a float number instead of a `np.array`, which is a little confusing.

### LayerNorm

We need manually `broadcast_to()` in some operations with different input shapes.

### BatchNorm

Careful with the testing behaviour.

### Dropout

1. During training, do not forget to divided by `(1 - p)`
2. Dropout will not be applied during evaluation.

## Question 3

### SGD & Adam

1. Use a dictionary to store `u` and `v`
2. `u` and `v` are initialized with zeros.
3. Add `weight_decay` into grad first, and then use this grad to momentum upgrade.

## Question 4

## Question 5

Cannot pass the last test case, but passed all the mugrade tests with `shuffle=False` in `dataloader`. When set `shuffle=True`, I failed on the last mugrade test... Quite confusing...
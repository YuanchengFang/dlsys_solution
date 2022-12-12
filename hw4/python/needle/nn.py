"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight= Parameter(
            init.kaiming_uniform(
                in_features, 
                out_features, 
                shape=(in_features, out_features),
                device=device,
                dtype=dtype, 
                requires_grad=True
            )
        )
        # NOTE: this line below will change type(self.bias) from 'Parameter' into 'ndl.Tensor'!
        # self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True)).reshape((1, out_features))
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(
                    out_features, 
                    1,
                    shape=(out_features, 1),
                    device=device,
                    dtype=dtype, 
                    requires_grad=True
                    ).reshape((1, out_features)
                )
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X_mul_weight = X @ self.weight
        if self.bias:
            return X_mul_weight + self.bias.broadcast_to(X_mul_weight.shape)
        else:
            return X_mul_weight
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        length = 1
        for i in X.shape[1:]:
            length *= i
        return X.reshape((X.shape[0], length))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x)) ** (-1)
        # return ops.exp(x) / (1 + ops.exp(x))
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        exp_sum = ops.logsumexp(logits, axes=(1, )).sum()
        z_y_sum = (logits * init.one_hot(logits.shape[1], y, logits.device, logits.dtype)).sum()
        return (exp_sum - z_y_sum) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        mean = x.sum((0, )) / batch_size
        # NOTE reshape before broadcast
        x_minus_mean = x - mean.reshape((1, x.shape[1])).broadcast_to(x.shape)
        var = (x_minus_mean ** 2).sum((0, )) / batch_size
        
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data

            x_std = ((var + self.eps) ** 0.5).reshape((1, x.shape[1])).broadcast_to(x.shape)
            x_normed = x_minus_mean / x_std
            return x_normed * self.weight.reshape((1, self.dim)).broadcast_to(x.shape) + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        else:
            # NOTE no momentum here!
            x_normed = (x - self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)) / (self.running_var.reshape((1, self.dim)).broadcast_to(x.shape) + self.eps) ** 0.5
            # NOTE testing time also need self.weight and self.bias
            return x_normed * self.weight.reshape((1, self.dim)).broadcast_to(x.shape) + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype, requires_grad=True))
        # NOTE bias initialized to 0!!!
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        feature_size = x.shape[1]
        # NOTE reshape before broadcast
        mean = x.sum(axes=(1, )).reshape((batch_size, 1)) / feature_size
        
        # NOTE need manual broadcast_to!!!
        x_minus_mean = x - mean.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        x_std = ((x_minus_mean ** 2).sum(axes=(1, )).reshape((batch_size, 1)) / feature_size + self.eps) ** 0.5
        # NOTE need manual broadcast_to!!!
        normed = x_minus_mean / x_std.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        
        return self.weight.reshape((x.shape[0], 1)).broadcast_to(x.shape) * normed + self.bias.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # NOTE Dropout is training only
        # NOTE 1 - self.p
        mask = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype=x.dtype)
        if self.training:
            x_mask = x * mask
            return x_mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # NOTE: input NCHW, while conv op is NHWC
        shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.weight = Parameter(init.kaiming_uniform(
            self.in_channels * kernel_size * kernel_size, 
            self.out_channels * kernel_size * kernel_size, 
            shape=shape,
            device=device,
            dtype=dtype,
            requires_grad=True
        ))
        if bias:
            self.bias = Parameter(
                init.rand(
                    int(self.out_channels),
                    low= -1 / (in_channels * kernel_size**2)**0.5,
                    high= 1 / (in_channels * kernel_size**2)**0.5,
                    device=device,
                    dtype=dtype,
                    requires_grad=True
                )
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N, C, H, W = x.shape
        x = x.transpose((1, 2)).transpose((2, 3))
        out = ops.conv(x, self.weight, stride=self.stride, padding=self.kernel_size//2)
        if self.bias:
            out += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(out.shape)
        # NOTE: out N H W C_out
        return out.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.W_ih = Parameter(
            init.rand(
                input_size, 
                hidden_size, 
                low=-1/hidden_size**0.5,
                high=1/hidden_size**0.5,
                device=device,
                dtype=dtype,
                requires_grad=True
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size, 
                hidden_size, 
                low=-1/hidden_size**0.5,
                high=1/hidden_size**0.5,
                device=device,
                dtype=dtype,
                requires_grad=True
            )
        )
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size, 
                    low=-1/hidden_size**0.5,
                    high=1/hidden_size**0.5,
                    device=device,
                    dtype=dtype,
                    requires_grad=True
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    hidden_size, 
                    low=-1/hidden_size**0.5,
                    high=1/hidden_size**0.5,
                    device=device,
                    dtype=dtype,
                    requires_grad=True
                )
            )
        self.nonlinearity = Tanh() if nonlinearity == 'tanh' else ReLU()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        out = X @ self.W_ih
        if self.bias:
            out += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
        if h is not None:
            out += h @ self.W_hh
        if self.bias:
            out += self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
        return self.nonlinearity(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers
        rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]
        for i in range(num_layers - 1):
            rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        self.rnn_cells = rnn_cells
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        Xs = ops.split(X, 0)
        hs = ops.split(h0, 0) if h0 is not None else [None] * self.num_layers
        out = []
        for t, x in enumerate(Xs):
            hiddens = []
            for l, model in enumerate(self.rnn_cells):
                x = model(x, hs[l])
                hiddens.append(x)
            out.append(x)
            hs = hiddens
        out = ops.stack(out, 0)
        hs = ops.stack(hs, 0)
        return out, hs
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        il, hl = input_size, hidden_size
        self.W_ih = Parameter(
            init.rand(il, 4*hl, low=-1/hl**0.5, high=1/hl**0.5,device=device,dtype=dtype,requires_grad=True)
        )
        self.W_hh = Parameter(
            # NOTE: hl, 4*hl
            init.rand(hl, 4*hl, low=-1/hl**0.5, high=1/hl**0.5,device=device,dtype=dtype,requires_grad=True)
        )
        if bias:
            self.bias_ih = Parameter(
                init.rand(4*hl, low=-1/hl**0.5,high=1/hl**0.5,device=device,dtype=dtype,requires_grad=True)
            )
            self.bias_hh = Parameter(
                init.rand(4*hl, low=-1/hl**0.5,high=1/hl**0.5,device=device,dtype=dtype,requires_grad=True)
            )
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        h0, c0 = (None, None) if h is None else h
        hl = self.hidden_size
        out = X @ self.W_ih
        if h0 is not None:
            out += h0 @ self.W_hh
        if self.bias:
            out += self.bias_ih.reshape((1, 4*hl)).broadcast_to((bs, 4*hl))
            out += self.bias_hh.reshape((1, 4*hl)).broadcast_to((bs, 4*hl))
        out_list = ops.split(out, 1)
        # NOTE out_list is a TensorTuple, cannot slice
        i = ops.stack(tuple([out_list[i] for i in range(0, hl)]), 1)
        f = ops.stack(tuple([out_list[i] for i in range(hl, 2*hl)]), 1)
        g = ops.stack(tuple([out_list[i] for i in range(2*hl, 3*hl)]), 1)
        o = ops.stack(tuple([out_list[i] for i in range(3*hl, 4*hl)]), 1)


        g = self.tanh(g)
        i, f, o = self.sigmoid(i), self.sigmoid(f), self.sigmoid(o)
        
        c1 = i * g if c0 is None else f * c0 + i * g
        h1 = o * self.tanh(c1)
        return (h1, c1)
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers

        lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]
        for i in range(num_layers - 1):
            lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        self.lstm_cells = lstm_cells
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        Xs = ops.split(X, 0)
        h0, c0 = (None, None) if h is None else h
        hs = [None] * self.num_layers if h0 is None else ops.split(h0, 0)
        cs = [None] * self.num_layers if c0 is None else ops.split(c0, 0)
        out = []
        for t, x in enumerate(Xs):
            hiddens = []
            cells = []
            for l, model in enumerate(self.lstm_cells):
                x, c_out = model(x, (hs[l], cs[l]))
                hiddens.append(x)
                cells.append(c_out)
            out.append(x)
            hs = hiddens
            cs = cells
        out = ops.stack(out, 0)
        hs = ops.stack(hs, 0)
        cs = ops.stack(cs, 0)
        return out, (hs, cs)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # NOTE: Do not use nn.Linear(), the weight do not need grad!
        self.weight = Parameter(init.randn(
            num_embeddings, embedding_dim, device=device, dtype=dtype))

        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        one_hot_vectors = self.one_hot = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype)
        seq_len, bs, em = one_hot_vectors.shape
        one_hot_vectors = one_hot_vectors.reshape((seq_len*bs, em))
        out = one_hot_vectors @ self.weight
        return out.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION

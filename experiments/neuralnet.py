# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam


class SimpleNN:
    """A neural network with a single hidden layer, RBF activation function,
    one input node and a linear output node.
    """

    def __init__(
        self, width, num_iters=100, step_size=0.001, checkpoint=None, seed=None
    ):
        self.width = width
        # N weights and N intercepts to the first hidden layer,
        # plus N weights and 1 intercept to the output layer
        self.n_weights = width * 3 + 1
        self.num_iters = num_iters
        self.step_size = step_size
        self.checkpoint = checkpoint or np.inf
        self.random = np.random.RandomState(seed)

    @staticmethod
    def rbf(x):
        alpha, c = 1, 0
        return np.exp(-alpha * (x - c) ** 2)

    def fit(self, X, y):
        def objective(weights, iteration):
            # The sum of squared errors
            squared_error = (y - self.predict(X, weights)) ** 2
            return np.sum(squared_error)

        def callback(weights, iteration, g):
            it = iteration + 1
            if it % self.checkpoint == 0 or it in {1, self.num_iters}:
                obj = objective(weights, iteration)
                padding = int(np.log10(self.num_iters) + 1)
                print(f"[Iteration {it:{padding}d}] Sum of squared errors: {obj:.6f}")

        # Ensure that X is two-dimensional
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y)

        # Reinitialize the weights vector
        weights_init = self.random.normal(size=self.n_weights)

        # Run optimizatio
        self.weights = adam(
            grad(objective),
            weights_init,
            num_iters=self.num_iters,
            step_size=self.step_size,
            callback=callback,
        )

    def predict(self, X, weights=None):
        # Reuse the weights if none are supplied
        weights = self.weights if weights is None else weights
        assert weights.shape[-1] == self.n_weights
        dimensions = weights.ndim

        # Ensure that X and weights are two-dimensional
        X = np.asarray(X).reshape(-1, 1)
        weights = weights.reshape(-1, self.n_weights).T
    
        # Input to the first hidden layer
        w = weights[: self.width]
        b = weights[np.newaxis, self.width : self.width * 2]
        outputs = np.einsum("nk,wi->nwi", X, w) + b
        inputs = self.rbf(outputs)

        # Output layer
        w = weights[self.width * 2 : self.width * 3]
        b = weights[self.width * 3, :, np.newaxis]
        outputs = np.einsum("nwi,wi->in", inputs, w) + b

        # Return one-dimensional predictions if weights where one-dimensional
        if dimensions == 1:
            return outputs.ravel()
        return outputs

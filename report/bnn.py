import jax.numpy as np
import numpyro
import numpyro.distributions as dist


def activation(x):
    """The non-linearity used in our neural network
    """
    return np.tanh(x)


def feedforward(X, Y, width=5, hidden=1, sigma=1.0, noise=1.0):
    """An implementation of feedforward Bayesian neural network with a fixed width of hidden layers
    and linear output node.
    """
    if Y is not None:
        assert Y.shape[1] == 1
    DX, DY, DH = X.shape[1], 1, width

    # Sample first layer
    i = 0
    w = numpyro.sample(f"w{i}", dist.Normal(np.zeros((DX, DH)), np.ones((DX, DH)) * sigma))
    b = numpyro.sample(f"b{i}", dist.Normal(np.zeros((DX, DH)), np.ones((DX, DH)) * sigma))
    z = activation(np.matmul(X, w) + b)  # N DH  <= first layer of activations

    for i in range(1, hidden):
        w = numpyro.sample(f"w{i}", dist.Normal(np.zeros((DH, DH)), np.ones((DH, DH)) * sigma))
        b = numpyro.sample(f"b{i}", dist.Normal(np.zeros((1, DH)), np.ones((1, DH)) * sigma))
        z = activation(np.matmul(z, w) + b)  # N DH  <= subsequent layers of activations

    # Sample final layer of weights and neural network output
    i += 1
    w = numpyro.sample(f"w{i}", dist.Normal(np.zeros((DH, DY)), np.ones((DH, DY)) * sigma))
    b = numpyro.sample(f"b{i}", dist.Normal(0, sigma))
    z = np.matmul(z, w) + b  # N DY  <= output of the neural network

    # Likelihood
    numpyro.sample("Y", dist.Normal(z, noise), obs=Y)

import jax.numpy as np
import numpyro
import numpyro.distributions as dist
import scipy.stats


def activation(x):
    """The non-linearity used in the neural network
    """
    return np.tanh(x)


def feedforward(X, Y, width=5, hidden=1, sigma=1.0, noise=1.0):
    """An implementation of feedforward Bayesian neural network with a fixed width of hidden layers
    and linear output node using NumPyro.

    Args:
        X: input array
        Y: output 1-dimensional array
        width: number of nodes in each hidden layer (default: {5})
        hidden: number of hidden layers (default: {1})
        sigma: the standard deviation of the normal prior on the weights (default: {1.0})
        noise: the standard deviation of the normal noise in the likelihood (default: {1.0})
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


def get_noise_model(noise=1.0):
    """ Construct a frozen scipy distribution that represents the noise model of the BNN.

    The returned distribution is a Guassian for this implementation.
    The resulting function takes a parameter as the mean.

    Args:
        noise: standard deviation
    
    Returns:
        a scipy distribution
    """

    return lambda mu: scipy.stats.norm(loc=mu, scale=noise)

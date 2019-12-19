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

# +
# Import regular numpy in addition to JAX's numpy
import numpy
import pandas as pd

from jax import lax, vmap
import jax.numpy as np
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI
from numpyro.optim import Adam
from numpyro.contrib.autoguide import AutoContinuousELBO, AutoDiagonalNormal

import matplotlib.pyplot as plt


# -

def generate_data(func, points, seed=0):
    """Generate a dataframe containing the covariate X, and observations Y
    """
    numpy.random.seed(seed)

    data = []
    for segment in points:
        x = numpy.linspace(*segment["xlim"], num=segment["n_points"])
        distribution = func(x)
        # Generate observations
        y = distribution.rvs()
        df = pd.DataFrame({"x": x, "y": y})
        data.append(df)

    return pd.concat(data, ignore_index=True)


# +
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
        z = activation(np.matmul(z, w) + b)  # N DH  <= second layer of activations

    # Sample final layer of weights and neural network output
    i += 1
    w = numpyro.sample(f"w{i}", dist.Normal(np.zeros((DH, DY)), np.ones((DH, DY)) * sigma))
    b = numpyro.sample(f"b{i}", dist.Normal(0, sigma))
    z = np.matmul(z, w) + b  # N DY  <= output of the neural network

    # Likelihood
    numpyro.sample("Y", dist.Normal(z, noise), obs=Y)


# -

def sample(
    model, num_samples, num_warmup, num_chains, seed=0, chain_method="parallel", summary=True, **kwargs
):
    """Run the No-U-Turn sampler
    """
    rng_key = random.PRNGKey(seed)
    kernel = NUTS(model)
    # Note: sampling more than one chain doesn't show a progress bar
    mcmc = MCMC(kernel, num_warmup, num_samples, num_chains, chain_method=chain_method)
    mcmc.run(rng_key, **kwargs)

    if summary:
        mcmc.print_summary()

    # Return a fitted MCMC object
    return mcmc


# +
class ADVIResults:
    """A convenience class to work with the results of Variational Inference
    """

    def __init__(self, svi, guide, state, losses):
        self.svi = svi
        self.guide = guide
        self.state = state
        self.losses = losses

    def get_params(self):
        """Obtain the parameters of the variational distribution
        """
        return self.svi.get_params(self.state)

    def sample_posterior(self, rng_key, n_samples):
        """Sample from the posterior, making all necessary transformations of
        the reparametrized variational distribution.
        """
        params = self.get_params()
        posterior_samples = self.guide.sample_posterior(rng_key, params, sample_shape=(n_samples,))
        return posterior_samples

    def plot_loss(self):
        plt.plot(self.losses)
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.title(f"Negative ELBO by Iteration, Final Value {self.losses[-1]:.1f}")


def fit_advi(model, num_iter, learning_rate=0.01, seed=0):
    """Automatic Differentiation Variational Inference using a Normal variational distribution
    with a diagonal covariance matrix.
    """
    rng_key = random.PRNGKey(seed)
    adam = Adam(learning_rate)
    # Automatically create a variational distribution (aka "guide" in Pyro's terminology)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, adam, AutoContinuousELBO())
    svi_state = svi.init(rng_key)

    # Run optimization
    last_state, losses = lax.scan(lambda state, i: svi.update(state), svi_state, np.zeros(num_iter))
    results = ADVIResults(svi=svi, guide=guide, state=last_state, losses=losses)
    return results


# +
def predict(model, rng_key, samples, X):
    """Numpyro's helper function for prediction
    """
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None)
    return model_trace["Y"]["value"]


def simulate_posterior_predictive(model, mcmc_or_vi, X_test, n_samples=None, seed=1, noiseless=False):
    """Predict Y_test at inputs X_test for a Numpyro model
    """
    # Set random state
    rng_key = random.PRNGKey(seed)

    # Obtain samples of the posterior
    if isinstance(mcmc_or_vi, MCMC):
        posterior_samples = mcmc.get_samples()
        n_samples = mcmc_or_vi.num_samples * mcmc_or_vi.num_chains
    elif isinstance(mcmc_or_vi, ADVIResults):
        assert n_samples is not None, "The argument `n_samples` must be specified for Variational Inference"
        posterior_samples = mcmc_or_vi.sample_posterior(rng_key, n_samples)
    else:
        raise ValueError("The `mcmc_or_vi` argument must be of type MCMC or ADVIResults")

    # Generate samples from the posterior predictive
    vmap_args = (posterior_samples, random.split(rng_key, n_samples))
    predictions = vmap(lambda samples, rng_key: predict(model, rng_key, samples, X_test))(*vmap_args)
    predictions = predictions[..., 0]

    # Optionally, return mean predictions (the variance of which is epistemic uncertainty)
    if noiseless:
        raise NotImplemented("A model with zero noise should be passed instead")

    return predictions


# +
def plot_true_function(func, df, title=None):
    x = numpy.linspace(df.x.min(), df.x.max(), num=1000)
    distribution = func(x)
    lower, upper = distribution.interval(0.95)

    plt.fill_between(
        x, lower, upper, color="tab:orange", alpha=0.1, label="True 95% Interval",
    )
    plt.scatter(df.x, df.y, s=10, color="lightgrey", label="Observations")
    plt.plot(x, distribution.mean(), color="tab:red", label="True Mean")
    if title is not None:
        plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)


def plot_posterior_predictive(x, y, title=None, func=None, df=None):
    if func is not None and df is not None:
        plot_true_function(func, df)

    x = x.ravel()
    lower, upper = numpy.percentile(y, [2.5, 97.5], axis=0)
    plt.fill_between(x, lower, upper, color="tab:blue", alpha=0.1, label=f"Predicted 95% Interval")
    plt.plot(x, y.mean(axis=0), color="tab:blue", label=f"Predicted Mean")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

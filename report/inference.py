from jax import lax, vmap

# JAX-wrapped Numpy
import jax.numpy as np
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, SVI
from numpyro.optim import Adam
from numpyro.contrib.autoguide import AutoContinuousELBO, AutoDiagonalNormal

import matplotlib.pyplot as plt


def sample(
    model,
    num_samples,
    num_warmup,
    num_chains=2,
    seed=0,
    chain_method="parallel",
    summary=True,
    **kwargs,
):
    """Run the No-U-Turn sampler

    Args:
        model: an NumPyro model function
        num_samples: number of samples to draw
        num_warmup: number of samples to use for tuning
        num_chains: number of chains to draw (default: {2})
        **kwargs: other arguments to be passed to the model function
        seed: random seed (default: {0})
        chain_method: one of NumPyro's sampling methods — "parallel" / "sequential" /
            "vectorized" (default: {"parallel"})
        summary: print diagnostics, including the Effective sample size and the
            Gelman-Rubin test (default: {True})

    Returns:
        mcmc: A fitted MCMC object
    """
    rng_key = random.PRNGKey(seed)
    kernel = NUTS(model)
    # Note: sampling more than one chain doesn't show a progress bar
    mcmc = MCMC(kernel, num_warmup, num_samples, num_chains, chain_method=chain_method)
    mcmc.run(rng_key, **kwargs)

    if summary:
        mcmc.print_summary()

    return mcmc


def predict(model, rng_key, samples, X):
    """NumPyro's helper function for prediction

    Used internally only by sample_pp().

    Args:
        model: a NumPyro model function
        rng_key: JAX's random generator key of type PRNGKey
        samples: posterior samples, an array
        X: input values for which to generate posterior predictive

    Returns:
        samples of the posterior predictive, an array
    """
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None)
    return model_trace["Y"]["value"]


def simulate_pp(model, mcmc_or_vi, X_test, n_samples=None, seed=1, noiseless=False):
    """Simulate posterior predictive: predict Y_test at each input of X_test

    Args:
        model: a NumPyro model function
        mcmc_or_vi: a fitted MCMC or VI object
        X_test: the X's for which to generate posterior predictive
        n_samples: number of samples to generate (required in case of VI,
            otherwise the same as the number of posterior MCMC samples) (default: {None})
        seed: random seed (default: {1})
        noiseless: return the posterior predictive without the noise (default: {False})

    Returns:
        predictions: samples of the posterior predictive,
        an array of shape (n_samples, X_test.shape[0])

    Raises:
        ValueError: if n_samples isn't specified for Variational Inference
        NotImplemented: of noiseless is set (something to attend to in the future)
    """
    # Set random state
    rng_key = random.PRNGKey(seed)

    # Obtain samples of the posterior
    if isinstance(mcmc_or_vi, MCMC):
        posterior_samples = mcmc_or_vi.get_samples()
        n_samples = mcmc_or_vi.num_samples * mcmc_or_vi.num_chains
    elif isinstance(mcmc_or_vi, ADVIResults):
        assert (
            n_samples is not None
        ), "The argument `n_samples` must be specified for Variational Inference"
        posterior_samples = mcmc_or_vi.sample_posterior(rng_key, n_samples)
    else:
        raise ValueError("The `mcmc_or_vi` argument must be of type MCMC or ADVIResults")

    # Generate samples from the posterior predictive
    vmap_args = (posterior_samples, random.split(rng_key, n_samples))
    predictions = vmap(lambda samples, rng_key: predict(model, rng_key, samples, X_test))(
        *vmap_args
    )
    predictions = predictions[..., 0]

    # Optionally, return mean predictions (the variance of which is epistemic uncertainty)
    if noiseless:
        raise NotImplemented("A model with zero noise should be passed instead")

    return predictions


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

        Args:
            rng_key: random number generator key, of type PRNGKey
            n_samples: number of samples to draw from the variational posterior

        Returns:
            posterior_samples: an array of shape (n_samples,)
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

    Args:
        model: a NumPyro's model function
        num_iter: number of iterations of gradient descent (Adam)
        learning_rate: the step size for the Adam algorithm (default: {0.01})
        seed: random seed (default: {0})

    Returns:
        a set of results of type ADVIResults
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


def get_metrics(mcmc):
    """Extract diagnostic metrics from a fitted MCMC model: the minimum effective sample size
    and the maximum Gelman-Rubin test value.

    Args:
        mcmc: a fitted MCMC object

    Returns:
        metrics: a dictionary of the metrics
    """
    summary_dict = summary(mcmc._states["z"])

    min_ess = np.inf
    max_rhat = 0

    for name, stats_dict in summary_dict.items():
        min_ess = min(min_ess, stats_dict["n_eff"].min())
        max_rhat = max(max_rhat, stats_dict["r_hat"].max())

    metrics = {"min_ess": min_ess, "max_rhat": max_rhat}
    return metrics

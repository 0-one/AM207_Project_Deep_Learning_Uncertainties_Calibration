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


def sample(
    model,
    num_samples,
    num_warmup,
    num_chains,
    seed=0,
    chain_method="parallel",
    summary=True,
    **kwargs,
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


def predict(model, rng_key, samples, X):
    """NumPyro's helper function for prediction
    """
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None)
    return model_trace["Y"]["value"]


def simulate_pp(model, mcmc_or_vi, X_test, n_samples=None, seed=1, noiseless=False):
    """Simulate posterior predictive: predict Y_test at each input of X_test
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


def get_metrics(mcmc):
    """Extract diagnostic metrics from a fitted MCMC model: the minimum effective sample size
    and the maximum Gelman-Rubin test value.
    """
    summary_dict = summary(mcmc._states["z"])

    min_ess = np.inf
    max_rhat = 0

    for name, stats_dict in summary_dict.items():
        min_ess = min(min_ess, stats_dict["n_eff"].min())
        max_rhat = max(max_rhat, stats_dict["r_hat"].max())

    return {"min_ess": min_ess, "max_rhat": max_rhat}

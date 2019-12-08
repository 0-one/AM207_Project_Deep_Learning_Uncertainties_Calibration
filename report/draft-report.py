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

# + {"slideshow": {"slide_type": "skip"}, "cell_type": "markdown"}
# **Requirements:** Please install NumPyro by running:
#
# ```$ pip install --upgrade numpyro```

# + {"slideshow": {"slide_type": "skip"}}
from functools import partial

# Regular Numpy
import numpy
import numpyro
import scipy.stats

import matplotlib.pyplot as plt
# %matplotlib inline

from bnn import feedforward
from inference import sample, simulate_pp, fit_advi, get_metrics
from data import *
from plotting import *

# + {"slideshow": {"slide_type": "skip"}}
# Configure matplotlib format and default dimensions
# %config InlineBackend.figure_formats = ['svg']
plt.rc("figure", figsize=(7, 3.5))

# Perform inference using a CPU and 2 cores
numpyro.set_platform("cpu")
numpyro.set_host_device_count(2)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Accurate Uncertainties for Deep Learning Using Calibrated Regression
#
# Analysis of the paper by Kuleshov et al. (2018)
#
# Project team: Piotr Pekala, Benjamin Yuen, Dmitry Vukolov, Alp Kutlualp
#
# ### Outline
#
# 1. Problem statement: miscalibration and its sources
# 2. Related work
# 3. Proposed calibration algorithm
# 4. Experiments
# 5. Evaluation of the claims
# 6. Future work

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # The Issue of Miscalibration
#
# **Problem statement:** Proper quantification of uncertainty is crucial for applying statistical models to real-world situations. The Bayesian approach to modeling provides us with a principled way of obtaining such uncertainty estimates. Yet, due to various reasons, such estimates are often inaccurate. For example, a 95% posterior predictive interval does not contain the true outcome with 95% probability.
#
# **Context:** <mark>why is this problem important or interesting? any examples?</mark>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Sources of Miscalibration
#
# Below we demonstrate that the problem of miscalibration exists and show why it exists for **Bayesian neural networks** in regression tasks. We focus on the following sources of miscalibration:
# - The **prior** is wrong, e.g. too strong and overcertain
# - The **likelihood function** is wrong. There is bias, i.e. the neural network is too simple and is unable to model the data.
# - The **noise** specification in the likelihood is wrong.
# - The **inference** is approximate or is performed incorrectly.
#
# Our aim is to establish a causal link between each aspect of the model building process and a bad miscalibrated outcome.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Methodology
#
# 1. **Data Generation:** We generate the data from a known true function with Gaussian noise. We then build multiple feedforward BNN models using:
#   - different network architectures
#   - several priors on the weights, depending on model complexity
#   - different variance of the Gaussian noise in the likelihood function
#   
# 2. **Inference**: We then obtain the posterior of the model by:
#   - sampling from it with the No-U-Turn algorithm
#   - approximating the posterior using Variational Inference with reparametrization and isotropic normals
#   
# 3. **Diagnostics**: We check for convergence using trace plots, the effective sample size, and Gelman-Rubin tests. In the case of variational inference, we track the ELBO during optimization. The simulated posterior predictive is evaluated visually.
#
# The probabilistic library [NumPyro](https://github.com/pyro-ppl/numpyro) provides fast implementations of both algorithms, which we make use of in this research. Due to time constraints we do not perform multiple random restarts, so the results may be subject to randomness.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Example: a Toy Dataset
#
# Using a simple data-generating function $y_i = 0.1 x^3_i + \varepsilon$, where $\varepsilon \sim \mathcal{N}(0, 0.5^2)$ and a series of BNN models we evaluate the impact of our design choices on the posterior predictive.

# + {"slideshow": {"slide_type": "-"}}
# Define the true function and generate observations
func = lambda x: scipy.stats.norm(loc=0.1 * x ** 3, scale=0.5)
func.latex = r"$y_i = 0.1x_i^3 + \varepsilon$"

data_points = [
    {"n_points": 40, "xlim": [-4, -1]},
    {"n_points": 40, "xlim": [1, 4]},
]
df = generate_data(func, points=data_points, seed=4)

# Plot the data
plot_true_function(func, df, title=f"True Function: {func.latex}")


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Proper Posterior Predictive
#
# A neural network with 50 nodes in a single hidden layer, well-chosen prior and noise, as well as correctly performed inference using sampling result in a posterior predictive that adequately reflects both epistemic and aleatoric uncertainty:

# + {"slideshow": {"slide_type": "skip"}}
def sample_and_plot(df, func, *, hidden, width, sigma, noise, num_samples, num_warmup, num_chains):
    """A helper function to instantiate the model, sample from the posterior, simulate
    the posterior predictive and plot it against the observations and the true function.
    """
    # Observations
    X = df[["x"]].values
    Y = df[["y"]].values
    X_test = numpy.linspace(X.min(), X.max(), num=1000)[:, numpy.newaxis]
    
    # Instantiate the model with prior standard deviation and likelihood noise
    model = partial(feedforward, X=X, Y=Y, width=width, hidden=hidden, sigma=sigma, noise=noise)

    # Run the No-U-Turn sampler
    mcmc = sample(model, num_samples, num_warmup, num_chains, seed=0, summary=False)
    
    # Generate the posterior predictive and plot the results
    posterior_predictive = simulate_pp(model, mcmc, X_test, seed=1)
    plot_posterior_predictive(
        X_test,
        posterior_predictive,
        func=func,
        df=df,
        title=f"BNN with {width} Nodes in {hidden} Hidden Layer{'' if hidden == 1 else 's'},\n"
        f"Weight Uncertainty {sigma}, Noise {noise}, NUTS Sampling",
    )
    
    # Print the diagnostic tests
    diagnostics=get_metrics(mcmc)
    message = ('Minimum ESS: {min_ess:,.2f}\n'
               'Max Gelman-Rubin: {max_rhat:.2f}').format(**diagnostics)
    plt.gcf().text(0.95, 0.15, message)


# + {"slideshow": {"slide_type": "-"}}
# Parameters of a Bayesian neural network
model_params = {
    # Number of hidden layers
    "hidden": 1,
    # Width of hidden layers
    "width": 50,
    # Standard deviation of the prior
    "sigma": 1.25,
    # Standard deviation of the likelihood
    "noise": 0.5,
}

# NUTS sampler parameters
sampler_params = {
    "num_chains": 2,
    "num_samples": 2000,
    "num_warmup": 2000,
}

# Run the No-U-Turn sampler, generate the posterior predictive and plot
sample_and_plot(df, func, **model_params, **sampler_params)
# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# Naturally, our claims regarding the adequacy of epistemic uncertainty are subjective due to absence of universal quantitative metrics.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Wrong Prior: Too Wide
#
# Higher variance of the prior results in a significantly larger and most likely unreasonable epistemic uncertainty:

# + {"slideshow": {"slide_type": "-"}}
model_params = {
    "hidden": 1,
    "width": 50,
    "sigma": 1.8,
    "noise": 0.5,
}
# Run the No-U-Turn sampler, generate the posterior predictive and plot
sample_and_plot(df, func, **model_params, **sampler_params)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Wrong Prior: Too Narrow
#
# Lower variance of the prior prevents the model from adequately reflecting epistemic uncertainty in areas where no data is available. It also introduces bias: a neural network with 50 nodes in a single hidden layer (i.e. 151 weights) is unable to fit a cubic function:

# + {"slideshow": {"slide_type": "-"}}
model_params = {
    "hidden": 1,
    "width": 50,
    "sigma": 0.5,
    "noise": 0.5,
}
# Run the No-U-Turn sampler, generate the posterior predictive and plot
sample_and_plot(df, func, **model_params, **sampler_params)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Wrong Prior: Extremely Restrictive
#
# The bias becomes apparent with an even narrower prior on the weights. This is a major issue with the model that needs to be fixed. Calibration is inappropriate in this case.

# + {"slideshow": {"slide_type": "-"}}
model_params = {
    "hidden": 1,
    "width": 50,
    "sigma": 0.1,
    "noise": 0.5,
}
# Run the No-U-Turn sampler, generate the posterior predictive and plot
sample_and_plot(df, func, **model_params, **sampler_params)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Wrong Likelihood Function

# + {"slideshow": {"slide_type": "-"}}


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Wrong Noise

# + {"slideshow": {"slide_type": "-"}}


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Approximate Inference

# + {"slideshow": {"slide_type": "-"}}


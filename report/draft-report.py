# -*- coding: utf-8 -*-
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

# Ordinary Numpy (without an autograd/JAX wrapper around it)
import numpy as np
import numpyro
import pandas as pd
import scipy.stats
from sklearn.isotonic import IsotonicRegression
from IPython.display import display

import matplotlib.pyplot as plt
# %matplotlib inline

from bnn import feedforward
from inference import sample, simulate_pp, fit_advi, get_metrics
from data import generate_data
from plotting import *

# + {"slideshow": {"slide_type": "skip"}}
# Configure matplotlib format and default dimensions
# %config InlineBackend.figure_formats = ['svg']
plt.rc("figure", figsize=(7, 3.5))

# Perform inference using a CPU and 2 cores
numpyro.set_platform("cpu")
numpyro.set_host_device_count(2)


# + {"slideshow": {"slide_type": "skip"}}
def sample_and_plot(df, func, *, hidden, width, sigma, noise, num_samples, num_warmup, num_chains):
    """A helper function to instantiate the model, sample from the posterior, simulate
    the posterior predictive and plot it against the observations and the true function.
    """
    # Observations
    X = df[["x"]].values
    Y = df[["y"]].values
    X_test = np.linspace(X.min(), X.max(), num=1000)[:, np.newaxis]
    
    # Instantiate the model with a network architecture, prior standard deviation and likelihood noise
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
    
    # Return the fitted MCMC object to enable detailed diagnostics, e.g. mcmc.print_summary()
    return mcmc


# + {"slideshow": {"slide_type": "skip"}}
def fit_and_plot(df, func, *, hidden, width, sigma, noise, num_iter, learning_rate):
    """A helper function to instantiate the model, approximate the posterior using Variational Inference
    with reparametrization and isotropic Gaussians, simulate the posterior predictive and plot it
    against the observations and the true function.
    """
    # Observations
    X = df[["x"]].values
    Y = df[["y"]].values
    X_test = np.linspace(X.min(), X.max(), num=1000)[:, np.newaxis]
    
    # Instantiate the model with a network architecture, prior standard deviation and likelihood noise
    model = partial(feedforward, X=X, Y=Y, width=width, hidden=hidden, sigma=sigma, noise=noise)

    # Approximate the posterior using Automatic Differentiation Variational Inference
    vi = fit_advi(model, num_iter=num_iter, learning_rate=learning_rate, seed=0)
    
    # Generate the posterior predictive and plot the results
    posterior_predictive = simulate_pp(model, vi, X_test, n_samples=1000, seed=1)
    plot_posterior_predictive(
        X_test,
        posterior_predictive,
        func=func,
        df=df,
        title=f"BNN with {width} Nodes in {hidden} Hidden Layer{'' if hidden == 1 else 's'},\n"
        f"Weight Uncertainty {sigma}, Noise {noise}, VI Approximation",
    )
    # Return the variation inference object to enable diagnostics, e.g. vi.plot_loss()
    return vi


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Accurate Uncertainties for Deep Learning Using Calibrated Regression
#
# Analysis of the paper by [Kuleshov et al. (2018)](https://arxiv.org/pdf/1807.00263)
#
# Project team: Piotr Pekala, Benjamin Yuen, Dmitry Vukolov, Alp Kutlualp

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Outline
#
# 1. Problem Statement: Miscalibration and its Sources
# 2. Existing work
# 3. Contribution: The Calibration Algorithm
# 4. Technical Details
# 5. Experiments
# 6. Evaluation of the claims
# 7. Future work

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Problem Statement

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # The Issue of Miscalibration
#
# **Problem statement:** Proper quantification of uncertainty is crucial for applying statistical models to real-world situations. The Bayesian approach to modeling provides us with a principled way of obtaining such uncertainty estimates. Yet, due to various reasons, such estimates are often inaccurate. For example, a 95% posterior predictive interval does not contain the true outcome with 95% probability. Such a model is *miscalibrated*.
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
# A neural network with 50 nodes in a single hidden layer, well-chosen prior and noise, as well as correctly performed inference using sampling produce a posterior predictive that adequately reflects both epistemic and aleatoric uncertainty:

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

# Run the No-U-Turn sampler, generate the posterior predictive and plot it
mcmc = sample_and_plot(df, func, **model_params, **sampler_params)
# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# Naturally, our claims regarding the adequacy of epistemic uncertainty are subjective due to the absence of universal quantitative metrics.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Wrong Prior: Too Wide
#
# The prior on the network weights defines epistemic uncertainty. A higher than necessary variance of the prior results in a significantly larger and most likely unreasonable epistemic uncertainty:

# + {"slideshow": {"slide_type": "-"}}
model_params = {
    "hidden": 1,
    "width": 50,
    "sigma": 1.8,
    "noise": 0.5,
}
# Run the No-U-Turn sampler, generate the posterior predictive and plot it
mcmc = sample_and_plot(df, func, **model_params, **sampler_params)

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
# Run the No-U-Turn sampler, generate the posterior predictive and plot it
mcmc = sample_and_plot(df, func, **model_params, **sampler_params)

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
# Run the No-U-Turn sampler, generate the posterior predictive and plot it
mcmc = sample_and_plot(df, func, **model_params, **sampler_params)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Wrong Likelihood Function
#
# Similar to the previous example, a BNN may demonstrate bias by beeing too simple architecturally. That is difficult to demonstrate for a dataset generated by a cubic function (can be described by just 4 points). Still, we can try a combination of the prior and noise specification from the proper model and reduce the number of nodes. The sampler does not converge in this setup.

# + {"slideshow": {"slide_type": "-"}}
model_params = {
    "hidden": 1,
    "width": 2,
    "sigma": 1.25,
    "noise": 0.5,
}
# Run the No-U-Turn sampler, generate the posterior predictive and plot it
mcmc = sample_and_plot(df, func, **model_params, **sampler_params)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Link Between Prior and Network Architecture
#
# The appropriate level of the prior variance depends on the network complexity. A simpler network with 10 nodes and the same prior variance as our original benchmark model predicts much lower epistemic uncertainty. Therefore, the prior has to be selected for each particular network configuration.

# + {"slideshow": {"slide_type": "-"}}
model_params = {
    "hidden": 1,
    "width": 10,
    "sigma": 1.25,
    "noise": 0.5,
}
# Run the No-U-Turn sampler, generate the posterior predictive and plot it
mcmc = sample_and_plot(df, func, **model_params, **sampler_params)


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Wrong Noise: Too High
#
# The noise in the likelihood function corresponds to aleatoric uncertainty. The effect of the wrong noise specification is that aleatoric uncertainty is captured incorrectly. In the model below the noise is still Gaussian, but has a higher variance than the true one:
# -

model_params = {
    "hidden": 1,
    "width": 50,
    "sigma": 1.25,
    "noise": 1.0,
}
# Run the No-U-Turn sampler, generate the posterior predictive and plot it
mcmc = sample_and_plot(df, func, **model_params, **sampler_params)

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# This might be a good candidate for calibration. Alternatively, one could find ways for the network to learn the noise from the data or put an additional prior on the variance of the noise.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Wrong Noise: Too Small
#
# Similarly, if the noise is too small, the resulting aleatoric uncertainty captured by the posterior predictive will be unrealistically low:

# + {"slideshow": {"slide_type": "-"}}
model_params = {
    "hidden": 1,
    "width": 50,
    "sigma": 1.25,
    "noise": 0.25,
}
# Run the No-U-Turn sampler, generate the posterior predictive and plot it
mcmc = sample_and_plot(df, func, **model_params, **sampler_params)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Approximate Inference
#
# Using approximate methods of inference is also likely to lead to a miscalibrated posterior predictive. In the example below, Variational Inference with reparametrization on a network with 50 nodes produces too low epistemic uncertainty and slightly larger aleatoric uncertainty. Calibration may be used for correcting the latter.

# + {"slideshow": {"slide_type": "-"}}
model_params = {
    "hidden": 1,
    "width": 50,
    "sigma": 1.25,
    "noise": 0.5,
}

vi_params = {
    "num_iter": 500_000,
    "learning_rate": 0.001,
}

# Approximate the posterior with Variational Inference, generate the posterior predictive and plot it
vi = fit_and_plot(df, func, **model_params, **vi_params)


# + {"slideshow": {"slide_type": "skip"}}
# Plot the negative ELBO for diagnostics
vi.plot_loss()


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Existing Work

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Literature Review
#
# <mark>To be filled...</mark>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Contribution: The Calibration Algorithm

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # The Calibration Algorithm
#
# **Proposition:** The authors of the paper propose a simple **calibration algorithm** for regression. In the case of Bayesian models, the procedure ensures that uncertainty estimates are well-calibrated, given enough data. In other words, the resulting posterior predictive aligns with the data, i.e. the observations fall within each posterior predictive interval  (e.g. the 95% interval) with a corresponding probability.
#
# ###### Unique Contribution:
# - The procedure is universally applicable to any **regression** model, be it Bayesian or frequentist. It extends previous work on calibration methods for classification.
# - Compared to alternative approaches, the method doesn't require modification of the model. Instead, the algorithm is applied to the output of any existing model in a postprocessing step.
#
# **Claim:** The authors claim that the method outperforms other techniques by consistently producing well-calibrated forecasts, given enough i.i.d. data. Based on their experiments, the procedure also improves predictive performance in several tasks, such as time-series forecasting and reinforcement learning.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Technical Details

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Recalibration of Regression Models
#
# The algorithm has two main steps (from Algorithm 1 listing in the paper):
# 1. Construct a recalibration dataset $\mathcal{D}$:
# $$
# \mathcal{D} = \left\{\left(\left[H\left(x_t\right)\right]\left(y_t\right), \hat P\left(\left[H\left(x_t\right)\right]\left(y_t\right)\right)\right)\right\}_{t=1}^T
# $$
# where:
#  - $T$ is the number of observations
#  - $H(x_t)$ is a CDF of the posterior predictive evaluated at $x_t$
#  - $H(x_t)(y_t)$ is the predicted quantile of $y_t$
#  - $\hat P(p)$ is the empirical quantile of $y_t$, computed as:
#  
#   $$
#   \hat P(p)=\left|\left\{y_t\mid \left[H\left(x_t\right)\right]\left(y_t\right)\lt p, t=1\ldots T\right\}\right|/T
#   $$
#
#
# 2. Train a model $R$ (e.g. isotonic regression) on $\mathcal{D}$.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # The Algorithm Step-by-Step
#
# Suppose we have the following hypothetical posterior predictive, which is heteroscedastic and is underestimating uncertainty. For each value of the covariate $X$, the posterior predictive provides us with a conditional distribution $f(Y|X)$:

# + {"slideshow": {"slide_type": "-"}}
def func(x, base_std):
    """The basis for the true function and a hypothetical posterior predictive
    """
    std = np.abs(x) * base_std
    std = np.where(std < 0.5, 0.5, std)
    return scipy.stats.norm(loc=0.1 * x ** 3, scale=std)


def true_func(x):
    """The true function: y_i = 0.1 x_i^3 + epsilon_i (heteroscedastic)
    """
    return func(x, base_std=1.5)


def ppc(x):
    """A miscalibrated posterior predictive that underestimates uncertainty
    """
    return func(x, base_std=0.5)


# Generate observations for an equally sized train set and a test set
data_points = [
    {"n_points": 200, "xlim": [-4, 4]},
]
df = generate_data(true_func, points=data_points, seed=4)
df_test = generate_data(true_func, points=data_points, seed=0)


def ppc_quantiles(y, x=df.x):
    """Return the predicted quantiles of y_t evaluated at x_t.
    Here we do it analytically, but in practice the quantiles are estimated based on the samples.
    """
    return ppc(x).cdf(y)

# Visualize an illustrative miscalibrated posterior predictive
plot_illustration(ppc, df, title=f"Miscalibrated Posterior Predictive", conditionals=True)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 1: Construct a Recalibration Dataset
#
# The first step of the calibration algorithm is to obtain predictive conditional distributions for each 𝑋 in the dataset. If no closed-form is available we simulate the posterior predictive based on the samples of the posterior:

# + {"slideshow": {"slide_type": "-"}}
plot_table()

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# An alternative, more commonly used notation for $H(x_t)$ is $F_t$ (a CDF)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 1a: Compute the Predicted Quantiles
#
# The observed $Y$ for each data point falls somewhere within those conditional distributions. We evaluate the conditional CDFs at each observed value of the response $Y$ to obtain the predicted quantiles. In the absence of analytical form, we simply count the proportion of samples that are less than $y_t$. This gives us the estimated quantile of $y_t$ at $x_t$ in the posterior predictive distribution:

# + {"slideshow": {"slide_type": "-"}}
plot_table(mark_y=True, show_quantiles="predicted")

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 1b: Estimate the Empirical Quantiles
#
# We next find the empirical quantiles, which are defined as the proportion of observations that have lower quantile values than that of the current observation. This is equivalent to finding the empirical CDF of the predicted quantiles. The mapping of predicted quantiles and the empirical quantiles will form a recalibration dataset:

# + {"slideshow": {"slide_type": "-"}}
# Estimate the predicted quantiles
predicted_quantiles = ppc_quantiles(df.y)

# Compute the empirical quantiles
T = predicted_quantiles.shape[0]
empirical_quantiles = (predicted_quantiles.reshape(1, -1) 
                       <= predicted_quantiles.reshape(-1, 1)).sum(axis=1) / T

# Plot the predicted against the empirical quantiles
plot_ecdf(predicted_quantiles)
plt.scatter(predicted_quantiles, empirical_quantiles, color="tab:blue", zorder=2);

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 1c: Form a Recalibration Dataset
#
# The mapping is obtained for all observations in the dataset. Note that in this example the first two observations have different conditional distributions, but the same values of the predicted and empirical quantiles. The calibration procedure doesn't distinguish between such cases:

# + {"slideshow": {"slide_type": "-"}}
plot_table(mark_y=True, show_quantiles="all")

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 1c: Form a Recalibration Dataset
#
# The inverse S-curve of the recalibration dataset in our example is characteristic of a posterior predictive that underestimates uncertainty. The diagnonal line denotes perfect calibration:

# + {"slideshow": {"slide_type": "-"}}
# Visualize the recalibration dataset
plt.scatter(predicted_quantiles, empirical_quantiles)
plt.plot([0, 1], [0, 1], color="tab:grey", linestyle="--")
plt.xlabel("Predicted Cumulative Distribution")
plt.ylabel("Empirical Cumulative Distribution")
plt.title("Recalibration Dataset");

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 2: Train a Model
#
# We then train a model (e.g. isotonic regression) on the recalibration dataset and use it to output the actual probability of any given quantile or interval. Here the 95% posterior predictive interval corresponds to a much narrower calibrated interval:

# + {"slideshow": {"slide_type": "-"}}
# Fit isotonic regression
ir = IsotonicRegression(out_of_bounds="clip")
ir.fit(predicted_quantiles, empirical_quantiles)

# Obtain actual calibrated quantiles
calibrated_quantiles = ir.predict([0.025, 0.5, 0.975])
table = (
    pd.DataFrame(
        {"Predicted quantiles": [0.025, 0.5, 0.975], "Calibrated quantiles": calibrated_quantiles}
    )
    .round(3)
    .style.hide_index()
)
display(table)

# + {"slideshow": {"slide_type": "-"}, "cell_type": "markdown"}
# Ideally, the model should be fit on a separate calibration set in order to reduce overfitting. Alternatively, multiple models can be trained in a way similar to cross-validation:
#
# - use $K-1$ folds for training
# - use 1 fold for calibration
# - at prediction time, the output is the average of $K$ models

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Detailed Steps (Part 1)
#
# Concretely, for models without closed form posterior predictive CDF, the calibration algorithm is restated as:
# 1. Generate $N$ samples from the posterior, $\theta = \left\{\theta_n, n=1\ldots N\right\}$.
# 2. For each observation, $t \in 1\ldots T$
#     * Generate $N$ samples of posterior predictive, $s_{t_n}$, from $\theta$ and evaluated at $x_t$
#     * Let $p_t$ be the quantile of $y_t$. Estimate the quantile of $y_t$ as
#     $$p_t = \widehat{\left[H(x_t)\right](y_t)} = \left|\left\{s_{t_n}\mid s_{t_n} \le y_t,n=1\ldots N\right\}\right|/N$$
# 3. For each $t$
#     * calculate $\hat P\left(\widehat{\left[H\left(x_t\right)\right]\left(y_t\right)}\right) = \hat P\left(p_t\right)$ as
#     $$\hat P\left(p_t\right) = \left|\left\{p_u\mid p_u\lt p_t, u=1\ldots T\right\}\right|/T
#     $$
#     That is, find the proportion of observations that have lower quantile values than that of the current observation.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Detailed Steps (Part 2)
#
# 4. Construct $\mathcal{D} = \left\{\left(\widehat{\left[H\left(x_t\right)\right]\left(y_t\right)}, \hat P\left(\widehat{\left[H\left(x_t\right)\right]\left(y_t\right)}\right)\right)\right\}_{t=1}^T$
# 5. Train calibration transformation using $\mathcal{D}$ via isotonic regression (or other models). Running prediction on the trained model results in a transformation $R$, $[0,1] \to [0,1]$. We can compose the calibrated model as $R\circ H\left(x_t\right)$.
# 6. To find the calibrated confidence intervals, we need to remap the original upper and lower limits. For example, the upper limit $y_{t\ high}$ is mapped to the calibrated value $y_{t\ high}'$ as:
# $$y_{t\ high}'=\left[H\left(x_t\right)\right]^{-1}\left(R^{-1}\left\{\left[H\left(x_t\right)\right]\left(y_{t\ high}\right)\right\}\right)$$

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Diagnostics
#
# As a visual diagnostic tool, the authors suggest using a calibration plot that shows the true frequency of points in each quantile or confidence interval compared to the predicted fraction of points in that interval. Well-calibrated models should be close to a diagonal line:

# + {"slideshow": {"slide_type": "-"}}
# Perform calibration on a hold-out test dataset
predicted_quantiles_test = ppc_quantiles(df_test.y)
calibration_plot(predicted_quantiles_test, model=ir)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Quantitative metrics
#
# Several alternatives are available, each with specific advantages and disadvantages:
#
# **1. Calibration error**
# $$cal(F_1, y_1, ..., F_N, y_N ) = \sum_{j=1}^m w_j \cdot (p_j − \hat{p}_j)^2$$
# Provides a synthetic measure representing the overall *'distance'* of the points on the calibration curve from the $45^\circ$ straight line. The weights ($w_j$) might be used to reduce the importance of intervals containing few observations. Value of $0$ indicates perfect calibration. Sensitive to binning.
#
# **2. Predictive RMSE**
# $$\sqrt{\frac{1}{N}\sum_{n=1}^{N}||y_n-\mathbb{E}_{q(W)}[f(x_n,W)]||_{2}^{2}}$$
# Measures the model *fit* to the observed data by normalizing the difference between observations and the mean of the posterior predictive. Minimizing RMSE does not guarantee calibration of the model.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Quantitative metrics cont.
#
# **3. Mean prediction interval width**
# $$\frac{1}{N}\sum_{n=1}^{N}\hat{y}_{n}^{high} - \hat{y}_{n}^{low},$$
# where $\hat{y}_{n}^{high}$ and $\hat{y}_{n}^{low}$ are - respectively - the 97.5 and 2.5 percentiles of the predicted outputs for $x_n$.
# Average difference between the the upper and lower bounds of predictive intervals evaluated for all the observations (different significance values might be used to define the predictive intervals). By itself provides information on the precision of the prediction (*confidence* with which a prediction is made) rather than calibration or miscalibration of the model. However may be used in conjunction with PICP.
#
# **4. Prediction interval coverage probability (PICP)**
# $$\frac{1}{N}\sum_{n=1}^{N}\mathbb{1}_{y_n\leq\hat{y}_{n}^{high}} \cdot \mathbb{1}_{y_n\geq\hat{y}_{n}^{low}}$$
# Calculates the share of observations covered by 95% (or any other, selected) predictive intervals. Alignment of the PICP with the probability mass assigned to the predictive interval generating it may misleadingly point to proper calibration if true noise distribution belongs to a different family than the posterior predictive. Requires a large sample of observations. 

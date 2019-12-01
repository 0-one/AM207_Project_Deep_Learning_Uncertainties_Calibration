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
import warnings

from autograd import numpy as np
import arviz as az
import pymc3 as pm
import scipy.stats
import theano

import matplotlib.pyplot as plt
# %matplotlib inline

# %run helpers.ipynb
# %run neuralnet.ipynb

# +
# Ignore PyMC3 FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Display all columns of a dataframe
pd.set_option('display.max_columns', None)

# Make plots larger by default
plt.rc('figure', dpi=100)
# -

# # True function: $y_i = 0.1 x_i^3 + \varepsilon$, where $\varepsilon \sim \mathcal{N}(0, 0.5^2)$

# +
# Define the true function and generate observations
func = lambda x: scipy.stats.norm(loc=0.1 * x**3, scale=0.5)
func.latex = r'$y_i = 0.1x_i^3 + \varepsilon$'

data_points = [
    { 'n_points': 40, 'xlim': [-4, -1] },
    { 'n_points': 40, 'xlim': [1, 4] },
]
df = generate_data(func, points=data_points, seed=4)

# Plot the data
plot_true_function(func, df, title=f'True Function: {func.latex}')

# +
# Fit a neural network with a speficied number of nodes in a single hidden layer
width = 50
nn = SimpleNN(width=width, num_iters=5_000, step_size=0.01, checkpoint=1_000, seed=0)
nn.fit(df.x, df.y)

# Plot MLE predictions. Overfitting is usually expected.
y_pred = nn.predict(df.x)
plt.plot(df.x, y_pred, label='MLE Prediction')
plot_true_function(func, df, title=f'Network Fit, {width} Nodes in 1 Hidden Layer')
# -

# # Sampling from the posterior

# +
# Standard deviation of the prior. Should be appropriate for a specific network configuration.
sigma = 1.0
# Standard deviation of the likelihood, 0.5 is the true noise.
noise = 0.25

x_input = theano.shared(df[['x']].values)
y_output = theano.shared(df['y'].values)

# Build a hierarchical Bayesian neural network. Initialize with MLE.
model = build_model(x_input, y_output, sigma, noise, width, n_weights=nn.n_weights, init=nn.weights)

# Visualize the model
pm.model_to_graphviz(model)
# -

# Sample from the posterior using the No-U-Turn sampler
trace = pm.sample(draws=1000, tune=1000, init="adapt_diag", target_accept=0.9,
                  cores=2, random_seed=[1, 2], model=model)

# # Diagnostics of sampling

# Convert the trace to arviz format with the possibility to select specific weights for plotting
data = az.from_pymc3(trace=trace,
                     coords={'weight': range(nn.n_weights)}, 
                     dims={'weights': ['weight']})

# Print the Effective Sample Size and the Gelman-Rubin Test
tests = az.summary(data, round_to=2)[['ess_mean', 'r_hat']]
print(f'Minimum ESS: {tests.ess_mean.min():,.2f}')
print(f'Max Gelman-Rubin: {tests.r_hat.max():.2f}')
tests.T

# Plot selected weights for illustration
az.plot_trace(data, coords={'weight': range(2, 5)});

# +
# Optionally plot autocorrelation (the list may be long). 
# The effective sample size above is a good alternative.
# az.plot_autocorr(data, combined=True);
# -

# # Automatic differentiation variational inference

with model:
    mean_field = pm.fit(20_000, method='advi', obj_n_mc=10, obj_optimizer=pm.adagrad())
    trace_advi = mean_field.sample(10_000)

# # Posterior predictive

# +
# Simulate data from the posterior predictive using the NUTS posterior
x_test = np.linspace(df.x.min(), df.x.max(), num=1000)
posterior_predictive = simulate_posterior_predictive(trace, nn, x_test, n_samples=10_000)

# Plot the results: truth vs prediction
plot_posterior_predictive(x_test, posterior_predictive, func=func, df=df,
                          title=f'NUTS, Weight Uncertainty {sigma}, Noise {noise},\n'
                                f'{width} Nodes in 1 Hidden Layer')
diagnostics = (f'Minimum ESS: {tests.ess_mean.min():,.2f}\n'
               f'Max Gelman-Rubin: {tests.r_hat.max():.2f}')
plt.gcf().text(0.95, 0.15, diagnostics)
plt.savefig(f"NUTS_hidden_1_width_{width}_sigma_{sigma}_noise_{noise}.png", dpi=plt.gcf().dpi, bbox_inches="tight")

# +
# Simulate data from the posterior predictive using ADVI approximation of the posterior
posterior_predictive = simulate_posterior_predictive(trace_advi, nn, x_test, n_samples=10_000)

# Plot the results: truth vs prediction
plot_posterior_predictive(x_test, posterior_predictive, func=func, df=df,
                          title=f'ADVI, Weight Uncertainty {sigma}, Noise {noise},\n'
                                f'{width} Nodes in 1 Hidden Layer')
# -

# ---

# # True function: $y_i = 0.5 x_i + 3 \sin(x_i) + \varepsilon_i$ (heteroscedastic)

# +
# Define the true function and generate observations
func = lambda x: scipy.stats.norm(loc=0.5*x + 3*np.sin(x), scale=np.exp(x/10))
func.latex = r'$y_i = 0.5x_i + 3 \sin(x_i) + \varepsilon_i$'

data_points = [
    { 'n_points': 40, 'xlim': [-10, -2] },
    { 'n_points': 40, 'xlim': [2, 10] },
]
df = generate_data(func, points=data_points, seed=4)

# Plot the data
plot_true_function(func, df, title=f'True Function: {func.latex}')
# -



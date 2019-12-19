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
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import pymc3 as pm

import matplotlib.pyplot as plt


# +
def generate_data(func, points, seed=0):
    """Generate a dataframe containing the covariate X, and observations Y
    """
    np.random.seed(seed)

    data = []
    for segment in points:
        x = np.linspace(*segment["xlim"], num=segment["n_points"])
        distribution = func(x)
        # Generate observations
        y = distribution.rvs()
        df = pd.DataFrame({"x": x, "y": y})
        data.append(df)

    return pd.concat(data, ignore_index=True)


def build_model(x_input, y_output, sigma, noise, width, n_weights, init):
    """Return a PyMC3 model for a Bayesian Neural Network with a single hidden layer
    and an RBF activation function.
    """
    with pm.Model() as model:
        # Prior
        weights = pm.Normal("weights", mu=0, sigma=sigma, shape=n_weights, testval=init)

        # Input to the first hidden layer
        w = weights[:width].reshape((1, -1))
        b = weights[width : width * 2].reshape((1, -1))
        act_in = pm.math.dot(x_input, w) + b
        act_out = pm.math.exp(-1 * act_in ** 2)  # RBF activation function

        # Output layer
        w = weights[width * 2 : width * 3].reshape((-1, 1))
        b = weights[width * 3]
        y_pred = (pm.math.dot(act_out, w) + b).ravel()

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=y_pred, sigma=noise, observed=y_output)

    return model


def simulate_posterior_predictive(trace, nnet, x_test, n_samples=1000, noiseless=False):
    """A much faster alternative to pm.sample_posterior_predictive()
    for constant variance of the noise.
    """
    weights = trace["weights"]
    n_weights = weights.shape[0]

    # Retrieve the standard deviation of the noise from the PyMC3 model
    noise = trace._straces[0].model.observed_RVs[0].distribution.sigma.value.item()

    # Pick weights with replacement
    indices = np.random.choice(n_weights, size=n_samples, replace=True)

    # Iterate over indices to avoid huge matrix multiplication
    y_pred = np.zeros((n_samples, x_test.shape[0]))
    for i in range(indices.shape[0]):
        y_pred[i] = nnet.predict(x_test, weights[indices[i]])

    # Optionally, return mean predictions (variance of y_pred is epistemic uncertainty)
    if noiseless:
        return y_pred
    return y_pred + np.random.normal(loc=0, scale=noise, size=y_pred.shape)


def storage_path(**spec):
    path = "./models/{algorithm}_hidden_{hidden}_width_{width}_sigma_{sigma}_noise_{noise}".format(
        **spec
    )
    return Path(path)


def save_model(func, df, nnet, trace, path=None, **spec):
    if path is None:
        p = storage_path(**spec)
        path = p.with_suffix(p.suffix + ".pickle")
    else:
        path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    model = trace._straces[0].model
    doc = {
        "func": func,
        "df": df,
        "nnet": nnet,
        "model": model,
        "trace": trace,
        "specification": spec,
    }
    with path.open("wb") as f:
        cloudpickle.dump(doc, f)


def load_model(path=None, **spec):
    if path is None:
        p = storage_path(**spec)
        path = p.with_suffix(p.suffix + ".pickle")
    else:
        path = Path(path)

    with path.open("rb") as f:
        data = cloudpickle.load(f)

    return data


# +
def plot_true_function(func, df, title=None):
    x = np.linspace(df.x.min(), df.x.max(), num=1000)
    distribution = func(x)
    lower, upper = distribution.interval(0.95)

    plt.fill_between(
        x, lower, upper, color="tab:orange", alpha=0.1, label="True 95% Interval",
    )
    plt.scatter(df.x, df.y, color="lightgrey", label="Observations")
    plt.plot(x, distribution.mean(), color="tab:red", label="True Mean")
    if title is not None:
        plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)


def plot_posterior_predictive(x, y, title=None, func=None, df=None):
    if func is not None and df is not None:
        plot_true_function(func, df)

    lower, upper = np.percentile(y, [2.5, 97.5], axis=0)
    plt.fill_between(
        x, lower, upper, color="tab:blue", alpha=0.1, label=f"Predicted 95% Interval"
    )
    plt.plot(x, y.mean(axis=0), color="tab:blue", label=f"Predicted Mean")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

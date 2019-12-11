from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from bnn import feedforward
from calibration import QuantileCalibration
from inference import sample, simulate_pp, fit_advi, run_diagnostics
from plotting import plot_posterior_predictive


def build_model(df, *, hidden, width, sigma, noise):
    """Instantiate the a feedforward BNN model with a network architecture,
    prior standard deviation and likelihood noise.

    Args:
        df: a pandas DataFrame of observations (x, y)
        hidden: the number of hidden layers in a BNN
        width: the number of nodes in each hidden player
        sigma: the standard deviation of the prior on the network weights
        noise: the standard deviation of the likelihood noise

    Returns:
        model: an instantiated NumPyro model function
    """
    X = df[["x"]].values
    Y = df[["y"]].values

    model = partial(feedforward, X=X, Y=Y, width=width, hidden=hidden, sigma=sigma, noise=noise)

    return model


def sample_and_plot(
    df, func, *, hidden, width, sigma, noise, num_samples, num_warmup, num_chains=2
):
    """A helper function to instantiate the model, sample from the posterior, simulate
    the posterior predictive and plot it against the observations and the true function.

    Args:
        df: a pandas DataFrame of observations (x, y)
        func: the true function, a scipy.stats distribution for plotting
        hidden: the number of hidden layers in a BNN
        width: the number of nodes in each hidden player
        sigma: the standard deviation of the prior on the network weights
        noise: the standard deviation of the likelihood noise
        num_samples: the number of samples to draw in each chain
        num_warmup: the number of samples to use for tuning in each chain
        num_chains: the number of chains to draw (default: {2})

    Returns:
        mcmc: a fitted MCMC inference object.
        mcmc.print_summary() can be used for detailed diagnostics
    """
    # Instantiate the model
    model = build_model(df, width=width, hidden=hidden, sigma=sigma, noise=noise)

    # Run the No-U-Turn sampler
    mcmc = sample(model, num_samples, num_warmup, num_chains, seed=0, summary=False)

    # Generate the posterior predictive
    X_test = np.linspace(df.x.min(), df.x.max(), num=1000)[:, np.newaxis]
    posterior_predictive = simulate_pp(model, mcmc, X_test, seed=1)

    # Plot the posterior predictive
    plot_posterior_predictive(
        X_test,
        posterior_predictive,
        func=func,
        df=df,
        title=f"BNN with {width} Nodes in {hidden} Hidden Layer{'' if hidden == 1 else 's'},\n"
        f"Weight Uncertainty {sigma}, Noise {noise}, NUTS Sampling",
    )

    # Print the diagnostic tests
    diagnostics = run_diagnostics(mcmc)
    message = ("Minimum ESS: {min_ess:,.2f}\n" "Max Gelman-Rubin: {max_rhat:.2f}").format(
        **diagnostics
    )
    plt.gcf().text(0.95, 0.15, message)

    # Return the fitted MCMC object to enable detailed diagnostics, e.g. mcmc.print_summary()
    return mcmc


def fit_and_plot(df, func, *, hidden, width, sigma, noise, num_iter, learning_rate):
    """A helper function to instantiate the model, approximate the posterior using Variational Inference
    with reparametrization and isotropic Gaussians, simulate the posterior predictive and plot it
    against the observations and the true function.

    Args:
        df: a pandas DataFrame of observations (x, y)
        func: the true function, a scipy.stats distribution for plotting
        hidden: the number of hidden layers in a BNN
        width: the number of nodes in each hidden player
        sigma: the standard deviation of the prior on the network weights
        noise: the standard deviation of the likelihood noise
        num_iter: the number of iterations of gradient descent (Adam)
        learning_rate: the step size for the Adam algorithm (default: {0.01})

    Returns:
        vi: an collection of fitted VI objects, an instance of ADVIResults.
        vi.losses or vi.plot_loss() can be used for diagnostics of the ELBO.
    """
    # Instantiate the model
    model = build_model(df, width=width, hidden=hidden, sigma=sigma, noise=noise)

    # Approximate the posterior using Automatic Differentiation Variational Inference
    vi = fit_advi(model, num_iter=num_iter, learning_rate=learning_rate, seed=0)

    # Generate the posterior predictive and plot the results
    X_test = np.linspace(df.x.min(), df.x.max(), num=1000)[:, np.newaxis]
    posterior_predictive = simulate_pp(model, vi, X_test, n_samples=5000, seed=1)

    # Plot the posterior predictive
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


def calibrate(df_main, df_hold, *, hidden, width, sigma, noise, inference="NUTS", **kwargs):
    """A helper function to instantiate BNNs for both datasets, sample from the posterior,
    simulate the posterior predictives and train isotonic regression.

    Args:
        df_main: the main dataset (x, y) that need to be recalibrated, a pandas DataFrame
        df_hold: the hold-out dataset (x, y) to train isotonic regression on, a pandas DataFrame
        hidden: the number of hidden layers in a BNN
        width: the number of nodes in each hidden player
        sigma: the standard deviation of the prior on the network weights
        noise: the standard deviation of the likelihood noise
        inference: a method of interence, either "NUTS" or "VI" (default: {"NUTS"})
        **kwargs: additional arguments passed along to the NUTS sampler or to the VI optimizer

    Returns:
        res_main, res_holdout: dictionaries holding fitted inference objects, including
            the posterior predictive
        qc: an instance of QuantileCalibration, containing isotonic regression trained
            in forward an in reverse modes.
    """
    assert inference in {"NUTS", "VI"}, "Inference method must be one of 'NUTS' or 'VI'"
    results = []

    # Obtain the posterior predictives for both datasets
    for df in [df_main, df_hold]:
        # Instantiate the model
        model = build_model(df, width=width, hidden=hidden, sigma=sigma, noise=noise)

        if inference == "NUTS":
            # Sample from the posterior using the No-U-Turn sampler
            infer = sample(model, seed=0, summary=False, **kwargs)
            # Use all posterior samples to generate the posterior predictive
            n_samples = None
        elif inference == "VI":
            infer = fit_advi(model, seed=0, **kwargs)
            n_samples = 5000

        X_test = np.linspace(df.x.min(), df.x.max(), num=1000)[:, np.newaxis]

        # Simulate the posterior predictive for equally spaced values of X for plotting
        post_pred = simulate_pp(model, infer, X_test, n_samples=n_samples, seed=1)
        # Simulate the posterior predictive for all X's in the dataset
        post_pred_train = simulate_pp(model, infer, df[["x"]].values, n_samples=n_samples, seed=1)

        # Collect the results
        results.append(
            {
                "df": df,
                "model": model,
                "infer": infer,
                "X_test": X_test,
                "post_pred": post_pred,
                "post_pred_train": post_pred_train,
            }
        )

    res_main, res_holdout = results

    # Train isotonic regression on the hold-out dataset
    qc = QuantileCalibration()
    qc.fit(res_holdout["df"].y, res_holdout["post_pred_train"])

    return res_main, res_holdout, qc

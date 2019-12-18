import numpy as np
import scipy.stats
from numpyro.infer import MCMC

import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle

from code.calibration import calculate_quantiles, calibrate_posterior_predictive
from code.inference import run_diagnostics
from code.metrics import calibration_error, picp, log_likelihood

from code.calibration import calibrate_posterior_predictive

# Colors used for plotting the posterior predictives
COLORS = {
    "true": "tab:orange",
    "predicted": "tab:blue",
    "calibrated": "tab:pink",
    "observations": "lightgrey",
}
# Transparency for the posterior predictives
FILL_ALPHA = 0.15


def plot_true_function(
    func, df, point_estimate="mean", interval=0.95, title=None, legend=True, ax=None
):
    """Plot the true function and the observations

    Args:
        func: a scipy.stats distribution
        df: a pandas DataFrame containing observations (x, y)
        point_estimate: either a mean or a median (default: {"mean"})
        interval: the width of the predictive interval (default: {0.95})
        title: an optional plot title (default: {None})
        legend: whether to show a legend (default: {True})
        ax: matplotlib axis to draw on, if any (default: {None})
    """
    assert point_estimate in {"mean", "median"}, "Point estimate must be either 'mean' or 'median'"
    assert 0 <= interval <= 1

    x = np.linspace(df.x.min(), df.x.max(), num=1000)
    distribution = func(x)
    lower, upper = distribution.interval(interval)
    point_est = distribution.mean() if point_estimate == "mean" else distribution.median()

    ax = ax or plt.gca()
    ax.fill_between(
        x,
        lower,
        upper,
        color=COLORS["true"],
        alpha=FILL_ALPHA,
        label=f"True {interval*100:.0f}% Interval",
    )
    ax.scatter(df.x, df.y, s=10, color=COLORS["observations"], label="Observations")
    ax.plot(x, point_est, color=COLORS["true"], label="True Mean")
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)


def plot_posterior_predictive(
    x,
    post_pred,
    func=None,
    df=None,
    point_estimate="mean",
    interval=0.95,
    title=None,
    legend=True,
    ax=None,
):
    """Plot the posterior predictive along with the observations and the true function

    Args:
        x: an array of X's of shape (N,), (N, 1) or (1, N)
        post_pred: the posterior predictive, array of shape (M, N),
            where M is the number of samples for each X (e.g. 1000)
        func: the true function, a scipy.stats distribution (default: {None})
        df: a pandas DataFrame of observations (x,y) (default: {None})
        point_estimate: either a mean of a median (default: {"mean"})
        interval: the width of the predictive interval (default: {0.95})
        title: an optional plot title (default: {None})
        legend: whether to show a legend (default: {True})
        ax: matplotlib axis to draw on, if any (default: {None})
    """
    assert point_estimate in {"mean", "median"}, "Point estimate must be either 'mean' or 'median'"
    assert 0 <= interval <= 1

    ax = ax or plt.gca()

    if func is not None and df is not None:
        plot_true_function(
            func, df, point_estimate=point_estimate, interval=interval, legend=legend, ax=ax
        )

    x = x.ravel()
    lower, upper = np.percentile(post_pred, [2.5, 97.5], axis=0)
    point_est = post_pred.mean(axis=0) if point_estimate == "mean" else np.median(post_pred, axis=0)

    ax.fill_between(
        x,
        lower,
        upper,
        color=COLORS["predicted"],
        alpha=FILL_ALPHA,
        label=f"{interval*100:.0f}% Predictive Interval",
    )
    ax.plot(x, point_est, color=COLORS["predicted"], label=f"Predicted Mean")
    ax.set_title(title)
    if legend:
        ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)


def plot_illustration(ppc_func, df, conditionals=True, interval=0.95, title=None):
    """Visualize the miscalibrated posterior predictive to illustrate
    the calibration algorithm.

    Used on the slide "The Algorithm Step-by-Step"

    Args:
        ppc_func: a scipy.stats distribution for the posterior predictive
        df: a pandas DataFrame of observations (x, y)
        conditionals: whether to plot the conditional densities (default: {True})
        interval: the width of the predictive interval (default: {0.95})
        title: an optional plot title (default: {None})
    """
    # Plot the observations and the predictive interval
    assert 0 <= interval <= 1

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(df.x.min(), df.x.max(), num=1000)
    distribution = ppc_func(x)
    lower, upper = distribution.interval(interval)

    ax.fill_between(
        x,
        lower,
        upper,
        color=COLORS["predicted"],
        alpha=FILL_ALPHA,
        label=f"{interval*100:.0f}% Predictive Interval",
    )
    ax.scatter(df.x, df.y, s=10, color=COLORS["observations"], label="Observations")
    ax.plot(x, distribution.mean(), color=COLORS["predicted"], label="Predicted Mean")
    ax.set(xlabel="X", ylabel="Y")
    if title is not None:
        ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    ax.set_ylim([-12, 12])

    if not conditionals:
        return

    # Plot the conditional distribution of Y given X=x_0
    ax2 = fig.add_axes([0.2, 0.235, 0.075, 0.4])
    ax2.axison = False

    base = ax2.transData
    rot = transforms.Affine2D().rotate_deg(90)

    x_ = np.linspace(-3, 3, num=100)
    density = scipy.stats.norm(loc=0, scale=0.75)
    ax2.plot(x_, density.pdf(x_), transform=rot + base, color="tab:gray")

    # Plot the conditional distribution of Y given X=x_1
    ax3 = fig.add_axes([0.5, 0.405, 0.075, 0.2])
    ax3.axison = False

    base = ax3.transData
    rot = transforms.Affine2D().rotate_deg(90)

    x_ = np.linspace(-3, 3, num=100)
    density = scipy.stats.norm(loc=0, scale=0.55)
    ax3.plot(x_, density.pdf(x_), transform=rot + base, color="tab:gray")

    ax.annotate("$f(Y|x_0)$", [-3.23, 4])
    ax.annotate("$f(Y|x_1)$", [0.2, 3.5])


def plot_table(mark_y=False, show_quantiles=None):
    """Display a table, accompaining the step-by-step illustration of the algorithm

    Args:
        mark_y: whether to draw dashed vertical lines for the location of y (default: {False})
        show_quantiles: the values of quantiles to display, "predicted" / "all" or None
            (default: {None})
    """
    if show_quantiles == "all":
        table_params = {"ncols": 5, "figsize": (10, 4)}
        columns = [
            "Observation",
            "PDF $f(Y|x_t)$",
            "CDF $H(x_t)$",
            "$H(x_t)(y_t)$",
            r"$\hat{P}(p)$",
        ]
    elif show_quantiles == "predicted":
        table_params = {"ncols": 4, "figsize": (8, 4)}
        columns = ["Observation", "PDF $f(Y|x_t)$", "CDF $H(x_t)$", "$H(x_t)(y_t)$"]
    else:
        table_params = {"ncols": 3, "figsize": (6, 4)}
        columns = ["Observation", "PDF $f(Y|x_t)$", "CDF $H(x_t)$"]

    fig, ax = plt.subplots(nrows=5, **table_params)

    for a in ax.flatten():
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_xticks([])
        a.set_yticks([])
        a.margins(0.2)

    plt.subplots_adjust(wspace=0.0, hspace=0)

    for i, column in enumerate(columns):
        ax[0, i].set_title(column)

    rows = ["$(x_0, y_0)$", "$(x_1, y_1)$", r"$\ldots$", r"$\ldots$", "$(x_t, y_t)$"]
    for i, row in enumerate(rows):
        ax[i, 0].annotate(row, xy=[0.5, 0.5], size=15, ha="center", va="center")

    x_ = np.linspace(-3, 3, num=100)
    scales = [1, 0.5, 0.75, 1.25, 1.5]
    for i, std in enumerate(scales):
        ax[i, 1].plot(x_, scipy.stats.norm.pdf(x_, loc=0, scale=std))
        ax[i, 2].plot(x_, scipy.stats.norm.cdf(x_, loc=0, scale=std))

    # Illustrative predictive and empirical quantiles (obtained via isotonic regression)
    quantiles = [0.8, 0.8, 0.2, 0.4, 0.6]
    empirical = [0.638443, 0.638443, 0.3569105, 0.44220539, 0.56308756]

    if mark_y:
        for i, quantile in enumerate(quantiles):
            distribution = scipy.stats.norm(loc=0, scale=scales[i])
            value = distribution.ppf(quantile)
            ax[i, 1].plot([value] * 2, [0, distribution.pdf(value)], linestyle="--")
            ax[i, 2].plot([value] * 2, [0, distribution.cdf(value)], linestyle="--")

    if show_quantiles in ["predicted", "all"]:
        for i, quantile in enumerate(quantiles):
            ax[i, 3].annotate(quantile, xy=[0.5, 0.5], size=15, ha="center", va="center")

    if show_quantiles in ["all"]:
        for i, quantile in enumerate(empirical):
            ax[i, 4].annotate(f"{quantile:.2f}", xy=[0.5, 0.5], size=15, ha="center", va="center")

        ax4 = plt.gcf().add_axes([0.595, 0.59, 0.303, 0.28])
        ax4.axison = False
        ax4.add_patch(
            Rectangle(
                (0, 0.01),
                0.99,
                0.99,
                linewidth=1,
                linestyle="--",
                edgecolor="tab:red",
                facecolor="none",
            )
        )


def plot_ecdf(predicted_quantiles):
    """Visualize the empirical CDF

    Used in the step-by-step illustration of the calibration algorithm.

    Args:
        predicted_quantiles: the values of the quantiles for each observed Y
    """
    plt.hist(predicted_quantiles, bins=50, cumulative=True, density=True, alpha=0.3)
    plt.title("CDF of Predicted Quantiles")
    plt.xlabel("Predicted Quantiles, $H(x_t)(y_t)$")
    plt.ylabel(r"Empirical Quantiles, $\hat{P}(p_t)$")


def calibration_plot(predicted_quantiles, model):
    """Visualize a calibration plot suggested by the authors

    Args:
        predicted_quantiles: the values of the quantiles for each Y in the dataset
        model: an isotonic regression object (trained in forward mode from predicted to empirical
            quantiles)
    """
    # Choose equally spaced quantiles
    expected_quantiles = np.linspace(0, 1, num=11).reshape(-1, 1)

    # Compute the probabilities of predicted quantiles at the discrete quantile levels
    T = predicted_quantiles.shape[0]
    observed_uncalibrated = (predicted_quantiles.reshape(1, -1) <= expected_quantiles).sum(
        axis=1
    ) / T

    # Use the model to output the actual probabilities of any quantile
    calibrated_quantiles = model.predict(predicted_quantiles)
    # Estimate the observed calibrated quantiles
    observed_calibrated = (calibrated_quantiles.reshape(1, -1) <= expected_quantiles).sum(
        axis=1
    ) / T

    # Plot the results
    plt.plot(
        expected_quantiles,
        observed_uncalibrated,
        marker="o",
        color=COLORS["predicted"],
        label="Uncalibrated",
    )
    plt.plot(
        expected_quantiles,
        observed_calibrated,
        marker="o",
        color=COLORS["calibrated"],
        label="Calibrated",
    )
    plt.plot([0, 1], [0, 1], color="tab:grey", linestyle="--", zorder=0)
    plt.title("Calibration Plot")
    plt.xlabel("Expected Quantiles")
    plt.ylabel("Observed Quantiles")
    plt.legend()

def plot_calibration_results(results, qc, func, interval=0.95, figsize=(8.5, 3.5),
                                point_est="median"):
    """Plot the posterior predictive before and after calibration

    Args:
        x: an array of X's of the shape (N,), (N, 1) or (1, N)
        post_pred: the posterior predictive, array of shape (M, N),
            where M is the number of samples for each X (e.g. 1000)
        qc: a fitted QuantileCalibration object
        df: a pandas DataFrame of observations (x, y)
        func: the true function, a scipy.stats distribution
        interval: the width of the predictive interval (default: {0.95})
        figsize: the overall size of the matplotlib figure, which will be split in
            two subplots (default: {(8.5, 3.5)})
        point_est: indicate whether to use mean or median as the point estimate
    """
    assert point_est in {"mean", "median"}, "Point estimate must be either 'mean' or 'median'"


    x = results["X_test"].ravel()
    post_pred = results["post_pred"]
    if point_est == "mean":
        calibrated_post_pred = calibrate_posterior_predictive(results["post_pred"], qc)
    post_pred_x = results["post_pred_x"]

    # we need this anyway for expected log likelihood metric
    calibrated_post_pred_x = calibrate_posterior_predictive(results["post_pred_x"], qc)

    df = results["df"]

    assert 0 <= interval <= 1
    q_alpha = (1 - interval) / 2
    low, high = 1 - interval - q_alpha, interval + q_alpha
    q = [low, 0.5, high]
    quantiles = [q, qc.inverse_transform(q)]
    titles = ["Before Calibration", "After Calibration"]

    fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    for i, axis in enumerate(ax):
        # Plot the true function
        distribution = func(x)
        lower, upper = distribution.interval(interval)
        true_interval = axis.fill_between(
            x,
            lower,
            upper,
            color=COLORS["true"],
            alpha=FILL_ALPHA,
            label=f"True {interval*100:.0f}% Interval",
        )
        axis.scatter(df.x, df.y, s=3, color=COLORS["observations"], label="Observations")
        if point_est == "mean":
            point_est_value = distribution.mean()
            true_label = "True Mean"
        else:
            point_est_value = distribution.median()
            true_label = "True Median"
        true_point_est = axis.plot(x, point_est_value, color=COLORS["true"], label=true_label)
        axis.set_title(titles[i])

        lower, median, upper = np.quantile(post_pred, quantiles[i], axis=0)
        predicted_interval = axis.fill_between(
            x,
            lower,
            upper,
            color=COLORS["predicted"],
            alpha=FILL_ALPHA,
            label=f"{interval*100:.0f}% Predictive Interval",
        )
        if point_est == "mean":
            mean = np.mean(calibrated_post_pred, axis=0)
            predicted_point_est = axis.plot(
                x, mean, color=COLORS["predicted"], label=f"Predicted Mean"
            )
        else:
            predicted_point_est = axis.plot(
                x, median, color=COLORS["predicted"], label=f"Predicted Median"
            )

    # Compute the calibration error and PICP, before calibration
    uncalibrated_quantiles = calculate_quantiles(post_pred_x.T, df[["y"]].values)
    cal_error = calibration_error(uncalibrated_quantiles)
    picp_value = picp(uncalibrated_quantiles, interval=interval)

    likelihood_func = results["noise_model"]

    loglikelihood = log_likelihood(likelihood_func, post_pred_x, df[["y"]].values)
    ll_message = f"\nE[Log Likelihood] {loglikelihood:.3f}"

    ax[0].text(
        0.96,
        0.06,
        f"Calibr. {cal_error:.3f}\nPICP  {picp_value:.3f}" + ll_message,
        horizontalalignment="right",
        transform=ax[0].transAxes,
    )
    # After calibration:
    calibrated_quantiles = qc.transform(uncalibrated_quantiles)
    cal_error = calibration_error(calibrated_quantiles)
    picp_value = picp(calibrated_quantiles, interval=interval)

    loglikelihood = log_likelihood(likelihood_func, calibrated_post_pred_x,
                                    df[["y"]].values)
    ll_message = f"\nE[LogLikelihoo]d {loglikelihood:.3f}"

    ax[1].text(
        0.96,
        0.06,
        f"Calibr. {cal_error:.3f}\nPICP  {picp_value:.3f}" + ll_message,
        horizontalalignment="right",
        transform=ax[1].transAxes,
    )

    # Add a legend under the plots
    handles = [true_interval, true_point_est[0], predicted_interval, predicted_point_est[0]]
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.1, 1, 1))


def check_convergence(res_main, res_holdout, func, plot=True, point_estimate="median"):
    """Print basic diagnostic metrics for each trained dataset and optinally plot the
    posterior predictives for visual checks.

    The diagnostic metrics are the Effective Sample Size and the Gelman-Rubin test. These
    are only available for posteriors obtained via sampling. For VI posteriors one needs
    to perform a visual check.

    Args:
        res_main: a dictionary of fitted objects for the main dataset (model, inference
            object, the posterior predictive, etc.)
        res_holdout: a similar dictionary of fitted objects for the hold-out dataset
        func: the true function, a scipy.stats distribution
        plot: whether to plot the posterior predictives. If set to False, only textual
            information will be printed, if available (default: {True})
        point_estimate: either a median or a mean (default: {"median"})
    """
    assert point_estimate in {"mean", "median"}, "Point estimate must be either 'mean' or 'median'"

    data = {"Main dataset": res_main, "Hold-out dataset": res_holdout}

    for name, res in data.items():
        if isinstance(res["infer"], MCMC):
            # Compute basic diagnostic tests for an MCMC model
            diagnostics = run_diagnostics(res["infer"])
        else:
            diagnostics = None

        if plot:
            plt.figure()
            plot_posterior_predictive(
                res["X_test"],
                res["post_pred"],
                func=func,
                df=res["df"],
                title=name,
                point_estimate=point_estimate,
            )

            # Print the results of diagnostic tests
            if diagnostics:
                message = ("Minimum ESS: {min_ess:,.2f}\nMax Gelman-Rubin: {max_rhat:.2f}").format(
                    **diagnostics
                )
                plt.gcf().text(0.95, 0.15, message)
        else:
            if diagnostics:
                print(
                    "{name}: minimum ESS {min_ess:,.2f}, "
                    "maximum Gelman-Rubin {max_rhat:.2f}".format(name=name, **diagnostics)
                )

def plot_calibration_slice(result, slice_locations):
    """Plots calibrated vs uncalibrated posterior predictive cross-sections.
    """

    cal_post_pred = calibrate_posterior_predictive(result['post_pred'], qc)
    slices = np.floor(cal_post_pred.shape[1] * slice_locations).astype(int)

    uncal_lower_limit = np.min(np.apply_along_axis(lambda x: np.quantile(x, 0.02),
                                                   0, result['post_pred'][:,slices]))
    cal_lower_limit = np.min(np.apply_along_axis(lambda x: np.quantile(x, 0.02),
                                                 0, cal_post_pred[:,slices]))
    lower_limit = min(uncal_lower_limit, cal_lower_limit)

    uncal_upper_limit = np.max(np.apply_along_axis(lambda x: np.quantile(x, 0.98),
                                                   0, result['post_pred'][:,slices]))
    cal_upper_limit = np.max(np.apply_along_axis(lambda x: np.quantile(x, 0.98),
                                                 0, cal_post_pred[:,slices]))
    upper_limit = max(uncal_upper_limit, cal_upper_limit)

    x_values = result['X_test'][slices]

    fig, ax = plt.subplots(1,2)
    for idx, s in enumerate(slices):
        pp_df = pd.DataFrame({'calibrated':cal_post_pred[:,slices[idx]],
                              'uncalibrated':result['post_pred'][:,slices[idx]]})
        pp_df.plot.kde(ax=ax[idx], xlim=(lower_limit, upper_limit))
        ax[idx].set_title(f'Posterior Predictive at x={x_values[idx][0]:.2f}')
        ax[idx].set_xlabel('y')
    fig.tight_layout()
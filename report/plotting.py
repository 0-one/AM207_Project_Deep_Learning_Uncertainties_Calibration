import numpy
import matplotlib.pyplot as plt


def plot_true_function(func, df, title=None):
    x = numpy.linspace(df.x.min(), df.x.max(), num=1000)
    distribution = func(x)
    lower, upper = distribution.interval(0.95)

    plt.fill_between(
        x, lower, upper, color="tab:orange", alpha=0.1, label="True 95% Interval",
    )
    plt.scatter(df.x, df.y, s=10, color="lightgrey", label="Observations")
    plt.plot(x, distribution.mean(), color="tab:orange", label="True Mean")
    if title is not None:
        plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)


def plot_posterior_predictive(x, y, title=None, func=None, df=None):
    """Plot the posterior predictive along with the observations and the true function
    """
    if func is not None and df is not None:
        plot_true_function(func, df)

    x = x.ravel()
    lower, upper = numpy.percentile(y, [2.5, 97.5], axis=0)
    plt.fill_between(x, lower, upper, color="tab:blue", alpha=0.1, label=f"Predicted 95% Interval")
    plt.plot(x, y.mean(axis=0), color="tab:blue", label=f"Predicted Mean")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

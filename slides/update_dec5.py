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

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Project update
#
#
# In the past few weeks we:
# - Implemented the calibration algorithm
# - Started applying it to some of the miscalibrated posterior predictives
# - Researched the calibration metrics
# - Formulated some of the properties of the algorithm

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # The calibration algorithm
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
#  - $\hat P(p)=\left|\left\{y_t\mid \left[H\left(x_t\right)\right]\left(y_t\right)\lt p, t=1\ldots T\right\}\right|/T$, i.e. the empirical quantile of $y_t$
#
#
# 2. Train a model $R$ (e.g. isotonic regression) on $\mathcal{D}$.

# + {"slideshow": {"slide_type": "skip"}}
import warnings

import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats
import theano
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle

# %matplotlib inline

# + {"slideshow": {"slide_type": "skip"}}
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rc("figure", dpi=100)


# + {"slideshow": {"slide_type": "skip"}}
def calculate_quantiles(samples, y):
    """ Function to compute quantiles of y within the given samples.

    For efficiency, this function does not distinguish even and odd
    number of samples, N. This should not really be an issue for large
    N.

    Paramters:
        y: array of shape (T, 1)
        samples: array of shape (T, N)
                Note: if sample has shape (1, N), then it is broadcasted
                to (T, N) by numpy. This happens in step 3 of the
                calibration algorithm.

    Returns:
        quantile of each of the y values, shape (T,)
    """
    N = samples.shape[1]

    return np.sum(samples <= y, axis=1) / N


def make_cal_dataset(y, post_pred):
    """ Function to construct the calibration dataset.

    The function returns two arrays, cal_y and cal_X. They are to be
    used to train the calibration transformation.

    Notation: documentation assumes we have T observations.

    Parateters:
        y: array of shape (T, 1)
        ppc: posterior predictive, array of shape (T, N)

    Returns:
        cal_y: shape (T,)
        cal_X: shape (T,)
    """
    y = np.asarray(y).reshape(-1, 1)
    T = y.shape[0]
    N = post_pred.shape[1]

    # compute quantiles of y observation
    # quant_y.shape = (T,)
    quant_y = calculate_quantiles(post_pred, y)

    # p_hat.shape = (T,)
    p_hat = calculate_quantiles(quant_y.reshape(-1, T), quant_y.reshape(T, -1))

    return (quant_y, p_hat)


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


def plot_posterior_predictive(ppc_func, df, conditionals=False, title=None):
    # Plot the observations and 95% predictive interval
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(df.x.min(), df.x.max(), num=1000)
    distribution = ppc_func(x)
    lower, upper = distribution.interval(0.95)

    ax.fill_between(
        x, lower, upper, color="tab:orange", alpha=0.1, label="95% Predictive Interval",
    )
    ax.scatter(df.x, df.y, s=10, color="lightgrey", label="Observations")
    ax.plot(x, distribution.mean(), color="tab:orange", label="Predicted Mean")
    ax.set(xlabel="X", ylabel="Y")
    if title is not None:
        ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.5, 1), borderaxespad=0)
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
    if show_quantiles == "all":
        table_params = {"ncols": 5, "figsize": (10, 4)}
        columns = [
            "Observation",
            "PDF $f(Y|x_t)$",
            "CDF $H(x_t)$",
            "$H(x_t)(y_t)$",
            "$\hat{P}(p)$",
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

    rows = ["$(x_0, y_0)$", "$(x_1, y_1)$", "$\ldots$", "$\ldots$", "$(x_t, y_t)$"]
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
            ax[i, 3].annotate(
                quantile, xy=[0.5, 0.5], size=15, ha="center", va="center"
            )

    if show_quantiles in ["all"]:
        for i, quantile in enumerate(empirical):
            ax[i, 4].annotate(
                f"{quantile:.2f}", xy=[0.5, 0.5], size=15, ha="center", va="center"
            )

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
    plt.hist(predicted_quantiles, bins=50, cumulative=True, density=True, alpha=0.5)
    plt.title("CDF of Predicted Quantiles")
    plt.xlabel("Predicted Quantiles, $H(x_t)(y_t)$")
    plt.ylabel("Empirical Quantiles, $\hat{P}(p_t)$")


def calibration_plot(predicted_quantiles, model):
    """Visualize a calibration plot suggested by the authors
    """
    # Choose equally spaced confidence levels
    expected_quantiles = np.linspace(0, 1, num=11).reshape(-1, 1)

    # Compute the probabilities of predicted quantiles at the discrete confidence levels
    T = predicted_quantiles.shape[0]
    observed_uncalibrated = (
        predicted_quantiles.reshape(1, -1) <= expected_quantiles
    ).sum(axis=1) / T

    # Use the model to output the actual probabilities of any quantile
    calibrated_quantiles = model.predict(predicted_quantiles)
    # Estimate the observed calibrated confidence levels
    observed_calibrated = (
        calibrated_quantiles.reshape(1, -1) <= expected_quantiles
    ).sum(axis=1) / T

    # Plot the results
    plt.plot(
        expected_quantiles,
        observed_uncalibrated,
        marker="o",
        color="tab:orange",
        label="Uncalibrated",
    )
    plt.plot(
        expected_quantiles,
        observed_calibrated,
        marker="o",
        color="tab:pink",
        label="Calibrated",
    )
    plt.plot([0, 1], [0, 1], color="tab:grey", linestyle="--", zorder=0)
    plt.title("Calibration Plot")
    plt.xlabel("Expected Confidence Levels")
    plt.ylabel("Observed Confidence Levels")
    plt.legend()


def plot_distributions(
    df,
    posterior_predictive=None,
    plot_uncalibrated=True,
    true_func=None,
    calibrated_quantiles=None,
    title=None,
):
    """Plot the true function, the calibrated and the uncalibrated 95% intervals
    """
    # Plot the observations
    plt.scatter(df.x, df.y, s=10, color="lightgrey", zorder=0, label="Observations")

    # Assuming the shape of the posterior predictive is compatible with the dataset
    x = df.x

    if plot_uncalibrated and posterior_predictive is not None:
        # Plot the uncalibrated median and the 95% predictive interval
        low, mid, high = np.percentile(posterior_predictive, [2.5, 50, 97.5], axis=1)
        plt.fill_between(
            x, low, high, color="tab:orange", alpha=0.1, label="95% Predictive Interval"
        )
        plt.plot(x, mid, color="tab:orange", label="Predicted Median")

    if true_func is not None:
        # Plot true function
        distribution = true_func(x)
        low, high = distribution.interval(0.95)
        plt.fill_between(
            x,
            low,
            high,
            color="tab:blue",
            alpha=0.2,
            zorder=0,
            label="95% True Interval",
        )
        plt.plot(
            x, distribution.median(), color="tab:blue", zorder=0, label="True Median"
        )

    if calibrated_quantiles is not None:
        # Plot the calibrated median and the 95% predictive interval
        low, mid, high = np.quantile(posterior_predictive, calibrated_quantiles, axis=1)
        plt.fill_between(
            x, low, high, color="tab:pink", alpha=0.2, label="95% Calibrated Interval"
        )
        plt.plot(x, mid, color="tab:pink", label="Calibrated Median")

    plt.gca().set(xlabel="X", ylabel="Y", title=title)
    plt.legend(bbox_to_anchor=(1.5, 1), borderaxespad=0)


def bayesian_poly_model(x_input, y_output):
    """Sample from the posterior of a miscalibrated Bayesian polynomial regression
    and return the simulated posterior predictive.
    """
    with pm.Model() as model:
        coefs = pm.Normal("coefs", mu=0, sigma=1, shape=4)
        y_pred = pm.math.dot(x_input, coefs)
        y_obs = pm.Normal("y_obs", mu=y_pred, sigma=0.5, observed=y_output)

        # Obtain 2000 samples from the posterior (1000 * 2 chains)
        trace = pm.sample(1000, tune=1000)
        # Simulate the posterior predictive
        posterior_predictive = pm.sample_posterior_predictive(trace, progressbar=False)[
            "y_obs"
        ].T

    return posterior_predictive


# + {"slideshow": {"slide_type": "skip"}}
def func(x, base_std):
    """Basis for the true function and a hypothetical posterior predictive
    """
    std = np.abs(x) * base_std
    std = np.where(std < 0.5, 0.5, std)
    return scipy.stats.norm(loc=0.1 * x ** 3, scale=std)


def true_func(x):
    """True function: y_i = 0.1 x_i^3 + epsilon_i (heteroscedastic)
    """
    return func(x, base_std=1.5)


def ppc(x):
    """A miscalibrated posterior predictive that underestimates uncertainty
    """
    return func(x, base_std=0.5)


# + {"slideshow": {"slide_type": "skip"}}
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


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Setup: The posterior predictive
#
# Say we have the following hypothetical posterior predictive, which is underestimating uncertainty:

# + {"slideshow": {"slide_type": "fragment"}}
plot_posterior_predictive(ppc, df, title=f"Miscalibrated Posterior Predictive")

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Setup: The posterior predictive
#
# For each value of the covariate $X$, the posterior predictive provides us with a conditional distribution $f(Y|X)$:

# + {"slideshow": {"slide_type": "fragment"}}
plot_posterior_predictive(ppc, df, title=f"Miscalibrated Posterior Predictive",
                          conditionals=True)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 1: Construct a recalibration dataset
#
# The first step of the calibration algorithm is to obtain predictive conditional distributions for each $X$ in the dataset. If no closed-form is available we simulate the posterior predictive based on the samples of the posterior:

# + {"slideshow": {"slide_type": "fragment"}}
plot_table()

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# Alternative notation: $H(x_t) = F_t$ (a CDF)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 1a: Compute the predicted quantiles
#
# The observed $Y$ for each data point falls somewhere within those conditional distributions:

# + {"slideshow": {"slide_type": "fragment"}}
plot_table(mark_y=True)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 1a: Compute the predicted quantiles
#
# We evaluate the conditional CDFs at each observed value of the response $Y$ to obtain the predicted quantiles. In the absence of analytical form, we simply count the proportion of samples that are less than $y_t$. This gives us the estimated quantile of $y_t$ at $x_t$ in the posterior predictive distribution:

# + {"slideshow": {"slide_type": "fragment"}}
plot_table(mark_y=True, show_quantiles="predicted")

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 1b: Estimate the empirical quantiles
#
# We next find the empirical quantiles, which are defined as the proportion of observations that have lower quantile values than that of the current observation. This is equivalent to finding the empirical CDF of the predicted quantiles:

# + {"slideshow": {"slide_type": "fragment"}}
# Estimate the predicted quantiles and plot their CDF
predicted_quantiles = ppc_quantiles(df.y)
plot_ecdf(predicted_quantiles)

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 1b: Estimate the empirical quantiles
#
# The mapping of predicted quantiles and the empirical quantiles will form a recalibration dataset:

# + {"slideshow": {"slide_type": "fragment"}}
# Compute the empirical quantiles and plot them against predicted quantiles
T = predicted_quantiles.shape[0]
empirical_quantiles = (predicted_quantiles.reshape(1, -1) 
                       <= predicted_quantiles.reshape(-1, 1)).sum(axis=1) / T
plot_ecdf(predicted_quantiles)
plt.scatter(predicted_quantiles, empirical_quantiles, color="tab:blue", zorder=2);

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 1c: Form a recalibration dataset
#
# The mapping is obtained for all observations in the dataset. Note that in this example the first two observations have different conditional distributions, but the same values of the predicted and empirical quantiles. The calibration procedure doesn't distinguish between such cases:

# + {"slideshow": {"slide_type": "fragment"}}
plot_table(mark_y=True, show_quantiles="all")

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 1c: Form a recalibration dataset
#
# The inverse S-curve of the recalibration dataset in our example is characteristic of a posterior predictive that underestimates uncertainty:

# + {"slideshow": {"slide_type": "fragment"}}
plt.scatter(predicted_quantiles, empirical_quantiles)
plt.plot([0, 1], [0, 1], color="tab:grey", linestyle="--")
plt.xlabel("Predicted Cumulative Distribution")
plt.ylabel("Empirical Cumulative Distribution")
plt.title("Recalibration Dataset");

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Step 2: Train a model
#
# We then train a model (e.g. isotonic regression) on the recalibration dataset and use it to output the actual probability of any given quantile or interval:

# + {"slideshow": {"slide_type": "fragment"}}
# Fit isotonic regression
ir = IsotonicRegression(out_of_bounds="clip")
ir.fit(predicted_quantiles, empirical_quantiles)
# Obtain actual calibrated quantiles
calibrated_quantiles = ir.predict([0.025, 0.5, 0.975])
(pd.DataFrame({"Predicted quantiles": [0.025, 0.5, 0.975],
               "Calibrated quantiles": calibrated_quantiles})
 .round(3).style.hide_index())

# + {"slideshow": {"slide_type": "fragment"}, "cell_type": "markdown"}
# Ideally, the model should be fit on a separate calibration set in order to reduce overfitting. Alternatively, multiple models can be trained in a way similar to cross-validation:
#
# - use $K-1$ folds for training
# - use 1 fold for calibration
# - at prediction time, the output is the average of $K$ models

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Detailed steps (part 1)
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
# # Detailed step (part 2)
#
# 4. Construct $\mathcal{D} = \left\{\left(\widehat{\left[H\left(x_t\right)\right]\left(y_t\right)}, \hat P\left(\widehat{\left[H\left(x_t\right)\right]\left(y_t\right)}\right)\right)\right\}_{t=1}^T$
# 5. Train calibration transformation using $\mathcal{D}$ via isotonic regression (or other models). Running prediction on the trained model results in a transformation $R$, $[0,1] \to [0,1]$. We can compose the calibrated model as $R\circ H\left(x_t\right)$.
# 6. To find the calibrated confidence intervals, we need to remap the original upper and lower limits. For example, the upper limit $y_{t\ high}$ is mapped to the calibrated value $y_{t\ high}'$ as:
# $$y_{t\ high}'=\left[H\left(x_t\right)\right]^{-1}\left(R^{-1}\left\{\left[H\left(x_t\right)\right]\left(y_{t\ high}\right)\right\}\right)$$

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Diagnostics
#
# As a visual diagnostic tool, the authors suggest using a calibration plot that shows the true frequency of points in each quantile or confidence interval compared to the predicted fraction of points in that interval. Well-calibrated models should be close to a diagonal line:

# + {"slideshow": {"slide_type": "fragment"}}
# Perform calibration on a hold-out test dataset
predicted_quantiles_test = ppc_quantiles(df_test.y)
calibration_plot(predicted_quantiles_test, model=ir)


# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Quantitative metrics
#
# Several alternatives available, each with specific advantages and disadvantages:
#
# **1. Calibration error**
# $$cal(F_1, y_1, ..., F_N, y_N ) = \sum_{j=1}^m w_j \cdot (p_j − \hat{p}_j)^2$$
# Provides a synthetic measure representing the overall *'distance'* of the points on the calibration curve from the $45^\circ$ straight line. The weights ($w_j$) might be used to reduce the importance of intervals containing few observations. Value of $0$ indicates perfect calibration. Sensitive to binning.
#
# **2. Predictive RMSE**
# $$\sqrt{\frac{1}{N}\sum_{n=1}^{N}||y_n-\mathbb{E}_{q(W)}[f(x_n,W)]||_{2}^{2}}$$<br>
# Measures the model *fit* to the observed data by normalizing the difference between observations and the mean of the posterior predictive. Minimizing RMSE does not guarantee calibration of the model.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Quantitative metrics cont.
#
# **3. Mean prediction interval width**
# $$\frac{1}{N}\sum_{n=1}^{N}\hat{y}_{n}^{high} - \hat{y}_{n}^{low},$$<br>
# where $\hat{y}_{n}^{high}$ and $\hat{y}_{n}^{low}$ are - respectively - the 97.5 and 2.5 percentiles of the predicted outputs for $x_n$.
# Average difference between the the upper and lower bounds of predictive intervals evaluated for all the observations (different significance values might be used to define the predictive intervals). By itself provides information on the precision of the prediction (*confidence* with which a prediction is made) rather than calibration or miscalibration of the model. However may be used in conjunction with prediction interval coverage probability.
#
# **4. Prediction interval coverage probability**
# $$\frac{1}{N}\sum_{n=1}^{N}\mathbb{1}_{y_n\leq\hat{y}_{n}^{high}} \cdot \mathbb{1}_{y_n\geq\hat{y}_{n}^{low}} $$<br>
# Calculates the share of observations covered by 95% (or any other, selected) predictive intervals. Alignment of the PICP with the probability mass assigned to the predictive interval generating it may misleadingly point to proper calibration if true noise distribution belongs to a different family than the posterior predictive. Requires a large sample of observations. 

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Properties of the algorithm (to be tested)
#
# - **Sufficient data:** The procedure requires sufficient i.i.d. data to perform the recalibration. In its absence, point estimates like the median or any other quantile may become worse. 
# - **Marginal probabilities:** Predicted quantiles are mapped to empirical quantiles uniformly over the entire input space, irrespective of the value of $X$. In other words, quantile-based calibration is based on a marginal probability, not a conditional probability.
# - **Calibration on average:** The algorithm does not ensure calibration of a particular prediction, and only aims for the quantile to be calibrated on average over all predictions
# - **Arbitrary mappings:** Can arbitrarily change uncertainties to achieve the calibration objective since isotonic regression is always able to find a perfect fit
# - **Uninformative calibration:** Does not distinguish between informative and uninformative (random) uncertainties. Any posterior predictive, even a completely wrong one, can be perfectly calibrated (up to a sampling error) according to the quantile definition of calibration. This may make the estimates meaningless for further analysis.

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Example: Bayesian polynomial regression
#
#
# True function:
# $$
# y_i = 0.1x_i^3 + \varepsilon, \text{ where } \varepsilon \sim \mathcal{N}(0, 1)
# $$

# + {"slideshow": {"slide_type": "skip"}}
def polynomial(x):
    """True data-generating function"""
    return scipy.stats.norm(loc=0.1 * x ** 3, scale=1)

# Generate observations for an equally sized main dataset and a calibration set
data_points = [
    {"n_points": 200, "xlim": [-4, 4]},
]
df = generate_data(polynomial, points=data_points, seed=4)
df_cal = generate_data(polynomial, points=data_points, seed=0)

# + {"slideshow": {"slide_type": "fragment"}}
plot_distributions(df, true_func = polynomial,
                   title=r"True Function: $y_i = 0.1x_i^3 + \varepsilon$")

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Example: Bayesian polynomial regression
#
# Bayesian polynomial regression model (with unrealistically low noise in the likelihood):
#
# $$
# a, b, c, d \sim \mathcal{N}(0, 1) \\
# f(X) = a X^3 + b X^2 + c X + d \\
# Y_{observed} \sim \mathcal{N}(f(X), 0.5)
# $$

# + {"slideshow": {"slide_type": "skip"}}
# Create polynomial features: 1, X, X^2 and X^3
poly = PolynomialFeatures(degree=3)
X_train = poly.fit_transform(df[["x"]])
X_cal = poly.fit_transform(df_cal[["x"]])

posterior_predictive = bayesian_poly_model(X_train, df.y)
posterior_predictive_cal = bayesian_poly_model(X_cal, df_cal.y)

# + {"slideshow": {"slide_type": "fragment"}}
plot_distributions(df, posterior_predictive,
                   title="Miscalibrated Posterior Predictive")

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Results of calibration

# + {"slideshow": {"slide_type": "skip"}}
# Remove the if statement below when all of the procedures are implemented
INVERTER = False

from statsmodels.distributions.empirical_distribution import monotone_fn_inverter

def calibrate_model_pred(R, get_post_pred, posterior_samples, x, y):
    posterior_predictive_samples = get_post_pred(posterior_samples, x)
    emp_cdf = lambda uncal_y: calculate_quantiles(posterior_predictive_samples,
                                                  uncal_y)
    y_p = emp_cdf(y)
    inv_R = monotone_fn_inverter(cal_transform, y_p)
    inv_H = monotone_fn_inverter(emp_cdf, inv_R)

    return inv_H

if INVERTER:
    # get_post_pred is a function that returns the posterior_predictive given x and posterior
    # trace_from_pymc3 is the set of posterior samples
    calibrate_mode_pred(ir.predict, get_post_pred, trace_from_pymc3, x, y)

# + {"slideshow": {"slide_type": "fragment"}}
# Build a recalibration dataset using out-of-sample data
predicted, empirical = make_cal_dataset(df_cal.y, posterior_predictive_cal)
# Fit the recalibration dataset in reverse: from empirical to predicted
ir = IsotonicRegression(out_of_bounds="clip")
ir.fit(empirical, predicted)
calibrated_quantiles = ir.predict([0.025, 0.5, 0.975])
plot_distributions(df, posterior_predictive, 
                   plot_uncalibrated=False, true_func=polynomial,
                   calibrated_quantiles=calibrated_quantiles,
                   title="Quantile Calibrated Posterior Predictive")

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Open questions
#
# - Would like to confirm validity of the implementation approach, since sampling is not explicitly mentioned in the paper.
# - What about the variance introduced by estimating $\left[H\left(x_t\right)\right]\left(y_t\right)$ – how will this affect the calibration? Should we take this into account? Perhaps we can:
#  - compare calibration for conjugate models without sampling (in "Detailed Step" 2) with calibration using sampling;
#  - repeatedly calibrate the same model using sampling to study calibration variance.
# - For non-Bayesian models (MLE), can this calibration via sampling method be applied to bootstraps?
# - If the idea above is valid, would it be a viable alternative to arbitrarily constructing a CDF (proposed in section 3.4 of the paper)?
# - Since the calibration model (step 5) is trained on a single dataset for all values of $X$, we expect the transformation to be applied uniformly w.r.t. $X$. Is it reasonable to expect intervals in certain regions to get overly compressed/expanded after calibration?
# - What is the effect on the expected log likelihood of the calibrated model? Something worth investigating?
#

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Next steps
#
#
# - Further research on miscalibration arising from:
#
#     - heteroscedastic noise,
#     - non-Gaussian noise,
#     - wrong likelihood function,
#     - and different choices of data-generating functions
#     
# - Application of the calibration algorithm and subsequent analysis of its effects on the posterior predictive, as well as on point estimates (median, mean, MAP)
# - Quantitative assessment of calibration results in terms of selected metrics

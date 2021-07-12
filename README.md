# Calibration of Bayesian Neural Networks

Analysis of the paper by [Kuleshov et al. (2018)](https://arxiv.org/pdf/1807.00263) — Accurate Uncertainties for Deep Learning Using Calibrated Regression.

Harvard University<br>
Class: AM 207 — Stochastic Methods for Data Analysis, Inference and Optimization<br>
Deliverables: [Project Report](https://raw.githubusercontent.com/0-one/AM207_Project_Deep_Learning_Uncertainties_Calibration/master/report/report.slides.pdf) and the source code in this repository

Table of Contents
-----------------

* [Summary of Research](#summary-of-research)
   * [The Issue of Miscalibration](#the-issue-of-miscalibration)
   * [Sources of Miscalibration](#sources-of-miscalibration)
   * [Contribution of the Reviewed Paper](#contribution-of-the-reviewed-paper)
   * [Evaluation of the Claims](#evaluation-of-the-claims)
* [Reproducing the Results](#reproducing-the-results)
* [Repository Structure](#repository-structure)

## Summary of Research

### The Issue of Miscalibration

Proper quantification of uncertainty is crucial for applying statistical models to real-world situations. The Bayesian approach to modeling provides us with a principled way of obtaining such uncertainty estimates. Yet, due to various reasons, such estimates are often inaccurate. For example, a 95% posterior predictive interval does not contain the true outcome with a 95% probability. Such a model is *miscalibrated*.

### Sources of Miscalibration

In our project, we first demonstrate that the problem of miscalibration exists and show why it exists for **Bayesian neural networks** (BNNs) in regression tasks. We focus on the following sources of miscalibration:
- The **prior** is wrong, e.g. too strong and overly certain
- The **likelihood function** is wrong. There is bias, i.e. the neural network is too simple and is unable to model the data.
- The **noise** specification in the likelihood is wrong
- The **inference** is approximate or is performed incorrectly

Our aim is to establish a causal link between each aspect of the model-building process and a bad miscalibrated outcome.

### Contribution of the Reviewed Paper

**Proposition:** [[Kuleshov et al., 2018]](https://arxiv.org/abs/1807.00263) propose a simple **calibration algorithm** for regression. The method is heavily inspired by Platt scaling [[Platt, 1999]](https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods), which consists of training an additional sigmoid function to map potentially non-probabilistic outputs of a classifier to empirical probabilities.

**Unique contribution:** The paper contributes to the subject literature by:

- extending the recalibration methods used so far for classification tasks (Platt scaling) to regression;
- proposing a procedure that is universally applicable to any regression model, be it Bayesian or frequentist, and does not require modification of the model. Instead, the algorithm is applied to the output of any existing model in a postprocessing step.

**Claim:** The authors claim that the method outperforms other techniques by consistently producing well-calibrated forecasts, given enough i.i.d. data. Based on their experiments, the procedure also improves predictive performance in several tasks, such as time-series forecasting and reinforcement learning.

### Evaluation of the Claims

We evaluate the claims through a series of experiments on synthetic datasets and different sources of miscalibration. Our methodology is as follows:

1. **Data Generation and Model Building:** We generate the data from a known true function with Gaussian or non-Gaussian noise. We then build multiple feedforward BNN models using:
   - different network architectures
   - several priors on the weights, depending on model complexity
   - different variance of the Gaussian noise in the likelihood function

2. **Inference**: We obtain the posterior of the model by:

   - sampling from it with the No-U-Turn algorithm
   - approximating the posterior using Variational Inference with reparametrization and isotropic Gaussians

   We check for convergence using trace plots, the effective sample size, and Gelman-Rubin tests. In the case of variational inference, we track the ELBO during optimization.

3. **Recalibration**: Finally, we apply the proposed recalibration algorithm to the obtained model. We then visually compare the posterior predictives before and after calibration to the true distribution of the data. This allows us to identify scenarios where the algorithm works well and the cases of failure.

See the full version of the [project report](https://raw.githubusercontent.com/0-one/AM207_Project_Deep_Learning_Uncertainties_Calibration/master/report/report.slides.pdf) for the summary of findings and conclusions.

## Reproducing the Results

The final report depends on the following Python data science stack: 

- NumPy
- SciPy
- pandas
- Dask
- scikit-learn
- matplotlib
- Jupyter Notebook

It also requires the probabilistic library [NumPyro](https://github.com/pyro-ppl/numpyro) (based on [JAX](https://github.com/google/jax)), which provides fast implementations of sampling and variational inference algorithms.

Use the provided conda environment specification to satisfy all dependencies:

```shell
$ conda env create -f report/environment.yml
$ conda activate am207
```

Intermediate experiments preceding the final report also make use of PyMC3 and [autograd](https://github.com/HIPS/autograd). These are considerably slower to run and are optional to reproduce the project results.

## Repository Structure

| Directory   | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| calibration | Initial implementation of the calibration algorithm and metrics |
| experiments | Sources of miscalibration in Bayesian neural networks        |
| report      | The final report together with all the code in the corresponding `./code` subfolder |
| slides      | Intermediate meetings in the course of the project           |

We use Jupyter Notebooks `*.ipynb` throughout the repository. The `./report/code` folder is the only one that contains the original source code in `*.py` format. All of the remaining `*.py` files can be ignored: [Jupytext](https://github.com/mwouts/jupytext) produced those from the corresponding notebooks to enable clear commit diffs for the project team.


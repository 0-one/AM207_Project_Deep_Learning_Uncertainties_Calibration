import numpy as np


def calibration_error(predicted_quantiles, levels=10):
    """Compute the calibration error

    The result is sensitive to binning.
    Will use the supplied number of equally spaced quantiles.

    Args:
        predicted_quantiles: a 1-dimensional array.
            The values of the quantiles for each Y in the dataset.
            Equivalent to H(x_t)(y_t) in the paper's notation.
        levels: the number of quantiles in the 0-1 range (default: {10})

    Returns:
        cal_error: the calibration error
    """

    # Assume equally spaced quantiles
    expected_quantiles = np.linspace(0, 1, num=levels + 1)

    # Compute the probabilities of predicted quantiles at the discrete quantile levels
    T = predicted_quantiles.shape[0]
    empirical_quantiles = (
        predicted_quantiles.reshape(1, -1) <= expected_quantiles.reshape(-1, 1)
    ).sum(axis=1) / T

    # Compute the calibration error
    cal_error = np.sum((empirical_quantiles - expected_quantiles) ** 2)

    return cal_error

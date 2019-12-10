import numpy as np
from sklearn.isotonic import IsotonicRegression


def calculate_quantiles(samples, y):
    """ Function to compute quantiles of y within the given samples.

    For efficiency, this function does not distinguish even and odd
    number of samples, N. This should not really be an issue for large
    N.

    Args:
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

    Args:
        y: array of shape (T,) or (T, 1)
        post_pred: the posterior predictive, array of shape (N, T).

    Returns:
        cal_y: shape (T,)
        cal_X: shape (T,)
    """
    y = np.asarray(y).reshape(-1, 1)
    T = y.shape[0]

    # compute quantiles of y observation
    # quant_y.shape = (T,)
    quant_y = calculate_quantiles(post_pred.T, y)

    # p_hat.shape = (T,)
    p_hat = calculate_quantiles(quant_y.reshape(-1, T), quant_y.reshape(T, -1))

    return quant_y, p_hat


class QuantileCalibration:
    """Quantile calibration based on Kuleshov et al. (2018):
    https://arxiv.org/abs/1807.00263

    Learns the relationship between predicted and empirical quantiles of the
    posterior predictive based on observations using isotonic regression.
    """

    def __init__(self):
        self.isotonic = None
        self.isotonic_inverse = None

    def fit(self, y, post_pred):
        """Train isotonic regression on predicted and empirical quantiles

        Constructs a recalibration dataset from the posterior predictive and
        observations of the response variable Y. Learns the inverse relationship
        between the two using isotonic regression.

        Args:
            y: the response variable, array of shape (T,) or (T, 1)
            post_pred: samples of the posterior predictive, array of shape (N, T)

        Returns:
            self: a fitted instance of the QuantileCalibration class
        """

        assert y.shape[0] == post_pred.shape[1], "y.shape[0] must match post_pred.shape[1]"

        # Build a recalibration dataset
        predicted, empirical = make_cal_dataset(y, post_pred)

        # Fit the recalibration dataset in forward mode: from predicted to empirical
        self.isotonic = IsotonicRegression(out_of_bounds="clip")
        self.isotonic.fit(predicted, empirical)

        # Fit the recalibration dataset in reverse: from empirical to predicted
        self.isotonic_inverse = IsotonicRegression(out_of_bounds="clip")
        self.isotonic_inverse.fit(empirical, predicted)

        return self

    def transform(self, quantiles):
        """Forward transform the values of the predicted quantiles to the
        empirical quantiles using a previously learned relationship.

        Args:
            quantiles: a 1-dimensional array

        Returns:
            empirical_quantiles: the values of the empirical quantiles corresponding
            to the predicted quantiles in the posterior predictive,
            a 1-dimensional array
        """
        assert self.isotonic is not None, "The calibration instance must be fit first"
        empirical_quantiles = self.isotonic.transform(quantiles)
        return empirical_quantiles

    def inverse_transform(self, quantiles):
        """Inverse transform the values of the desired (empirical) quantiles to the
        predicted quantiles using a previously learned relationship.

        Args:
            quantiles: a 1-dimensional array

        Returns:
            predicted_quantiles: the values of the predicted quantiles corresponding
            to the desired quantiles in the posterior predictive,
            a 1-dimensional array
        """
        assert self.isotonic_inverse is not None, "The calibration instance must be fit first"
        predicted_quantiles = self.isotonic_inverse.transform(quantiles)
        return predicted_quantiles
